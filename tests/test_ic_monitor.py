"""
测试 ic_monitor 模块 - 滞后兑现IC监控器

测试内容：
1. 未满horizon+delay交易日不得计算IC
2. 重复update不增加history
3. label_name参数传递
"""
import pytest
import pandas as pd
import numpy as np
from src.features.ic_monitor import RealizedICCalculator, RealizedICMonitor


def create_test_panels(n_stocks=100, n_dates=200, seed=42):
    """创建测试用面板数据"""
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
    symbols = [f'00000{i}.SZ' for i in range(n_stocks)]

    panel_data = []
    for d in dates:
        for s in symbols:
            earnings_yield = np.random.randn()
            fwd_return = np.random.randn() * 0.05 + 0.002 * earnings_yield
            panel_data.append({
                'trade_date': d,
                'symbol': s,
                'earnings_yield': earnings_yield,
                'roe': np.random.randn(),
                'fwd_return_20d': fwd_return
            })

    full_panel = pd.DataFrame(panel_data)
    feature_panel = full_panel[['trade_date', 'symbol', 'earnings_yield', 'roe']]
    label_panel = full_panel[['trade_date', 'symbol', 'fwd_return_20d']]

    return feature_panel, label_panel, dates


class TestRealizedICCalculator:
    """测试 RealizedICCalculator"""

    def test_label_name_parameter(self):
        """测试 label_name 参数是否正确传递"""
        calc = RealizedICCalculator(horizon=20, trade_delay=1, label_name='custom_return')
        assert calc.label_name == 'custom_return'

        calc_default = RealizedICCalculator()
        assert calc_default.label_name == 'fwd_return_20d'

    def test_realized_signal_dates_requires_minimum_lookback(self):
        """测试：未满horizon+delay交易日不得有可兑现信号"""
        feature_panel, label_panel, dates = create_test_panels()

        calc = RealizedICCalculator(horizon=20, trade_delay=1)

        early_date = dates[20]  # 只有20天数据，不够horizon+delay=21
        realized = calc.get_realized_signal_dates(feature_panel, label_panel, early_date)
        assert len(realized) == 0, "horizon+delay=21天，但数据只有20天，不应有可兑现信号"

        sufficient_date = dates[40]  # 40天数据，远超21天要求
        realized = calc.get_realized_signal_dates(feature_panel, label_panel, sufficient_date)
        assert len(realized) > 0, "数据充足，应有可兑现信号"

    def test_realized_dates_grow_with_time(self):
        """测试：随时间推移，可兑现信号日数量增加"""
        feature_panel, label_panel, dates = create_test_panels()

        calc = RealizedICCalculator(horizon=20, trade_delay=1)

        realized_counts = []
        for i in [30, 50, 70, 90]:
            date = dates[i]
            realized = calc.get_realized_signal_dates(feature_panel, label_panel, date)
            realized_counts.append(len(realized))

        for i in range(1, len(realized_counts)):
            assert realized_counts[i] >= realized_counts[i-1], \
                "随时间推移，可兑现信号日数量应单调增加"


class TestRealizedICMonitor:
    """测试 RealizedICMonitor"""

    def test_monitor_passes_label_name_to_calculator(self):
        """测试：Monitor正确传递label_name给Calculator"""
        monitor = RealizedICMonitor(
            factor_names=['earnings_yield'],
            label_horizon=20,
            label_name='custom_label'
        )
        assert monitor.label_name == 'custom_label'
        assert monitor.calculator.label_name == 'custom_label'

    def test_duplicate_update_does_not_increase_history(self):
        """测试：重复update不增加history"""
        feature_panel, label_panel, dates = create_test_panels()

        monitor = RealizedICMonitor(
            factor_names=['earnings_yield', 'roe'],
            label_horizon=20
        )

        rebal_date = dates[60]

        # 第一次update
        result1 = monitor.update(rebal_date, feature_panel, label_panel)
        initial_history_len = len(monitor.get_ic_history())
        initial_processed = len(monitor._processed_keys)

        # 第二次update（相同日期相同signal_date）
        result2 = monitor.update(rebal_date, feature_panel, label_panel)
        final_history_len = len(monitor.get_ic_history())
        final_processed = len(monitor._processed_keys)

        # 应该被去重，不增加history
        assert final_history_len == initial_history_len, "重复update不应增加history"
        assert final_processed == initial_processed, "processed_keys不应变化"

    def test_different_dates_add_to_history(self):
        """测试：不同日期的update会增加history"""
        feature_panel, label_panel, dates = create_test_panels()

        monitor = RealizedICMonitor(
            factor_names=['earnings_yield'],
            label_horizon=20
        )

        # 第一次update
        result1 = monitor.update(dates[60], feature_panel, label_panel)
        history_len_after_first = len(monitor.get_ic_history())

        # 第二次update（不同日期）
        result2 = monitor.update(dates[70], feature_panel, label_panel)
        history_len_after_second = len(monitor.get_ic_history())

        # 应该增加
        assert history_len_after_second > history_len_after_first, \
            "不同日期的update应增加history"

    def test_min_samples_threshold(self):
        """测试：样本数不足时不返回结果"""
        feature_panel, label_panel, dates = create_test_panels(n_stocks=10)

        calc = RealizedICCalculator(horizon=20, trade_delay=1, label_name='fwd_return_20d')

        rebal_date = dates[40]
        results = calc.calc_realized_ic(
            feature_panel, label_panel, rebal_date,
            ['earnings_yield'],
            min_samples=50  # 要求50个样本，但只有10个股票
        )

        assert len(results) == 0, "样本数不足时应返回空结果"

    def test_weights_initialized_to_one(self):
        """测试：权重初始化为1.0"""
        monitor = RealizedICMonitor(
            factor_names=['factor_a', 'factor_b'],
            label_horizon=20
        )

        weights = monitor.get_weights()
        assert weights['factor_a'] == 1.0
        assert weights['factor_b'] == 1.0

    def test_offline_factors_list_empty_initially(self):
        """测试：初始状态下无下线因子"""
        monitor = RealizedICMonitor(
            factor_names=['factor_a'],
            label_horizon=20
        )

        assert len(monitor.get_offline_factors()) == 0


class TestICCalculationCorrectness:
    """测试IC计算正确性"""

    def test_ic_with_strong_signal(self):
        """测试：强信号因子的IC应为正值"""
        np.random.seed(123)
        dates = pd.date_range('2024-01-01', periods=200, freq='B')
        symbols = [f'00000{i}.SZ' for i in range(100)]

        panel_data = []
        for d in dates:
            for s in symbols:
                earnings_yield = np.random.randn()
                fwd_return = np.random.randn() * 0.05 + 0.01 * earnings_yield  # 正相关
                panel_data.append({
                    'trade_date': d,
                    'symbol': s,
                    'earnings_yield': earnings_yield,
                    'fwd_return_20d': fwd_return
                })

        feature_panel = pd.DataFrame(panel_data)[['trade_date', 'symbol', 'earnings_yield']]
        label_panel = pd.DataFrame(panel_data)[['trade_date', 'symbol', 'fwd_return_20d']]

        monitor = RealizedICMonitor(factor_names=['earnings_yield'], label_horizon=20)

        rebal_date = dates[60]
        result_df = monitor.update(rebal_date, feature_panel, label_panel)

        if not result_df.empty:
            ic_mean = result_df['ic'].mean()
            assert ic_mean > 0, "正相关因子的IC均值应为正"
