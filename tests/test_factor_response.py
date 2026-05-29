"""
测试 factor_response 模块 - 因子响应状态机

测试内容：
1. REDUCED不得1次正IC直接active
2. OFFLINE不得自动恢复
3. WATCH状态下需连续正IC才能恢复
4. 状态机慢速变化原则
"""
import pytest
import pandas as pd
import numpy as np
from src.features.factor_response import (
    FactorResponseMonitor,
    FactorStatus,
)


def create_monitor(factor_names=None, **kwargs):
    """创建测试用监控器"""
    if factor_names is None:
        factor_names = ['factor_a']
    return FactorResponseMonitor(
        factor_names=factor_names,
        ic_threshold_watch=0.0,
        ic_threshold_offline=-0.03,
        consecutive_days_watch=5,
        consecutive_days_reduce=10,
        consecutive_days_offline=15,
        consecutive_days_recover=20,
        **kwargs
    )


class TestStateTransitionRules:
    """测试状态转换规则"""

    def test_immediate_offline_on_strong_negative_ic(self):
        """测试：IC < -0.03 立即下线"""
        monitor = create_monitor(['factor_a'])

        monitor.update(pd.Timestamp('2024-01-01'), {'factor_a': -0.05})

        snapshot = monitor.get_health_snapshot()
        assert snapshot[snapshot['factor_name'] == 'factor_a']['status'].values[0] == FactorStatus.OFFLINE.value
        assert monitor.get_weights()['factor_a'] == 0.0

    def test_consecutive_negative_5_goes_to_watch(self):
        """测试：连续5次IC<0进入观察"""
        monitor = create_monitor(['factor_a'])

        for i in range(5):
            monitor.update(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i), {'factor_a': -0.01})

        snapshot = monitor.get_health_snapshot()
        assert snapshot[snapshot['factor_name'] == 'factor_a']['status'].values[0] == FactorStatus.WATCH.value

    def test_consecutive_negative_10_goes_to_reduced(self):
        """测试：连续10次IC<0降权50%"""
        monitor = create_monitor(['factor_a'])

        for i in range(10):
            monitor.update(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i), {'factor_a': -0.01})

        snapshot = monitor.get_health_snapshot()
        assert snapshot[snapshot['factor_name'] == 'factor_a']['status'].values[0] == FactorStatus.REDUCED.value
        assert monitor.get_weights()['factor_a'] == 0.5


class TestNoSkipRecovery:
    """测试：不能跳级恢复"""

    def test_reduced_cannot_recover_with_single_positive_ic(self):
        """测试：REDUCED状态不能1次正IC就恢复ACTIVE"""
        monitor = create_monitor(['factor_a'])

        for i in range(10):
            monitor.update(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i), {'factor_a': -0.01})

        assert monitor.get_weights()['factor_a'] == 0.5

        monitor.update(pd.Timestamp('2024-01-11'), {'factor_a': 0.01})

        snapshot = monitor.get_health_snapshot()
        assert snapshot[snapshot['factor_name'] == 'factor_a']['status'].values[0] == FactorStatus.REDUCED.value
        assert monitor.get_weights()['factor_a'] == 0.5, "REDUCED状态不能单次正IC恢复"

    def test_reduced_stays_reduced_until_offline_or_full_recovery(self):
        """测试：REDUCED保持降权状态直到满足恢复条件"""
        monitor = create_monitor(['factor_a'])

        for i in range(10):
            monitor.update(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i), {'factor_a': -0.01})

        for i in range(19):
            monitor.update(pd.Timestamp('2024-01-11') + pd.Timedelta(days=i), {'factor_a': 0.01})

        snapshot = monitor.get_health_snapshot()
        assert snapshot[snapshot['factor_name'] == 'factor_a']['status'].values[0] == FactorStatus.REDUCED.value

    def test_offline_cannot_autorecover(self):
        """测试：OFFLINE状态不能自动恢复"""
        monitor = create_monitor(['factor_a'])

        for i in range(15):
            monitor.update(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i), {'factor_a': -0.01})

        assert monitor.get_weights()['factor_a'] == 0.0

        for i in range(30):
            monitor.update(pd.Timestamp('2024-01-16') + pd.Timedelta(days=i), {'factor_a': 0.01})

        snapshot = monitor.get_health_snapshot()
        status = snapshot[snapshot['factor_name'] == 'factor_a']['status'].values[0]
        assert status in [FactorStatus.OFFLINE.value, FactorStatus.RESERVE.value], \
            "OFFLINE状态不能自动恢复到ACTIVE"
        assert monitor.get_weights()['factor_a'] == 0.0, "OFFLINE因子权重应为0"

    def test_watch_needs_20_consecutive_positive_to_recover(self):
        """测试：WATCH状态需要20次连续正IC才能恢复ACTIVE"""
        monitor = create_monitor(['factor_a'])

        for i in range(5):
            monitor.update(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i), {'factor_a': -0.01})

        assert monitor.get_weights()['factor_a'] == 1.0

        for i in range(19):
            monitor.update(pd.Timestamp('2024-01-06') + pd.Timedelta(days=i), {'factor_a': 0.01})

        snapshot = monitor.get_health_snapshot()
        assert snapshot[snapshot['factor_name'] == 'factor_a']['status'].values[0] == FactorStatus.WATCH.value, \
            "WATCH需要20次正IC才能恢复，19次不够"

        monitor.update(pd.Timestamp('2024-01-25'), {'factor_a': 0.01})

        snapshot = monitor.get_health_snapshot()
        assert snapshot[snapshot['factor_name'] == 'factor_a']['status'].values[0] == FactorStatus.ACTIVE.value
        assert monitor.get_weights()['factor_a'] == 1.0


class TestActiveStateMaintenance:
    """测试ACTIVE状态维护"""

    def test_active_stays_active_with_intermittent_negative(self):
        """测试：ACTIVE状态遇到间歇性负IC仍保持ACTIVE"""
        monitor = create_monitor(['factor_a'])

        for i in range(20):
            ic = 0.01 if i % 3 != 0 else -0.01
            monitor.update(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i), {'factor_a': ic})

        snapshot = monitor.get_health_snapshot()
        assert snapshot[snapshot['factor_name'] == 'factor_a']['status'].values[0] == FactorStatus.ACTIVE.value

    def test_active_with_continuous_positive_stays_active(self):
        """测试：持续正IC的因子保持ACTIVE"""
        monitor = create_monitor(['factor_a'])

        for i in range(30):
            monitor.update(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i), {'factor_a': 0.01})

        snapshot = monitor.get_health_snapshot()
        assert snapshot[snapshot['factor_name'] == 'factor_a']['status'].values[0] == FactorStatus.ACTIVE.value
        assert monitor.get_weights()['factor_a'] == 1.0


class TestConsecutiveCounting:
    """测试连续计数逻辑"""

    def test_consecutive_negative_resets_on_positive(self):
        """测试：正IC后连续负计数重置"""
        monitor = create_monitor(['factor_a'])

        for i in range(3):
            monitor.update(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i), {'factor_a': -0.01})

        assert monitor._consecutive_negative['factor_a'] == 3

        monitor.update(pd.Timestamp('2024-01-04'), {'factor_a': 0.01})

        assert monitor._consecutive_negative['factor_a'] == 0
        assert monitor._consecutive_positive['factor_a'] == 1

    def test_consecutive_positive_resets_on_negative(self):
        """测试：负IC后连续正计数重置"""
        monitor = create_monitor(['factor_a'])

        monitor.update(pd.Timestamp('2024-01-01'), {'factor_a': 0.01})
        monitor.update(pd.Timestamp('2024-01-02'), {'factor_a': 0.01})
        monitor.update(pd.Timestamp('2024-01-03'), {'factor_a': -0.01})

        assert monitor._consecutive_positive['factor_a'] == 0
        assert monitor._consecutive_negative['factor_a'] == 1


class TestMultipleFactors:
    """测试多因子场景"""

    def test_different_factors_have_independent_states(self):
        """测试：不同因子状态独立"""
        monitor = create_monitor(['factor_a', 'factor_b'])

        for i in range(15):
            monitor.update(
                pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
                {'factor_a': 0.01, 'factor_b': -0.01}
            )

        snapshot = monitor.get_health_snapshot()
        a_status = snapshot[snapshot['factor_name'] == 'factor_a']['status'].values[0]
        b_status = snapshot[snapshot['factor_name'] == 'factor_b']['status'].values[0]

        assert a_status == FactorStatus.ACTIVE.value
        assert b_status == FactorStatus.OFFLINE.value
        assert monitor.get_weights()['factor_a'] == 1.0
        assert monitor.get_weights()['factor_b'] == 0.0

    def test_weights_sum_reflects_active_factors(self):
        """测试：权重总和反映活跃因子数量"""
        monitor = create_monitor(['factor_a', 'factor_b', 'factor_c'])

        monitor.update(pd.Timestamp('2024-01-01'), {'factor_a': 0.01, 'factor_b': -0.05, 'factor_c': 0.01})

        weights = monitor.get_weights()
        total_weight = sum(weights.values())
        assert total_weight == 2.0, "factor_b下线后，总权重应为2.0"


class TestExportAndRecords:
    """测试导出和记录功能"""

    def test_records_dataframe_structure(self):
        """测试：记录DataFrame结构正确"""
        monitor = create_monitor(['factor_a'])

        for i in range(3):
            monitor.update(pd.Timestamp('2024-01-01') + pd.Timedelta(days=i), {'factor_a': 0.01})

        records = monitor.get_records()
        assert not records.empty
        assert 'trade_date' in records.columns
        assert 'factor_name' in records.columns
        assert 'ic' in records.columns
        assert 'status' in records.columns
        assert 'action' in records.columns
        assert len(records) == 3

    def test_export_config_contains_required_fields(self):
        """测试：导出配置包含必要字段"""
        monitor = create_monitor(['factor_a'])

        config = monitor.export_config()
        assert 'monitor_config' in config
        assert 'current_status' in config
        assert 'current_weights' in config
        assert config['monitor_config']['ic_threshold_offline'] == -0.03
