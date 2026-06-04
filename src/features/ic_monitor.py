"""
滞后兑现IC监控器 - Realized IC Monitor

核心理念：IC需要未来收益，在调仓日D只能更新那些signal_date+horizon已经兑现的数据。

调仓日D的逻辑：
1. 找到已兑现信号日: signal_date + label_horizon + trade_delay <= D
2. 用已兑现信号日计算RankIC
3. 更新FactorResponseMonitor状态机
4. 输出健康快照到artifact

使用方式:
    from src.features.ic_monitor import RealizedICMonitor
    
    monitor = RealizedICMonitor(
        factor_names=['earnings_yield', 'roe'],
        label_horizon=20,
        trade_delay=1,
    )
    
    for rebalance_date in rebalance_dates:
        realized_ic = monitor.calc_realized_ic(
            feature_panel,      # 因子值面板
            label_panel,        # 收益标签面板
            current_date=rebalance_date,
        )
        monitor.update(rebalance_date, realized_ic)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class RealizedICResult:
    """单次滞后兑现IC计算结果"""
    trade_date: pd.Timestamp
    signal_date: pd.Timestamp
    factor_name: str
    ic: float
    rank_ic: float
    n_samples: int


class RealizedICCalculator:
    """
    滞后兑现IC计算器
    
    核心约束：只能使用已经兑现的数据
    realized_date = signal_date + horizon + trade_delay <= current_date
    """

    def __init__(self, horizon: int = 20, trade_delay: int = 1, label_name: str = 'fwd_return_20d'):
        self.horizon = horizon
        self.trade_delay = trade_delay
        self.label_name = label_name

    def get_realized_signal_dates(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> list[pd.Timestamp]:
        """
        获取当前可兑现的信号日列表
        
        核心理念：调仓日D能用的最新信号日S，需要满足 S + horizon + delay 个交易日已经过去
        
        实现方式：基于label_panel中每行标签对应的实际可观测日期判断
        label_panel应包含: signal_date(信号日), label_end_date(收益兑现日)
        
        Args:
            feature_panel: 因子面板，必须有trade_date和symbol列
            label_panel: 标签面板，必须有signal_date和可选的label_end_date列
            current_date: 当前调仓日
        
        Returns:
            可以计算IC的信号日列表
        """
        # 方案1: 使用label_end_date列
        if 'label_end_date' in label_panel.columns:
            return self._get_by_label_end_date(
                feature_panel, label_panel, current_date
            )

        # 方案2: 基于交易日序列位置判断
        return self._get_by_trading_day_index(
            feature_panel, label_panel, current_date
        )

    def _get_by_label_end_date(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> list[pd.Timestamp]:
        """使用label_end_date判断"""
        # 只取 label_end_date <= current_date 的信号
        realized = label_panel[label_panel['label_end_date'] <= current_date]

        available_signal_dates = realized['signal_date'].unique()
        return sorted(available_signal_dates)

    def _get_by_trading_day_index(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> list[pd.Timestamp]:
        """
        基于交易日序列位置判断已兑现
        
        逻辑：
        1. 获取所有交易日的排序索引
        2. 当前日期对应索引 current_idx
        3. 信号日对应索引 signal_idx
        4. 需要 signal_idx + horizon + trade_delay <= current_idx
        """
        # 获取所有交易日序列（按日期排序）
        all_dates = sorted(feature_panel['trade_date'].unique())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        if current_date not in date_to_idx:
            # 找最近的交易日
            valid_dates = [d for d in all_dates if d <= current_date]
            if not valid_dates:
                return []
            current_idx = date_to_idx.get(valid_dates[-1], 0)
        else:
            current_idx = date_to_idx[current_date]

        # 计算需要信号日索引 <= current_idx - (horizon + delay)
        required_offset = self.horizon + self.trade_delay
        max_signal_idx = current_idx - required_offset

        if max_signal_idx < 0:
            return []

        # 选取所有满足条件的信号日
        available_signal_dates = [
            d for d in all_dates
            if date_to_idx[d] <= max_signal_idx
        ]

        return sorted(available_signal_dates)

    def calc_realized_ic(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        current_date: pd.Timestamp,
        factor_names: list[str],
        min_samples: int = 50,
    ) -> dict[str, RealizedICResult]:
        """
        计算滞后兑现IC
        
        Args:
            feature_panel: 因子面板 [trade_date, symbol, factor1, factor2, ...]
            label_panel: 标签面板 [trade_date, symbol, fwd_return_20d]
            注意: label_panel的trade_date是信号日(对应forward_return的起始日)
            current_date: 当前调仓日
            factor_names: 要计算的因子列表
            min_samples: 最小样本数
        
        Returns:
            {factor_name: RealizedICResult}
        """
        realized_dates = self.get_realized_signal_dates(
            feature_panel, label_panel, current_date
        )

        if not realized_dates:
            return {}

        # 取最近的信号日进行计算
        signal_date = realized_dates[-1]

        # 获取该信号日的因子值
        factor_data = feature_panel[
            feature_panel['trade_date'] == signal_date
        ][['trade_date', 'symbol'] + [f for f in factor_names if f in feature_panel.columns]]

        # 获取该信号日对应的forward_return
        # label_panel的trade_date就是信号日
        label_cols = ['trade_date', 'symbol', self.label_name]
        available_cols = [c for c in label_cols if c in label_panel.columns]
        label_data = label_panel[
            label_panel['trade_date'] == signal_date
        ][available_cols]

        # 合并
        merged = factor_data.merge(
            label_data,
            on=['trade_date', 'symbol'],
            how='inner'
        )

        results = {}

        for factor_name in factor_names:
            if factor_name not in merged.columns:
                continue

            valid = merged[[self.label_name, factor_name]].dropna()

            if len(valid) < min_samples:
                continue

            if valid[factor_name].std() < 1e-10:
                continue

            # IC (Pearson)
            ic = valid[self.label_name].corr(valid[factor_name])

            # RankIC (Spearman)
            rank_ic, _ = spearmanr(valid[self.label_name], valid[factor_name])

            if not np.isnan(rank_ic):
                results[factor_name] = RealizedICResult(
                    trade_date=current_date,
                    signal_date=signal_date,
                    factor_name=factor_name,
                    ic=ic if not np.isnan(ic) else 0.0,
                    rank_ic=rank_ic,
                    n_samples=len(valid),
                )

        return results


class RealizedICMonitor:
    """
    滞后兑现IC监控器
    
    整合RealizedICCalculator和FactorResponseMonitor
    """

    def __init__(
        self,
        factor_names: list[str],
        label_horizon: int = 20,
        trade_delay: int = 1,
        min_realized_samples: int = 50,
        label_name: str = 'fwd_return_20d',
    ):
        from src.features.factor_response import FactorResponseMonitor

        self.factor_names = factor_names
        self.label_horizon = label_horizon
        self.trade_delay = trade_delay
        self.label_name = label_name

        self.calculator = RealizedICCalculator(
            horizon=label_horizon,
            trade_delay=trade_delay,
            label_name=label_name,
        )

        self.response_monitor = FactorResponseMonitor(
            factor_names=factor_names,
            ic_threshold_watch=0.0,
            ic_threshold_offline=-0.03,
            consecutive_days_watch=5,
            consecutive_days_reduce=10,
            consecutive_days_offline=15,
            consecutive_days_recover=20,
        )

        self._ic_results: list[RealizedICResult] = []
        self._last_signal_date: dict[str, pd.Timestamp] = {}
        self._processed_keys: set[tuple] = set()  # {(factor, signal_date)}
        self.min_realized_samples = min_realized_samples

    def update(
        self,
        rebalance_date: pd.Timestamp,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        更新监控器
        
        Args:
            rebalance_date: 当前调仓日
            feature_panel: 因子面板
            label_panel: 收益标签面板
        
        Returns:
            RealizedICResult列表
        """
        # 计算滞后兑现IC
        ic_results = self.calculator.calc_realized_ic(
            feature_panel=feature_panel,
            label_panel=label_panel,
            current_date=rebalance_date,
            factor_names=self.factor_names,
            min_samples=self.min_realized_samples,
        )

        if not ic_results:
            return pd.DataFrame()

        # 去重：跳过已处理的 (factor, signal_date)
        new_results = {}
        skipped_count = 0
        for factor_name, result in ic_results.items():
            key = (factor_name, result.signal_date)
            if key in self._processed_keys:
                skipped_count += 1
                continue
            self._processed_keys.add(key)
            new_results[factor_name] = result

        # 更新响应监控器（只传入新结果）
        if new_results:
            ic_data = {k: v.rank_ic for k, v in new_results.items()}
            self.response_monitor.update(rebalance_date, ic_data)

            # 记录新结果
            self._ic_results.extend(new_results.values())

        # 记录信号日（用于信息）
        for factor_name, result in new_results.items():
            self._last_signal_date[factor_name] = result.signal_date

        return pd.DataFrame([
            {
                'rebalance_date': r.trade_date,
                'signal_date': r.signal_date,
                'factor': r.factor_name,
                'ic': r.ic,
                'rank_ic': r.rank_ic,
                'n_samples': r.n_samples,
            }
            for r in new_results.values()
        ])

    def get_weights(self) -> dict[str, float]:
        """获取当前因子权重"""
        return self.response_monitor.get_weights()

    def get_health_snapshot(self) -> pd.DataFrame:
        """获取健康快照"""
        return self.response_monitor.get_health_snapshot()

    def get_records(self) -> pd.DataFrame:
        """获取响应记录"""
        return self.response_monitor.get_records()

    def get_ic_history(self) -> pd.DataFrame:
        """获取IC历史"""
        if not self._ic_results:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'rebalance_date': r.trade_date,
                'signal_date': r.signal_date,
                'factor': r.factor_name,
                'ic': r.ic,
                'rank_ic': r.rank_ic,
                'n_samples': r.n_samples,
            }
            for r in self._ic_results
        ])

    def get_offline_factors(self) -> list[str]:
        """获取已下线的因子"""
        snapshot = self.get_health_snapshot()
        if snapshot.empty:
            return []
        return snapshot[snapshot['status'] == 'offline']['factor_name'].tolist()

    def get_action_summary(self) -> pd.DataFrame:
        """获取需要采取的行动"""
        return self.response_monitor.get_action_summary()

    def export_for_artifact(self) -> dict:
        """导出用于artifact"""
        return {
            'ic_history': self.get_ic_history(),
            'health_snapshot': self.get_health_snapshot(),
            'weights': self.get_weights(),
            'records': self.get_records(),
            'action_summary': self.get_action_summary(),
            'offline_factors': self.get_offline_factors(),
            'config': {
                'label_horizon': self.label_horizon,
                'trade_delay': self.trade_delay,
                'factor_names': self.factor_names,
            },
        }


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='B')
    symbols = [f'00000{i}.SZ' for i in range(10)]

    # 模拟因子面板
    feature_panel = pd.DataFrame([
        {'trade_date': d, 'symbol': s, 'earnings_yield': np.random.randn(), 'roe': np.random.randn()}
        for d in dates
        for s in symbols
    ])

    # 模拟标签面板
    label_panel = pd.DataFrame([
        {'trade_date': d, 'symbol': s, 'fwd_return_20d': np.random.randn() * 0.05}
        for d in dates
        for s in symbols
    ])

    monitor = RealizedICMonitor(
        factor_names=['earnings_yield', 'roe'],
        label_horizon=20,
        trade_delay=1,
    )

    # 模拟调仓日
    rebalance_dates = dates[30::10]

    for rebal_date in rebalance_dates[:5]:
        print(f"\n调仓日: {rebal_date}")
        ic_df = monitor.update(rebal_date, feature_panel, label_panel)
        if not ic_df.empty:
            print(ic_df.to_string(index=False))
        else:
            print("无可用兑现数据")

    print("\n\n健康快照:")
    print(monitor.get_health_snapshot())

    print("\n\n权重:")
    print(monitor.get_weights())
