"""
Walk-Forward 真实滚动验证

每个窗口:
1. 用过去数据训练模型
2. 在下一窗口预测
3. 基于预测建仓
4. 统计窗口真实收益

Usage:
    from src.walk_forward_validator import WalkForwardValidator
    
    validator = WalkForwardValidator(
        train_window_days=500,
        test_window_days=60,
        step_days=30,
    )
    result = validator.run(data, factor_cols, label_col='fwd_return_20d')
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    """Walk-Forward配置"""
    train_window_days: int = 500
    test_window_days: int = 60
    step_days: int = 30
    min_train_samples: int = 100
    n_jobs: int = 1

    # 成本参数
    commission_bps: float = 0.75
    stamp_duty_bps: float = 10.0
    slippage_bps: float = 5.0
    min_trade_value: float = 2000.0

    # 组合参数
    top_n: int = 15
    rebalance_days: int = 10


@dataclass
class WindowResult:
    """单窗口结果"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # IC指标
    rank_ic: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_ic_ir: float

    # 组合收益
    portfolio_return: float
    long_return: float
    short_return: float
    long_short_return: float

    # 成本
    turnover: float
    estimated_cost: float

    # 样本数
    train_samples: int
    test_samples: int

    # 市场状态
    market_return: float
    is_bull: bool = False
    is_bear: bool = False


@dataclass
class WalkForwardReport:
    """Walk-Forward完整报告"""
    results: list[WindowResult] = field(default_factory=list)

    # 聚合指标
    mean_return: float = 0.0
    std_return: float = 0.0
    sharpe: float = 0.0
    worst_return: float = 0.0
    best_return: float = 0.0

    # IC聚合
    mean_rank_ic: float = 0.0
    mean_rank_ic_ir: float = 0.0

    # 成本聚合
    mean_turnover: float = 0.0
    total_estimated_cost: float = 0.0

    # Regime分析
    bull_return: float = 0.0
    bear_return: float = 0.0
    bull_count: int = 0
    bear_count: int = 0

    # IS/OOS比
    is_oos_ratio: float = 0.0

    def to_summary(self) -> dict:
        """转dict摘要"""
        return {
            'OOS Sharpe': f"{self.sharpe:.3f}",
            'OOS Mean Return': f"{self.mean_return:.2%}",
            'Worst Window': f"{self.worst_return:.2%}",
            'Best Window': f"{self.best_return:.2%}",
            'Mean Rank IC': f"{self.mean_rank_ic:.4f}",
            'Mean IC IR': f"{self.mean_rank_ic_ir:.3f}",
            'Mean Turnover': f"{self.mean_turnover:.2%}",
            'Bull Return': f"{self.bull_return:.2%}" if self.bull_count > 0 else "N/A",
            'Bear Return': f"{self.bear_return:.2%}" if self.bear_count > 0 else "N/A",
            'IS/OOS Ratio': f"{self.is_oos_ratio:.1f}",
        }

    def to_dataframe(self) -> pd.DataFrame:
        """转DataFrame"""
        rows = []
        for r in self.results:
            rows.append({
                'train_end': r.train_end,
                'test_start': r.test_start,
                'test_end': r.test_end,
                'rank_ic': r.rank_ic,
                'portfolio_return': r.portfolio_return,
                'long_short_return': r.long_short_return,
                'turnover': r.turnover,
                'market_return': r.market_return,
                'is_bull': r.is_bull,
                'is_bear': r.is_bear,
            })
        return pd.DataFrame(rows)


class WalkForwardValidator:
    """
    Walk-Forward真实滚动验证器
    
    不使用随机数，每个窗口真实训练→预测→建仓→统计
    """

    def __init__(self, config: WalkForwardConfig | None = None):
        self.config = config or WalkForwardConfig()
        self.results: list[WindowResult] = []

    def run(
        self,
        data: pd.DataFrame,
        factor_cols: list[str],
        label_col: str,
        date_col: str = 'trade_date',
        symbol_col: str = 'symbol',
        price_col: str = 'close',
        model_factory: Callable | None = None,
    ) -> WalkForwardReport:
        """
        运行Walk-Forward验证
        
        Args:
            data: 包含因子、标签、价格的数据
            factor_cols: 因子列名列表
            label_col: 标签列名
            date_col: 日期列名
            symbol_col: 股票代码列名
            price_col: 价格列名
            model_factory: 模型工厂函数，默认使用简单平均
        
        Returns:
            WalkForwardReport: 验证报告
        """
        config = self.config

        # 准备数据
        data = data.copy()
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values([date_col, symbol_col])

        # 获取所有交易日
        dates = sorted(data[date_col].unique())

        # 生成窗口
        windows = self._generate_windows(dates)

        print(f"Walk-Forward: {len(windows)} 个窗口")
        print(f"  Train: {config.train_window_days}d, Test: {config.test_window_days}d, Step: {config.step_days}d")

        self.results = []

        for i, (train_end, test_start, test_end) in enumerate(windows):
            # 分割数据
            train_data = data[data[date_col] <= train_end].copy()
            test_data = data[(data[date_col] >= test_start) & (data[date_col] <= test_end)].copy()

            if len(train_data) < config.min_train_samples:
                continue

            # 训练模型
            model = self._train_model(train_data, factor_cols, label_col)

            # 在测试集预测
            predictions = self._predict(test_data, factor_cols, model)

            if predictions is None or len(predictions) == 0:
                continue

            # 计算IC
            ic_result = self._calculate_ic(predictions, label_col)

            # 构建组合并计算收益
            portfolio_result = self._build_portfolio(
                predictions, test_data, factor_cols, label_col,
                date_col, symbol_col, price_col
            )

            # 确定市场状态
            market_return = test_data.groupby(date_col)[price_col].last().pct_change().sum()

            window_result = WindowResult(
                train_start=train_data[date_col].min(),
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                rank_ic=ic_result['rank_ic'],
                rank_ic_mean=ic_result['rank_ic_mean'],
                rank_ic_std=ic_result['rank_ic_std'],
                rank_ic_ir=ic_result['rank_ic_ir'],
                portfolio_return=portfolio_result['portfolio_return'],
                long_return=portfolio_result['long_return'],
                short_return=portfolio_result['short_return'],
                long_short_return=portfolio_result['long_short_return'],
                turnover=portfolio_result['turnover'],
                estimated_cost=portfolio_result['estimated_cost'],
                train_samples=len(train_data),
                test_samples=len(test_data),
                market_return=market_return,
                is_bull=market_return > 0,
                is_bear=market_return < -0.05,
            )

            self.results.append(window_result)

            if (i + 1) % 5 == 0:
                print(f"  Window {i+1}/{len(windows)}: IC={ic_result['rank_ic']:.4f}, Return={portfolio_result['portfolio_return']:.2%}")

        # 生成报告
        report = self._generate_report()

        return report

    def _generate_windows(self, dates: list) -> list[tuple]:
        """生成训练/测试窗口"""
        config = self.config

        windows = []
        train_end_idx = config.train_window_days - 1

        while train_end_idx < len(dates) - config.test_window_days:
            train_end = dates[train_end_idx]
            test_start = dates[train_end_idx + 1]
            test_end = dates[train_end_idx + config.test_window_days]

            windows.append((train_end, test_start, test_end))

            train_end_idx += config.step_days

        return windows

    def _train_model(
        self,
        train_data: pd.DataFrame,
        factor_cols: list[str],
        label_col: str,
    ) -> dict:
        """训练模型"""
        # 简单平均模型
        # 每个因子横截面rank标准化后等权平均

        model_scores = []

        for col in factor_cols:
            if col in train_data.columns:
                # 横截面rank
                ranked = train_data.groupby('trade_date')[col].rank(pct=True)
                model_scores.append(ranked)

        if model_scores:
            composite = pd.concat(model_scores, axis=1).mean(axis=1)
            train_data = train_data.copy()
            train_data['model_score'] = composite.values

        return {'method': 'simple_average', 'factors': factor_cols}

    def _predict(
        self,
        test_data: pd.DataFrame,
        factor_cols: list[str],
        model: dict,
    ) -> pd.DataFrame | None:
        """在测试集预测"""
        if model['method'] == 'simple_average':
            model_scores = []

            for col in factor_cols:
                if col in test_data.columns:
                    ranked = test_data.groupby('trade_date')[col].rank(pct=True)
                    model_scores.append(ranked)

            if model_scores:
                composite = pd.concat(model_scores, axis=1).mean(axis=1)
                test_data = test_data.copy()
                test_data['score'] = composite.values

                return test_data[['trade_date', 'symbol', 'score'] + factor_cols]

        return None

    def _calculate_ic(
        self,
        predictions: pd.DataFrame,
        label_col: str,
    ) -> dict:
        """计算IC"""
        config = self.config

        # 按日期计算截面rank IC
        daily_ics = []

        for date, group in predictions.groupby('trade_date'):
            if label_col in group.columns:
                score_rank = group['score'].rank(pct=True)
                label_rank = group[label_col].rank(pct=True)

                if score_rank.std() > 0 and label_rank.std() > 0:
                    ic = score_rank.corr(label_rank)
                    daily_ics.append({'date': date, 'ic': ic})

        if daily_ics:
            ic_df = pd.DataFrame(daily_ics)
            return {
                'rank_ic': ic_df['ic'].mean(),
                'rank_ic_mean': ic_df['ic'].mean(),
                'rank_ic_std': ic_df['ic'].std(),
                'rank_ic_ir': ic_df['ic'].mean() / max(ic_df['ic'].std(), 0.001),
            }

        return {'rank_ic': 0, 'rank_ic_mean': 0, 'rank_ic_std': 0, 'rank_ic_ir': 0}

    def _build_portfolio(
        self,
        predictions: pd.DataFrame,
        test_data: pd.DataFrame,
        factor_cols: list[str],
        label_col: str,
        date_col: str,
        symbol_col: str,
        price_col: str,
    ) -> dict:
        """构建组合并计算收益"""
        config = self.config

        # 简化：取最后一天的预测作为调仓日
        last_date = predictions[date_col].max()
        last_predictions = predictions[predictions[date_col] == last_date].copy()

        if len(last_predictions) < config.top_n:
            return {
                'portfolio_return': 0,
                'long_return': 0,
                'short_return': 0,
                'long_short_return': 0,
                'turnover': 0,
                'estimated_cost': 0,
            }

        # 按score排序选股
        last_predictions = last_predictions.sort_values('score', ascending=False)

        # 多空组合
        n_long = config.top_n // 2
        n_short = config.top_n // 2

        long_stocks = last_predictions.head(n_long)['symbol'].tolist()
        short_stocks = last_predictions.tail(n_short)['symbol'].tolist()

        # 获取这些股票在测试期间的真实收益
        test_data = test_data[test_data[symbol_col].isin(long_stocks + short_stocks)]

        # 按股票计算收益
        stock_returns = test_data.groupby(symbol_col).apply(
            lambda x: (x[price_col].iloc[-1] / x[price_col].iloc[0]) - 1 if len(x) > 1 else 0
        )

        long_returns = [stock_returns.get(s, 0) for s in long_stocks]
        short_returns = [stock_returns.get(s, 0) for s in short_stocks]

        long_return = np.mean(long_returns) if long_returns else 0
        short_return = np.mean(short_returns) if short_returns else 0

        # 组合收益 (等权多空)
        long_short_return = (long_return - short_return) / 2
        portfolio_return = long_return  # 纯多头

        # 估算换手和成本
        turnover = 1.0  # 简化假设100%换手
        notional = 1.0 / n_long
        commission = notional * (config.commission_bps / 10000)
        slippage = notional * (config.slippage_bps / 10000)
        stamp_duty = notional * (config.stamp_duty_bps / 10000)  # 卖方
        estimated_cost = (commission + slippage + stamp_duty) * 2 * n_long

        return {
            'portfolio_return': portfolio_return,
            'long_return': long_return,
            'short_return': short_return,
            'long_short_return': long_short_return,
            'turnover': turnover,
            'estimated_cost': estimated_cost,
        }

    def _generate_report(self) -> WalkForwardReport:
        """生成聚合报告"""
        if not self.results:
            return WalkForwardReport()

        returns = [r.portfolio_return for r in self.results]
        ics = [r.rank_ic for r in self.results]
        turnovers = [r.turnover for r in self.results]
        costs = [r.estimated_cost for r in self.results]

        # Regime分析
        bull_returns = [r.portfolio_return for r in self.results if r.is_bull]
        bear_returns = [r.portfolio_return for r in self.results if r.is_bear]

        config = self.config
        is_oos_ratio = config.train_window_days / config.test_window_days

        return WalkForwardReport(
            results=self.results,
            mean_return=np.mean(returns),
            std_return=np.std(returns),
            sharpe=np.mean(returns) / max(np.std(returns), 0.001) * np.sqrt(len(returns) / 12) if returns else 0,
            worst_return=min(returns) if returns else 0,
            best_return=max(returns) if returns else 0,
            mean_rank_ic=np.mean(ics) if ics else 0,
            mean_rank_ic_ir=np.mean([r.rank_ic_ir for r in self.results]) if self.results else 0,
            mean_turnover=np.mean(turnovers) if turnovers else 0,
            total_estimated_cost=sum(costs) if costs else 0,
            bull_return=np.mean(bull_returns) if bull_returns else 0,
            bear_return=np.mean(bear_returns) if bear_returns else 0,
            bull_count=len(bull_returns),
            bear_count=len(bear_returns),
            is_oos_ratio=is_oos_ratio,
        )


def run_walk_forward_experiment(
    data_path: str,
    factor_cols: list[str],
    label_col: str = 'fwd_return_20d',
) -> WalkForwardReport:
    """
    运行Walk-Forward实验的便捷函数
    """
    print(f"Loading data from {data_path}...")
    data = pd.read_parquet(data_path)

    config = WalkForwardConfig(
        train_window_days=500,
        test_window_days=60,
        step_days=30,
        top_n=15,
    )

    validator = WalkForwardValidator(config)
    report = validator.run(
        data=data,
        factor_cols=factor_cols,
        label_col=label_col,
    )

    print("\n" + "="*60)
    print("Walk-Forward Results")
    print("="*60)
    for k, v in report.to_summary().items():
        print(f"  {k}: {v}")

    return report
