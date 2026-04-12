"""
策略选优分析工具

提供策略比较所需的各种分析:
- 分年度表现
- 市场状态分析 (牛市/熊市/震荡)
- 参数敏感性分析
- 基线对照矩阵

使用方式:
    from src.evaluation.strategy_comparison import StrategyComparator
    
    comparator = StrategyComparator()
    comparator.add_strategy('ridge', nav_ridge, trades_ridge, rank_ic_ridge)
    comparator.add_strategy('baseline', nav_baseline, trades_baseline, rank_ic_baseline)
    
    result = comparator.compare()
    print(result.to_markdown())
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import asdict, dataclass, field
from typing import Optional
import json

from src.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class YearlyMetrics:
    """年度指标"""
    year: int
    total_return: float
    sharpe: float
    max_drawdown: float
    turnover: float
    num_trades: int
    ic_mean: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegimeMetrics:
    """市场状态指标"""
    regime: str
    days: int
    strategy_return: float
    benchmark_return: float
    excess_return: float
    sharpe: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SensitivityResult:
    """参数敏感性结果"""
    parameter: str
    values: list
    metric_name: str
    metric_values: list
    sensitivity_score: float  # 越高越敏感
    
    def to_dict(self) -> dict:
        return {
            'parameter': self.parameter,
            'values': self.values,
            'metric_name': self.metric_name,
            'metric_values': self.metric_values,
            'sensitivity_score': self.sensitivity_score,
            'interpretation': self._interpret(),
        }
    
    def _interpret(self) -> str:
        if self.sensitivity_score < 0.05:
            return "稳定: 参数变化对策略影响很小"
        elif self.sensitivity_score < 0.15:
            return "较稳定: 参数变化有轻微影响"
        elif self.sensitivity_score < 0.30:
            return "敏感: 参数变化有明显影响"
        else:
            return "极敏感: 参数变化导致策略表现大幅波动"


@dataclass
class ComparisonResult:
    """策略比较结果"""
    strategies: list[str]
    yearly_comparison: pd.DataFrame
    regime_comparison: pd.DataFrame
    baseline_comparison: pd.DataFrame
    metrics_summary: pd.DataFrame
    winner: str
    
    def to_markdown(self) -> str:
        lines = [
            "# Strategy Comparison Report",
            "",
            f"**Strategies Compared:** {', '.join(self.strategies)}",
            f"**Winner:** {self.winner}",
            "",
            "## Summary Metrics",
            "",
        ]
        
        if not self.metrics_summary.empty:
            lines.append(self.metrics_summary.to_markdown(index=True))
            lines.append("")
        
        lines.extend([
            "## Yearly Comparison",
            "",
        ])
        if not self.yearly_comparison.empty:
            lines.append(self.yearly_comparison.to_markdown(index=False))
            lines.append("")
        
        lines.extend([
            "## Regime Comparison",
            "",
        ])
        if not self.regime_comparison.empty:
            lines.append(self.regime_comparison.to_markdown(index=False))
            lines.append("")
        
        lines.extend([
            "## Baseline Comparison",
            "",
        ])
        if not self.baseline_comparison.empty:
            lines.append(self.baseline_comparison.to_markdown(index=False))
            lines.append("")
        
        return "\n".join(lines)


def compute_yearly_metrics(
    nav: pd.DataFrame,
    trades: Optional[pd.DataFrame],
    rank_ic: Optional[pd.DataFrame],
) -> list[YearlyMetrics]:
    """计算分年度指标"""
    if nav.empty:
        return []
    
    nav = nav.copy()
    nav['trade_date'] = pd.to_datetime(nav['trade_date'])
    nav['year'] = nav['trade_date'].dt.year
    
    yearly_list = []
    for year, group in nav.groupby('year'):
        if len(group) < 20:
            continue
        
        group_sorted = group.sort_values('trade_date')
        nav_series = group_sorted.set_index('trade_date')['nav']
        
        total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
        daily_returns = nav_series.pct_change().dropna()
        
        if len(daily_returns) > 0:
            annual_return = daily_returns.mean() * 252
            volatility = daily_returns.std() * (252 ** 0.5)
            sharpe = annual_return / volatility if volatility > 0 else 0.0
        else:
            sharpe = 0.0
        
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_dd = abs(drawdown.min())
        
        # 换手率 - 支持多种字段名
        if trades is not None and not trades.empty:
            # execution_date 或 trade_date
            date_col = 'execution_date' if 'execution_date' in trades.columns else 'trade_date'
            trades[date_col] = pd.to_datetime(trades[date_col])
            year_trades = trades[trades[date_col].dt.year == year]
            # trade_value 或 notional
            value_col = 'trade_value' if 'trade_value' in year_trades.columns else 'notional'
            if value_col in year_trades.columns:
                turnover = year_trades[value_col].sum()
            else:
                turnover = 0.0
            num_trades = len(year_trades)
        else:
            turnover = 0.0
            num_trades = 0
        
        # IC
        if rank_ic is not None and not rank_ic.empty:
            rank_ic['trade_date'] = pd.to_datetime(rank_ic['trade_date'])
            year_ic = rank_ic[rank_ic['trade_date'].dt.year == year]
            ic_mean = year_ic['rank_ic'].mean() if not year_ic.empty else 0.0
        else:
            ic_mean = 0.0
        
        yearly_list.append(YearlyMetrics(
            year=int(year),
            total_return=float(total_return),
            sharpe=float(sharpe),
            max_drawdown=float(max_dd),
            turnover=float(turnover),
            num_trades=int(num_trades),
            ic_mean=float(ic_mean),
        ))
    
    return yearly_list


def compute_regime_metrics(
    nav: pd.DataFrame,
    benchmark_returns: pd.Series,
    threshold_up: float = 0.001,
    threshold_down: float = -0.001,
) -> list[RegimeMetrics]:
    """计算市场状态分解"""
    if nav.empty or benchmark_returns.empty:
        return []
    
    nav = nav.copy()
    nav['trade_date'] = pd.to_datetime(nav['trade_date'])
    
    # 合并基准收益
    nav['benchmark_return'] = nav['trade_date'].map(benchmark_returns)
    nav = nav.dropna(subset=['benchmark_return'])
    
    # 分类市场状态
    def classify_regime(ret):
        if ret > threshold_up:
            return 'bull'
        elif ret < threshold_down:
            return 'bear'
        else:
            return 'neutral'
    
    nav['regime'] = nav['benchmark_return'].apply(classify_regime)
    
    # 计算每个状态的指标
    regime_list = []
    for regime in ['bull', 'bear', 'neutral']:
        regime_data = nav[nav['regime'] == regime]
        if len(regime_data) < 5:
            continue
        
        regime_data = regime_data.sort_values('trade_date')
        nav_series = regime_data.set_index('trade_date')['nav']
        
        strategy_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1 if len(nav_series) > 1 else 0.0
        benchmark_return = regime_data['benchmark_return'].sum()
        excess_return = strategy_return - benchmark_return
        
        daily_returns = nav_series.pct_change().dropna()
        if len(daily_returns) > 0:
            sharpe = daily_returns.mean() * (252 ** 0.5) / (daily_returns.std() * (252 ** 0.5)) if daily_returns.std() > 0 else 0.0
        else:
            sharpe = 0.0
        
        regime_list.append(RegimeMetrics(
            regime=regime,
            days=len(regime_data),
            strategy_return=float(strategy_return),
            benchmark_return=float(benchmark_return),
            excess_return=float(excess_return),
            sharpe=float(sharpe),
        ))
    
    return regime_list


def compute_sensitivity(
    nav_base: pd.DataFrame,
    parameter: str,
    values: list,
    metric_name: str = 'sharpe',
) -> SensitivityResult:
    """
    计算参数敏感性。
    
    注意: 这是一个框架函数，实际的参数变化需要外部实现。
    这里返回的是基础结构。
    """
    return SensitivityResult(
        parameter=parameter,
        values=values,
        metric_name=metric_name,
        metric_values=[0.0] * len(values),
        sensitivity_score=0.0,
    )


class StrategyComparator:
    """
    策略比较器
    
    用于比较多策略的年度、regime和基线表现。
    """
    
    def __init__(self):
        self._strategies: dict[str, dict] = {}
    
    def add_strategy(
        self,
        name: str,
        nav: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
        rank_ic: Optional[pd.DataFrame] = None,
        benchmark_nav: Optional[pd.DataFrame] = None,
    ):
        """添加策略数据"""
        self._strategies[name] = {
            'nav': nav,
            'trades': trades,
            'rank_ic': rank_ic,
            'benchmark_nav': benchmark_nav,
        }
    
    def compare(self) -> ComparisonResult:
        """执行比较"""
        if len(self._strategies) < 2:
            raise ValueError("Need at least 2 strategies to compare")
        
        strategies = list(self._strategies.keys())
        
        # 年度比较
        yearly_data = []
        for name, data in self._strategies.items():
            yearly = compute_yearly_metrics(data['nav'], data['trades'], data['rank_ic'])
            for y in yearly:
                yearly_data.append({
                    'strategy': name,
                    'year': y.year,
                    'return': y.total_return,
                    'sharpe': y.sharpe,
                    'max_dd': y.max_drawdown,
                    'turnover': y.turnover,
                    'ic_mean': y.ic_mean,
                })
        
        yearly_comparison = pd.DataFrame(yearly_data) if yearly_data else pd.DataFrame()
        
        # Regime比较
        regime_data = []
        base_benchmark = None
        for name, data in self._strategies.items():
            if base_benchmark is None and data['benchmark_nav'] is not None:
                bench = data['benchmark_nav'].copy()
                bench['trade_date'] = pd.to_datetime(bench['trade_date'])
                bench = bench.sort_values('trade_date')
                # benchmark_nav 列可能是 'nav' 或 'benchmark_nav'
                bm_col = 'benchmark_nav' if 'benchmark_nav' in bench.columns else 'nav'
                base_benchmark = bench.set_index('trade_date')[bm_col].pct_change().dropna()
        
        for name, data in self._strategies.items():
            if base_benchmark is not None:
                regimes = compute_regime_metrics(data['nav'], base_benchmark)
                for r in regimes:
                    regime_data.append({
                        'strategy': name,
                        'regime': r.regime,
                        'days': r.days,
                        'return': r.strategy_return,
                        'excess': r.excess_return,
                        'sharpe': r.sharpe,
                    })
        
        regime_comparison = pd.DataFrame(regime_data) if regime_data else pd.DataFrame()
        
        # 基线比较
        baseline_data = []
        baseline_name = strategies[0]
        for name, data in self._strategies.items():
            if name == baseline_name:
                continue
            
            nav = data['nav']
            baseline_nav = self._strategies[baseline_name]['nav']
            
            if not nav.empty and not baseline_nav.empty:
                strategy_ret = nav['nav'].iloc[-1] / nav['nav'].iloc[0] - 1
                baseline_ret = baseline_nav['nav'].iloc[-1] / baseline_nav['nav'].iloc[0] - 1
                
                baseline_data.append({
                    'strategy': name,
                    'vs_baseline': baseline_name,
                    'strategy_return': strategy_ret,
                    'baseline_return': baseline_ret,
                    'excess_return': strategy_ret - baseline_ret,
                })
        
        baseline_comparison = pd.DataFrame(baseline_data) if baseline_data else pd.DataFrame()
        
        # 指标汇总
        metrics_data = []
        for name, data in self._strategies.items():
            nav = data['nav']
            if nav.empty:
                continue
            
            nav_series = nav.set_index('trade_date')['nav']
            total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
            daily_returns = nav_series.pct_change().dropna()
            
            if len(daily_returns) > 0:
                sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * (252 ** 0.5))
            else:
                sharpe = 0.0
            
            cummax = nav_series.cummax()
            max_dd = abs(((nav_series - cummax) / cummax).min())
            
            ic_mean = data['rank_ic']['rank_ic'].mean() if data['rank_ic'] is not None and not data['rank_ic'].empty else 0.0
            
            metrics_data.append({
                'strategy': name,
                'total_return': total_return,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'ic_mean': ic_mean,
            })
        
        metrics_summary = pd.DataFrame(metrics_data)
        
        # 确定赢家
        if not metrics_summary.empty:
            metrics_summary['score'] = (
                metrics_summary['total_return'] * 0.3 +
                metrics_summary['sharpe'] * 0.3 +
                metrics_summary['ic_mean'] * 0.2 -
                metrics_summary['max_drawdown'] * 0.2
            )
            winner = metrics_summary.loc[metrics_summary['score'].idxmax(), 'strategy']
        else:
            winner = strategies[0]
        
        return ComparisonResult(
            strategies=strategies,
            yearly_comparison=yearly_comparison,
            regime_comparison=regime_comparison,
            baseline_comparison=baseline_comparison,
            metrics_summary=metrics_summary,
            winner=winner,
        )


def run_parameter_sensitivity(
    base_nav: pd.DataFrame,
    parameter_name: str,
    parameter_values: list,
    run_single_experiment_fn,
    metric: str = 'sharpe',
) -> SensitivityResult:
    """
    运行参数敏感性分析。
    
    Args:
        base_nav: 基准净值
        parameter_name: 参数名
        parameter_values: 参数值列表
        run_single_experiment_fn: 单次实验函数 (接收参数值, 返回净值)
        metric: 评估指标 (sharpe/total_return/max_drawdown)
    """
    metric_values = []
    
    for value in parameter_values:
        nav = run_single_experiment_fn(value)
        
        if nav.empty:
            metric_values.append(0.0)
            continue
        
        nav_series = nav.set_index('trade_date')['nav']
        
        if metric == 'sharpe':
            daily_returns = nav_series.pct_change().dropna()
            if len(daily_returns) > 0:
                val = (daily_returns.mean() * 252) / (daily_returns.std() * (252 ** 0.5))
            else:
                val = 0.0
        elif metric == 'total_return':
            val = nav_series.iloc[-1] / nav_series.iloc[0] - 1
        elif metric == 'max_drawdown':
            cummax = nav_series.cummax()
            val = abs(((nav_series - cummax) / cummax).min())
        else:
            val = 0.0
        
        metric_values.append(val)
    
    # 计算敏感性分数
    if len(metric_values) >= 2:
        metric_range = max(metric_values) - min(metric_values)
        metric_std = np.std(metric_values)
        sensitivity = metric_std / (abs(np.mean(metric_values)) + 1e-6)
    else:
        sensitivity = 0.0
    
    return SensitivityResult(
        parameter=parameter_name,
        values=parameter_values,
        metric_name=metric,
        metric_values=metric_values,
        sensitivity_score=sensitivity,
    )
