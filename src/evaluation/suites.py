from __future__ import annotations

import json

import pandas as pd

from src.core.module_registry import ModuleMetadata

from .diagnostics import compute_quantile_forward_returns, compute_selection_turnover, compute_signal_coverage, summarize_rank_ic
from .registry import EvaluationContext, EvaluationRegistry, EvaluationSuiteResult, register_evaluation_suite


def _compute_yearly_breakdown(nav: pd.DataFrame, label_panel: pd.DataFrame, rank_ic: pd.DataFrame) -> pd.DataFrame:
    """Compute yearly performance breakdown."""
    if nav.empty:
        return pd.DataFrame()
    
    nav = nav.copy()
    nav['trade_date'] = pd.to_datetime(nav['trade_date'])
    nav['year'] = nav['trade_date'].dt.year
    
    yearly = []
    for year, group in nav.groupby('year'):
        group_sorted = group.sort_values('trade_date')
        nav_series = group_sorted.set_index('trade_date')['nav']
        daily_returns = nav_series.pct_change().fillna(0)
        
        total_return = (nav_series.iloc[-1] / nav_series.iloc[0] - 1) if len(nav_series) > 1 else 0
        annual_return = total_return
        volatility = daily_returns.std() * (252 ** 0.5)
        sharpe = annual_return / volatility if volatility > 0 else 0
        max_dd = (nav_series / nav_series.cummax() - 1).min()
        
        yearly.append({
            'year': int(year),
            'total_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'trading_days': len(group),
        })
    
    return pd.DataFrame(yearly)


def _compute_regime_breakdown(
    nav: pd.DataFrame, 
    daily_bar: pd.DataFrame, 
    threshold_up: float = 0.0005,
    threshold_down: float = -0.0005
) -> dict:
    """Compute performance breakdown by market regime."""
    if daily_bar.empty or nav.empty:
        return {}
    
    daily_bar = daily_bar.copy()
    daily_bar['trade_date'] = pd.to_datetime(daily_bar['trade_date'])
    daily_bar = daily_bar.sort_values('trade_date')
    
    daily_bar['benchmark_return'] = daily_bar.groupby('trade_date')['close'].mean().pct_change()
    daily_bar = daily_bar.dropna(subset=['benchmark_return'])
    
    regime_map = {}
    for date, row in daily_bar.groupby('trade_date')['benchmark_return'].first().items():
        if row > threshold_up:
            regime_map[date] = 'bull'
        elif row < threshold_down:
            regime_map[date] = 'bear'
        else:
            regime_map[date] = 'neutral'
    
    nav = nav.copy()
    nav['trade_date'] = pd.to_datetime(nav['trade_date'])
    nav['regime'] = nav['trade_date'].map(regime_map).fillna('neutral')
    
    regime_stats = {}
    for regime, group in nav.groupby('regime'):
        if len(group) < 5:
            continue
        group_sorted = group.sort_values('trade_date')
        nav_series = group_sorted.set_index('trade_date')['nav']
        total_ret = (nav_series.iloc[-1] / nav_series.iloc[0] - 1) if len(nav_series) > 1 else 0
        regime_stats[regime] = {
            'days': len(group),
            'total_return': float(total_ret),
        }
    
    return regime_stats


def _compute_cost_sensitivity(
    nav: pd.DataFrame,
    trades: pd.DataFrame,
    cost_multipliers: list[float] = [0.5, 1.0, 1.5, 2.0, 3.0]
) -> pd.DataFrame:
    """Compute performance under different cost assumptions."""
    if nav.empty or trades.empty:
        return pd.DataFrame()
    
    # 使用backtest实际字段: fee
    base_cost = trades['fee'].sum() if 'fee' in trades.columns else 0.0
    results = []
    
    for mult in cost_multipliers:
        adjusted_nav = nav.copy()
        adjusted_nav['nav'] = adjusted_nav['nav'] + base_cost * (mult - 1)
        adjusted_nav['total_return'] = adjusted_nav['nav'] / adjusted_nav['nav'].iloc[0] - 1
        
        results.append({
            'cost_multiplier': mult,
            'total_return': float(adjusted_nav['total_return'].iloc[-1]),
            'total_cost': float(base_cost * mult),
        })
    
    return pd.DataFrame(results)


def _compute_baseline_comparison(
    strategy_nav: pd.DataFrame,
    benchmark_nav: pd.DataFrame
) -> dict:
    """Compute excess return vs benchmark."""
    if strategy_nav.empty or benchmark_nav.empty:
        return {}
    
    strategy_ret = strategy_nav['nav'].iloc[-1] / strategy_nav['nav'].iloc[0] - 1
    benchmark_ret = benchmark_nav['nav'].iloc[-1] / benchmark_nav['nav'].iloc[0] - 1
    
    return {
        'strategy_return': float(strategy_ret),
        'benchmark_return': float(benchmark_ret),
        'excess_return': float(strategy_ret - benchmark_ret),
    }


def _comprehensive_factor_diagnostics(context: EvaluationContext, params: dict) -> EvaluationSuiteResult:
    """Comprehensive evaluation including regime, yearly, and cost sensitivity."""
    quantiles = int(params.get('quantiles', 5))
    
    ic_summary = summarize_rank_ic(context.rank_ic)
    quantile_returns, quantile_summary = compute_quantile_forward_returns(
        signal_scores=context.signal_scores,
        label_panel=context.label_panel,
        label_name=context.label_name,
        quantiles=quantiles,
    )
    selection_turnover = compute_selection_turnover(context.target_weights)
    coverage = compute_signal_coverage(context.signal_scores, context.model_dataset, context.label_name)
    
    yearly_breakdown = pd.DataFrame()
    regime_breakdown = {}
    cost_sensitivity = pd.DataFrame()
    baseline_comparison = {}
    
    if hasattr(context, 'nav') and not context.nav.empty:
        yearly_breakdown = _compute_yearly_breakdown(context.nav, context.label_panel, context.rank_ic)
        
        if hasattr(context, 'daily_bar') and not context.daily_bar.empty:
            regime_breakdown = _compute_regime_breakdown(context.nav, context.daily_bar)
        
        if hasattr(context, 'trades') and not context.trades.empty:
            cost_sensitivity = _compute_cost_sensitivity(context.nav, context.trades)
        
        if hasattr(context, 'benchmark_nav') and not context.benchmark_nav.empty:
            baseline_comparison = _compute_baseline_comparison(context.nav, context.benchmark_nav)
    
    metrics = {
        'diagnostics_ic_mean': float(ic_summary['ic_mean'].iloc[0]) if not ic_summary.empty else 0.0,
        'diagnostics_ic_ir': float(ic_summary['ic_ir'].iloc[0]) if not ic_summary.empty else 0.0,
        'diagnostics_avg_selection_turnover': float(selection_turnover['selection_turnover'].iloc[1:].mean()) if len(selection_turnover) > 1 else 0.0,
        'diagnostics_avg_signal_coverage': float(coverage['coverage_ratio'].mean()) if not coverage.empty else 0.0,
    }
    
    if yearly_breakdown is not None and not yearly_breakdown.empty:
        metrics['diagnostics_avg_yearly_sharpe'] = float(yearly_breakdown['sharpe'].mean())
        metrics['diagnostics_worst_year_return'] = float(yearly_breakdown['total_return'].min())
        metrics['diagnostics_best_year_return'] = float(yearly_breakdown['total_return'].max())
    
    if baseline_comparison:
        metrics['diagnostics_excess_return'] = baseline_comparison.get('excess_return', 0)
    
    top_bucket_row = quantile_summary.loc[quantile_summary['bucket'] == str(quantiles)]
    if not top_bucket_row.empty:
        metrics['diagnostics_top_bucket_return'] = float(top_bucket_row['mean_forward_return'].iloc[0])
    long_short_row = quantile_summary.loc[quantile_summary['bucket'] == 'long_short']
    if not long_short_row.empty:
        metrics['diagnostics_long_short_return'] = float(long_short_row['mean_forward_return'].iloc[0])
    
    markdown = (
        '# Comprehensive Factor Diagnostics\n\n'
        '## Summary Metrics\n'
        f'```json\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n```\n\n'
        '## IC Summary\n'
        f"{ic_summary.to_markdown(index=False) if not ic_summary.empty else '_无 IC 摘要_'}\n\n"
        '## Quantile Summary\n'
        f"{quantile_summary.to_markdown(index=False) if not quantile_summary.empty else '_无分组收益摘要_'}\n\n"
        '## Yearly Breakdown\n'
        f"{yearly_breakdown.to_markdown(index=False) if not yearly_breakdown.empty else '_无年度数据_'}\n\n"
        '## Regime Breakdown\n'
        f"```json\n{json.dumps(regime_breakdown, ensure_ascii=False, indent=2)}\n```\n\n"
        '## Cost Sensitivity\n'
        f"{cost_sensitivity.to_markdown(index=False) if not cost_sensitivity.empty else '_无成本敏感性数据_'}\n\n"
        '## Baseline Comparison\n'
        f"```json\n{json.dumps(baseline_comparison, ensure_ascii=False, indent=2)}\n```\n"
    )
    
    return EvaluationSuiteResult(
        metrics=metrics,
        tables={
            'ic_summary': ic_summary,
            'quantile_returns': quantile_returns,
            'quantile_summary': quantile_summary,
            'selection_turnover': selection_turnover,
            'signal_coverage': coverage,
            'yearly_breakdown': yearly_breakdown,
            'regime_breakdown': pd.DataFrame([regime_breakdown]) if regime_breakdown else pd.DataFrame(),
            'cost_sensitivity': cost_sensitivity,
        },
        markdown=markdown,
    )


def _basic_factor_diagnostics(context: EvaluationContext, params: dict) -> EvaluationSuiteResult:
    quantiles = int(params.get('quantiles', 5))
    ic_summary = summarize_rank_ic(context.rank_ic)
    quantile_returns, quantile_summary = compute_quantile_forward_returns(
        signal_scores=context.signal_scores,
        label_panel=context.label_panel,
        label_name=context.label_name,
        quantiles=quantiles,
    )
    selection_turnover = compute_selection_turnover(context.target_weights)
    coverage = compute_signal_coverage(context.signal_scores, context.model_dataset, context.label_name)
    metrics = {
        'diagnostics_ic_mean': float(ic_summary['ic_mean'].iloc[0]) if not ic_summary.empty else 0.0,
        'diagnostics_ic_ir': float(ic_summary['ic_ir'].iloc[0]) if not ic_summary.empty else 0.0,
        'diagnostics_avg_selection_turnover': float(selection_turnover['selection_turnover'].iloc[1:].mean()) if len(selection_turnover) > 1 else 0.0,
        'diagnostics_avg_signal_coverage': float(coverage['coverage_ratio'].mean()) if not coverage.empty else 0.0,
    }
    top_bucket_row = quantile_summary.loc[quantile_summary['bucket'] == str(quantiles)]
    if not top_bucket_row.empty:
        metrics['diagnostics_top_bucket_return'] = float(top_bucket_row['mean_forward_return'].iloc[0])
    long_short_row = quantile_summary.loc[quantile_summary['bucket'] == 'long_short']
    if not long_short_row.empty:
        metrics['diagnostics_long_short_return'] = float(long_short_row['mean_forward_return'].iloc[0])

    markdown = (
        '# Factor Diagnostics\n\n'
        '## Summary Metrics\n'
        f'```json\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n```\n\n'
        '## IC Summary\n'
        f"{ic_summary.to_markdown(index=False) if not ic_summary.empty else '_无 IC 摘要_'}\n\n"
        '## Quantile Summary\n'
        f"{quantile_summary.to_markdown(index=False) if not quantile_summary.empty else '_无分组收益摘要_'}\n\n"
        '## Selection Turnover\n'
        f"{selection_turnover.head(20).to_markdown(index=False) if not selection_turnover.empty else '_无调仓换手摘要_'}\n"
    )
    return EvaluationSuiteResult(
        metrics=metrics,
        tables={
            'ic_summary': ic_summary,
            'quantile_returns': quantile_returns,
            'quantile_summary': quantile_summary,
            'selection_turnover': selection_turnover,
            'signal_coverage': coverage,
        },
        markdown=markdown,
    )


def default_evaluation_registry() -> EvaluationRegistry:
    registry = EvaluationRegistry()
    register_evaluation_suite(
        registry,
        ModuleMetadata(
            name='basic_factor_diagnostics',
            version='v1',
            category='evaluation_suite',
            description='IC, quantile return, signal coverage, and selection turnover diagnostics.',
            owner='platform',
            inputs=('signal_scores', 'label_panel', 'target_weights', 'rank_ic'),
            outputs=('ic_summary', 'quantile_returns', 'quantile_summary', 'selection_turnover', 'signal_coverage'),
            tags=('factor_research', 'diagnostics'),
        ),
        suite=_basic_factor_diagnostics,
    )
    register_evaluation_suite(
        registry,
        ModuleMetadata(
            name='comprehensive_factor_diagnostics',
            version='v1',
            category='evaluation_suite',
            description='Full diagnostics including IC, quantile returns, yearly breakdown, regime analysis, cost sensitivity, and baseline comparison.',
            owner='platform',
            inputs=('signal_scores', 'label_panel', 'target_weights', 'rank_ic', 'nav', 'trades', 'daily_bar', 'benchmark_nav'),
            outputs=('ic_summary', 'quantile_returns', 'quantile_summary', 'yearly_breakdown', 'regime_breakdown', 'cost_sensitivity', 'baseline_comparison'),
            tags=('factor_research', 'diagnostics', 'comprehensive'),
        ),
        suite=_comprehensive_factor_diagnostics,
    )
    return registry
