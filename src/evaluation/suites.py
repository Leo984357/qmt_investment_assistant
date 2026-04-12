from __future__ import annotations

import json

import pandas as pd

from src.core.module_registry import ModuleMetadata

from .diagnostics import compute_quantile_forward_returns, compute_selection_turnover, compute_signal_coverage, summarize_rank_ic
from .registry import EvaluationContext, EvaluationRegistry, EvaluationSuiteResult, register_evaluation_suite


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
    return registry
