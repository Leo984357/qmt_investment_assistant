from __future__ import annotations

import pandas as pd

from src.core.module_registry import ModuleMetadata

from .registry import SignalContext, SignalRegistry, SignalResult, register_signal


def _cross_sectional_score_signal(context: SignalContext, params: dict) -> SignalResult:
    signal_scores = context.predictions.copy()
    signal_scores['trade_date'] = pd.to_datetime(signal_scores['trade_date'])
    signal_scores = signal_scores.sort_values(['trade_date', 'score', 'symbol'], ascending=[True, False, True]).reset_index(drop=True)
    signal_scores['signal_rank'] = (
        signal_scores.groupby('trade_date')['score']
        .rank(method='first', ascending=False)
        .astype(int)
    )
    daily_stats = signal_scores.groupby('trade_date', as_index=False).agg(
        scored_count=('symbol', 'size'),
        top_score=('score', 'max'),
        median_score=('score', 'median'),
        bottom_score=('score', 'min'),
        fallback_rate=('fallback_used', 'mean'),
    )
    summary = {
        'signal_dates': int(signal_scores['trade_date'].nunique()),
        'avg_scored_universe': float(daily_stats['scored_count'].mean()) if not daily_stats.empty else 0.0,
        'fallback_rate': float(signal_scores['fallback_used'].mean()) if 'fallback_used' in signal_scores.columns and not signal_scores.empty else 0.0,
    }
    return SignalResult(
        signal_scores=signal_scores,
        summary=summary,
        tables={'daily_signal_stats': daily_stats},
    )


def default_signal_registry() -> SignalRegistry:
    registry = SignalRegistry()
    register_signal(
        registry,
        ModuleMetadata(
            name='cross_sectional_score',
            version='v1',
            category='signal',
            description='Pass-through cross-sectional score signal with rank and daily signal statistics.',
            owner='platform',
            inputs=('predictions',),
            outputs=('signal_scores', 'daily_signal_stats'),
            tags=('score', 'cross_sectional'),
        ),
        generator=_cross_sectional_score_signal,
    )
    return registry
