from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_rank_ic(rank_ic: pd.DataFrame) -> pd.DataFrame:
    if rank_ic.empty:
        return pd.DataFrame([{'ic_count': 0, 'ic_mean': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'ic_positive_rate': 0.0, 'ic_abs_mean': 0.0}])
    series = pd.to_numeric(rank_ic['rank_ic'], errors='coerce').dropna()
    ic_std = float(series.std(ddof=0)) if not series.empty else 0.0
    ic_mean = float(series.mean()) if not series.empty else 0.0
    return pd.DataFrame(
        [
            {
                'ic_count': int(series.size),
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_ir': ic_mean / ic_std if ic_std > 1e-12 else 0.0,
                'ic_positive_rate': float((series > 0).mean()) if not series.empty else 0.0,
                'ic_abs_mean': float(series.abs().mean()) if not series.empty else 0.0,
            }
        ]
    )


def compute_signal_quantiles(signal_scores: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    if signal_scores.empty:
        return pd.DataFrame(columns=['trade_date', 'symbol', 'score', 'quantile'])
    frame = signal_scores[['trade_date', 'symbol', 'score']].copy()
    frame['trade_date'] = pd.to_datetime(frame['trade_date'])
    pct_rank = frame.groupby('trade_date')['score'].rank(method='first', pct=True)
    frame['quantile'] = np.ceil(pct_rank * max(int(quantiles), 1)).clip(1, max(int(quantiles), 1)).astype(int)
    return frame


def compute_quantile_forward_returns(signal_scores: pd.DataFrame, label_panel: pd.DataFrame, label_name: str, quantiles: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    quantile_frame = compute_signal_quantiles(signal_scores, quantiles)
    if quantile_frame.empty:
        return (
            pd.DataFrame(columns=['trade_date', 'quantile', 'avg_forward_return', 'count']),
            pd.DataFrame(columns=['bucket', 'mean_forward_return', 'std_forward_return', 'observations']),
        )
    merged = quantile_frame.merge(label_panel[['trade_date', 'symbol', label_name]], on=['trade_date', 'symbol'], how='left')
    merged = merged.dropna(subset=[label_name]).copy()
    daily = (
        merged.groupby(['trade_date', 'quantile'], as_index=False)
        .agg(
            avg_forward_return=(label_name, 'mean'),
            count=('symbol', 'size'),
        )
    )
    summary = (
        daily.groupby('quantile', as_index=False)
        .agg(
            mean_forward_return=('avg_forward_return', 'mean'),
            std_forward_return=('avg_forward_return', 'std'),
            observations=('trade_date', 'size'),
        )
        .rename(columns={'quantile': 'bucket'})
    )
    summary['bucket'] = summary['bucket'].astype(str)
    top_bucket = int(daily['quantile'].max())
    bottom_bucket = int(daily['quantile'].min())
    spread = (
        daily.loc[daily['quantile'] == top_bucket, ['trade_date', 'avg_forward_return']]
        .rename(columns={'avg_forward_return': 'top_return'})
        .merge(
            daily.loc[daily['quantile'] == bottom_bucket, ['trade_date', 'avg_forward_return']].rename(columns={'avg_forward_return': 'bottom_return'}),
            on='trade_date',
            how='inner',
        )
    )
    if not spread.empty:
        spread['long_short_spread'] = spread['top_return'] - spread['bottom_return']
        spread_summary = pd.DataFrame(
            [
                {
                    'bucket': 'long_short',
                    'mean_forward_return': float(spread['long_short_spread'].mean()),
                    'std_forward_return': float(spread['long_short_spread'].std(ddof=0)),
                    'observations': int(len(spread)),
                }
            ]
        )
        summary = pd.concat([summary, spread_summary], ignore_index=True)
    return daily.sort_values(['trade_date', 'quantile']).reset_index(drop=True), summary


def compute_selection_turnover(target_weights: pd.DataFrame) -> pd.DataFrame:
    if target_weights.empty:
        return pd.DataFrame(columns=['signal_date', 'prev_signal_date', 'selected_count', 'overlap_count', 'selection_turnover'])
    selected = target_weights.groupby('signal_date')['symbol'].agg(lambda values: tuple(sorted(set(values)))).reset_index()
    rows = []
    previous_symbols: set[str] | None = None
    previous_date = None
    for row in selected.itertuples(index=False):
        current_symbols = set(row.symbol)
        if previous_symbols is None:
            rows.append(
                {
                    'signal_date': row.signal_date,
                    'prev_signal_date': pd.NaT,
                    'selected_count': len(current_symbols),
                    'overlap_count': 0,
                    'selection_turnover': 0.0,
                }
            )
        else:
            overlap = len(current_symbols & previous_symbols)
            rows.append(
                {
                    'signal_date': row.signal_date,
                    'prev_signal_date': previous_date,
                    'selected_count': len(current_symbols),
                    'overlap_count': overlap,
                    'selection_turnover': 1.0 - overlap / max(len(current_symbols), 1),
                }
            )
        previous_symbols = current_symbols
        previous_date = row.signal_date
    return pd.DataFrame(rows)


def compute_signal_coverage(signal_scores: pd.DataFrame, model_dataset: pd.DataFrame, label_name: str) -> pd.DataFrame:
    if signal_scores.empty:
        return pd.DataFrame(columns=['trade_date', 'candidate_count', 'labeled_count', 'scored_count', 'coverage_ratio', 'labeled_ratio'])
    candidate = model_dataset.groupby('trade_date', as_index=False).agg(
        candidate_count=('symbol', 'size'),
        labeled_count=(label_name, lambda series: int(series.notna().sum())),
    )
    scored = signal_scores.groupby('trade_date', as_index=False).agg(scored_count=('symbol', 'size'))
    coverage = candidate.merge(scored, on='trade_date', how='left').fillna({'scored_count': 0})
    coverage['coverage_ratio'] = coverage['scored_count'] / coverage['candidate_count'].replace(0, np.nan)
    coverage['labeled_ratio'] = coverage['scored_count'] / coverage['labeled_count'].replace(0, np.nan)
    return coverage.fillna(0.0).sort_values('trade_date').reset_index(drop=True)
