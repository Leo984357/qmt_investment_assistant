from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RollingModelResult:
    predictions: pd.DataFrame
    split_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    model_registry: pd.DataFrame


class SimpleAverageModel:
    def __init__(
        self,
        feature_names: list[str],
    ):
        self.feature_names = feature_names

    def fit_walk_forward(
        self,
        dataset: pd.DataFrame,
        feature_names: list[str],
        label_name: str,
        rebalance_dates: list[pd.Timestamp],
        artifact_dir: Path,
        label_horizon: int = 0,
    ) -> RollingModelResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        unique_dates = sorted(pd.to_datetime(dataset['trade_date'].unique()))
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        prediction_frames: list[pd.DataFrame] = []
        metric_rows: list[dict] = []
        importance_frames: list[pd.DataFrame] = []
        registry_rows: list[dict] = []
        total_splits = len(rebalance_dates)
        progress_every = max(1, total_splits // 20) if total_splits else 1
        start_ts = perf_counter()

        for split_idx, signal_date in enumerate(rebalance_dates, start=1):
            signal_date = pd.Timestamp(signal_date)
            if signal_date not in date_to_idx:
                continue

            test_df = dataset.loc[dataset['trade_date'] == signal_date].dropna(subset=feature_names).copy()
            if test_df.empty:
                continue

            score = test_df[feature_names].mean(axis=1)
            test_df['score'] = score
            test_df['fallback_used'] = False

            pred = test_df[['trade_date', 'symbol', 'score', 'fallback_used']].copy()
            pred['model_type'] = 'simple_average'
            pred['signal_date'] = signal_date
            prediction_frames.append(pred)

            importance_frames.append(pd.DataFrame({
                'feature_name': feature_names,
                'importance_gain': [1.0] * len(feature_names),
                'signal_date': [signal_date] * len(feature_names),
            }))

            registry_rows.append({
                'signal_date': signal_date,
                'model_type': 'simple_average',
                'train_samples': 0,
                'valid_samples': 0,
                'train_dates': 0,
                'valid_dates': 0,
            })

            if split_idx % progress_every == 0:
                elapsed = perf_counter() - start_ts
                logger.info('SimpleAverage progress %d/%d splits (%.1fs)', split_idx, total_splits, elapsed)

        predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        split_metrics = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame()
        feature_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
        model_registry = pd.DataFrame(registry_rows) if registry_rows else pd.DataFrame()

        return RollingModelResult(
            predictions=predictions,
            split_metrics=split_metrics,
            feature_importance=feature_importance,
            model_registry=model_registry,
        )
