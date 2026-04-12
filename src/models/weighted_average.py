from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.core.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class RollingModelResult:
    predictions: pd.DataFrame
    split_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    model_registry: pd.DataFrame


class ICWeightedAverageModel:
    def __init__(
        self,
        feature_names: list[str],
        ic_weights: dict[str, float] | None = None,
        lookback_days: int = 60,
    ):
        self.feature_names = feature_names
        self.ic_weights = ic_weights or {}
        self.lookback_days = lookback_days
        
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

            idx = date_to_idx[signal_date]
            train_end_idx = max(0, idx - int(label_horizon) - 1)
            train_start_idx = max(0, train_end_idx - self.lookback_days)
            train_dates = unique_dates[train_start_idx:train_end_idx]
            
            train_df = dataset.loc[dataset['trade_date'].isin(train_dates)].dropna(subset=feature_names + [label_name]).copy()
            
            ic_scores = {}
            for f in feature_names:
                valid = train_df[['trade_date', f, label_name]].dropna()
                if len(valid) > 30:
                    ic = valid.groupby('trade_date').apply(
                        lambda x: x[f].corr(x[label_name]), include_groups=False
                    ).mean()
                    ic_scores[f] = max(ic, 0.001)
                else:
                    ic_scores[f] = 0.001
            
            weights = np.array([ic_scores.get(f, 0.001) for f in feature_names])
            weights = weights / weights.sum()
            
            test_df = dataset.loc[dataset['trade_date'] == signal_date].dropna(subset=feature_names).copy()
            if test_df.empty:
                continue

            factor_matrix = test_df[feature_names].values
            score = np.dot(factor_matrix, weights)
            test_df['score'] = score
            test_df['fallback_used'] = False

            pred = test_df[['trade_date', 'symbol', 'score', 'fallback_used']].copy()
            pred['model_type'] = 'ic_weighted_average'
            pred['signal_date'] = signal_date
            prediction_frames.append(pred)

            importance_frames.append(pd.DataFrame({
                'feature_name': feature_names,
                'importance_gain': weights,
                'signal_date': [signal_date] * len(feature_names),
            }))

            registry_rows.append({
                'signal_date': signal_date,
                'model_type': 'ic_weighted_average',
                'train_samples': len(train_df),
                'valid_samples': 0,
                'train_dates': len(train_dates),
                'valid_dates': 0,
                **{f'ic_{f}': ic_scores.get(f, 0) for f in feature_names},
            })

            if split_idx % progress_every == 0:
                elapsed = perf_counter() - start_ts
                logger.info('ICWeightedAverage progress %d/%d splits (%.1fs)', split_idx, total_splits, elapsed)

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
