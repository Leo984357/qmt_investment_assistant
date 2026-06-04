from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RollingModelResult:
    predictions: pd.DataFrame
    split_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    model_registry: pd.DataFrame


class RidgeRegressionModel:
    def __init__(
        self,
        feature_names: list[str],
        alpha: float = 1.0,
    ):
        self.feature_names = feature_names
        self.alpha = alpha
        self._models: dict = {}

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

            current_idx = date_to_idx[signal_date]

            # 训练窗口 (需要 embargo 避免标签泄漏)
            # embargo = label_horizon + 一些缓冲 (通常 5-10 天)
            embargo_days = max(label_horizon, 20)
            embargo_idx = current_idx - embargo_days if current_idx >= embargo_days else 0
            train_start_idx = max(0, embargo_idx - 500)
            train_dates = unique_dates[train_start_idx:embargo_idx]

            train_df = dataset[
                (dataset['trade_date'].isin(train_dates)) &
                dataset['trade_date'].notna() &
                dataset[label_name].notna()
            ].dropna(subset=feature_names + [label_name]).copy()

            if len(train_df) < 50:
                # Fallback to simple average
                test_df = dataset.loc[dataset['trade_date'] == signal_date].dropna(subset=feature_names).copy()
                if test_df.empty:
                    continue

                score = test_df[feature_names].mean(axis=1)
                test_df['score'] = score
                test_df['fallback_used'] = True

                pred = test_df[['trade_date', 'symbol', 'score', 'fallback_used']].copy()
                pred['model_type'] = 'ridge_regression'
                pred['signal_date'] = signal_date
                prediction_frames.append(pred)

                registry_rows.append({
                    'signal_date': signal_date,
                    'model_type': 'ridge_regression',
                    'train_samples': len(train_df),
                    'valid_samples': 0,
                    'train_dates': len(train_dates),
                    'valid_dates': 0,
                })
                continue

            # 训练Ridge回归
            X_train = train_df[feature_names].values
            y_train = train_df[label_name].values

            # 标准化
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0) + 1e-8
            X_train_norm = (X_train - X_mean) / X_std

            # 训练
            model = Ridge(alpha=self.alpha)
            model.fit(X_train_norm, y_train)

            # 测试集预测
            test_df = dataset.loc[dataset['trade_date'] == signal_date].dropna(subset=feature_names).copy()
            if test_df.empty:
                continue

            X_test = test_df[feature_names].values
            X_test_norm = (X_test - X_mean) / X_std

            score = model.predict(X_test_norm)
            test_df['score'] = score
            test_df['fallback_used'] = False

            pred = test_df[['trade_date', 'symbol', 'score', 'fallback_used']].copy()
            pred['model_type'] = 'ridge_regression'
            pred['signal_date'] = signal_date
            prediction_frames.append(pred)

            # 特征重要性（系数绝对值）
            importance_frames.append(pd.DataFrame({
                'feature_name': feature_names,
                'importance_gain': np.abs(model.coef_),
                'signal_date': [signal_date] * len(feature_names),
            }))

            # 度量
            y_pred_train = model.predict(X_train_norm)
            train_ic = np.corrcoef(y_train, y_pred_train)[0, 1]

            registry_rows.append({
                'signal_date': signal_date,
                'model_type': 'ridge_regression',
                'train_samples': len(train_df),
                'valid_samples': len(test_df),
                'train_dates': len(train_dates),
                'valid_dates': 1,
            })

            metric_rows.append({
                'signal_date': signal_date,
                'train_ic': train_ic,
            })

            # 保存模型
            self._models[signal_date] = {
                'model': model,
                'X_mean': X_mean,
                'X_std': X_std,
            }

            if split_idx % progress_every == 0:
                elapsed = perf_counter() - start_ts
                logger.info('Ridge progress %d/%d splits (%.1fs)', split_idx, total_splits, elapsed)

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
