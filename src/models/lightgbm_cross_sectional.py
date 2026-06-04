from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
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


class CrossSectionalLightGBMModel:
    def __init__(
        self,
        params: dict,
        train_window_days: int,
        valid_window_days: int,
        min_train_samples: int,
        training_embargo_days: int,
        registry_stage: str,
        fallback_model: str,
        fallback_feature: str | None = None,
        score_blend_feature: str | None = None,
        score_blend_weight_model: float = 1.0,
        score_blend_weight_feature: float = 0.0,
        label_clip: tuple[float, float] | None = None,
    ):
        self.params = params
        self.train_window_days = train_window_days
        self.valid_window_days = valid_window_days
        self.min_train_samples = min_train_samples
        self.training_embargo_days = max(int(training_embargo_days), 0)
        self.registry_stage = registry_stage
        self.fallback_model = fallback_model
        self.fallback_feature = fallback_feature
        self.score_blend_feature = score_blend_feature
        self.score_blend_weight_model = float(score_blend_weight_model)
        self.score_blend_weight_feature = float(score_blend_weight_feature)
        self.label_clip = label_clip

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
            label_safe_end_exclusive = max(
                0,
                idx - max(int(label_horizon), 0) - self.training_embargo_days + 1,
            )
            if label_safe_end_exclusive <= 0:
                continue
            valid_end_idx = label_safe_end_exclusive
            valid_start_idx = max(0, valid_end_idx - self.valid_window_days)
            train_span = self.train_window_days + (1 if self.valid_window_days <= 0 else 0)
            train_start_idx = max(0, valid_start_idx - train_span)
            train_end_idx = valid_start_idx if self.valid_window_days > 0 else valid_end_idx
            train_dates = unique_dates[train_start_idx:train_end_idx]
            valid_dates = unique_dates[valid_start_idx:valid_end_idx]

            train_df = dataset.loc[dataset['trade_date'].isin(train_dates)].dropna(subset=feature_names + [label_name]).copy()
            valid_df = dataset.loc[dataset['trade_date'].isin(valid_dates)].dropna(subset=feature_names + [label_name]).copy()
            test_df = dataset.loc[dataset['trade_date'] == signal_date].dropna(subset=feature_names).copy()
            if test_df.empty:
                continue

            use_fallback = len(train_df) < self.min_train_samples or train_df['trade_date'].nunique() < 20
            model_path = None
            model_type = self._fallback_model_type() if use_fallback else self._active_model_type()

            if use_fallback:
                scores = self._fallback_score(test_df, feature_names)
                valid_scores = self._fallback_score(valid_df, feature_names) if not valid_df.empty else pd.Series(dtype=float)
            else:
                train_label = train_df[label_name].copy()
                if self.label_clip is not None:
                    train_label = train_label.clip(lower=float(self.label_clip[0]), upper=float(self.label_clip[1]))
                try:
                    train_dataset = lgb.Dataset(train_df[feature_names], label=train_label)
                    valid_datasets = [train_dataset]
                    valid_names = ['train']
                    callbacks = []
                    if not valid_df.empty:
                        valid_dataset = lgb.Dataset(valid_df[feature_names], label=valid_df[label_name], reference=train_dataset)
                        valid_datasets.append(valid_dataset)
                        valid_names.append('valid')
                        callbacks.append(lgb.early_stopping(30, verbose=False))
                    booster = lgb.train(
                        self._train_params(),
                        train_set=train_dataset,
                        num_boost_round=self._num_boost_round(),
                        valid_sets=valid_datasets,
                        valid_names=valid_names,
                        callbacks=callbacks,
                    )
                    raw_scores = pd.Series(booster.predict(test_df[feature_names]), index=test_df.index)
                    if raw_scores.nunique(dropna=True) < 2:
                        raise ValueError('模型预测退化为常数，回退到 fallback。')
                    raw_valid_scores = pd.Series(booster.predict(valid_df[feature_names]), index=valid_df.index) if not valid_df.empty else pd.Series(dtype=float)
                    scores = self._post_process_scores(test_df, raw_scores)
                    valid_scores = self._post_process_scores(valid_df, raw_valid_scores) if not valid_df.empty else pd.Series(dtype=float)
                    model_path = artifact_dir / f'lightgbm_{signal_date.date()}.txt'
                    booster.save_model(str(model_path))
                    importance_frames.append(
                        pd.DataFrame(
                            {
                                'feature_name': feature_names,
                                'importance_gain': booster.feature_importance(importance_type='gain'),
                                'signal_date': signal_date,
                            }
                        )
                    )
                except Exception:
                    use_fallback = True
                    model_path = None
                    model_type = self._fallback_model_type()
                    scores = self._fallback_score(test_df, feature_names)
                    valid_scores = self._fallback_score(valid_df, feature_names) if not valid_df.empty else pd.Series(dtype=float)

            prediction_frames.append(
                test_df[['trade_date', 'symbol']].assign(
                    score=scores.values,
                    model_type=model_type,
                    fallback_used=use_fallback,
                )
            )

            metric_rows.append(
                {
                    'signal_date': signal_date,
                    'train_start_date': train_dates[0] if train_dates else pd.NaT,
                    'train_end_date': train_dates[-1] if train_dates else pd.NaT,
                    'valid_start_date': valid_dates[0] if valid_dates else pd.NaT,
                    'valid_end_date': valid_dates[-1] if valid_dates else pd.NaT,
                    'train_rows': int(len(train_df)),
                    'valid_rows': int(len(valid_df)),
                    'test_rows': int(len(test_df)),
                    'valid_rank_ic': self._rank_ic(valid_df[label_name], valid_scores) if not valid_df.empty else 0.0,
                    'valid_rmse': self._rmse(valid_df[label_name], valid_scores) if not valid_df.empty else 0.0,
                    'fallback_used': use_fallback,
                }
            )
            registry_rows.append(
                {
                    'signal_date': signal_date,
                    'model_type': model_type,
                    'registry_stage': self.registry_stage,
                    'model_path': str(model_path) if model_path else '',
                    'fallback_model': self.fallback_model if use_fallback else '',
                }
            )

            if split_idx == 1 or split_idx == total_splits or split_idx % progress_every == 0:
                elapsed = perf_counter() - start_ts
                logger.info(
                    'Model walk-forward progress %s/%s signal_date=%s train_rows=%s test_rows=%s fallback=%s elapsed=%.1fs',
                    split_idx,
                    total_splits,
                    signal_date.date(),
                    len(train_df),
                    len(test_df),
                    use_fallback,
                    elapsed,
                )

        if not prediction_frames:
            raise ValueError('模型阶段没有生成任何预测结果。')

        feature_importance = (
            pd.concat(importance_frames, ignore_index=True).groupby('feature_name', as_index=False)['importance_gain'].mean()
            if importance_frames
            else pd.DataFrame({'feature_name': feature_names, 'importance_gain': np.zeros(len(feature_names))})
        )
        return RollingModelResult(
            predictions=pd.concat(prediction_frames, ignore_index=True).sort_values(['trade_date', 'score'], ascending=[True, False]),
            split_metrics=pd.DataFrame(metric_rows),
            feature_importance=feature_importance.sort_values('importance_gain', ascending=False).reset_index(drop=True),
            model_registry=pd.DataFrame(registry_rows),
        )

    def _num_boost_round(self) -> int:
        return int(self.params.get('n_estimators', self.params.get('num_boost_round', 100)))

    def _train_params(self) -> dict:
        params = dict(self.params)
        params.pop('n_estimators', None)
        random_state = params.pop('random_state', None)
        if random_state is not None and 'seed' not in params:
            params['seed'] = int(random_state)
        min_child_samples = params.pop('min_child_samples', None)
        if min_child_samples is not None and 'min_data_in_leaf' not in params:
            params['min_data_in_leaf'] = int(min_child_samples)
        reg_lambda = params.pop('reg_lambda', None)
        if reg_lambda is not None and 'lambda_l2' not in params:
            params['lambda_l2'] = float(reg_lambda)
        return params

    def _fallback_score(self, df: pd.DataFrame, feature_names: list[str]) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float, index=df.index)
        if self.fallback_feature and self.fallback_feature in df.columns:
            return self._cross_sectional_zscore(df, df[self.fallback_feature])
        return self._linear_fallback_score(df, feature_names)

    @staticmethod
    def _linear_fallback_score(df: pd.DataFrame, feature_names: list[str]) -> pd.Series:
        validated_weights = {
            'mom250': 0.3,
            'mom120': 0.2,
            'roe': 0.2,
            'earnings_yield': 0.2,
            'operating_margin': 0.1,
        }
        score = pd.Series(0.0, index=df.index, dtype=float)
        for feature_name in feature_names:
            score = score + df[feature_name].fillna(0.0) * validated_weights.get(feature_name, 0.0)
        return score

    def _post_process_scores(self, df: pd.DataFrame, raw_scores: pd.Series) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float, index=df.index)
        if not self.score_blend_feature or self.score_blend_weight_feature == 0.0 or self.score_blend_feature not in df.columns:
            return raw_scores.fillna(0.0)
        model_score = self._cross_sectional_zscore(df, raw_scores)
        feature_score = self._cross_sectional_zscore(df, df[self.score_blend_feature])
        blended = self.score_blend_weight_model * model_score + self.score_blend_weight_feature * feature_score
        return blended.fillna(0.0)

    @staticmethod
    def _cross_sectional_zscore(df: pd.DataFrame, values: pd.Series) -> pd.Series:
        frame = df[['trade_date']].copy()
        frame['value'] = values.astype(float)
        frame['value'] = frame.groupby('trade_date')['value'].transform(CrossSectionalLightGBMModel._zscore_series)
        return frame['value'].fillna(0.0)

    @staticmethod
    def _zscore_series(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if pd.isna(std) or std <= 1e-12:
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    def _fallback_model_type(self) -> str:
        if self.fallback_feature:
            return f'fallback_{self.fallback_feature}_zscore'
        return 'fallback_linear_blend'

    def _active_model_type(self) -> str:
        if self.score_blend_feature and self.score_blend_weight_feature != 0.0:
            return f'lightgbm_{self.score_blend_feature}_blend'
        return 'lightgbm_regression'

    @staticmethod
    def _rank_ic(actual: pd.Series, predicted: pd.Series) -> float:
        if len(actual) < 2 or len(predicted) < 2:
            return 0.0
        if actual.nunique(dropna=True) < 2 or predicted.nunique(dropna=True) < 2:
            return 0.0
        corr = actual.corr(predicted, method='spearman')
        return round(float(0.0 if pd.isna(corr) else corr), 6)

    @staticmethod
    def _rmse(actual: pd.Series, predicted: pd.Series) -> float:
        if len(actual) == 0 or len(predicted) == 0:
            return 0.0
        return round(float(np.sqrt(np.mean((actual.values - predicted.values) ** 2))), 6)
