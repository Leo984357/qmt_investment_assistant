import numpy as np
import pandas as pd
import pytest

from src.models.lightgbm_cross_sectional import CrossSectionalLightGBMModel, RollingModelResult


class TestCrossSectionalLightGBMModel:
    def test_init_default_params(self):
        model = CrossSectionalLightGBMModel(
            params={'n_estimators': 100},
            train_window_days=60,
            valid_window_days=20,
            min_train_samples=50,
            training_embargo_days=1,
            registry_stage='dev',
            fallback_model='simple_average',
        )
        assert model.params['n_estimators'] == 100
        assert model.train_window_days == 60
        assert model.fallback_model == 'simple_average'
        assert model.score_blend_weight_model == 1.0
        assert model.score_blend_weight_feature == 0.0

    def test_negative_embargo_clamped_to_zero(self):
        model = CrossSectionalLightGBMModel(
            params={},
            train_window_days=60,
            valid_window_days=20,
            min_train_samples=50,
            training_embargo_days=-5,
            registry_stage='dev',
            fallback_model='simple_average',
        )
        assert model.training_embargo_days == 0

    def test_label_clip_propagated(self):
        model = CrossSectionalLightGBMModel(
            params={},
            train_window_days=60,
            valid_window_days=20,
            min_train_samples=50,
            training_embargo_days=1,
            registry_stage='dev',
            fallback_model='simple_average',
            label_clip=(-0.1, 0.1),
        )
        assert model.label_clip == (-0.1, 0.1)

    def test_score_blend_weights(self):
        model = CrossSectionalLightGBMModel(
            params={},
            train_window_days=60,
            valid_window_days=20,
            min_train_samples=50,
            training_embargo_days=1,
            registry_stage='dev',
            fallback_model='simple_average',
            score_blend_weight_model=0.7,
            score_blend_weight_feature=0.3,
            score_blend_feature='rev5',
        )
        assert model.score_blend_weight_model == 0.7
        assert model.score_blend_weight_feature == 0.3
        assert model.score_blend_feature == 'rev5'


class TestRollingModelResult:
    def test_dataclass_fields(self):
        preds = pd.DataFrame({'trade_date': ['2024-01-01'], 'symbol': ['A'], 'score': [0.5]})
        metrics = pd.DataFrame({'split': ['valid'], 'mse': [0.1]})
        importance = pd.DataFrame({'feature': ['f1'], 'gain': [0.8]})
        registry = pd.DataFrame({'model_id': ['1'], 'stage': ['dev']})

        result = RollingModelResult(
            predictions=preds,
            split_metrics=metrics,
            feature_importance=importance,
            model_registry=registry,
        )
        assert result.predictions.iloc[0]['score'] == 0.5
        assert result.split_metrics.iloc[0]['mse'] == 0.1
