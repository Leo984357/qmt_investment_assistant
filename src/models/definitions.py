from __future__ import annotations

from src.core.module_registry import ModuleMetadata

from .lightgbm_cross_sectional import CrossSectionalLightGBMModel
from .registry import ModelRegistry, register_model
from .ridge_regression import RidgeRegressionModel
from .simple_average import SimpleAverageModel
from .weighted_average import ICWeightedAverageModel


def default_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    register_model(
        registry,
        ModuleMetadata(
            name='lightgbm_regression',
            version='v1',
            category='model',
            description='Rolling cross-sectional LightGBM regression with fallback support.',
            owner='platform',
            inputs=('feature_panel', 'label_panel', 'rebalance_dates'),
            outputs=('predictions', 'split_metrics', 'feature_importance', 'model_registry'),
            tags=('cross_sectional', 'lightgbm', 'walk_forward'),
        ),
        builder=lambda config: CrossSectionalLightGBMModel(
            params=config.params,
            train_window_days=config.train_window_days,
            valid_window_days=config.valid_window_days,
            min_train_samples=config.min_train_samples,
            training_embargo_days=getattr(config, 'training_embargo_days', 0),
            registry_stage=config.registry_stage,
            fallback_model=config.fallback_model,
            fallback_feature=config.fallback_feature,
            score_blend_feature=config.score_blend_feature,
            score_blend_weight_model=config.score_blend_weight_model,
            score_blend_weight_feature=config.score_blend_weight_feature,
            label_clip=config.label_clip,
        ),
    )
    register_model(
        registry,
        ModuleMetadata(
            name='simple_average',
            version='v1',
            category='model',
            description='Simple average of z-scored features as baseline.',
            owner='platform',
            inputs=('feature_panel', 'rebalance_dates'),
            outputs=('predictions', 'split_metrics', 'feature_importance', 'model_registry'),
            tags=('baseline', 'simple', 'average'),
        ),
        builder=lambda config: SimpleAverageModel(
            feature_names=[],
        ),
    )
    register_model(
        registry,
        ModuleMetadata(
            name='ic_weighted_average',
            version='v1',
            category='model',
            description='IC-weighted average of features based on historical IC.',
            owner='platform',
            inputs=('feature_panel', 'label_panel', 'rebalance_dates'),
            outputs=('predictions', 'split_metrics', 'feature_importance', 'model_registry'),
            tags=('baseline', 'weighted', 'ic'),
        ),
        builder=lambda config: ICWeightedAverageModel(
            feature_names=[],
            lookback_days=60,
        ),
    )
    register_model(
        registry,
        ModuleMetadata(
            name='ridge_regression',
            version='v1',
            category='model',
            description='Ridge regression baseline - tests if LightGBM improvement is from non-linear patterns.',
            owner='platform',
            inputs=('feature_panel', 'label_panel', 'rebalance_dates'),
            outputs=('predictions', 'split_metrics', 'feature_importance', 'model_registry'),
            tags=('baseline', 'linear', 'ridge', 'regularized'),
        ),
        builder=lambda config: RidgeRegressionModel(
            feature_names=[],
            alpha=config.params.get('alpha', 1.0) if config.params else 1.0,
        ),
    )
    return registry
