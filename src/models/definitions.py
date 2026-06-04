from __future__ import annotations

from src.core.module_registry import ModuleMetadata

from .adaptive_weighted import AdaptiveICWeightedModel, AdaptiveWeightConfig, FactorDecayConfig
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
    register_model(
        registry,
        ModuleMetadata(
            name='adaptive_ic_weighted',
            version='v1',
            category='model',
            description='Adaptive IC-weighted average with exponential decay and factor decay monitoring.',
            owner='platform',
            inputs=('feature_panel', 'label_panel', 'rebalance_dates'),
            outputs=('predictions', 'split_metrics', 'feature_importance', 'model_registry'),
            tags=('adaptive', 'weighted', 'ic', 'decay'),
        ),
        builder=lambda config: AdaptiveICWeightedModel(
            feature_names=[],
            config=AdaptiveWeightConfig(
                ic_lookback=config.params.get('ic_lookback', 120) if config.params else 120,
                ic_half_life=config.params.get('ic_half_life', 20) if config.params else 20,
                decay_config=FactorDecayConfig(
                    decay_lookback=config.params.get('decay_lookback', 60) if config.params else 60,
                    decay_threshold_ic=config.params.get('decay_threshold_ic', -0.03) if config.params else -0.03,
                    decay_consecutive_neg=config.params.get('decay_consecutive_neg', 5) if config.params else 5,
                    decay_weight_cut=config.params.get('decay_weight_cut', 0.5) if config.params else 0.5,
                    decay_min_positive_rate=config.params.get('decay_min_positive_rate', 0.35) if config.params else 0.35,
                ),
                enable_decay_monitor=config.params.get('enable_decay_monitor', True) if config.params else True,
                min_factor_weight=config.params.get('min_factor_weight', 0.05) if config.params else 0.05,
            ),
        ),
    )
    return registry
