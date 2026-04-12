import pytest

from src.core.config import ConfigError, DecisionConfig, ExecutionConfig, StrategyConfig


def test_strategy_config_rejects_invalid_weights():
    with pytest.raises(ConfigError):
        StrategyConfig.from_mapping({'max_single_weight': 1.2})


def test_decision_config_rejects_negative_threshold():
    with pytest.raises(ConfigError):
        DecisionConfig.from_mapping({'min_rebalance_l1': -0.1})


def test_execution_config_rejects_unknown_mode():
    with pytest.raises(ConfigError):
        ExecutionConfig.from_mapping({'mode': 'paper'})
