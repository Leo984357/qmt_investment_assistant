import pytest

from src.core.config import (
    AppConfig,
    CommonConfig,
    ConfigError,
    DecisionConfig,
    ExecutionConfig,
    StrategyConfig,
)


class TestCommonConfig:
    def test_defaults(self):
        cfg = CommonConfig()
        assert cfg.project_name == 'QMT Investment Assistant'
        assert cfg.latest_n_days == 180

    def test_from_mapping(self):
        cfg = CommonConfig.from_mapping({'project_name': 'Test', 'latest_n_days': 90})
        assert cfg.project_name == 'Test'
        assert cfg.latest_n_days == 90

    def test_rejects_negative_latest_n_days(self):
        with pytest.raises(ConfigError, match='latest_n_days'):
            CommonConfig.from_mapping({'latest_n_days': -1})

    def test_rejects_zero_latest_n_days(self):
        with pytest.raises(ConfigError, match='latest_n_days'):
            CommonConfig.from_mapping({'latest_n_days': 0})

    def test_rejects_unknown_keys(self):
        with pytest.raises(ConfigError, match='未知配置'):
            CommonConfig.from_mapping({'invalid_key': 'value'})


class TestStrategyConfig:
    def test_defaults(self):
        cfg = StrategyConfig()
        assert cfg.top_n == 5
        assert cfg.max_single_weight == 0.25

    def test_from_mapping(self):
        cfg = StrategyConfig.from_mapping({'top_n': 10, 'min_price': 5.0})
        assert cfg.top_n == 10
        assert cfg.min_price == 5.0

    def test_rejects_zero_top_n(self):
        with pytest.raises(ConfigError, match='top_n'):
            StrategyConfig.from_mapping({'top_n': 0})

    def test_rejects_small_momentum_lookback(self):
        with pytest.raises(ConfigError, match='momentum_lookback'):
            StrategyConfig.from_mapping({'momentum_lookback': 1})

    def test_rejects_long_lookback_not_greater_than_momentum(self):
        with pytest.raises(ConfigError, match='long_lookback'):
            StrategyConfig.from_mapping({'momentum_lookback': 20, 'long_lookback': 15})

    def test_rejects_max_single_weight_exceeding_target_gross(self):
        with pytest.raises(ConfigError, match='max_single_weight'):
            StrategyConfig.from_mapping({'max_single_weight': 0.9, 'target_gross': 0.8})

    def test_rejects_negative_min_price(self):
        with pytest.raises(ConfigError, match='min_price'):
            StrategyConfig.from_mapping({'min_price': -1})


class TestDecisionConfig:
    def test_defaults(self):
        cfg = DecisionConfig()
        assert cfg.min_rebalance_l1 == 0.12

    def test_rejects_negative_values(self):
        with pytest.raises(ConfigError, match='min_rebalance_l1'):
            DecisionConfig.from_mapping({'min_rebalance_l1': -0.1})

    def test_accepts_zero_min_rebalance(self):
        cfg = DecisionConfig.from_mapping({'min_rebalance_l1': 0})
        assert cfg.min_rebalance_l1 == 0


class TestExecutionConfig:
    def test_defaults(self):
        cfg = ExecutionConfig()
        assert cfg.mode == 'mock'

    def test_accepts_qmt_mode(self):
        cfg = ExecutionConfig.from_mapping({'mode': 'qmt'})
        assert cfg.mode == 'qmt'

    def test_rejects_unknown_mode(self):
        with pytest.raises(ConfigError, match='mode'):
            ExecutionConfig.from_mapping({'mode': 'paper'})

    def test_rejects_non_positive_lot_size(self):
        with pytest.raises(ConfigError, match='lot_size'):
            ExecutionConfig.from_mapping({'lot_size': 0})

    def test_rejects_negative_slippage(self):
        with pytest.raises(ConfigError, match='slippage'):
            ExecutionConfig.from_mapping({'slippage_bps': -1})


class TestAppConfig:
    def test_db_path_property(self):
        cfg = AppConfig(
            common=CommonConfig(db_path='db/test.sqlite'),
            strategy=StrategyConfig(),
            decision=DecisionConfig(),
            execution=ExecutionConfig(),
        )
        assert str(cfg.db_path).endswith('db/test.sqlite')

    def test_to_dict_structure(self):
        cfg = AppConfig(
            common=CommonConfig(),
            strategy=StrategyConfig(),
            decision=DecisionConfig(),
            execution=ExecutionConfig(),
        )
        d = cfg.to_dict()
        assert 'common' in d
        assert 'strategy' in d
        assert 'decision' in d
        assert 'execution' in d
