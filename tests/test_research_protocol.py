from pathlib import Path

import pytest
import yaml

from src.experiment.spec import load_experiment_spec


def _base_config(registry_stage: str = 'research') -> dict:
    return {
        'name': f'protocol_{registry_stage}',
        'description': 'Protocol validation test.',
        'data': {
            'source': 'baostock_ashare',
            'snapshot_id': 'test',
            'start_date': '2022-01-01',
            'end_date': '2024-12-31',
            'universe_name': 'HS300',
            'universe_mode': 'point_in_time',
            'price_adjust': 'qfq',
        },
        'features': {
            'set_name': 'protocol_test',
            'version': 'v1',
            'names': ['mom250'],
        },
        'label': {'name': 'fwd_return_20d', 'horizon': 20},
        'model': {
            'family': 'simple_average',
            'version': 'v1',
            'registry_stage': registry_stage,
            'fallback_model': 'mom250_zscore',
            'train_window_days': 500,
            'valid_window_days': 0,
            'test_window_days': 60,
            'min_train_samples': 100,
        },
        'signal': {'name': 'cross_sectional_score', 'version': 'v1', 'params': {}},
        'portfolio': {
            'top_n': 40,
            'weighting': 'equal',
            'gross_exposure': 1.0,
            'defensive_gross': 0.5,
            'max_single_weight': 0.04,
            'cash_buffer': 0.0,
            'min_trade_value': 3000,
            'market_filter_lookback': 60,
            'market_filter_threshold': 0.0,
        },
        'backtest': {
            'initial_cash': 1000000,
            'lot_size': 100,
            'commission_bps': 0.75,
            'stamp_duty_bps': 10,
            'slippage_bps': 5,
            'rebalance_frequency_days': 10,
            'trade_delay_days': 1,
        },
        'evaluation': {'suite': 'basic_factor_diagnostics', 'version': 'v1', 'params': {}},
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / 'experiment.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return path


def test_research_config_without_protocol_defaults_to_discovery(tmp_path: Path):
    path = _write_config(tmp_path, _base_config('research'))

    spec = load_experiment_spec(path)

    assert spec.research_protocol.stage == 'discovery'
    assert spec.research_protocol.data_mined is True


def test_production_config_without_explicit_safe_protocol_is_rejected(tmp_path: Path):
    path = _write_config(tmp_path, _base_config('production'))

    with pytest.raises(ValueError, match='violates research protocol'):
        load_experiment_spec(path)


def test_production_config_requires_locked_non_data_mined_protocol(tmp_path: Path):
    config = _base_config('production')
    config['research_protocol'] = {
        'stage': 'production',
        'hypothesis_id': 'mom250_long_horizon_v1',
        'candidate_id': 'hs300_mom250_v1',
        'data_mined': False,
        'frozen_after': '2024-12-31',
        'validation_window': {'start': '2023-01-01', 'end': '2023-12-31'},
        'holdout_window': {'start': '2024-01-01', 'end': '2024-12-31'},
        'allowed_change_after_freeze': 'none',
    }
    config['multiple_testing'] = {
        'enabled': True,
        'search_space_id': 'mom250_locked_search_v1',
        'candidate_count': 1,
        'factor_trial_count': 1,
        'parameter_trial_count': 1,
        'correction_method': 'single_candidate_locked_holdout',
        'random_baseline': True,
        'permutation_test': False,
        'white_noise_baseline': False,
        'stability_over_best': True,
    }
    config['risk_attribution'] = {
        'enabled': True,
        'industry_exposure': True,
        'style_exposure': True,
        'beta_exposure': True,
        'benchmark_relative_active_risk': True,
        'neutralized_ic': True,
        'return_attribution': True,
    }
    path = _write_config(tmp_path, config)

    spec = load_experiment_spec(path)

    assert spec.research_protocol.stage == 'production'
    assert spec.research_protocol.data_mined is False
    assert spec.research_protocol.allowed_change_after_freeze == 'none'
    assert spec.multiple_testing.enabled is True
    assert spec.risk_attribution.enabled is True


def test_production_config_requires_multiple_testing_and_risk_attribution(tmp_path: Path):
    config = _base_config('production')
    config['research_protocol'] = {
        'stage': 'production',
        'hypothesis_id': 'mom250_long_horizon_v1',
        'candidate_id': 'hs300_mom250_v1',
        'data_mined': False,
        'frozen_after': '2024-12-31',
        'validation_window': {'start': '2023-01-01', 'end': '2023-12-31'},
        'holdout_window': {'start': '2024-01-01', 'end': '2024-12-31'},
        'allowed_change_after_freeze': 'none',
    }
    path = _write_config(tmp_path, config)

    with pytest.raises(ValueError, match='violates professional research controls'):
        load_experiment_spec(path)


def test_regime_detection_must_be_declared_as_overlay(tmp_path: Path):
    config = _base_config('research')
    config['portfolio']['regime_detection'] = True
    path = _write_config(tmp_path, config)

    with pytest.raises(ValueError, match='overlay strategy'):
        load_experiment_spec(path)
