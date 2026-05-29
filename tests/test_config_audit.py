from pathlib import Path

import yaml

from src.cli import _audit_config


def test_production_unknown_factor_is_rejected(tmp_path: Path):
    config = {
        'name': 'unknown_factor_prod',
        'data': {
            'source': 'baostock_ashare',
            'snapshot_id': 'test',
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'universe_name': 'HS300',
            'universe_mode': 'point_in_time',
            'price_adjust': 'qfq',
        },
        'features': {
            'set_name': 'unknown',
            'version': 'v1',
            'names': ['factor_not_in_catalog'],
        },
        'label': {'name': 'fwd_return_20d', 'horizon': 20},
        'model': {
            'family': 'simple_average',
            'version': 'v1',
            'registry_stage': 'production',
            'train_window_days': 500,
            'valid_window_days': 0,
            'min_train_samples': 10,
        },
        'signal': {'name': 'cross_sectional_score', 'version': 'v1', 'params': {}},
        'portfolio': {'top_n': 10, 'constructor': 'qmt_topn_equal_weight'},
        'backtest': {'initial_cash': 1000000, 'rebalance_frequency_days': 10},
        'evaluation': {'suite': 'basic_factor_diagnostics', 'version': 'v1', 'params': {}},
    }
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')

    audit = _audit_config(str(path))

    assert audit['status'] == 'error'
    assert 'Unknown factors' in audit['message']
