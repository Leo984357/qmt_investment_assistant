"""Tests for artifact numpy type serialization."""

import numpy as np
import pandas as pd
import yaml
import json
from pathlib import Path
from src.experiment.artifact import _to_native, ExperimentArtifact, ExperimentMetrics, save_artifact
import tempfile
import shutil


def test_to_native_handles_numpy_types():
    """Test that _to_native correctly converts numpy types to native Python types."""
    data = {
        'int64': np.int64(42),
        'float64': np.float64(3.14),
        'numpy_array': np.array([1, 2, 3]),
        'nested': {
            'float32': np.float32(2.71),
            'list_with_numpy': [np.int64(1), np.float64(2.0)],
        },
        'timestamp': pd.Timestamp('2024-01-01'),
        'regular_str': 'hello',
        'regular_int': 42,
    }
    
    result = _to_native(data)
    
    assert isinstance(result['int64'], int)
    assert result['int64'] == 42
    assert isinstance(result['float64'], float)
    assert result['float64'] == 3.14
    assert isinstance(result['numpy_array'], list)
    assert result['numpy_array'] == [1, 2, 3]
    assert isinstance(result['nested']['float32'], float)
    assert isinstance(result['nested']['list_with_numpy'][0], int)
    assert isinstance(result['timestamp'], str)
    assert result['regular_str'] == 'hello'
    assert result['regular_int'] == 42


def test_to_native_handles_nan():
    """Test that _to_native converts NaN to None."""
    data = {
        'nan_value': np.nan,
        'pd_na': pd.NA,
    }
    
    result = _to_native(data)
    
    assert result['nan_value'] is None
    assert result['pd_na'] is None


def test_artifact_yaml_serialization():
    """Test that ExperimentArtifact can be serialized to YAML without RepresenterError."""
    metrics = ExperimentMetrics(
        total_return=np.float64(0.25),
        sharpe_ratio=np.float64(1.234),
        max_drawdown=np.float64(0.08),
        annual_return=np.float64(0.18),
        annual_volatility=np.float64(0.15),
        ic_mean=np.float64(0.045),
        ic_ir=np.float64(0.52),
        avg_turnover=np.float64(0.025),
        total_cost=np.float64(5000.0),
        num_trades=np.int64(120),
        win_rate=np.float64(0.58),
        excess_return=np.float64(0.12),
    )
    
    # Add test for real backtest field names
    nav_df = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=100),
        'nav': np.linspace(1.0, 1.25, 100),
    })
    # Real backtest uses: fee, notional, trade_date (not execution_date, trade_value)
    trades_df = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=10),
        'symbol': ['000001.SZ'] * 10,
        'notional': [100000.0] * 10,
        'fee': [75.0] * 10,
    })
    
    # Test _compute_metrics with real field names
    rank_ic_df = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=50),
        'rank_ic': np.random.randn(50) * 0.05,
    })
    
    from src.experiment.artifact import _compute_metrics
    metrics = _compute_metrics(nav_df, trades_df, rank_ic_df, None)
    
    # Verify cost is correctly computed from fee
    assert metrics.total_cost == 750.0, f"Expected total_cost=750.0, got {metrics.total_cost}"
    # Verify num_trades
    assert metrics.num_trades == 10, f"Expected num_trades=10, got {metrics.num_trades}"
    
    # Test with cost column (old format)
    trades_old = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=5),
        'symbol': ['000001.SZ'] * 5,
        'trade_value': [50000.0] * 5,
        'cost': [50.0] * 5,
        'execution_date': pd.date_range('2024-01-01', periods=5),
    })
    metrics_old = _compute_metrics(nav_df, trades_old, rank_ic_df, None)
    assert metrics_old.total_cost == 250.0, f"Expected total_cost=250.0, got {metrics_old.total_cost}"


def test_artifact_turnover_from_nav():
    """Test that avg_turnover uses nav['turnover'] when available."""
    nav_df = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=100),
        'nav': np.linspace(1.0, 1.25, 100),
        'turnover': [0.05] * 100,  # 5% daily turnover
    })
    trades_df = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=10),
        'symbol': ['000001.SZ'] * 10,
        'notional': [100000.0] * 10,
        'fee': [75.0] * 10,
    })
    rank_ic_df = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=50),
        'rank_ic': np.random.randn(50) * 0.05,
    })
    
    from src.experiment.artifact import _compute_metrics
    metrics = _compute_metrics(nav_df, trades_df, rank_ic_df, None)
    
    # Should use nav['turnover'].mean() = 0.05, not notional-based calculation
    assert abs(metrics.avg_turnover - 0.05) < 1e-6, f"Expected avg_turnover=0.05, got {metrics.avg_turnover}"


def test_artifact_turnover_from_notional_divided_by_equity():
    """Test that avg_turnover divides notional by equity when fallback."""
    nav_df = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=100),
        'nav': np.linspace(1.0, 1.25, 100),
        'equity': np.linspace(1e6, 1.25e6, 100),  # 初始资金100万
    })
    # Daily notional = 50000, equity = 1e6, expected turnover = 5%
    trades_df = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=10),
        'symbol': ['000001.SZ'] * 10,
        'notional': [50000.0] * 10,  # 5万/天
        'fee': [75.0] * 10,
    })
    rank_ic_df = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=50),
        'rank_ic': np.random.randn(50) * 0.05,
    })
    
    from src.experiment.artifact import _compute_metrics
    metrics = _compute_metrics(nav_df, trades_df, rank_ic_df, None)
    
    # Should be notional/equity = 50000/1e6 = 0.05 = 5%
    expected_turnover = 50000.0 / 1e6
    assert abs(metrics.avg_turnover - expected_turnover) < 1e-6, \
        f"Expected avg_turnover={expected_turnover}, got {metrics.avg_turnover}"
    
    artifact = ExperimentArtifact(
        run_id='test1234',
        experiment_name='test_experiment',
        config_hash='abc123' * 10,
        data_snapshot_hash='def456' * 4,
        research_contract_version='v1',
        timestamp='2024-01-01T00:00:00',
        metrics=metrics,
        gate_passed=True,
        gate_score=85.5,
        gate_details={'checks': []},
        feature_names=['roe', 'earnings_yield'],
        model_family='ridge_regression',
        backtest_start='2022-01-01',
        backtest_end='2024-12-31',
        notes='Test artifact',
        tags=['test'],
    )
    
    # Test YAML serialization (the critical path that was failing)
    artifact_dict = artifact.to_dict()
    native_dict = _to_native(artifact_dict)
    
    # This should NOT raise RepresenterError
    yaml_output = yaml.safe_dump(native_dict, allow_unicode=True, sort_keys=False)
    assert isinstance(yaml_output, str)
    assert 'test_experiment' in yaml_output
    assert 'ridge_regression' in yaml_output
    
    # Test JSON serialization still works (needs native types)
    json_output = json.dumps(_to_native(artifact_dict), ensure_ascii=False, indent=2)
    assert isinstance(json_output, str)


def test_save_artifact_produces_yaml():
    """Test that save_artifact produces a valid YAML file."""
    metrics = ExperimentMetrics(
        total_return=0.25,
        sharpe_ratio=1.234,
        max_drawdown=0.08,
        annual_return=0.18,
        annual_volatility=0.15,
        ic_mean=0.045,
        ic_ir=0.52,
        avg_turnover=0.025,
        total_cost=5000.0,
        num_trades=120,
        win_rate=0.58,
        excess_return=0.12,
    )
    
    artifact = ExperimentArtifact(
        run_id='yaml_test',
        experiment_name='yaml_serialization_test',
        config_hash='a' * 64,
        data_snapshot_hash='b' * 16,
        research_contract_version='v1',
        timestamp='2024-01-01T00:00:00',
        metrics=metrics,
        gate_passed=True,
        gate_score=85.5,
        gate_details={'checks': []},
        feature_names=['roe'],
        model_family='ridge',
        backtest_start='2022-01-01',
        backtest_end='2024-12-31',
    )
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        saved_path = save_artifact(artifact, temp_dir)
        
        # Verify YAML file exists and is valid
        yaml_path = temp_dir / f"{artifact.run_id}.yaml"
        assert yaml_path.exists(), "YAML file should be created"
        
        # Verify it can be parsed back
        with open(yaml_path, 'r') as f:
            parsed = yaml.safe_load(f)
        assert parsed['experiment_name'] == 'yaml_serialization_test'
        assert parsed['gate_passed'] is True
        
        # Verify JSON also exists
        json_path = temp_dir / f"{artifact.run_id}.json"
        assert json_path.exists(), "JSON file should be created"
    finally:
        shutil.rmtree(temp_dir)
