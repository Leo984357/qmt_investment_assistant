"""Tests for evaluation suites cost sensitivity path."""

import numpy as np
import pandas as pd
from src.evaluation.suites import _compute_cost_sensitivity


def test_cost_sensitivity_uses_strategy_return():
    """Test that cost sensitivity computes strategy_return correctly.
    
    This tests the critical path that was broken: using strategy_return
    without defining it.
    """
    # Create minimal nav DataFrame with required fields
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    nav = pd.DataFrame({
        'trade_date': dates,
        'nav': np.linspace(1.0, 1.25, 100),  # 25% return
        'equity': np.linspace(1e6, 1.25e6, 100),
    })
    
    # Create trades with fee column
    trades = pd.DataFrame({
        'trade_date': dates[::10],
        'symbol': ['000001.SZ'] * 10,
        'execution_date': dates[::10],
        'trade_value': [100000] * 10,
        'fee': [75.0] * 10,  # 750 total fees
        'cost': [75.0] * 10,
    })
    
    # Run cost sensitivity analysis
    result = _compute_cost_sensitivity(nav, trades)
    
    # Verify it runs without NameError
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert 'total_return' in result.columns
    assert 'cost_multiplier' in result.columns
    
    # Verify strategy return is correctly calculated (25% - 750/1e6)
    # At multiplier 1.0, adjusted_return = strategy_return - cost_ratio
    # strategy_return = 0.25, cost_ratio = 750/1e6 = 0.00075
    base_row = result[result['cost_multiplier'] == 1.0].iloc[0]
    expected_return = 0.25 - 0.00075
    assert abs(base_row['total_return'] - expected_return) < 1e-6


def test_cost_sensitivity_handles_multiple_multipliers():
    """Test cost sensitivity with multiple cost multipliers."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    nav = pd.DataFrame({
        'trade_date': dates,
        'nav': np.linspace(1.0, 1.10, 50),  # 10% return
        'equity': np.linspace(1e6, 1.10e6, 50),
    })
    
    trades = pd.DataFrame({
        'trade_date': dates[::10],
        'symbol': ['000001.SZ'] * 5,
        'execution_date': dates[::10],
        'fee': [100.0] * 5,  # 500 total
        'cost': [100.0] * 5,
    })
    
    result = _compute_cost_sensitivity(nav, trades, cost_multipliers=[0.5, 1.0, 2.0, 3.0])
    
    assert len(result) == 4
    assert list(result['cost_multiplier']) == [0.5, 1.0, 2.0, 3.0]
    
    # At 3x multiplier, cost should be 3x base
    row_3x = result[result['cost_multiplier'] == 3.0].iloc[0]
    assert row_3x['total_cost'] == 1500.0  # 500 * 3


def test_cost_sensitivity_empty_nav():
    """Test cost sensitivity with empty nav returns empty DataFrame."""
    empty_nav = pd.DataFrame(columns=['trade_date', 'nav'])
    trades = pd.DataFrame({'fee': [100]})
    
    result = _compute_cost_sensitivity(empty_nav, trades)
    
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_cost_sensitivity_empty_trades():
    """Test cost sensitivity with empty trades returns empty DataFrame."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    nav = pd.DataFrame({
        'trade_date': dates,
        'nav': np.linspace(1.0, 1.10, 50),
        'equity': np.linspace(1e6, 1.10e6, 50),
    })
    empty_trades = pd.DataFrame(columns=['trade_date', 'symbol', 'fee'])
    
    result = _compute_cost_sensitivity(nav, empty_trades)
    
    assert isinstance(result, pd.DataFrame)
    # Empty trades means we can't compute cost sensitivity
    assert result.empty
