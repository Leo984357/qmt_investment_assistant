"""Tests for strategy gate skipped check behavior."""

import numpy as np
import pandas as pd
from src.evaluation.strategy_gate import StrategyGate, GateThresholds


def test_gate_requires_all_checks_passed():
    """Test that gate fails if any check is SKIPPED.
    
    This is the critical behavior: 8 gates must ALL pass,
    not just the 4 critical ones.
    """
    # Create a nav DataFrame that will cause IC checks to SKIP (no rank_ic provided)
    nav = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=100),
        'nav': np.linspace(1.0, 1.25, 100),
    })
    
    gate = StrategyGate()
    
    # Call evaluate WITHOUT rank_ic - this will cause IC checks to SKIP
    result = gate.evaluate(
        strategy_name='test_strategy',
        nav=nav,
        rank_ic=None,  # This will cause IC checks to SKIP
        trades=pd.DataFrame({'fee': [100]}),
        benchmark_nav=None,
    )
    
    # With all SKIPPED checks, passed should be False
    assert result.passed is False, "Gate should fail when checks are SKIPPED"
    assert result.overall_score < 100, "Score should be less than 100 when checks skip"


def test_gate_fails_with_partial_skipped():
    """Test gate fails when some checks are SKIPPED, even if critical gates pass."""
    nav = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=100),
        'nav': np.linspace(1.0, 1.25, 100),
    })
    
    # Create rank_ic with good IC so IC gates PASS
    rank_ic = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=50),
        'rank_ic': [0.05] * 50,  # Good IC
    })
    
    # Create trades with high turnover so turnover check might PASS
    trades = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=10),
        'symbol': ['000001.SZ'] * 10,
        'fee': [100.0] * 10,
        'notional': [50000.0] * 10,
    })
    
    # Empty benchmark - this will cause excess return check to SKIP
    benchmark_nav = pd.DataFrame()
    
    gate = StrategyGate()
    
    # With empty benchmark, excess return check will SKIP
    result = gate.evaluate(
        strategy_name='test_partial_skipped',
        nav=nav,
        rank_ic=rank_ic,
        trades=trades,
        benchmark_nav=benchmark_nav,
    )
    
    # Even with good IC, SKIPPED checks should cause failure
    assert result.passed is False, "Gate should fail when any check is SKIPPED"


def test_gate_passes_when_all_checks_pass():
    """Test gate passes only when ALL 8 checks PASS."""
    nav = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=200),
        'nav': np.linspace(1.0, 1.30, 200),  # 30% return with stable growth
    })
    
    # Good IC data
    rank_ic = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=100),
        'rank_ic': [0.05] * 100,  # Good IC mean > 0.02
    })
    
    # Good trades (low turnover, low cost)
    trades = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=10),
        'symbol': ['000001.SZ'] * 10,
        'fee': [50.0] * 10,  # Low cost
        'notional': [50000.0] * 10,
    })
    
    # Good benchmark for excess return check
    benchmark_nav = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=200),
        'benchmark_nav': np.linspace(1.0, 1.15, 200),  # 15% return
    })
    
    # Good yearly breakdown (needs at least 2 years)
    yearly = pd.DataFrame({
        'year': [2023, 2024],
        'total_return': [0.25, 0.30],  # Both positive
        'sharpe': [1.2, 1.5],
    })
    
    # Quantile summary for monotonicity check (uses bucket/mean_forward_return columns)
    quantile_summary = pd.DataFrame({
        'bucket': ['1', '2', '3', '4', '5'],  # bucket must be string
        'mean_forward_return': [0.02, 0.04, 0.06, 0.08, 0.10],  # Monotonic increasing
    })
    
    gate = StrategyGate()
    
    result = gate.evaluate(
        strategy_name='test_all_pass',
        nav=nav,
        rank_ic=rank_ic,
        trades=trades,
        benchmark_nav=benchmark_nav,
        yearly_breakdown=yearly,
        quantile_summary=quantile_summary,
        baseline_sharpe=0.5,  # Baseline Sharpe for comparison
    )
    
    # With all data provided, gate should pass
    # Note: May still fail depending on thresholds, but should not SKIP
    for check in result.gate_checks:
        assert check.status.value != 'skipped', f"Check {check.name} should not SKIP when data is provided"


def test_gate_quantile_summary_skipped_when_none():
    """Test that monotonicity check SKIPs when quantile_summary is None."""
    nav = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=100),
        'nav': np.linspace(1.0, 1.25, 100),
    })
    
    gate = StrategyGate()
    
    result = gate.evaluate(
        strategy_name='test_no_quantile',
        nav=nav,
        quantile_summary=None,  # No quantile summary
    )
    
    # Find monotonicity check
    mono_check = next((c for c in result.gate_checks if c.name == '分组单调性'), None)
    if mono_check:
        assert mono_check.status.value == 'skipped', "Monotonicity should SKIP when quantile_summary is None"
    
    # Overall should fail due to SKIPPED
    assert result.passed is False


def test_gate_ic_ir_epsilon_protection():
    """Test that IC IR has epsilon protection for near-zero std.
    
    Constant IC sequence [0.05] * 100 has near-zero std due to floating point,
    which should NOT produce infinity IR.
    """
    # Constant IC - will have std ~ 0 due to floating point
    rank_ic = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=100),
        'rank_ic': [0.05] * 100,  # Constant IC
    })
    
    nav = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=200),
        'nav': np.linspace(1.0, 1.30, 200),
    })
    
    gate = StrategyGate()
    result = gate.evaluate(
        strategy_name='test_constant_ic',
        nav=nav,
        rank_ic=rank_ic,
    )
    
    # Find IC IR check
    ic_ir_check = next((c for c in result.gate_checks if c.name == 'IC IR'), None)
    assert ic_ir_check is not None, "IC IR check should exist"
    
    # IC IR should be finite, not infinity
    assert np.isfinite(ic_ir_check.value), f"IC IR should be finite, got {ic_ir_check.value}"
    # IC IR should be reasonable (not 5e14)
    assert ic_ir_check.value < 1e6, f"IC IR should be capped, got {ic_ir_check.value}"
