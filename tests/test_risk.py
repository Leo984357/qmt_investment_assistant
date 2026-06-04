import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.risk.risk_engine import RiskConstraintEngine, RiskConstraints, RiskCheckResult


class TestRiskConstraints:
    def test_defaults(self):
        c = RiskConstraints()
        assert c.max_single_weight == 0.05
        assert c.min_positions == 10
        assert c.max_gross_exposure == 1.0


class TestRiskConstraintEngine:
    def test_init_with_defaults(self):
        engine = RiskConstraintEngine()
        assert engine.constraints.min_positions == 10

    def test_init_with_custom_constraints(self):
        c = RiskConstraints(max_single_weight=0.1)
        engine = RiskConstraintEngine(constraints=c)
        assert engine.constraints.max_single_weight == 0.1

    def test_apply_constraints_clamps_single_weight(self):
        engine = RiskConstraintEngine(RiskConstraints(max_single_weight=0.1, min_positions=1))
        weights = pd.DataFrame({
            'symbol': ['A', 'B', 'C'],
            'target_weight': [0.15, 0.05, 0.05],
            'gross_exposure': [0.20, 0.20, 0.20],
        })
        result, checks = engine.apply_constraints(
            target_weights=weights,
            current_positions={},
            prices={'A': 100, 'B': 50, 'C': 30},
            as_of_date=datetime(2024, 1, 1),
        )
        assert result['target_weight'].max() <= 0.1
        assert any(not c.passed for c in checks if c.check_name == 'max_single_weight')

    def test_apply_constraints_passes_within_limits(self):
        engine = RiskConstraintEngine(RiskConstraints(max_single_weight=0.2, min_positions=1))
        weights = pd.DataFrame({
            'symbol': ['A', 'B'],
            'target_weight': [0.1, 0.1],
            'gross_exposure': [0.2, 0.2],
        })
        result, checks = engine.apply_constraints(
            target_weights=weights,
            current_positions={},
            prices={'A': 100, 'B': 50},
            as_of_date=datetime(2024, 1, 1),
        )
        assert all(c.passed for c in checks)

    def test_normalizes_weights_to_gross_exposure(self):
        engine = RiskConstraintEngine(RiskConstraints(max_single_weight=0.5, max_gross_exposure=0.8))
        weights = pd.DataFrame({
            'symbol': ['A', 'B'],
            'target_weight': [0.3, 0.3],
            'gross_exposure': [0.8, 0.8],
        })
        result, _ = engine.apply_constraints(
            target_weights=weights,
            current_positions={},
            prices={'A': 100, 'B': 50},
            as_of_date=datetime(2024, 1, 1),
        )
        assert abs(result['target_weight'].sum() - 0.8) < 1e-6

    def test_empty_weights_returns_empty(self):
        engine = RiskConstraintEngine()
        weights = pd.DataFrame(columns=['symbol', 'target_weight'])
        result, _ = engine.apply_constraints(
            target_weights=weights,
            current_positions={},
            prices={},
            as_of_date=datetime(2024, 1, 1),
        )
        assert result.empty

    def test_history_accumulates(self):
        engine = RiskConstraintEngine(RiskConstraints(max_single_weight=0.05))
        weights = pd.DataFrame({
            'symbol': ['A'],
            'target_weight': [0.99],
        })
        engine.apply_constraints(
            target_weights=weights,
            current_positions={},
            prices={'A': 100},
            as_of_date=datetime(2024, 1, 1),
        )
        assert len(engine._history) > 0
