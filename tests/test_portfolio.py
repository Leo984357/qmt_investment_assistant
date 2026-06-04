import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.portfolio.construction import build_target_weights, _active_gross_exposure


class TestActiveGrossExposure:
    def test_full_exposure_when_market_above_short_ma(self):
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        market = pd.Series(np.linspace(100, 110, 200), index=dates)
        result = _active_gross_exposure(
            signal_date=dates[-1],
            market_reference=market,
            gross_exposure=0.95,
            defensive_gross=0.65,
            market_filter_lookback=20,
            market_filter_threshold=-0.02,
            risk_model='qmt_style_ladder',
            risk_ma_short_window=60,
            risk_ma_long_window=120,
            risk_momentum_window=20,
            risk_mid_exposure=0.85,
            risk_low_exposure=0.65,
            risk_crash_exposure=0.45,
        )
        assert result == 0.95

    def test_mid_exposure_when_below_short_ma_above_long_ma(self):
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        values = np.concatenate([np.linspace(100, 200, 140), np.linspace(199, 190, 60)])
        market = pd.Series(values, index=dates)
        result = _active_gross_exposure(
            signal_date=dates[-1],
            market_reference=market,
            gross_exposure=0.95,
            defensive_gross=0.65,
            market_filter_lookback=20,
            market_filter_threshold=-0.02,
            risk_model='qmt_style_ladder',
            risk_ma_short_window=60,
            risk_ma_long_window=120,
            risk_momentum_window=20,
            risk_mid_exposure=0.85,
            risk_low_exposure=0.65,
            risk_crash_exposure=0.45,
        )
        assert result == 0.85

    def test_uses_simple_momentum_when_not_qmt_ladder(self):
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        market = pd.Series(np.linspace(100, 95, 30), index=dates)
        result = _active_gross_exposure(
            signal_date=dates[-1],
            market_reference=market,
            gross_exposure=0.95,
            defensive_gross=0.65,
            market_filter_lookback=20,
            market_filter_threshold=0.0,
            risk_model='two_tier_momentum',
            risk_ma_short_window=60,
            risk_ma_long_window=120,
            risk_momentum_window=20,
            risk_mid_exposure=0.85,
            risk_low_exposure=0.65,
            risk_crash_exposure=0.45,
        )
        assert result == 0.65


class TestBuildTargetWeights:
    def test_returns_empty_on_no_predictions(self):
        result = build_target_weights(
            predictions=pd.DataFrame(columns=['trade_date', 'symbol', 'score', 'model_type', 'fallback_used']),
            universe_membership=pd.DataFrame(columns=['universe_name', 'trade_date', 'symbol']),
            tradability=pd.DataFrame(columns=['trade_date', 'symbol', 'is_st', 'is_new_listing', 'is_tradable']),
            market_reference=pd.Series(dtype=float),
            trade_calendar=pd.DataFrame(columns=['trade_date']),
            universe_name='test',
            top_n=5,
            gross_exposure=0.95,
            defensive_gross=0.65,
            max_single_weight=0.25,
            market_filter_lookback=20,
            market_filter_threshold=-0.02,
            trade_delay_days=1,
        )
        assert result.target_weights.empty
        assert result.filtered_candidates.empty

    def test_selects_top_n_candidates(self):
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        predictions = pd.DataFrame({
            'trade_date': [dates[0]] * 6,
            'symbol': [f'A{i}' for i in range(6)],
            'score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            'model_type': ['lgb'] * 6,
            'fallback_used': [False] * 6,
        })
        universe = pd.DataFrame({
            'universe_name': ['test'] * 5,
            'trade_date': [dates[0]] * 5,
            'symbol': [f'A{i}' for i in range(5)],
        })
        tradability = pd.DataFrame({
            'trade_date': [dates[1]] * 5,
            'symbol': [f'A{i}' for i in range(5)],
            'is_st': [False] * 5,
            'is_new_listing': [False] * 5,
            'is_tradable': [True] * 5,
        })
        calendar = pd.DataFrame({'trade_date': dates})
        market = pd.Series(np.ones(10), index=pd.date_range('2023-12-25', periods=10, freq='D'))

        result = build_target_weights(
            predictions=predictions,
            universe_membership=universe,
            tradability=tradability,
            market_reference=market,
            trade_calendar=calendar,
            universe_name='test',
            top_n=3,
            gross_exposure=0.95,
            defensive_gross=0.65,
            max_single_weight=0.25,
            market_filter_lookback=20,
            market_filter_threshold=-0.02,
            trade_delay_days=1,
        )
        assert len(result.target_weights) == 3
        assert result.target_weights['symbol'].tolist() == ['A0', 'A1', 'A2']
