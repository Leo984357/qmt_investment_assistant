from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PortfolioBuildResult:
    target_weights: pd.DataFrame
    filtered_candidates: pd.DataFrame


def build_target_weights(
    predictions: pd.DataFrame,
    universe_membership: pd.DataFrame,
    tradability: pd.DataFrame,
    market_reference: pd.Series,
    trade_calendar: pd.DataFrame,
    universe_name: str,
    top_n: int,
    gross_exposure: float,
    defensive_gross: float,
    max_single_weight: float,
    market_filter_lookback: int,
    market_filter_threshold: float,
    trade_delay_days: int,
    risk_model: str = 'two_tier_momentum',
    risk_ma_short_window: int = 60,
    risk_ma_long_window: int = 120,
    risk_momentum_window: int = 20,
    risk_mid_exposure: float = 0.85,
    risk_low_exposure: float = 0.65,
    risk_crash_exposure: float = 0.45,
    candidate_filter_mode: str = 'strict_ashare',
) -> PortfolioBuildResult:
    membership = universe_membership.loc[universe_membership['universe_name'] == universe_name].copy()
    membership['trade_date'] = pd.to_datetime(membership['trade_date'])
    tradability = tradability.copy()
    tradability['trade_date'] = pd.to_datetime(tradability['trade_date'])
    calendar = sorted(pd.to_datetime(trade_calendar['trade_date']).tolist())
    next_trade_date = {
        calendar[idx]: calendar[idx + trade_delay_days]
        for idx in range(len(calendar) - trade_delay_days)
    }
    market_reference = market_reference.sort_index()
    market_momentum = market_reference.pct_change(market_filter_lookback).fillna(0.0)

    target_frames: list[pd.DataFrame] = []
    filtered_frames: list[pd.DataFrame] = []

    for signal_date, signal_scores in predictions.groupby('trade_date'):
        signal_date = pd.Timestamp(signal_date)
        execution_date = next_trade_date.get(signal_date)
        if execution_date is None:
            continue

        universe_slice = membership.loc[membership['trade_date'] == signal_date, ['symbol']].copy()
        tradability_date = execution_date if candidate_filter_mode == 'strict_ashare' else signal_date
        tradability_slice = tradability.loc[tradability['trade_date'] == tradability_date, ['symbol', 'is_st', 'is_new_listing', 'is_tradable']]
        candidate = signal_scores[['trade_date', 'symbol', 'score', 'model_type', 'fallback_used']].copy()
        candidate = candidate.merge(universe_slice.assign(in_universe=True), on='symbol', how='left')
        candidate = candidate.merge(tradability_slice, on='symbol', how='left')
        for column in ['in_universe', 'is_st', 'is_new_listing', 'is_tradable']:
            candidate[column] = candidate[column].astype('boolean').fillna(False).astype(bool)
        candidate['reason'] = 'selected'
        candidate.loc[~candidate['in_universe'], 'reason'] = 'not_in_universe'
        if candidate_filter_mode == 'strict_ashare':
            candidate.loc[candidate['in_universe'] & candidate['is_st'], 'reason'] = 'is_st'
            candidate.loc[candidate['in_universe'] & candidate['is_new_listing'], 'reason'] = 'new_listing'
            candidate.loc[candidate['in_universe'] & ~candidate['is_st'] & ~candidate['is_new_listing'] & ~candidate['is_tradable'], 'reason'] = 'temporarily_restricted'

        eligible = candidate.loc[candidate['reason'] == 'selected'].sort_values('score', ascending=False).copy()
        selected = eligible.head(top_n).copy()
        if not selected.empty:
            selected_symbols = set(selected['symbol'])
            candidate.loc[candidate['reason'] == 'selected', 'reason'] = candidate['symbol'].map(
                lambda symbol: 'selected' if symbol in selected_symbols else 'below_cutoff'
            )
        filtered_frames.append(candidate.assign(signal_date=signal_date, execution_date=execution_date))

        if selected.empty:
            continue

        current_market_momentum = float(market_momentum.get(signal_date, 0.0))
        active_gross = _active_gross_exposure(
            signal_date=signal_date,
            market_reference=market_reference,
            gross_exposure=gross_exposure,
            defensive_gross=defensive_gross,
            market_filter_lookback=market_filter_lookback,
            market_filter_threshold=market_filter_threshold,
            risk_model=risk_model,
            risk_ma_short_window=risk_ma_short_window,
            risk_ma_long_window=risk_ma_long_window,
            risk_momentum_window=risk_momentum_window,
            risk_mid_exposure=risk_mid_exposure,
            risk_low_exposure=risk_low_exposure,
            risk_crash_exposure=risk_crash_exposure,
        )
        weight = min(max_single_weight, active_gross / max(len(selected), 1))
        target_frames.append(
            selected.reset_index(drop=True).assign(
                signal_date=signal_date,
                execution_date=execution_date,
                rank=lambda frame: range(1, len(frame) + 1),
                target_weight=weight,
                gross_exposure=active_gross,
                market_momentum=current_market_momentum,
            )[
                ['signal_date', 'execution_date', 'symbol', 'rank', 'score', 'target_weight', 'gross_exposure', 'market_momentum', 'model_type', 'fallback_used']
            ]
        )

    target_weights = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame(
        columns=['signal_date', 'execution_date', 'symbol', 'rank', 'score', 'target_weight', 'gross_exposure', 'market_momentum', 'model_type', 'fallback_used']
    )
    filtered_candidates = pd.concat(filtered_frames, ignore_index=True) if filtered_frames else pd.DataFrame(
        columns=['signal_date', 'execution_date', 'trade_date', 'symbol', 'score', 'model_type', 'fallback_used', 'in_universe', 'is_st', 'is_new_listing', 'is_tradable', 'reason']
    )
    return PortfolioBuildResult(target_weights=target_weights, filtered_candidates=filtered_candidates)


def _active_gross_exposure(
    signal_date: pd.Timestamp,
    market_reference: pd.Series,
    gross_exposure: float,
    defensive_gross: float,
    market_filter_lookback: int,
    market_filter_threshold: float,
    risk_model: str,
    risk_ma_short_window: int,
    risk_ma_long_window: int,
    risk_momentum_window: int,
    risk_mid_exposure: float,
    risk_low_exposure: float,
    risk_crash_exposure: float,
) -> float:
    if risk_model == 'qmt_style_ladder':
        history = market_reference.loc[market_reference.index <= signal_date].dropna()
        if len(history) < max(risk_ma_long_window, risk_momentum_window + 1):
            return float(gross_exposure)
        idx_now = float(history.iloc[-1])
        ma_short = float(history.tail(risk_ma_short_window).mean())
        ma_long = float(history.tail(risk_ma_long_window).mean())
        momentum = idx_now / float(history.iloc[-(risk_momentum_window + 1)]) - 1.0
        if idx_now >= ma_short:
            return float(gross_exposure)
        if idx_now >= ma_long:
            return float(risk_mid_exposure)
        if momentum < 0.0:
            return float(risk_crash_exposure)
        return float(risk_low_exposure)
    current_market_momentum = float(market_reference.pct_change(market_filter_lookback).fillna(0.0).get(signal_date, 0.0))
    return float(gross_exposure if current_market_momentum >= market_filter_threshold else defensive_gross)
