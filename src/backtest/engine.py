from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    nav: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame


def run_backtest(
    daily_bar: pd.DataFrame,
    target_weights: pd.DataFrame,
    tradability: pd.DataFrame,
    corporate_actions: pd.DataFrame | None,
    initial_cash: float,
    lot_size: int,
    commission_bps: float,
    stamp_duty_bps: float,
    slippage_bps: float,
    min_trade_value: float,
    execution_constraint_mode: str = 'strict_ashare',
) -> BacktestResult:
    bars = daily_bar.copy()
    bars['trade_date'] = pd.to_datetime(bars['trade_date'])
    tradability = tradability.copy()
    tradability['trade_date'] = pd.to_datetime(tradability['trade_date'])
    corporate_actions = corporate_actions.copy() if corporate_actions is not None else pd.DataFrame(columns=['event_date', 'symbol', 'event_type', 'event_value'])
    if not corporate_actions.empty:
        corporate_actions['event_date'] = pd.to_datetime(corporate_actions['event_date'])
    target_weights = target_weights.copy()
    target_weights['execution_date'] = pd.to_datetime(target_weights['execution_date'])
    target_weights['signal_date'] = pd.to_datetime(target_weights['signal_date'])

    open_matrix = bars.pivot(index='trade_date', columns='symbol', values='open').sort_index()
    close_matrix = bars.pivot(index='trade_date', columns='symbol', values='close').sort_index()
    tradability_index = tradability.set_index(['trade_date', 'symbol'])
    target_by_execution = {
        execution_date: frame.sort_values('rank').reset_index(drop=True)
        for execution_date, frame in target_weights.groupby('execution_date')
    }
    actions_by_date = {
        event_date: frame.reset_index(drop=True)
        for event_date, frame in corporate_actions.groupby('event_date')
    } if not corporate_actions.empty else {}

    positions: dict[str, int] = {}
    last_close: dict[str, float] = {}
    cash = float(initial_cash)
    nav_rows: list[dict] = []
    trade_rows: list[dict] = []
    position_rows: list[dict] = []

    for trade_date, close_prices in close_matrix.iterrows():
        open_prices = open_matrix.loc[trade_date]
        trade_notional = 0.0
        trade_cost = 0.0
        action_cash = 0.0

        if trade_date in actions_by_date:
            action_cash = _apply_corporate_actions(actions_by_date[trade_date], positions, trade_rows, trade_date)
            cash += action_cash

        if trade_date in target_by_execution:
            targets = target_by_execution[trade_date]
            total_equity = cash + sum(
                shares * _lookup_price(open_prices.get(symbol), close_prices.get(symbol), last_close.get(symbol))
                for symbol, shares in positions.items()
            )
            desired_shares: dict[str, int] = {}
            for row in targets.itertuples(index=False):
                ref_price = _lookup_price(open_prices.get(row.symbol), close_prices.get(row.symbol), None)
                if ref_price is None or ref_price <= 0:
                    continue
                target_value = total_equity * float(row.target_weight)
                raw_shares = int(np.floor(target_value / ref_price / lot_size) * lot_size)
                if raw_shares * ref_price < min_trade_value:
                    raw_shares = 0
                if raw_shares > 0:
                    desired_shares[row.symbol] = raw_shares

            for symbol in sorted(list(positions.keys())):
                current_shares = positions.get(symbol, 0)
                target_shares = desired_shares.get(symbol, 0)
                delta = target_shares - current_shares
                if delta >= 0:
                    continue
                if execution_constraint_mode == 'strict_ashare' and not _trade_flag(tradability_index, trade_date, symbol, 'can_sell'):
                    continue
                fill_price = _lookup_price(open_prices.get(symbol), close_prices.get(symbol), last_close.get(symbol))
                if fill_price is None or fill_price <= 0:
                    continue
                qty = min(current_shares, abs(delta))
                fill_price *= 1 - slippage_bps / 10000.0
                notional = qty * fill_price
                if notional < min_trade_value:
                    continue
                fee = notional * commission_bps / 10000.0 + notional * stamp_duty_bps / 10000.0
                cash += notional - fee
                positions[symbol] = current_shares - qty
                if positions[symbol] <= 0:
                    positions.pop(symbol, None)
                trade_notional += notional
                trade_cost += fee
                trade_rows.append(
                    {
                        'trade_date': trade_date,
                        'signal_date': targets['signal_date'].iloc[0],
                        'symbol': symbol,
                        'side': 'SELL',
                        'shares': qty,
                        'fill_price': round(fill_price, 4),
                        'notional': round(notional, 2),
                        'fee': round(fee, 2),
                    }
                )

            for row in targets.sort_values('target_weight', ascending=False).itertuples(index=False):
                current_shares = positions.get(row.symbol, 0)
                target_shares = desired_shares.get(row.symbol, 0)
                delta = target_shares - current_shares
                if delta <= 0:
                    continue
                if execution_constraint_mode == 'strict_ashare' and not _trade_flag(tradability_index, trade_date, row.symbol, 'can_buy'):
                    continue
                fill_price = _lookup_price(open_prices.get(row.symbol), close_prices.get(row.symbol), last_close.get(row.symbol))
                if fill_price is None or fill_price <= 0:
                    continue
                fill_price *= 1 + slippage_bps / 10000.0
                per_share_cash = fill_price * (1 + commission_bps / 10000.0)
                affordable = int(np.floor(cash / max(per_share_cash, 1e-9) / lot_size) * lot_size)
                qty = int(np.floor(min(delta, affordable) / lot_size) * lot_size)
                if qty <= 0:
                    continue
                notional = qty * fill_price
                if notional < min_trade_value:
                    continue
                fee = notional * commission_bps / 10000.0
                cash -= notional + fee
                positions[row.symbol] = current_shares + qty
                trade_notional += notional
                trade_cost += fee
                trade_rows.append(
                    {
                        'trade_date': trade_date,
                        'signal_date': row.signal_date,
                        'symbol': row.symbol,
                        'side': 'BUY',
                        'shares': qty,
                        'fill_price': round(fill_price, 4),
                        'notional': round(notional, 2),
                        'fee': round(fee, 2),
                    }
                )

        market_value = 0.0
        for symbol in list(positions.keys()):
            px = _lookup_price(close_prices.get(symbol), close_prices.get(symbol), last_close.get(symbol))
            if px is None or px <= 0:
                continue
            last_close[symbol] = px
            market_value += positions[symbol] * px
            position_rows.append(
                {
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'shares': positions[symbol],
                    'close_price': round(px, 4),
                    'market_value': round(positions[symbol] * px, 2),
                }
            )
        equity = cash + market_value
        nav_rows.append(
            {
                'trade_date': trade_date,
                'cash': round(cash, 2),
                'market_value': round(market_value, 2),
                'equity': round(equity, 2),
                'trade_notional': round(trade_notional, 2),
                'trade_cost': round(trade_cost, 2),
                'corporate_action_cash': round(action_cash, 2),
                'turnover': round(trade_notional / max(equity, 1e-9), 6),
            }
        )

    nav = pd.DataFrame(nav_rows)
    nav['daily_return'] = nav['equity'].pct_change().fillna(0.0)
    nav['nav'] = nav['equity'] / max(initial_cash, 1e-9)
    trades = pd.DataFrame(trade_rows)
    positions_df = pd.DataFrame(position_rows)
    return BacktestResult(nav=nav, trades=trades, positions=positions_df)


def _lookup_price(primary, secondary, fallback):
    for value in (primary, secondary, fallback):
        if value is None:
            continue
        if pd.isna(value):
            continue
        if float(value) > 0:
            return float(value)
    return None


def _trade_flag(tradability_index: pd.DataFrame, trade_date: pd.Timestamp, symbol: str, field: str) -> bool:
    try:
        return bool(tradability_index.at[(trade_date, symbol), field])
    except KeyError:
        return False


def _apply_corporate_actions(
    actions: pd.DataFrame,
    positions: dict[str, int],
    trade_rows: list[dict],
    trade_date: pd.Timestamp,
) -> float:
    action_cash = 0.0
    for row in actions.itertuples(index=False):
        current_shares = positions.get(row.symbol, 0)
        if current_shares <= 0:
            continue
        if row.event_type == 'cash_dividend':
            cash_delta = float(current_shares) * float(row.event_value)
            action_cash += cash_delta
            trade_rows.append(
                {
                    'trade_date': trade_date,
                    'signal_date': pd.NaT,
                    'symbol': row.symbol,
                    'side': 'DIVIDEND',
                    'shares': 0,
                    'fill_price': 0.0,
                    'notional': round(cash_delta, 2),
                    'fee': 0.0,
                }
            )
        elif row.event_type in {'stock_dividend', 'reserve_to_stock'}:
            bonus_shares = int(round(float(current_shares) * float(row.event_value)))
            if bonus_shares <= 0:
                continue
            positions[row.symbol] = current_shares + bonus_shares
            trade_rows.append(
                {
                    'trade_date': trade_date,
                    'signal_date': pd.NaT,
                    'symbol': row.symbol,
                    'side': 'STOCK_BONUS',
                    'shares': bonus_shares,
                    'fill_price': 0.0,
                    'notional': 0.0,
                    'fee': 0.0,
                }
            )
    return action_cash
