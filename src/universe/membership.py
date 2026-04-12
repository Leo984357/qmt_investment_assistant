from __future__ import annotations

import pandas as pd


def build_point_in_time_universe(universe_membership: pd.DataFrame, universe_name: str) -> pd.DataFrame:
    subset = universe_membership.loc[universe_membership['universe_name'] == universe_name].copy()
    subset['trade_date'] = pd.to_datetime(subset['trade_date'])
    return subset.sort_values(['trade_date', 'symbol']).reset_index(drop=True)


def build_current_snapshot_universe(
    universe_membership: pd.DataFrame,
    universe_name: str,
    trade_calendar: pd.DataFrame,
) -> pd.DataFrame:
    subset = build_point_in_time_universe(universe_membership, universe_name)
    if subset.empty:
        return subset
    latest_date = pd.to_datetime(subset['trade_date']).max()
    latest_snapshot = subset.loc[subset['trade_date'] == latest_date, ['symbol']].drop_duplicates().sort_values('symbol').reset_index(drop=True)
    trade_dates = pd.DataFrame({'trade_date': pd.to_datetime(trade_calendar['trade_date']).sort_values().drop_duplicates()})
    weight = 1.0 / max(len(latest_snapshot), 1)
    return trade_dates.merge(
        latest_snapshot.assign(
            universe_name=universe_name,
            universe_weight=weight,
            rebalance_id=pd.Timestamp(latest_date).strftime('%Y-%m-%d'),
        ),
        how='cross',
    )[['trade_date', 'universe_name', 'symbol', 'universe_weight', 'rebalance_id']].sort_values(['trade_date', 'symbol']).reset_index(drop=True)
