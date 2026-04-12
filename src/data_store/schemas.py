from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TableSchema:
    name: str
    required_columns: tuple[str, ...]
    key_columns: tuple[str, ...]


SCHEMAS: dict[str, TableSchema] = {
    'trade_calendar': TableSchema('trade_calendar', ('trade_date', 'is_trading_day'), ('trade_date',)),
    'security_master': TableSchema(
        'security_master',
        ('symbol', 'security_name', 'market', 'board', 'industry', 'listed_date', 'delisted_date', 'is_st_flag'),
        ('symbol',),
    ),
    'daily_bar': TableSchema(
        'daily_bar',
        ('trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'adj_close', 'status_flag'),
        ('trade_date', 'symbol'),
    ),
    'universe_membership': TableSchema(
        'universe_membership',
        ('trade_date', 'universe_name', 'symbol', 'universe_weight', 'rebalance_id'),
        ('trade_date', 'universe_name', 'symbol'),
    ),
    'tradability': TableSchema(
        'tradability',
        ('trade_date', 'symbol', 'is_tradable', 'can_buy', 'can_sell', 'is_suspended', 'up_limit_hit', 'down_limit_hit', 'is_st', 'is_new_listing'),
        ('trade_date', 'symbol'),
    ),
    'corporate_actions': TableSchema('corporate_actions', ('event_date', 'symbol', 'event_type', 'event_value'), ('event_date', 'symbol', 'event_type')),
}


def validate_schema(table_name: str, df: pd.DataFrame) -> None:
    schema = SCHEMAS[table_name]
    missing = [column for column in schema.required_columns if column not in df.columns]
    if missing:
        joined = ', '.join(missing)
        raise ValueError(f'{table_name} 缺少必需列: {joined}')
