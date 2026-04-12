from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from src.core.paths import DATA_DIR

ACCOUNT_PATH = DATA_DIR / 'snapshots' / 'mock_account.json'
POS_PATH = DATA_DIR / 'snapshots' / 'current_positions.csv'

DEFAULT_ACCOUNT = {'cash': 120000.0, 'total_equity': 200000.0}
DEFAULT_POSITIONS = [
    {'ticker': '000333.SZ', 'shares': 800, 'avg_cost': 42.3},
    {'ticker': '600036.SH', 'shares': 500, 'avg_cost': 34.8},
    {'ticker': '601318.SH', 'shares': 400, 'avg_cost': 51.2},
]


def reset_mock_account():
    ACCOUNT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACCOUNT_PATH.write_text(json.dumps(DEFAULT_ACCOUNT, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(DEFAULT_POSITIONS).to_csv(POS_PATH, index=False, encoding='utf-8-sig')


def ensure_mock_account():
    if not ACCOUNT_PATH.exists() or not POS_PATH.exists():
        reset_mock_account()


def read_account() -> dict:
    ensure_mock_account()
    return json.loads(ACCOUNT_PATH.read_text(encoding='utf-8'))


def read_positions(latest_prices: pd.Series) -> pd.DataFrame:
    ensure_mock_account()
    df = pd.read_csv(POS_PATH)
    df['reference_price'] = df['ticker'].map(latest_prices).fillna(df['avg_cost'])
    df['market_value'] = (df['shares'] * df['reference_price']).round(2)
    return df


def write_positions(df: pd.DataFrame) -> None:
    keep = df[['ticker', 'shares', 'avg_cost']].copy()
    keep = keep[keep['shares'] > 0].reset_index(drop=True)
    POS_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(POS_PATH, index=False, encoding='utf-8-sig')


def write_account(account: dict) -> None:
    ACCOUNT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACCOUNT_PATH.write_text(json.dumps(account, ensure_ascii=False, indent=2), encoding='utf-8')
