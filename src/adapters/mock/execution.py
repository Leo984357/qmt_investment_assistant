from __future__ import annotations
import pandas as pd

from src.core.config import ExecutionConfig
from src.core.paths import OUTPUT_DIR
from .account import read_account, write_account, read_positions, write_positions


def execute_plan(plan, latest_prices: pd.Series, cfg: ExecutionConfig) -> dict:
    account = read_account()
    positions = read_positions(latest_prices)
    pos_map = {row['ticker']: {'shares': int(row['shares']), 'avg_cost': float(row['avg_cost'])} for _, row in positions.iterrows()}
    fills = []
    commission_bps = float(cfg.commission_bps)
    stamp_duty_bps = float(cfg.stamp_duty_bps)

    orders = plan.orders.copy()
    side_rank = {'SELL': 0, 'BUY': 1}
    orders['side_rank'] = orders['side'].map(side_rank)
    orders = orders.sort_values(['side_rank', 'ticker']).drop(columns=['side_rank'])

    for _, row in orders.iterrows():
        ticker = row['ticker']
        side = row['side']
        shares = int(row['shares'])
        price = float(row['est_price'])
        if shares == 0:
            continue
        p = pos_map.get(ticker, {'shares': 0, 'avg_cost': price})

        if side == 'SELL':
            sell_qty = min(abs(shares), p['shares'])
            if sell_qty <= 0:
                continue
            notional = sell_qty * price
            fee = notional * commission_bps / 10000.0 + notional * stamp_duty_bps / 10000.0
            account['cash'] += notional - fee
            pos_map[ticker] = {'shares': p['shares'] - sell_qty, 'avg_cost': p['avg_cost']}
            fills.append({'ticker': ticker, 'side': side, 'shares': sell_qty, 'price': round(price, 4), 'notional': round(notional, 2), 'fee': round(fee, 2)})
        elif side == 'BUY':
            per_share_cost = price * (1 + commission_bps / 10000.0)
            affordable = int(account['cash'] // max(per_share_cost, 1e-9))
            buy_qty = min(max(shares, 0), affordable)
            if buy_qty <= 0:
                continue
            notional = buy_qty * price
            fee = notional * commission_bps / 10000.0
            account['cash'] -= notional + fee
            new_shares = p['shares'] + buy_qty
            new_cost = ((p['shares'] * p['avg_cost']) + (buy_qty * price)) / max(new_shares, 1)
            pos_map[ticker] = {'shares': new_shares, 'avg_cost': round(new_cost, 4)}
            fills.append({'ticker': ticker, 'side': side, 'shares': buy_qty, 'price': round(price, 4), 'notional': round(notional, 2), 'fee': round(fee, 2)})

    new_pos = pd.DataFrame([{'ticker': k, 'shares': v['shares'], 'avg_cost': v['avg_cost']} for k, v in pos_map.items()])
    write_positions(new_pos)
    market_value = 0.0
    if not new_pos.empty:
        market_value = float((new_pos['ticker'].map(latest_prices).fillna(new_pos['avg_cost']) * new_pos['shares']).sum())
    account['cash'] = round(max(account['cash'], 0.0), 2)
    account['total_equity'] = round(account['cash'] + market_value, 2)
    write_account(account)

    out = OUTPUT_DIR / 'decisions' / 'latest_mock_fills.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fills).to_csv(out, index=False, encoding='utf-8-sig')
    return {'fills_path': str(out), 'cash': account['cash'], 'total_equity': account['total_equity'], 'fill_count': len(fills)}
