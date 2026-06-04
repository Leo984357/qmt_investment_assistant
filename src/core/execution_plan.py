from __future__ import annotations

import numpy as np

from .config import ExecutionConfig
from .schemas import ExecutionPlan


def build_execution_plan(packet, latest_prices, total_equity: float, cfg: ExecutionConfig) -> ExecutionPlan:
    lot_size = int(cfg.lot_size)
    slippage_bps = float(cfg.slippage_bps)
    orders = packet.rebalance_delta.copy()
    orders['reference_price'] = orders['ticker'].map(latest_prices).fillna(0.0)
    warnings = []
    invalid_price_count = int((orders['reference_price'] <= 0).sum())
    if invalid_price_count:
        warnings.append(f'有 {invalid_price_count} 个标的缺少有效价格，已从执行计划中剔除。')
    orders = orders[orders['reference_price'] > 0].copy()
    target_raw = (orders['target_value'] / orders['reference_price']).replace([np.inf, -np.inf], 0).fillna(0)
    orders['target_shares'] = (np.floor(target_raw / lot_size) * lot_size).astype(int)
    orders['current_shares'] = orders['shares'].fillna(0).astype(int)
    orders['order_shares'] = (orders['target_shares'] - orders['current_shares']).astype(int)
    orders['side'] = orders['order_shares'].apply(lambda x: 'BUY' if x > 0 else ('SELL' if x < 0 else 'HOLD'))
    orders['est_price'] = orders['reference_price'] * (1 + slippage_bps / 10000.0 * (orders['order_shares'] > 0) - slippage_bps / 10000.0 * (orders['order_shares'] < 0))
    orders['est_notional'] = (orders['order_shares'].abs() * orders['est_price']).round(2)
    orders = orders[orders['side'] != 'HOLD'].copy()
    cash_required = float(orders.loc[orders['side'] == 'BUY', 'est_notional'].sum())
    if cash_required > total_equity:
        warnings.append('理论买入金额超过总权益，请检查目标仓位与现有持仓。')
    if orders.empty:
        warnings.append('无需要执行的订单。')
    return ExecutionPlan(
        as_of_date=packet.as_of_date,
        broker=str(cfg.mode),
        orders=orders[['ticker', 'side', 'order_shares', 'est_price', 'est_notional']].rename(columns={'order_shares': 'shares'}),
        cash_required=round(cash_required, 2),
        warnings=warnings,
    )
