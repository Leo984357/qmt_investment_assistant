from __future__ import annotations

import pandas as pd

from .config import DecisionConfig
from .schemas import DecisionPacket


def build_decision_packet(
    current_portfolio: pd.DataFrame,
    target_portfolio: pd.DataFrame,
    latest_prices: pd.Series,
    cash: float,
    total_equity: float,
    as_of_date: str,
    cfg: DecisionConfig,
) -> DecisionPacket:
    current = current_portfolio.copy()
    target = target_portfolio.copy()
    current['weight'] = current['market_value'] / max(total_equity, 1e-9)

    merged = pd.merge(
        current[['ticker', 'shares', 'weight', 'market_value']],
        target[['ticker', 'target_weight']],
        on='ticker',
        how='outer'
    ).fillna({'shares': 0, 'weight': 0.0, 'market_value': 0.0, 'target_weight': 0.0})
    merged['delta_weight'] = merged['target_weight'] - merged['weight']
    merged['reference_price'] = merged['ticker'].map(latest_prices).fillna(0.0)
    merged['target_value'] = merged['target_weight'] * total_equity
    merged['delta_value'] = merged['target_value'] - merged['market_value']
    min_trade_value = float(cfg.min_trade_value)
    small_trade_mask = merged['delta_value'].abs() < min_trade_value
    suppressed_trade_count = int(small_trade_mask.sum())
    if suppressed_trade_count:
        merged.loc[small_trade_mask, 'target_value'] = merged.loc[small_trade_mask, 'market_value']
        merged.loc[small_trade_mask, 'target_weight'] = merged.loc[small_trade_mask, 'weight']
        merged.loc[small_trade_mask, 'delta_value'] = 0.0
        merged.loc[small_trade_mask, 'delta_weight'] = 0.0
    merged = merged.sort_values('delta_weight', ascending=False).reset_index(drop=True)

    l1_turnover = float(merged['delta_weight'].abs().sum())
    change_names = int((merged['delta_weight'].abs() > 1e-6).sum())
    estimated_turnover_cost_bps = round(l1_turnover * float(cfg.turnover_penalty_bps), 4)
    constraints_hit = []
    rationale = []

    threshold = float(cfg.min_rebalance_l1)
    if l1_turnover < threshold:
        action = 'hold'
        rationale.append(f'组合偏离度 {l1_turnover:.3f} 低于阈值 {threshold:.3f}，维持现有仓位更稳妥。')
    else:
        action = 'rebalance'
        rationale.append(f'组合偏离度 {l1_turnover:.3f} 超过阈值 {threshold:.3f}，建议执行调仓。')

    if suppressed_trade_count:
        constraints_hit.append('存在低于最小成交额的调仓项')
        rationale.append(f'已忽略 {suppressed_trade_count} 个低于最小成交额 {min_trade_value:.0f} 的调仓项。')

    if change_names > int(cfg.max_name_changes):
        constraints_hit.append('调仓标的数量较多')
        rationale.append(f'本次涉及 {change_names} 个标的变化，需关注执行摩擦。')

    if cash / max(total_equity, 1e-9) < 0.05:
        constraints_hit.append('现金缓冲偏低')
        rationale.append('当前现金缓冲偏低，买入单需分批或先卖后买。')

    if estimated_turnover_cost_bps > 0:
        rationale.append(f'按配置估算，本次调仓摩擦约为 {estimated_turnover_cost_bps:.2f} bps。')

    concentration = float(target['target_weight'].max()) if not target.empty else 0.0
    risk_summary = {
        'estimated_turnover_l1': round(l1_turnover, 4),
        'estimated_turnover_cost_bps': estimated_turnover_cost_bps,
        'cash_ratio': round(cash / max(total_equity, 1e-9), 4),
        'target_max_weight': round(concentration, 4),
        'positions_after_rebalance': int((target['target_weight'] > 0).sum()),
    }

    return DecisionPacket(
        as_of_date=as_of_date,
        action=action,
        rationale=rationale,
        current_portfolio=current,
        target_portfolio=target,
        rebalance_delta=merged,
        risk_summary=risk_summary,
        constraints_hit=constraints_hit,
    )
