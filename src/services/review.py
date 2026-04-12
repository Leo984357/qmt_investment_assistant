from __future__ import annotations
import pandas as pd

from src.core.logging_utils import get_logger
from src.core.schemas import PostTradeReview
from src.core.experiment_store import record_run
from src.core.paths import OUTPUT_DIR
from src.core.reporting import write_review_report
from src.adapters.mock.account import read_positions, read_account
from src.services.workflow import WorkflowState, create_workflow_state, ensure_decision_packet

logger = get_logger(__name__)


def run_review(state: WorkflowState | None = None) -> dict:
    state = state or create_workflow_state()
    cfg = state.cfg
    logger.info('Starting review stage for strategy=%s', cfg.common.strategy_name)
    packet = ensure_decision_packet(state)
    latest_prices = state.latest_prices
    current_after = read_positions(latest_prices)
    acct = read_account()

    target = packet.target_portfolio[['ticker', 'target_weight']].copy()
    actual = current_after[['ticker', 'shares', 'market_value']].copy()
    actual['actual_weight'] = actual['market_value'] / max(acct['total_equity'], 1e-9)
    merged = pd.merge(target, actual[['ticker', 'actual_weight', 'shares']], on='ticker', how='outer').fillna(0.0)
    merged['weight_gap'] = merged['target_weight'] - merged['actual_weight']
    summary = {
        'cash': acct['cash'],
        'total_equity': acct['total_equity'],
        'mean_abs_weight_gap': round(float(merged['weight_gap'].abs().mean()), 4),
        'max_abs_weight_gap': round(float(merged['weight_gap'].abs().max()), 4),
    }
    reasons = []
    if summary['max_abs_weight_gap'] > 0.05:
        reasons.append('部分标的实际仓位和目标仓位偏差较大，需关注 lot size 与现金约束。')
    else:
        reasons.append('本次执行后整体仓位偏差可控。')
    next_actions = ['继续观察目标组合与实际仓位偏差。', '后续接入 QMT 后替换 mock fills 与账户读取。']
    review = PostTradeReview(packet.as_of_date, merged.sort_values('weight_gap', ascending=False), summary, reasons, next_actions)
    artifacts = write_review_report(OUTPUT_DIR / 'reports', packet.as_of_date, review)
    review_csv = OUTPUT_DIR / 'decisions' / 'latest_review_target_vs_actual.csv'
    review.target_vs_actual.to_csv(review_csv, index=False, encoding='utf-8-sig')
    artifacts['target_vs_actual_csv'] = str(review_csv)
    run_id = record_run(cfg.db_path, 'review', cfg.common.strategy_name, packet.as_of_date, summary, artifacts, note='复盘完成')
    logger.info('Review stage complete run_id=%s max_abs_weight_gap=%s', run_id, summary['max_abs_weight_gap'])
    return {'run_id': run_id, 'review': review, 'artifacts': artifacts}
