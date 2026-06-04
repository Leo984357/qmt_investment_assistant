from __future__ import annotations

from src.core.experiment_store import record_run
from src.core.logging_utils import get_logger
from src.core.paths import OUTPUT_DIR
from src.core.reporting import write_decision_report
from src.services.workflow import WorkflowState, create_workflow_state, ensure_account_snapshot, ensure_decision_packet

logger = get_logger(__name__)


def run_decision(state: WorkflowState | None = None) -> dict:
    state = state or create_workflow_state()
    cfg = state.cfg
    logger.info('Starting decision stage for strategy=%s', cfg.common.strategy_name)
    packet = ensure_decision_packet(state)
    account, _ = ensure_account_snapshot(state)
    latest_prices = state.latest_prices

    out_dir = OUTPUT_DIR / 'decisions'
    out_dir.mkdir(parents=True, exist_ok=True)
    delta_path = out_dir / 'latest_rebalance_delta.csv'
    packet.rebalance_delta.to_csv(delta_path, index=False, encoding='utf-8-sig')
    artifacts = write_decision_report(OUTPUT_DIR / 'reports', packet.as_of_date, packet)
    artifacts.update({'rebalance_delta_csv': str(delta_path)})
    metrics = {'action': packet.action, **packet.risk_summary}
    run_id = record_run(cfg.db_path, 'decision', cfg.common.strategy_name, packet.as_of_date, metrics, artifacts, note='决策包生成完成')
    logger.info('Decision stage complete run_id=%s action=%s', run_id, packet.action)
    return {'run_id': run_id, 'packet': packet, 'latest_prices': latest_prices, 'account': account}
