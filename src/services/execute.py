from __future__ import annotations

from src.core.logging_utils import get_logger
from src.core.experiment_store import record_run
from src.core.paths import OUTPUT_DIR
from src.adapters.mock.execution import execute_plan as execute_mock
from src.services.workflow import WorkflowState, create_workflow_state, ensure_account_snapshot, ensure_decision_packet, ensure_execution_plan

logger = get_logger(__name__)


def run_execute(state: WorkflowState | None = None) -> dict:
    state = state or create_workflow_state()
    cfg = state.cfg
    logger.info('Starting execution stage for strategy=%s mode=%s', cfg.common.strategy_name, cfg.execution.mode)
    packet = ensure_decision_packet(state)
    account, _ = ensure_account_snapshot(state)
    latest_prices = state.latest_prices
    plan = ensure_execution_plan(state)
    plan_path = OUTPUT_DIR / 'decisions' / 'latest_execution_plan.csv'
    plan.orders.to_csv(plan_path, index=False, encoding='utf-8-sig')

    if cfg.execution.mode == 'mock':
        execution_result = execute_mock(plan, latest_prices, cfg.execution)
    else:
        raise NotImplementedError('当前版本只支持 mock 模式执行。')
    state.execution_result = execution_result

    artifacts = {'execution_plan_csv': str(plan_path), **execution_result}
    metrics = {'cash_required': plan.cash_required, 'order_count': int(len(plan.orders))}
    run_id = record_run(cfg.db_path, 'execute', cfg.common.strategy_name, packet.as_of_date, metrics, artifacts, note='执行计划已完成')
    logger.info('Execution stage complete run_id=%s fill_count=%s', run_id, execution_result.get('fill_count', 0))
    return {'run_id': run_id, 'plan': plan, 'execution_result': execution_result, 'latest_prices': latest_prices, 'as_of_date': packet.as_of_date}
