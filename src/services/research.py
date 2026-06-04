from __future__ import annotations

from src.core.experiment_store import record_run
from src.core.logging_utils import get_logger
from src.core.paths import OUTPUT_DIR
from src.core.reporting import write_research_report
from src.services.workflow import WorkflowState, create_workflow_state, ensure_research

logger = get_logger(__name__)


def run_research(state: WorkflowState | None = None) -> dict:
    state = state or create_workflow_state()
    cfg = state.cfg
    logger.info('Starting research stage for strategy=%s', cfg.common.strategy_name)
    result = ensure_research(state)

    run_dir = OUTPUT_DIR / 'runs' / result.as_of_date
    run_dir.mkdir(parents=True, exist_ok=True)
    signals_path = run_dir / 'signals.csv'
    target_path = run_dir / 'target_portfolio.csv'
    equity_path = run_dir / 'equity_curve.csv'

    result.signals.to_csv(signals_path, index=False, encoding='utf-8-sig')
    result.target_portfolio.to_csv(target_path, index=False, encoding='utf-8-sig')
    result.equity_curve.to_csv(equity_path, encoding='utf-8-sig')

    artifacts = write_research_report(OUTPUT_DIR / 'reports', result.as_of_date, result.metrics, result.signals, result.target_portfolio, result.diagnostics)
    artifacts.update({'signals_csv': str(signals_path), 'target_csv': str(target_path), 'equity_csv': str(equity_path)})
    run_id = record_run(cfg.db_path, 'research', cfg.common.strategy_name, result.as_of_date, result.metrics, artifacts, note='研究运行完成')
    logger.info('Research stage complete run_id=%s as_of_date=%s', run_id, result.as_of_date)
    return {'run_id': run_id, 'as_of_date': result.as_of_date, 'metrics': result.metrics, 'artifacts': artifacts}
