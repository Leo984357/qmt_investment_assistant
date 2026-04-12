from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Sequence

from src.adapters.mock.account import reset_mock_account
from src.core.config import ConfigError, load_app_config
from src.core.experiment_store import latest_runs, normalize_payload
from src.data_sources.factory import build_data_source
from src.data_store.catalog import LocalResearchCatalog
from src.experiment.runner import build_source_data_spec_with_warmup, run_experiment
from src.experiment.spec import load_experiment_spec
from src.experiment.validation import validate_run
from src.core.logging_utils import configure_logging, get_logger
from src.ops.paths import ROOT
from src.services.decision import run_decision
from src.services.execute import run_execute
from src.services.pipeline import run_pipeline
from src.services.research import run_research
from src.services.review import run_review

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='qmt-assistant', description='QMT Investment Assistant command line interface.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('research', help='Run the research stage.')
    subparsers.add_parser('decision', help='Run the decision stage.')
    subparsers.add_parser('execute', help='Run the execution stage.')
    subparsers.add_parser('review', help='Run the review stage.')
    subparsers.add_parser('reset-mock', help='Reset the mock account snapshot.')
    subparsers.add_parser('gui', help='Launch the Tkinter GUI.')
    subparsers.add_parser('dashboard', help='Launch the Streamlit research dashboard.')

    pipeline_parser = subparsers.add_parser('pipeline', help='Run research, decision, execute, and review in one process.')
    pipeline_parser.add_argument('--reset-mock', action='store_true', help='Reset the mock account before running the pipeline.')

    experiment_parser = subparsers.add_parser('experiment', help='Run a structured research experiment.')
    experiment_parser.add_argument('--config', required=True, help='Path to the experiment YAML.')

    validate_parser = subparsers.add_parser('validate-run', help='Validate a completed experiment run.')
    validate_parser.add_argument('--run-id', help='Run id under artifacts/runs/. If omitted, validate the latest run.')
    validate_parser.add_argument('--path', help='Explicit run directory path.')

    bootstrap_parser = subparsers.add_parser('bootstrap-data', help='Bootstrap the local DuckDB + Parquet research catalog.')
    bootstrap_parser.add_argument('--config', default='configs/experiments/hs300_lightgbm.yaml', help='Experiment config used to derive the data source.')
    bootstrap_parser.add_argument('--force', action='store_true', help='Rebuild catalog tables even if they already exist.')

    runs_parser = subparsers.add_parser('runs', help='List recent experiment runs.')
    runs_parser.add_argument('--limit', type=int, default=10, help='Number of runs to return.')
    return parser


def _dispatch(args: argparse.Namespace):
    if args.command == 'research':
        return run_research()
    if args.command == 'decision':
        return run_decision()
    if args.command == 'execute':
        return run_execute()
    if args.command == 'review':
        return run_review()
    if args.command == 'pipeline':
        return run_pipeline(reset_account=args.reset_mock)
    if args.command == 'reset-mock':
        reset_mock_account()
        return {'status': 'ok', 'message': 'mock 账户已重置'}
    if args.command == 'experiment':
        return run_experiment(args.config)
    if args.command == 'validate-run':
        return validate_run(run_id=getattr(args, 'run_id', None), run_path=getattr(args, 'path', None))
    if args.command == 'bootstrap-data':
        spec = load_experiment_spec(args.config)
        catalog = LocalResearchCatalog()
        catalog.bootstrap(build_data_source(build_source_data_spec_with_warmup(spec)), force=args.force)
        return {'status': 'ok', 'tables': catalog.snapshot_manifest('silver')}
    if args.command == 'runs':
        cfg = load_app_config()
        return {'runs': latest_runs(cfg.db_path, limit=args.limit)}
    if args.command == 'gui':
        dashboard_path = ROOT / 'src' / 'ui' / 'dashboard.py'
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(dashboard_path)], check=True)
        return None
    if args.command == 'dashboard':
        dashboard_path = ROOT / 'src' / 'ui' / 'dashboard.py'
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(dashboard_path)], check=True)
        return None
    raise ValueError(f'未知命令: {args.command}')


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_logging()
    try:
        result = _dispatch(args)
    except ConfigError as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except NotImplementedError as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3
    except Exception as exc:
        logger.exception('CLI command failed command=%s', args.command)
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    if result is not None:
        print(json.dumps(normalize_payload(result), ensure_ascii=False, indent=2))
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == '__main__':
    run()
