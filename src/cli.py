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
from src.features.factor_catalog import build_default_catalog
from src.features.factor_catalog import FactorStatus

logger = get_logger(__name__)


def _audit_config(config_path: str) -> dict:
    """Audit experiment config for production safety."""
    from src.experiment.spec import load_experiment_spec
    
    issues = []
    warnings = []
    info = []
    
    try:
        spec = load_experiment_spec(config_path)
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to load config: {e}',
        }
    
    # Check registry_stage
    registry_stage = spec.model.registry_stage if hasattr(spec.model, 'registry_stage') else 'unknown'
    protocol_stage = spec.research_protocol.stage
    info.append(f"registry_stage: {registry_stage}")
    info.append(f"research_protocol.stage: {protocol_stage}")
    info.append(f"research_protocol.data_mined: {spec.research_protocol.data_mined}")
    info.append(f"multiple_testing.enabled: {spec.multiple_testing.enabled}")
    info.append(f"risk_attribution.enabled: {spec.risk_attribution.enabled}")
    info.append(f"overlay.enabled: {spec.overlay.enabled}")
    info.append(f"overlay.regime_exposure_enabled: {spec.overlay.regime_exposure_enabled}")
    
    # Research stage cannot produce formal conclusions
    if registry_stage == 'research':
        warnings.append({
            'severity': 'warning',
            'type': 'research_stage',
            'message': 'Research stage - results cannot be used for formal conclusions',
        })

    if protocol_stage in {'diagnostic', 'discovery'}:
        warnings.append({
            'severity': 'warning',
            'type': 'pre_validation_protocol',
            'message': f'{protocol_stage} stage - candidate generation only; requires validation/holdout before strategy claims',
        })
    elif protocol_stage == 'validation':
        warnings.append({
            'severity': 'warning',
            'type': 'validation_protocol',
            'message': 'Validation stage - do not modify strategy logic based on this run',
        })
    elif protocol_stage == 'holdout':
        warnings.append({
            'severity': 'warning',
            'type': 'holdout_protocol',
            'message': 'Holdout stage - final evidence only; no tuning allowed after this run',
        })
    
    # Check allow_rejected_factors and allow_observe_factors (from raw config)
    import yaml
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    allow_rejected = raw.get('allow_rejected_factors', False)
    allow_observe = raw.get('allow_observe_factors', False)
    
    if registry_stage == 'production' and allow_rejected:
        issues.append({
            'severity': 'error',
            'type': 'production_rejected_combo',
            'message': 'Production config cannot have allow_rejected_factors: true',
            'fix': 'Move rejected factors to diagnostic config or remove allow_rejected_factors',
        })
    
    # Check factor catalog status
    catalog = build_default_catalog()
    unknown_factors = []
    rejected_factors = []
    observe_factors = []
    
    for factor_name in spec.features.names:
        profile = catalog.get(factor_name)
        if profile is None:
            unknown_factors.append(factor_name)
        elif profile.status == FactorStatus.REJECT:
            rejected_factors.append(factor_name)
        elif profile.status == FactorStatus.OBSERVE:
            observe_factors.append(factor_name)
    
    if unknown_factors:
        finding = {
            'severity': 'error' if registry_stage == 'production' else 'warning',
            'type': 'unknown_factors',
            'factors': unknown_factors,
            'message': f'{len(unknown_factors)} factors not in catalog',
            'fix': 'Add to src/features/factor_catalog.py before production conclusions',
        }
        if registry_stage == 'production':
            issues.append(finding)
        else:
            warnings.append(finding)
    
    if rejected_factors:
        if registry_stage == 'production':
            issues.append({
                'severity': 'error',
                'type': 'rejected_in_production',
                'factors': rejected_factors,
                'message': 'Rejected factors in production config',
                'fix': 'Use diagnostic config or remove rejected factors',
            })
        else:
            warnings.append({
                'severity': 'warning',
                'type': 'rejected_in_diagnostic',
                'factors': rejected_factors,
                'message': 'Rejected factors in diagnostic config (acceptable for research)',
            })
    
    if observe_factors and registry_stage == 'production':
        if allow_observe:
            warnings.append({
                'severity': 'warning',
                'type': 'observe_in_production',
                'factors': observe_factors,
                'message': 'Observe-status factors in production (allowed by allow_observe_factors: true)',
            })
        else:
            issues.append({
                'severity': 'error',
                'type': 'observe_in_production',
                'factors': observe_factors,
                'message': 'Observe-status factors in production without allow_observe_factors: true',
                'fix': 'Set allow_observe_factors: true to allow, or move factors to research stage',
            })
    
    # Check enhancer
    enhancer_enabled = spec.enhancer.enabled if hasattr(spec.enhancer, 'enabled') else None
    info.append(f"enhancer_enabled: {enhancer_enabled}")
    
    # Determine overall status
    if registry_stage == 'research':
        status = 'warning'
        recommendation = 'Research only - NOT for formal conclusions'
    elif registry_stage == 'diagnostic':
        if issues:
            status = 'warning'
            recommendation = 'Diagnostic only; not production candidate (has issues)'
        elif warnings:
            status = 'warning'
            recommendation = 'Diagnostic only; not production candidate (see warnings)'
        else:
            status = 'pass'
            recommendation = 'Diagnostic only; not production candidate'
    elif registry_stage == 'production':
        if issues:
            status = 'reject' if any(i['severity'] == 'error' for i in issues) else 'warning'
            recommendation = 'NOT for production use until issues are resolved'
        elif warnings:
            status = 'warning'
            recommendation = 'Production candidate with caveats (see warnings)'
        else:
            status = 'pass'
            recommendation = 'Ready for production use'
    else:
        status = 'warning'
        recommendation = f'Unknown registry_stage: {registry_stage}'
    
    # Core factors
    core_factors = [p.name for p in catalog.get_core_factors()]
    used_core = [f for f in spec.features.names if f in core_factors]
    
    return {
        'status': status,
        'config': config_path,
        'registry_stage': registry_stage,
        'research_protocol_stage': protocol_stage,
        'data_mined': spec.research_protocol.data_mined,
        'multiple_testing_enabled': spec.multiple_testing.enabled,
        'risk_attribution_enabled': spec.risk_attribution.enabled,
        'overlay_enabled': spec.overlay.enabled or spec.overlay.regime_exposure_enabled,
        'recommendation': recommendation,
        'core_factors_used': used_core,
        'total_factors': len(spec.features.names),
        'issues': issues,
        'warnings': warnings,
        'info': info,
    }


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

    audit_parser = subparsers.add_parser('audit-config', help='Audit experiment config for production safety.')
    audit_parser.add_argument('--config', required=True, help='Path to the experiment YAML.')
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
    if args.command == 'audit-config':
        return _audit_config(args.config)
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
