from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.ops.paths import ARTIFACT_RUNS_DIR


def resolve_run_dir(run_id: str | None = None, run_path: str | Path | None = None) -> Path:
    if run_path is not None:
        path = Path(run_path)
        if not path.exists():
            raise FileNotFoundError(f'run 路径不存在: {path}')
        return path
    candidates = [path for path in ARTIFACT_RUNS_DIR.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError('artifacts/runs 下没有任何 run。')
    if run_id:
        path = ARTIFACT_RUNS_DIR / run_id
        if not path.exists():
            raise FileNotFoundError(f'run_id 不存在: {run_id}')
        return path
    return max(candidates, key=lambda path: path.stat().st_mtime)


def validate_run(run_id: str | None = None, run_path: str | Path | None = None) -> dict:
    run_dir = resolve_run_dir(run_id=run_id, run_path=run_path)
    summary = json.loads((run_dir / 'metadata' / 'run_summary.json').read_text(encoding='utf-8'))
    spec = yaml.safe_load((run_dir / 'config' / 'resolved_experiment.yaml').read_text(encoding='utf-8'))

    nav = pd.read_parquet(run_dir / 'backtest' / 'nav.parquet')
    trades = pd.read_parquet(run_dir / 'backtest' / 'trades.parquet')
    rank_ic = pd.read_parquet(run_dir / 'backtest' / 'rank_ic.parquet')
    target_weights = pd.read_parquet(run_dir / 'signals' / 'target_weights.parquet')
    predictions = pd.read_parquet(run_dir / 'signals' / 'predictions.parquet')

    lot_size = int(spec['backtest']['lot_size'])
    allowed_exposures = {
        round(float(spec['portfolio']['gross_exposure']), 6),
        round(float(spec['portfolio'].get('defensive_gross', spec['portfolio']['gross_exposure'])), 6),
        round(float(spec['portfolio']['risk_mid_exposure']), 6),
        round(float(spec['portfolio']['risk_low_exposure']), 6),
        round(float(spec['portfolio']['risk_crash_exposure']), 6),
    }

    checks: list[dict] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

    add_check('run_summary_exists', True, str(run_dir / 'metadata' / 'run_summary.json'))
    add_check('nav_not_empty', not nav.empty, f'nav_rows={len(nav)}')
    add_check('predictions_not_empty', not predictions.empty, f'prediction_rows={len(predictions)}')
    add_check('target_weights_not_empty', not target_weights.empty, f'target_rows={len(target_weights)}')

    if not target_weights.empty and not nav.empty:
        execution_start = pd.to_datetime(target_weights['execution_date']).min()
        nav_start = pd.to_datetime(nav['trade_date']).min()
        add_check('nav_starts_at_first_execution', nav_start == execution_start, f'nav_start={nav_start.date()} execution_start={execution_start.date()}')

    if not predictions.empty and not rank_ic.empty:
        signal_start = pd.to_datetime(predictions['trade_date']).min()
        rank_ic_start = pd.to_datetime(rank_ic['trade_date']).min()
        add_check('rank_ic_starts_at_signal_window', rank_ic_start >= signal_start, f'signal_start={signal_start.date()} rank_ic_start={rank_ic_start.date()}')

    if not target_weights.empty:
        gross_by_date = target_weights.groupby('signal_date')['gross_exposure'].first().round(6)
        invalid = sorted(set(gross_by_date.tolist()) - allowed_exposures)
        add_check('gross_exposure_in_allowed_buckets', not invalid, f'allowed={sorted(allowed_exposures)} invalid={invalid}')

        weight_sum = target_weights.groupby('signal_date')['target_weight'].sum().round(6)
        weight_vs_gross = (
            target_weights.groupby('signal_date')['gross_exposure'].first().round(6) - weight_sum
        ).abs().max()
        add_check('weights_sum_to_gross_exposure', float(weight_vs_gross or 0.0) <= 1e-6, f'max_abs_gap={float(weight_vs_gross or 0.0):.6f}')

        rank_frame = target_weights.groupby('signal_date')['rank'].agg(['min', 'max', 'nunique', 'size']).reset_index()
        rank_ok = bool(((rank_frame['min'] == 1) & (rank_frame['max'] == rank_frame['size']) & (rank_frame['nunique'] == rank_frame['size'])).all())
        add_check('ranks_are_contiguous', rank_ok, f'signal_dates={len(rank_frame)}')

    executable_trades = trades.loc[trades['side'].isin(['BUY', 'SELL'])].copy() if not trades.empty else pd.DataFrame()
    if not executable_trades.empty:
        buy_trades = executable_trades.loc[executable_trades['side'] == 'BUY'].copy()
        lot_ok = bool((buy_trades['shares'] % lot_size == 0).all()) if not buy_trades.empty else True
        add_check('exchange_lot_size_respected', lot_ok, f'lot_size={lot_size} buy_rows={len(buy_trades)}')
        add_check('no_negative_fill_price', bool((executable_trades['fill_price'] > 0).all()), f'trade_rows={len(executable_trades)}')

    if not nav.empty:
        add_check('cash_never_negative', bool((nav['cash'] >= -1e-6).all()), f'min_cash={float(nav["cash"].min()):.2f}')
        add_check('nav_never_nonpositive', bool((nav['nav'] > 0).all()), f'min_nav={float(nav["nav"].min()):.6f}')

    computed_fallback_rate = float(predictions['fallback_used'].mean()) if not predictions.empty and 'fallback_used' in predictions.columns else 0.0
    summary_fallback_rate = float(summary.get('fallback_rate', computed_fallback_rate))
    add_check(
        'fallback_rate_matches_summary',
        abs(computed_fallback_rate - summary_fallback_rate) <= 1e-9,
        f'computed={computed_fallback_rate:.6f} summary={summary_fallback_rate:.6f}',
    )

    passed = [check for check in checks if check['passed']]
    failed = [check for check in checks if not check['passed']]
    return {
        'run_id': summary.get('run_id', run_dir.name),
        'run_dir': str(run_dir),
        'passed': len(failed) == 0,
        'passed_count': len(passed),
        'failed_count': len(failed),
        'checks': checks,
    }
