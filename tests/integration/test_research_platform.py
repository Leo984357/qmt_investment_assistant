from pathlib import Path

from src.experiment.runner import run_experiment


def test_research_platform_end_to_end():
    summary = run_experiment(Path('tests/fixtures/test_experiment.yaml'))
    run_dir = Path(summary['artifact_dir'])

    assert summary['run_id'].startswith('smoke_platform_run_')
    assert summary['strategy_name'] == 'smoke_platform_run'
    assert summary['experiment_name'] == 'smoke_platform_run'
    assert summary['prediction_dates'] > 0
    assert (run_dir / 'metadata' / 'data_snapshot.json').exists()
    assert (run_dir / 'datasets' / 'model_dataset.parquet').exists()
    assert (run_dir / 'signals' / 'predictions.parquet').exists()
    assert (run_dir / 'signals' / 'signal_scores.parquet').exists()
    assert (run_dir / 'signals' / 'target_weights.parquet').exists()
    assert (run_dir / 'backtest' / 'nav.parquet').exists()
    assert (run_dir / 'evaluation' / 'ic_summary.parquet').exists()
    assert (run_dir / 'metadata' / 'experiment_manifest.json').exists()
    assert (run_dir / 'metadata' / 'artifact_inventory.parquet').exists()
    assert (run_dir / 'reports' / 'run_report.md').exists()
    assert (run_dir / 'reports' / 'factor_diagnostics.md').exists()
