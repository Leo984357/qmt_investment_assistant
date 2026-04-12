from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / 'configs'
EXPERIMENT_CONFIG_DIR = CONFIGS_DIR / 'experiments'

DATA_ROOT = ROOT / 'data'
RAW_DATA_DIR = DATA_ROOT / 'raw'
BRONZE_DATA_DIR = DATA_ROOT / 'bronze'
SILVER_DATA_DIR = DATA_ROOT / 'silver'
GOLD_DATA_DIR = DATA_ROOT / 'gold'
CATALOG_DIR = DATA_ROOT / 'catalog'

ARTIFACTS_DIR = ROOT / 'artifacts'
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / 'runs'
MLFLOW_DIR = ARTIFACTS_DIR / 'mlruns'


def ensure_platform_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        BRONZE_DATA_DIR,
        SILVER_DATA_DIR,
        GOLD_DATA_DIR,
        CATALOG_DIR,
        ARTIFACTS_DIR,
        ARTIFACT_RUNS_DIR,
        MLFLOW_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
