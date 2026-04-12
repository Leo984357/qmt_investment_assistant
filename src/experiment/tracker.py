from __future__ import annotations

from pathlib import Path

import mlflow

from src.ops.paths import MLFLOW_DIR, ensure_platform_dirs


class MLflowTracker:
    def __init__(self, experiment_name: str):
        ensure_platform_dirs()
        self.experiment_name = experiment_name
        tracking_db = (MLFLOW_DIR / 'mlflow.db').resolve()
        mlflow.set_tracking_uri(f'sqlite:///{tracking_db}')
        mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        run_name: str,
        params: dict,
        metrics: dict,
        tags: dict,
        artifact_dir: Path,
    ) -> str:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(tags)
            mlflow.log_params({key: value for key, value in params.items() if isinstance(value, (str, int, float, bool))})
            mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
            mlflow.log_artifacts(str(artifact_dir))
            return run.info.run_id
