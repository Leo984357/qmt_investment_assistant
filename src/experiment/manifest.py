from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_store.schemas import SCHEMAS
from src.experiment.spec import ExperimentSpec


def build_artifact_inventory(run_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(run_dir.rglob('*')):
        if not path.is_file():
            continue
        if path.name.startswith('.'):
            continue
        relative = path.relative_to(run_dir).as_posix()
        rows.append(
            {
                'relative_path': relative,
                'section': relative.split('/')[0],
                'size_bytes': path.stat().st_size,
                'sha256': _sha256(path),
                'modified_at': datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec='seconds'),
            }
        )
    return pd.DataFrame(rows)


def build_data_contract_report(table_frames: dict[str, pd.DataFrame]) -> list[dict]:
    report: list[dict] = []
    for table_name, schema in SCHEMAS.items():
        frame = table_frames.get(table_name)
        if frame is None:
            continue
        date_columns = [column for column in ('trade_date', 'event_date', 'listed_date', 'delisted_date') if column in frame.columns]
        date_min = None
        date_max = None
        if date_columns:
            stacked = pd.concat([pd.to_datetime(frame[column], errors='coerce') for column in date_columns], ignore_index=True)
            stacked = stacked.dropna()
            if not stacked.empty:
                date_min = str(stacked.min().date())
                date_max = str(stacked.max().date())
        report.append(
            {
                'table_name': table_name,
                'row_count': int(len(frame)),
                'column_count': int(len(frame.columns)),
                'columns': list(frame.columns),
                'key_columns': list(schema.key_columns),
                'required_columns': list(schema.required_columns),
                'date_min': date_min,
                'date_max': date_max,
            }
        )
    return report


def build_experiment_manifest(
    spec: ExperimentSpec,
    run_id: str,
    summary: dict,
    dataset_summary: dict,
    data_snapshot: list[dict],
    data_contract: list[dict],
    stage_timings: dict[str, float],
    registry_catalog: dict[str, list[dict]],
    selected_modules: dict,
    artifact_inventory: pd.DataFrame,
) -> dict:
    return {
        'run_id': run_id,
        'config_hash': spec.config_hash,
        'experiment_name': spec.name,
        'description': spec.description,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'spec': spec.to_dict(),
        'summary': summary,
        'dataset_summary': dataset_summary,
        'stage_timings_seconds': stage_timings,
        'data_snapshot_manifest': data_snapshot,
        'data_contract': data_contract,
        'selected_modules': selected_modules,
        'registry_catalog': registry_catalog,
        'artifact_count': int(len(artifact_inventory)),
        'artifact_sections': sorted(artifact_inventory['section'].drop_duplicates().tolist()) if not artifact_inventory.empty else [],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
