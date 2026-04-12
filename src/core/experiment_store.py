from __future__ import annotations
import json
import sqlite3
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


def normalize_payload(obj):
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: normalize_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_payload(x) for x in obj]
    item = getattr(obj, 'item', None)
    if callable(item):
        try:
            return item()
        except Exception:
            return obj
    return obj


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            stage TEXT NOT NULL,
            strategy_name TEXT NOT NULL,
            as_of_date TEXT,
            metrics_json TEXT,
            artifacts_json TEXT,
            note TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def record_run(db_path: Path, stage: str, strategy_name: str, as_of_date: str, metrics: dict, artifacts: dict, note: str = '') -> str:
    init_db(db_path)
    run_id = f"{stage}_{uuid.uuid4().hex[:12]}"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO experiment_runs (run_id, created_at, stage, strategy_name, as_of_date, metrics_json, artifacts_json, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        (
            run_id,
            datetime.now().isoformat(timespec='seconds'),
            stage,
            strategy_name,
            as_of_date,
            json.dumps(normalize_payload(metrics), ensure_ascii=False, indent=2),
            json.dumps(normalize_payload(artifacts), ensure_ascii=False, indent=2),
            note,
        )
    )
    conn.commit()
    conn.close()
    return run_id


def latest_runs(db_path: Path, limit: int = 20):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        'SELECT run_id, created_at, stage, strategy_name, as_of_date, note FROM experiment_runs ORDER BY created_at DESC LIMIT ?',
        conn,
        params=(limit,),
    )
    conn.close()
    return df
