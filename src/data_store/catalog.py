from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from src.data_sources.base import BaseResearchDataSource
from src.ops.paths import BRONZE_DATA_DIR, CATALOG_DIR, RAW_DATA_DIR, SILVER_DATA_DIR, ensure_platform_dirs

from .schemas import SCHEMAS, validate_schema


@dataclass(frozen=True)
class CatalogEntry:
    table_name: str
    zone: str
    path: str
    row_count: int
    sha256: str
    created_at: str


class LocalResearchCatalog:
    def __init__(self, catalog_path: Path | None = None):
        ensure_platform_dirs()
        self.catalog_path = catalog_path or (CATALOG_DIR / 'research_catalog.duckdb')
        self.zone_dirs = {'raw': RAW_DATA_DIR, 'bronze': BRONZE_DATA_DIR, 'silver': SILVER_DATA_DIR}
        self._init_catalog()

    def _init_catalog(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS table_registry (
                    table_name TEXT NOT NULL,
                    zone TEXT NOT NULL,
                    path TEXT NOT NULL,
                    row_count BIGINT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (table_name, zone)
                )
                """
            )

    def bootstrap(self, source: BaseResearchDataSource, force: bool = False) -> None:
        existing_tables = None
        required_tables = list(SCHEMAS.keys())
        if not force and all(self.table_exists(table_name, 'silver') for table_name in required_tables) and not getattr(source, 'supports_incremental', False):
            return
        if not force:
            existing_tables = {
                table_name: self.read_table(table_name)
                for table_name in required_tables
                if self.table_exists(table_name, 'silver')
            }
        tables = source.fetch_tables(existing_tables=existing_tables)
        for table_name, raw_df in tables.items():
            self.write_table(raw_df, table_name, zone='raw', validate=False)
            self.write_table(raw_df, table_name, zone='bronze', validate=False)
            silver_df = self._standardize_table(table_name, raw_df)
            self.write_table(silver_df, table_name, zone='silver', validate=True)

    def table_exists(self, table_name: str, zone: str = 'silver') -> bool:
        return self._table_path(table_name, zone).exists()

    def write_table(self, df: pd.DataFrame, table_name: str, zone: str = 'silver', validate: bool = True) -> Path:
        if validate:
            validate_schema(table_name, df)
        path = self._table_path(table_name, zone)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        entry = CatalogEntry(
            table_name=table_name,
            zone=zone,
            path=str(path),
            row_count=int(len(df)),
            sha256=self._sha256(path),
            created_at=datetime.now().isoformat(timespec='seconds'),
        )
        with self._connect() as conn:
            conn.execute('DELETE FROM table_registry WHERE table_name = ? AND zone = ?', [table_name, zone])
            conn.execute(
                'INSERT INTO table_registry (table_name, zone, path, row_count, sha256, created_at) VALUES (?, ?, ?, ?, ?, ?)',
                [entry.table_name, entry.zone, entry.path, entry.row_count, entry.sha256, entry.created_at],
            )
        return path

    def read_table(self, table_name: str, zone: str = 'silver') -> pd.DataFrame:
        path = self._table_path(table_name, zone)
        if not path.exists():
            raise FileNotFoundError(f'表 {table_name} 在 {zone} 区不存在: {path}')
        return pd.read_parquet(path)

    def snapshot_manifest(self, zone: str = 'silver') -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT table_name, zone, path, row_count, sha256, created_at FROM table_registry WHERE zone = ? ORDER BY table_name',
                [zone],
            ).fetchdf()
        return rows.to_dict(orient='records')

    def query(self, sql: str, zone: str = 'silver') -> pd.DataFrame:
        with duckdb.connect() as conn:
            for table_name in SCHEMAS:
                path = self._table_path(table_name, zone)
                if path.exists():
                    conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{path.as_posix()}')")
            return conn.execute(sql).df()

    def _standardize_table(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        standardized = df.copy()
        for column in standardized.columns:
            if column.endswith('_date') or column == 'trade_date' or column == 'event_date':
                standardized[column] = pd.to_datetime(standardized[column])
        sort_columns = [column for column in ('trade_date', 'event_date', 'symbol') if column in standardized.columns]
        if sort_columns:
            standardized = standardized.sort_values(sort_columns).reset_index(drop=True)
        return standardized

    def _table_path(self, table_name: str, zone: str) -> Path:
        return self.zone_dirs[zone] / f'{table_name}.parquet'

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _connect(self):
        last_error = None
        for _ in range(20):
            try:
                return duckdb.connect(str(self.catalog_path))
            except duckdb.IOException as exc:
                last_error = exc
                time.sleep(0.25)
        raise last_error
