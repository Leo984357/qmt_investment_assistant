from __future__ import annotations

from src.experiment.spec import DataSpec

from .baostock_ashare import BaoStockAShareConfig, BaoStockAShareDataSource
from .base import BaseResearchDataSource
from .mock_ashare import MockAShareConfig, MockAShareDataSource


def build_data_source(data_spec: DataSpec) -> BaseResearchDataSource:
    if data_spec.source == 'mock_ashare':
        return MockAShareDataSource(
            MockAShareConfig(
                start_date=data_spec.start_date,
                end_date=data_spec.end_date,
                n_symbols_master=int(data_spec.n_symbols_master or 320),
                n_universe=int(data_spec.n_universe or 300),
                seed=int(data_spec.seed or 7),
                universe_name=data_spec.universe_name,
            )
        )
    if data_spec.source == 'baostock_ashare':
        return BaoStockAShareDataSource(
            BaoStockAShareConfig(
                start_date=data_spec.start_date,
                end_date=data_spec.end_date,
                universe_name=data_spec.universe_name,
                formal_start_date=data_spec.formal_start_date,
                universe_mode=data_spec.universe_mode,
                universe_refresh_frequency_days=int(data_spec.universe_refresh_frequency_days or 1),
                price_adjust=data_spec.price_adjust or 'qfq',
                incremental=bool(data_spec.incremental),
            )
        )
    raise NotImplementedError(f'未实现的数据源: {data_spec.source}')
