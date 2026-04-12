from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseResearchDataSource(ABC):
    supports_incremental: bool = False

    @abstractmethod
    def fetch_tables(self, existing_tables: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
        raise NotImplementedError
