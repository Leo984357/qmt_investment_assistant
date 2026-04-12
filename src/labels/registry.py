from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class LabelSpec:
    name: str
    horizon: int
    description: str
    compute: Callable[[pd.DataFrame], pd.Series]
    version: str = 'v1'
    category: str = 'forward_return'
    owner: str = 'system'
    frequency: str = '1d'
    level: str = 'security'
    lag: int = 0
    future_safe: bool = False


class LabelRegistry:
    def __init__(self):
        self._specs: dict[str, LabelSpec] = {}

    def register(self, spec: LabelSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> LabelSpec:
        return self._specs[name]

    def inventory(self) -> pd.DataFrame:
        if not self._specs:
            return pd.DataFrame(columns=['label_name', 'version', 'category', 'owner', 'frequency', 'level', 'horizon', 'lag', 'future_safe', 'description'])
        rows = []
        for spec in self._specs.values():
            rows.append(
                {
                    'label_name': spec.name,
                    'version': spec.version,
                    'category': spec.category,
                    'owner': spec.owner,
                    'frequency': spec.frequency,
                    'level': spec.level,
                    'horizon': spec.horizon,
                    'lag': spec.lag,
                    'future_safe': spec.future_safe,
                    'description': spec.description,
                }
            )
        return pd.DataFrame(rows).sort_values('label_name').reset_index(drop=True)

    def compute_panel(self, daily_bar: pd.DataFrame, label_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        bars = daily_bar.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        spec = self.get(label_name)
        panel = bars[['trade_date', 'symbol']].copy()
        panel[label_name] = spec.compute(bars)
        metadata = pd.DataFrame(
            [
                {
                    'label_name': spec.name,
                    'version': spec.version,
                    'category': spec.category,
                    'owner': spec.owner,
                    'frequency': spec.frequency,
                    'level': spec.level,
                    'horizon': spec.horizon,
                    'lag': spec.lag,
                    'future_safe': spec.future_safe,
                    'description': spec.description,
                }
            ]
        )
        return panel, metadata
