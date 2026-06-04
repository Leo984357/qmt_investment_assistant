from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    inputs: tuple[str, ...]
    lookback: int
    description: str
    compute: Callable[[pd.DataFrame], pd.Series]
    version: str = 'v1'
    category: str = 'alpha'
    owner: str = 'system'
    frequency: str = '1d'
    level: str = 'security'
    lag: int = 0
    preprocessing: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    future_safe: bool = True
    economic_meaning: str = ''
    logic: str = ''
    failure_modes: str = ''


class FeatureRegistry:
    def __init__(self):
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def inventory(self) -> pd.DataFrame:
        if not self._specs:
            return pd.DataFrame(
                columns=[
                    'feature_name',
                    'version',
                    'category',
                    'owner',
                    'frequency',
                    'level',
                    'lookback',
                    'lag',
                    'inputs',
                    'dependencies',
                    'preprocessing',
                    'future_safe',
                    'economic_meaning',
                    'logic',
                    'failure_modes',
                    'description',
                ]
            )
        rows = []
        for spec in self._specs.values():
            rows.append(
                {
                    'feature_name': spec.name,
                    'version': spec.version,
                    'category': spec.category,
                    'owner': spec.owner,
                    'frequency': spec.frequency,
                    'level': spec.level,
                    'lookback': spec.lookback,
                    'lag': spec.lag,
                    'inputs': ','.join(spec.inputs),
                    'dependencies': ','.join(spec.dependencies),
                    'preprocessing': ','.join(spec.preprocessing),
                    'future_safe': spec.future_safe,
                    'economic_meaning': spec.economic_meaning,
                    'logic': spec.logic,
                    'failure_modes': spec.failure_modes,
                    'description': spec.description,
                }
            )
        return pd.DataFrame(rows).sort_values('feature_name').reset_index(drop=True)

    def compute_panel(
        self,
        daily_bar: pd.DataFrame,
        feature_names: list[str],
        winsorize_limits: tuple[float, float] = (0.01, 0.99),
        zscore: bool = True,
        fill_missing: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        bars = daily_bar.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        panel = bars[['trade_date', 'symbol']].copy()
        inventories = []
        for feature_name in feature_names:
            spec = self.get(feature_name)
            panel[feature_name] = spec.compute(bars)
            inventories.append(
                {
                    'feature_name': spec.name,
                    'version': spec.version,
                    'category': spec.category,
                    'owner': spec.owner,
                    'frequency': spec.frequency,
                    'level': spec.level,
                    'lookback': spec.lookback,
                    'lag': spec.lag,
                    'inputs': ','.join(spec.inputs),
                    'dependencies': ','.join(spec.dependencies),
                    'preprocessing': ','.join(spec.preprocessing),
                    'future_safe': spec.future_safe,
                    'economic_meaning': spec.economic_meaning,
                    'logic': spec.logic,
                    'failure_modes': spec.failure_modes,
                    'description': spec.description,
                }
            )
        panel = self._cross_sectional_post_process(panel, feature_names, winsorize_limits, zscore, fill_missing)
        long_form = panel.melt(id_vars=['trade_date', 'symbol'], var_name='feature_name', value_name='feature_value')
        inventory = pd.DataFrame(inventories)
        return panel, long_form, inventory

    @staticmethod
    def _cross_sectional_post_process(
        panel: pd.DataFrame,
        feature_names: list[str],
        winsorize_limits: tuple[float, float],
        zscore: bool,
        fill_missing: bool,
    ) -> pd.DataFrame:
        result = panel.copy()
        for feature_name in feature_names:
            result[feature_name] = result.groupby('trade_date')[feature_name].transform(
                lambda series: FeatureRegistry._winsorize_series(series, winsorize_limits)
            )
            if fill_missing:
                result[feature_name] = result.groupby('trade_date')[feature_name].transform(
                    FeatureRegistry._fill_missing_series
                )
            if zscore:
                result[feature_name] = result.groupby('trade_date')[feature_name].transform(FeatureRegistry._zscore_series)
        return result

    @staticmethod
    def _winsorize_series(series: pd.Series, limits: tuple[float, float]) -> pd.Series:
        series = pd.to_numeric(series, errors='coerce')
        if series.notna().sum() < 3:
            return series
        lower = series.quantile(limits[0])
        upper = series.quantile(limits[1])
        return series.clip(lower=lower, upper=upper)

    @staticmethod
    def _fill_missing_series(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().sum() == 0:
            return pd.Series(0.0, index=series.index, dtype=float)
        return numeric.fillna(numeric.median())

    @staticmethod
    def _zscore_series(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if pd.isna(std) or std <= 1e-12:
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std
