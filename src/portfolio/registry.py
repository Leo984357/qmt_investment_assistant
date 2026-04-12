from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.core.module_registry import ModuleMetadata, ModuleRegistry, RegisteredModule


@dataclass(frozen=True)
class PortfolioContext:
    signal_scores: pd.DataFrame
    universe_membership: pd.DataFrame
    tradability: pd.DataFrame
    market_reference: pd.Series
    trade_calendar: pd.DataFrame
    universe_name: str
    portfolio_config: Any
    backtest_config: Any


PortfolioBuilder = Callable[[PortfolioContext], Any]


class PortfolioRegistry(ModuleRegistry[PortfolioBuilder]):
    def build(self, name: str, context: PortfolioContext) -> Any:
        return self.get(name).implementation(context)


def register_portfolio(registry: PortfolioRegistry, metadata: ModuleMetadata, builder: PortfolioBuilder) -> None:
    registry.register(RegisteredModule(metadata=metadata, implementation=builder))
