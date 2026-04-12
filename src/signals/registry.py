from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from src.core.module_registry import ModuleMetadata, ModuleRegistry, RegisteredModule


@dataclass(frozen=True)
class SignalContext:
    predictions: pd.DataFrame
    model_dataset: pd.DataFrame
    feature_names: list[str]
    label_name: str


@dataclass
class SignalResult:
    signal_scores: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


SignalGenerator = Callable[[SignalContext, dict[str, Any]], SignalResult]


class SignalRegistry(ModuleRegistry[SignalGenerator]):
    def run(self, name: str, context: SignalContext, params: dict[str, Any] | None = None) -> SignalResult:
        return self.get(name).implementation(context, params or {})


def register_signal(registry: SignalRegistry, metadata: ModuleMetadata, generator: SignalGenerator) -> None:
    registry.register(RegisteredModule(metadata=metadata, implementation=generator))
