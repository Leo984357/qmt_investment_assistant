from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.core.module_registry import ModuleMetadata, ModuleRegistry, RegisteredModule


@dataclass(frozen=True)
class ModelRunContext:
    dataset: pd.DataFrame
    feature_names: list[str]
    label_name: str
    rebalance_dates: list[pd.Timestamp]
    artifact_dir: Any


ModelBuilder = Callable[[Any], Any]


class ModelRegistry(ModuleRegistry[ModelBuilder]):
    def build(self, name: str, config: Any) -> Any:
        return self.get(name).implementation(config)


def register_model(registry: ModelRegistry, metadata: ModuleMetadata, builder: ModelBuilder) -> None:
    registry.register(RegisteredModule(metadata=metadata, implementation=builder))
