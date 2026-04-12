from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from src.core.module_registry import ModuleMetadata, ModuleRegistry, RegisteredModule


@dataclass(frozen=True)
class EvaluationContext:
    signal_scores: pd.DataFrame
    label_panel: pd.DataFrame
    label_name: str
    model_dataset: pd.DataFrame
    target_weights: pd.DataFrame
    rank_ic: pd.DataFrame
    nav: pd.DataFrame
    trades: pd.DataFrame


@dataclass
class EvaluationSuiteResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    markdown: str = ''


EvaluationSuite = Callable[[EvaluationContext, dict[str, Any]], EvaluationSuiteResult]


class EvaluationRegistry(ModuleRegistry[EvaluationSuite]):
    def run(self, name: str, context: EvaluationContext, params: dict[str, Any] | None = None) -> EvaluationSuiteResult:
        return self.get(name).implementation(context, params or {})


def register_evaluation_suite(registry: EvaluationRegistry, metadata: ModuleMetadata, suite: EvaluationSuite) -> None:
    registry.register(RegisteredModule(metadata=metadata, implementation=suite))
