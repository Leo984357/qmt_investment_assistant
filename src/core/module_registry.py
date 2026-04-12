from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Generic, TypeVar

import pandas as pd

T = TypeVar('T')


@dataclass(frozen=True)
class ModuleMetadata:
    name: str
    version: str
    category: str
    description: str
    owner: str = 'system'
    level: str = 'security'
    frequency: str = '1d'
    inputs: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_record(self) -> dict:
        record = asdict(self)
        record['inputs'] = ','.join(self.inputs)
        record['outputs'] = ','.join(self.outputs)
        record['tags'] = ','.join(self.tags)
        return record


@dataclass(frozen=True)
class RegisteredModule(Generic[T]):
    metadata: ModuleMetadata
    implementation: T


class ModuleRegistry(Generic[T]):
    def __init__(self):
        self._modules: dict[str, RegisteredModule[T]] = {}

    def register(self, module: RegisteredModule[T]) -> None:
        self._modules[module.metadata.name] = module

    def get(self, name: str) -> RegisteredModule[T]:
        return self._modules[name]

    def names(self) -> list[str]:
        return sorted(self._modules)

    def inventory(self) -> pd.DataFrame:
        if not self._modules:
            return pd.DataFrame(columns=['name', 'version', 'category', 'description', 'owner', 'level', 'frequency', 'inputs', 'outputs', 'tags'])
        return pd.DataFrame([module.metadata.to_record() for module in self._modules.values()]).sort_values('name').reset_index(drop=True)
