from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR, ROOT


class ConfigError(ValueError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f'配置文件 {path} 顶层必须是对象。')
    return data


def _ensure_known_keys(section_name: str, data: Mapping[str, Any], allowed_keys: set[str]) -> None:
    unknown_keys = sorted(set(data) - allowed_keys)
    if unknown_keys:
        joined = ', '.join(unknown_keys)
        raise ConfigError(f'{section_name} 存在未知配置项: {joined}')


def _as_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f'{field_name} 必须是非空字符串。')
    return value.strip()


def _as_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f'{field_name} 必须是整数。') from exc


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f'{field_name} 必须是数值。') from exc


@dataclass(frozen=True)
class CommonConfig:
    project_name: str = 'QMT Investment Assistant'
    db_path: str = 'db/meta.sqlite'
    timezone: str = 'Asia/Shanghai'
    strategy_name: str = 'demo_rotation'
    latest_n_days: int = 180

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> CommonConfig:
        raw = dict(data or {})
        defaults = cls()
        _ensure_known_keys('common', raw, {'project_name', 'db_path', 'timezone', 'strategy_name', 'latest_n_days'})
        cfg = cls(
            project_name=_as_str(raw.get('project_name', defaults.project_name), 'common.project_name'),
            db_path=_as_str(raw.get('db_path', defaults.db_path), 'common.db_path'),
            timezone=_as_str(raw.get('timezone', defaults.timezone), 'common.timezone'),
            strategy_name=_as_str(raw.get('strategy_name', defaults.strategy_name), 'common.strategy_name'),
            latest_n_days=_as_int(raw.get('latest_n_days', defaults.latest_n_days), 'common.latest_n_days'),
        )
        if cfg.latest_n_days <= 0:
            raise ConfigError('common.latest_n_days 必须大于 0。')
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyConfig:
    top_n: int = 5
    rebalance_cash_buffer: float = 0.05
    momentum_lookback: int = 20
    long_lookback: int = 60
    min_price: float = 2.0
    max_single_weight: float = 0.25
    target_gross: float = 0.95

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> StrategyConfig:
        raw = dict(data or {})
        defaults = cls()
        _ensure_known_keys(
            'strategy',
            raw,
            {'top_n', 'rebalance_cash_buffer', 'momentum_lookback', 'long_lookback', 'min_price', 'max_single_weight', 'target_gross'},
        )
        cfg = cls(
            top_n=_as_int(raw.get('top_n', defaults.top_n), 'strategy.top_n'),
            rebalance_cash_buffer=_as_float(raw.get('rebalance_cash_buffer', defaults.rebalance_cash_buffer), 'strategy.rebalance_cash_buffer'),
            momentum_lookback=_as_int(raw.get('momentum_lookback', defaults.momentum_lookback), 'strategy.momentum_lookback'),
            long_lookback=_as_int(raw.get('long_lookback', defaults.long_lookback), 'strategy.long_lookback'),
            min_price=_as_float(raw.get('min_price', defaults.min_price), 'strategy.min_price'),
            max_single_weight=_as_float(raw.get('max_single_weight', defaults.max_single_weight), 'strategy.max_single_weight'),
            target_gross=_as_float(raw.get('target_gross', defaults.target_gross), 'strategy.target_gross'),
        )
        if cfg.top_n <= 0:
            raise ConfigError('strategy.top_n 必须大于 0。')
        if cfg.momentum_lookback < 2:
            raise ConfigError('strategy.momentum_lookback 必须至少为 2。')
        if cfg.long_lookback <= cfg.momentum_lookback:
            raise ConfigError('strategy.long_lookback 必须大于 strategy.momentum_lookback。')
        if cfg.min_price <= 0:
            raise ConfigError('strategy.min_price 必须大于 0。')
        if not 0 < cfg.rebalance_cash_buffer < 1:
            raise ConfigError('strategy.rebalance_cash_buffer 必须在 0 和 1 之间。')
        if not 0 < cfg.max_single_weight <= 1:
            raise ConfigError('strategy.max_single_weight 必须在 0 和 1 之间。')
        if not 0 < cfg.target_gross <= 1:
            raise ConfigError('strategy.target_gross 必须在 0 和 1 之间。')
        if cfg.max_single_weight > cfg.target_gross:
            raise ConfigError('strategy.max_single_weight 不能大于 strategy.target_gross。')
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionConfig:
    min_rebalance_l1: float = 0.12
    min_trade_value: float = 1000.0
    max_name_changes: int = 6
    turnover_penalty_bps: float = 12.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> DecisionConfig:
        raw = dict(data or {})
        defaults = cls()
        _ensure_known_keys('decision', raw, {'min_rebalance_l1', 'min_trade_value', 'max_name_changes', 'turnover_penalty_bps'})
        cfg = cls(
            min_rebalance_l1=_as_float(raw.get('min_rebalance_l1', defaults.min_rebalance_l1), 'decision.min_rebalance_l1'),
            min_trade_value=_as_float(raw.get('min_trade_value', defaults.min_trade_value), 'decision.min_trade_value'),
            max_name_changes=_as_int(raw.get('max_name_changes', defaults.max_name_changes), 'decision.max_name_changes'),
            turnover_penalty_bps=_as_float(raw.get('turnover_penalty_bps', defaults.turnover_penalty_bps), 'decision.turnover_penalty_bps'),
        )
        if cfg.min_rebalance_l1 < 0:
            raise ConfigError('decision.min_rebalance_l1 不能为负数。')
        if cfg.min_trade_value < 0:
            raise ConfigError('decision.min_trade_value 不能为负数。')
        if cfg.max_name_changes < 0:
            raise ConfigError('decision.max_name_changes 不能为负数。')
        if cfg.turnover_penalty_bps < 0:
            raise ConfigError('decision.turnover_penalty_bps 不能为负数。')
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = 'mock'
    lot_size: int = 100
    slippage_bps: float = 8.0
    commission_bps: float = 0.75
    stamp_duty_bps: float = 10.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> ExecutionConfig:
        raw = dict(data or {})
        defaults = cls()
        _ensure_known_keys('execution', raw, {'mode', 'lot_size', 'slippage_bps', 'commission_bps', 'stamp_duty_bps'})
        cfg = cls(
            mode=_as_str(raw.get('mode', defaults.mode), 'execution.mode'),
            lot_size=_as_int(raw.get('lot_size', defaults.lot_size), 'execution.lot_size'),
            slippage_bps=_as_float(raw.get('slippage_bps', defaults.slippage_bps), 'execution.slippage_bps'),
            commission_bps=_as_float(raw.get('commission_bps', defaults.commission_bps), 'execution.commission_bps'),
            stamp_duty_bps=_as_float(raw.get('stamp_duty_bps', defaults.stamp_duty_bps), 'execution.stamp_duty_bps'),
        )
        if cfg.mode not in {'mock', 'qmt'}:
            raise ConfigError('execution.mode 只支持 mock 或 qmt。')
        if cfg.lot_size <= 0:
            raise ConfigError('execution.lot_size 必须大于 0。')
        if cfg.slippage_bps < 0 or cfg.commission_bps < 0 or cfg.stamp_duty_bps < 0:
            raise ConfigError('execution 中的 bps 配置不能为负数。')
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AppConfig:
    common: CommonConfig
    strategy: StrategyConfig
    decision: DecisionConfig
    execution: ExecutionConfig

    @property
    def db_path(self) -> Path:
        return ROOT / self.common.db_path

    def to_dict(self) -> dict[str, Any]:
        return {
            'common': self.common.to_dict(),
            'strategy': self.strategy.to_dict(),
            'decision': self.decision.to_dict(),
            'execution': self.execution.to_dict(),
        }


def load_app_config() -> AppConfig:
    common = CommonConfig.from_mapping(_load_yaml(CONFIG_DIR / 'common.yaml'))
    strategy = StrategyConfig.from_mapping(_load_yaml(CONFIG_DIR / 'strategy.yaml').get('strategy', {}))
    decision = DecisionConfig.from_mapping(_load_yaml(CONFIG_DIR / 'decision.yaml').get('decision', {}))
    execution = ExecutionConfig.from_mapping(_load_yaml(CONFIG_DIR / 'execution.yaml').get('execution', {}))
    return AppConfig(common=common, strategy=strategy, decision=decision, execution=execution)
