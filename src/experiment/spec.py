from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.ops.paths import CONFIGS_DIR


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_date_config_value(value: Any, field_name: str, allow_latest_completed: bool = False) -> str:
    text = str(value).strip()
    if allow_latest_completed and text == 'latest_completed':
        return text
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
        raise ValueError(f'{field_name} 必须是 YYYY-MM-DD，收到 {text!r}')
    ts = yaml.safe_load(f"ts: {text}")['ts']
    ts = getattr(ts, 'strftime', lambda *_: text)('%Y-%m-%d') if ts is not None else text
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', ts):
        raise ValueError(f'{field_name} 解析失败，收到 {text!r}')
    year = int(ts[:4])
    if year < 1990 or year > 2100:
        raise ValueError(f'{field_name} 年份超出允许范围，收到 {ts!r}')
    return ts


@dataclass(frozen=True)
class DataSpec:
    source: str
    snapshot_id: str
    start_date: str
    end_date: str
    universe_name: str
    formal_start_date: str | None = None
    universe_mode: str = 'point_in_time'
    bootstrap_if_missing: bool = True
    n_symbols_master: int | None = None
    n_universe: int | None = None
    seed: int | None = None
    incremental: bool = True
    universe_refresh_frequency_days: int = 1
    price_adjust: str = 'qfq'


@dataclass(frozen=True)
class FeatureSetSpec:
    set_name: str
    version: str
    names: list[str]
    winsorize_limits: tuple[float, float] = (0.01, 0.99)
    zscore: bool = True
    fill_missing: bool = False


@dataclass(frozen=True)
class LabelSpec:
    name: str
    horizon: int


@dataclass(frozen=True)
class ModelSpec:
    family: str
    version: str
    registry_stage: str
    fallback_model: str
    train_window_days: int
    valid_window_days: int
    min_train_samples: int
    training_embargo_days: int = 0
    fallback_feature: str | None = None
    score_blend_feature: str | None = None
    score_blend_weight_model: float = 1.0
    score_blend_weight_feature: float = 0.0
    label_clip: tuple[float, float] | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalSpec:
    name: str = 'cross_sectional_score'
    version: str = 'v1'
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioSpec:
    top_n: int
    weighting: str
    gross_exposure: float
    defensive_gross: float
    max_single_weight: float
    cash_buffer: float
    min_trade_value: float
    market_filter_lookback: int
    market_filter_threshold: float
    risk_model: str = 'two_tier_momentum'
    risk_ma_short_window: int = 60
    risk_ma_long_window: int = 120
    risk_momentum_window: int = 20
    risk_mid_exposure: float = 0.85
    risk_low_exposure: float = 0.65
    risk_crash_exposure: float = 0.45
    candidate_filter_mode: str = 'strict_ashare'
    constructor: str = 'qmt_topn_equal_weight'
    # 新增参数
    min_single_weight: float = 0.005
    max_industry_weight: float = 0.25
    industry_neutral: bool = False
    max_turnover: float = 0.50
    turnover_penalty: float = 0.10


@dataclass(frozen=True)
class BacktestSpec:
    initial_cash: float
    lot_size: int
    commission_bps: float
    stamp_duty_bps: float
    slippage_bps: float
    rebalance_frequency_days: int
    trade_delay_days: int
    anchor_mode: str = 'fixed'
    anchor_date: str | None = None
    execution_constraint_mode: str = 'strict_ashare'


@dataclass(frozen=True)
class EvaluationSpec:
    suite: str = 'basic_factor_diagnostics'
    version: str = 'v1'
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    description: str
    data: DataSpec
    features: FeatureSetSpec
    label: LabelSpec
    model: ModelSpec
    signal: SignalSpec
    portfolio: PortfolioSpec
    backtest: BacktestSpec
    evaluation: EvaluationSpec

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flattened_params(self) -> dict[str, Any]:
        result: dict[str, Any] = {'experiment_name': self.name, 'config_hash': self.config_hash}
        for section_name, section in self.to_dict().items():
            if isinstance(section, dict):
                for key, value in section.items():
                    if isinstance(value, dict):
                        for nested_key, nested_value in value.items():
                            result[f'{section_name}.{key}.{nested_key}'] = nested_value
                    else:
                        result[f'{section_name}.{key}'] = value
            else:
                result[section_name] = section
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False), encoding='utf-8')


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    experiment_path = Path(path)
    defaults = {}
    for default_name in ('data.yaml', 'backtest.yaml'):
        default_path = CONFIGS_DIR / default_name
        if default_path.exists():
            defaults = _deep_merge(defaults, yaml.safe_load(default_path.read_text(encoding='utf-8')) or {})
    raw = yaml.safe_load(experiment_path.read_text(encoding='utf-8')) or {}
    resolved = _deep_merge(defaults, raw)
    model_section = dict(resolved['model'])
    signal_section = dict(resolved.get('signal', {'name': 'cross_sectional_score', 'version': 'v1', 'params': {}}))
    evaluation_section = dict(resolved.get('evaluation', {'suite': 'basic_factor_diagnostics', 'version': 'v1', 'params': {}}))
    data_section = dict(resolved['data'])
    data_section['start_date'] = _normalize_date_config_value(data_section['start_date'], 'data.start_date')
    data_section['end_date'] = _normalize_date_config_value(data_section['end_date'], 'data.end_date', allow_latest_completed=True)
    label_clip = model_section.get('label_clip')
    if label_clip is not None:
        model_section['label_clip'] = tuple(label_clip)
    portfolio_section = dict(resolved['portfolio'])
    if 'constructor' not in portfolio_section:
        portfolio_section['constructor'] = 'qmt_topn_equal_weight'
    backtest_section = dict(resolved['backtest'])
    if backtest_section.get('anchor_date'):
        backtest_section['anchor_date'] = _normalize_date_config_value(backtest_section['anchor_date'], 'backtest.anchor_date')
    return ExperimentSpec(
        name=resolved['name'],
        description=resolved.get('description', ''),
        data=DataSpec(**data_section),
        features=FeatureSetSpec(
            set_name=resolved['features']['set_name'],
            version=resolved['features']['version'],
            names=list(resolved['features']['names']),
            winsorize_limits=tuple(resolved['features'].get('winsorize_limits', (0.01, 0.99))),
            zscore=bool(resolved['features'].get('zscore', True)),
            fill_missing=bool(resolved['features'].get('fill_missing', False)),
        ),
        label=LabelSpec(**resolved['label']),
        model=ModelSpec(**model_section),
        signal=SignalSpec(**signal_section),
        portfolio=PortfolioSpec(**portfolio_section),
        backtest=BacktestSpec(**backtest_section),
        evaluation=EvaluationSpec(**evaluation_section),
    )
