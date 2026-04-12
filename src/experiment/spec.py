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
    test_window_days: int = 60
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
class EnhancerSpec:
    enabled: bool = True
    buffer_retain_threshold_rank: int = 50
    buffer_max_retain_ratio: float = 0.6
    smoother_step_ratio: float = 0.5
    smoother_min_change_threshold: float = 0.001
    cost_min_alpha_threshold: float = 0.002
    cost_cost_to_alpha_ratio: float = 0.3


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
    enhancer: EnhancerSpec = EnhancerSpec()

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


def _load_research_contract() -> dict:
    """Load the active research contract for validation."""
    contract_path = CONFIGS_DIR / 'research_contract_v1.yaml'
    if contract_path.exists():
        return yaml.safe_load(contract_path.read_text(encoding='utf-8')) or {}
    return {}


def _validate_against_contract(spec_dict: dict, contract: dict) -> list[str]:
    """
    Validate experiment config against research contract.
    
    Returns list of violations (empty = valid).
    """
    violations = []
    
    if not contract:
        return violations
    
    # Check universe
    if 'universe' in contract:
        expected_universe = contract['universe'].get('name')
        if expected_universe and spec_dict.get('data', {}).get('universe_name') != expected_universe:
            violations.append(f"Universe mismatch: expected {expected_universe}, got {spec_dict.get('data', {}).get('universe_name')}")
    
    # Check label
    if 'label' in contract:
        expected_label = contract['label'].get('name')
        if expected_label and spec_dict.get('label', {}).get('name') != expected_label:
            violations.append(f"Label mismatch: expected {expected_label}, got {spec_dict.get('label', {}).get('name')}")
        
        expected_horizon = contract['label'].get('horizon')
        if expected_horizon and spec_dict.get('label', {}).get('horizon') != expected_horizon:
            violations.append(f"Label horizon mismatch: expected {expected_horizon}, got {spec_dict.get('label', {}).get('horizon')}")
    
    # Check rebalance frequency
    if 'frequency' in contract:
        expected_rebalance = contract['frequency'].get('rebalance_days')
        if expected_rebalance and spec_dict.get('backtest', {}).get('rebalance_frequency_days') != expected_rebalance:
            violations.append(f"Rebalance frequency mismatch: expected {expected_rebalance}, got {spec_dict.get('backtest', {}).get('rebalance_frequency_days')}")
    
    # Check cost parameters
    if 'cost' in contract:
        expected_commission = contract['cost'].get('commission_bps')
        if expected_commission and spec_dict.get('backtest', {}).get('commission_bps') != expected_commission:
            violations.append(f"Commission mismatch: expected {expected_commission}bps, got {spec_dict.get('backtest', {}).get('commission_bps')}bps")
        
        expected_stamp = contract['cost'].get('stamp_duty_bps')
        if expected_stamp and spec_dict.get('backtest', {}).get('stamp_duty_bps') != expected_stamp:
            violations.append(f"Stamp duty mismatch: expected {expected_stamp}bps, got {spec_dict.get('backtest', {}).get('stamp_duty_bps')}bps")
    
    # Check walk-forward
    if 'walk_forward' in contract:
        expected_train = contract['walk_forward'].get('train_window_days')
        if expected_train and spec_dict.get('model', {}).get('train_window_days') != expected_train:
            violations.append(f"Train window mismatch: expected {expected_train}, got {spec_dict.get('model', {}).get('train_window_days')}")
        
        expected_test = contract['walk_forward'].get('test_window_days')
        if expected_test and spec_dict.get('model', {}).get('test_window_days') != expected_test:
            violations.append(f"Test window mismatch: expected {expected_test}, got {spec_dict.get('model', {}).get('test_window_days')}")
    
    return violations


def load_experiment_spec(path: str | Path, validate_contract: bool = True) -> ExperimentSpec:
    experiment_path = Path(path)
    defaults = {}
    for default_name in ('data.yaml', 'backtest.yaml'):
        default_path = CONFIGS_DIR / default_name
        if default_path.exists():
            defaults = _deep_merge(defaults, yaml.safe_load(default_path.read_text(encoding='utf-8')) or {})
    raw = yaml.safe_load(experiment_path.read_text(encoding='utf-8')) or {}
    resolved = _deep_merge(defaults, raw)
    
    # Patch resolved with ModelSpec defaults before validation
    # This ensures validation sees defaults, not raw None/missing values
    model_defaults = {
        'test_window_days': 60,
        'training_embargo_days': 0,
    }
    for key, default_val in model_defaults.items():
        if resolved.get('model', {}).get(key) is None:
            resolved.setdefault('model', {})[key] = default_val
    
    # Validate against research contract
    if validate_contract:
        contract = _load_research_contract()
        violations = _validate_against_contract(resolved, contract)
        if violations:
            raise ValueError(
                f"Experiment '{resolved.get('name', path)}' violates research contract:\n" +
                "\n".join(f"  - {v}" for v in violations)
            )
    
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
    enhancer_section = dict(resolved.get('enhancer', {}))
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
        enhancer=EnhancerSpec(**enhancer_section) if enhancer_section else EnhancerSpec(),
    )
