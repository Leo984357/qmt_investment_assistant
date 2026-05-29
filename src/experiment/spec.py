from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.features.factor_catalog import build_default_catalog
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
    # 市场状态检测参数
    regime_detection: bool = False
    regime_short_window: int = 20
    regime_long_window: int = 120
    regime_volatility_window: int = 20
    regime_bull_exposure: float = 1.0
    regime_bear_exposure: float = 0.3
    regime_trending_exposure: float = 0.7
    regime_ranging_exposure: float = 0.5


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
class ResearchProtocolSpec:
    stage: str = 'discovery'
    hypothesis_id: str | None = None
    candidate_id: str | None = None
    data_mined: bool = True
    frozen_after: str | None = None
    discovery_window: dict[str, str] = field(default_factory=dict)
    validation_window: dict[str, str] = field(default_factory=dict)
    holdout_window: dict[str, str] = field(default_factory=dict)
    allowed_change_after_freeze: str = 'research_only'
    notes: str = ''


@dataclass(frozen=True)
class MultipleTestingSpec:
    enabled: bool = False
    search_space_id: str | None = None
    candidate_count: int = 0
    factor_trial_count: int = 0
    parameter_trial_count: int = 0
    correction_method: str = 'none'
    random_baseline: bool = False
    permutation_test: bool = False
    white_noise_baseline: bool = False
    stability_over_best: bool = False
    notes: str = ''


@dataclass(frozen=True)
class RiskAttributionSpec:
    enabled: bool = False
    industry_exposure: bool = False
    style_exposure: bool = False
    beta_exposure: bool = False
    benchmark_relative_active_risk: bool = False
    neutralized_ic: bool = False
    return_attribution: bool = False
    max_single_industry_active_weight: float | None = None
    notes: str = ''


@dataclass(frozen=True)
class OverlaySpec:
    enabled: bool = False
    regime_exposure_enabled: bool = False
    hypothesis_id: str | None = None
    data_mined: bool = True
    frozen_after: str | None = None
    notes: str = ''


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
    research_protocol: ResearchProtocolSpec = field(default_factory=ResearchProtocolSpec)
    multiple_testing: MultipleTestingSpec = field(default_factory=MultipleTestingSpec)
    risk_attribution: RiskAttributionSpec = field(default_factory=RiskAttributionSpec)
    overlay: OverlaySpec = field(default_factory=OverlaySpec)
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


# Rejected factors list - hard block for production experiments
# Based on factor health check results (2026-04)
REJECTED_FACTORS = {
    # 波动率因子 - A股波动率溢价不成立
    'vol20', 'vol60', 'vol120',
    # 短期动量 - A股反转效应更强
    'mom20', 'mom60',
    # 短期反转
    'rev5', 'rev10',
    # 流动性因子 - 覆盖率问题
    'liq20', 'liq60',
    # 价格位置因子 - IC显著为负
    'high_low_pos20', 'high_low_pos60',
    # WorldQuant Alpha - 显著负向
    'alpha_004',  # 简化版IC=0.144是错误的，原版IC=-0.001
    'alpha_018', 'alpha_031', 'alpha_032', 'alpha_039',
    'alpha_087', 'alpha_088',
}


def _validate_factors(spec_dict: dict, allow_rejected: bool = False, registry_stage: str = 'production') -> list[str]:
    """
    Validate factor names against rejected list and factor catalog.
    
    Args:
        spec_dict: Experiment config dict
        allow_rejected: If True, allow rejected factors (for diagnostic/research experiments)
        registry_stage: production, research, or diagnostic
    
    Returns:
        List of violations (empty = valid)
    """
    violations = []
    features = spec_dict.get('features', {}).get('names', [])
    
    # Validate registry_stage
    valid_stages = {'production', 'research', 'diagnostic'}
    if registry_stage not in valid_stages:
        violations.append(f"Invalid registry_stage: {registry_stage}. Must be one of {valid_stages}")
        return violations
    
    # Research stage: only warn, don't block
    if registry_stage == 'research':
        if rejected_found := [f for f in features if f in REJECTED_FACTORS]:
            violations.append(
                f"Research config contains rejected factors: {rejected_found}. "
                f"Results are for research only, not formal conclusions."
            )
        return violations
    
    # 1. Check rejected list for production/diagnostic
    rejected_found = [f for f in features if f in REJECTED_FACTORS]
    
    if rejected_found and not allow_rejected:
        violations.append(
            f"Rejected factors found: {rejected_found}. "
            f"Set allow_rejected_factors: true in experiment config to bypass (diagnostic only)."
        )
    
    # 2. Check production + allow_rejected combo
    if registry_stage == 'production' and allow_rejected:
        violations.append(
            f"Production config cannot have allow_rejected_factors: true. "
            f"Move rejected factors to diagnostic configs."
        )
    
    # 3. Production factors must be in the catalog. Unknown factors can be
    # explored in research/diagnostic, but cannot support formal conclusions.
    if registry_stage == 'production':
        catalog = build_default_catalog()
        unknown_factors = []
        for f in features:
            profile = catalog.get(f)
            if profile is None:
                unknown_factors.append(f)
        
        if unknown_factors:
            violations.append(
                f"Unknown factors (not in catalog): {unknown_factors}. "
                f"Add to src/features/factor_catalog.py or keep the config in research/diagnostic stage."
            )
    
    return violations


def _default_protocol_for_stage(registry_stage: str) -> dict[str, Any]:
    if registry_stage == 'diagnostic':
        return {
            'stage': 'diagnostic',
            'data_mined': True,
            'allowed_change_after_freeze': 'research_only',
            'notes': 'Auto-filled diagnostic protocol. Diagnostic results cannot support formal conclusions.',
        }
    if registry_stage == 'research':
        return {
            'stage': 'discovery',
            'data_mined': True,
            'allowed_change_after_freeze': 'research_only',
            'notes': 'Auto-filled discovery protocol. Discovery results require independent validation.',
        }
    return {
        'stage': 'production',
        'data_mined': True,
        'allowed_change_after_freeze': 'research_only',
        'notes': 'Auto-filled unsafe production protocol. Add explicit research_protocol before production use.',
    }


def _normalize_protocol_window(protocol: dict[str, Any], key: str) -> dict[str, str]:
    window = dict(protocol.get(key) or {})
    if not window:
        return {}
    for field_name in ('start', 'end'):
        if field_name in window and window[field_name] is not None:
            window[field_name] = _normalize_date_config_value(window[field_name], f'research_protocol.{key}.{field_name}')
    return window


def _resolve_research_protocol(resolved: dict[str, Any]) -> dict[str, Any]:
    registry_stage = resolved.get('model', {}).get('registry_stage', 'production')
    protocol = _deep_merge(_default_protocol_for_stage(registry_stage), dict(resolved.get('research_protocol') or {}))
    for window_name in ('discovery_window', 'validation_window', 'holdout_window'):
        protocol[window_name] = _normalize_protocol_window(protocol, window_name)
    if protocol.get('frozen_after'):
        protocol['frozen_after'] = _normalize_date_config_value(protocol['frozen_after'], 'research_protocol.frozen_after')
    return protocol


def _window_is_complete(protocol: dict[str, Any], key: str) -> bool:
    window = protocol.get(key) or {}
    return bool(window.get('start') and window.get('end'))


def _validate_research_protocol(resolved: dict[str, Any], protocol: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    registry_stage = resolved.get('model', {}).get('registry_stage', 'production')
    stage = protocol.get('stage')
    valid_stages = {'diagnostic', 'discovery', 'validation', 'holdout', 'production'}
    valid_change_modes = {'research_only', 'none', 'risk_only', 'cost_only'}

    if stage not in valid_stages:
        violations.append(f"Invalid research_protocol.stage: {stage!r}. Must be one of {sorted(valid_stages)}")
        return violations

    if protocol.get('allowed_change_after_freeze') not in valid_change_modes:
        violations.append(
            "Invalid research_protocol.allowed_change_after_freeze: "
            f"{protocol.get('allowed_change_after_freeze')!r}. Must be one of {sorted(valid_change_modes)}"
        )

    if registry_stage == 'diagnostic' and stage != 'diagnostic':
        violations.append("Diagnostic configs must use research_protocol.stage: diagnostic")

    if registry_stage == 'research' and stage not in {'diagnostic', 'discovery', 'validation', 'holdout'}:
        violations.append("Research configs cannot use research_protocol.stage: production")

    if registry_stage == 'production' and stage != 'production':
        violations.append("Production configs must use research_protocol.stage: production")

    if stage in {'validation', 'holdout', 'production'}:
        for field_name in ('hypothesis_id', 'candidate_id', 'frozen_after'):
            if not protocol.get(field_name):
                violations.append(f"research_protocol.{field_name} is required for {stage} stage")

    if stage in {'holdout', 'production'} and not _window_is_complete(protocol, 'holdout_window'):
        violations.append(f"research_protocol.holdout_window.start/end is required for {stage} stage")

    if stage == 'production':
        if bool(protocol.get('data_mined')):
            violations.append("Production configs must set research_protocol.data_mined: false")
        if not _window_is_complete(protocol, 'validation_window'):
            violations.append("research_protocol.validation_window.start/end is required for production stage")
        if protocol.get('allowed_change_after_freeze') == 'research_only':
            violations.append("Production configs must freeze logic: allowed_change_after_freeze cannot be research_only")

    return violations


def _resolve_overlay(resolved: dict[str, Any]) -> dict[str, Any]:
    overlay = dict(resolved.get('overlay') or {})
    if overlay.get('frozen_after'):
        overlay['frozen_after'] = _normalize_date_config_value(overlay['frozen_after'], 'overlay.frozen_after')
    return overlay


def _validate_professional_controls(
    resolved: dict[str, Any],
    protocol: dict[str, Any],
    multiple_testing: dict[str, Any],
    risk_attribution: dict[str, Any],
    overlay: dict[str, Any],
) -> list[str]:
    violations: list[str] = []
    registry_stage = resolved.get('model', {}).get('registry_stage', 'production')
    protocol_stage = protocol.get('stage')
    features = resolved.get('features', {}).get('names', [])
    portfolio = resolved.get('portfolio', {})
    legacy_regime_enabled = bool(portfolio.get('regime_detection', False))
    overlay_enabled = bool(overlay.get('enabled', False) or overlay.get('regime_exposure_enabled', False))

    if legacy_regime_enabled and not overlay_enabled:
        violations.append(
            "portfolio.regime_detection is an overlay strategy, not execution enhancement. "
            "Declare overlay.regime_exposure_enabled: true and document overlay.hypothesis_id."
        )

    if registry_stage != 'production' and protocol_stage not in {'holdout', 'production'}:
        return violations

    if not bool(multiple_testing.get('enabled', False)):
        violations.append("multiple_testing.enabled is required for holdout/production conclusions")
    if not multiple_testing.get('search_space_id'):
        violations.append("multiple_testing.search_space_id is required for holdout/production conclusions")
    if int(multiple_testing.get('candidate_count', 0) or 0) <= 0:
        violations.append("multiple_testing.candidate_count must record how many candidates were tried")
    if int(multiple_testing.get('factor_trial_count', 0) or 0) < len(features):
        violations.append("multiple_testing.factor_trial_count must be at least the number of used factors")
    if int(multiple_testing.get('parameter_trial_count', 0) or 0) <= 0:
        violations.append("multiple_testing.parameter_trial_count must record parameter/model trials")
    if multiple_testing.get('correction_method', 'none') in {'none', '', None}:
        violations.append("multiple_testing.correction_method cannot be 'none' for holdout/production conclusions")
    if not any(
        bool(multiple_testing.get(key, False))
        for key in ('random_baseline', 'permutation_test', 'white_noise_baseline')
    ):
        violations.append("multiple_testing requires at least one random/permutation/white-noise baseline")
    if not bool(multiple_testing.get('stability_over_best', False)):
        violations.append("multiple_testing.stability_over_best must be true; best-run selection is not sufficient")

    if not bool(risk_attribution.get('enabled', False)):
        violations.append("risk_attribution.enabled is required for holdout/production conclusions")
    for key in (
        'industry_exposure',
        'style_exposure',
        'beta_exposure',
        'benchmark_relative_active_risk',
        'neutralized_ic',
        'return_attribution',
    ):
        if not bool(risk_attribution.get(key, False)):
            violations.append(f"risk_attribution.{key} is required for holdout/production conclusions")

    if overlay_enabled:
        if not overlay.get('hypothesis_id'):
            violations.append("overlay.hypothesis_id is required when any overlay strategy is enabled")
        if bool(overlay.get('data_mined', True)):
            violations.append("overlay.data_mined must be false for holdout/production conclusions")
        if not overlay.get('frozen_after'):
            violations.append("overlay.frozen_after is required when overlay affects holdout/production results")

    return violations


def load_experiment_spec(
    path: str | Path,
    validate_contract: bool = True,
    allow_rejected_factors: bool = False,
) -> ExperimentSpec:
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
    
    # Validate factor names against rejected list and catalog
    registry_stage = resolved.get('model', {}).get('registry_stage', 'production')
    allow_from_config = bool(allow_rejected_factors or resolved.get('allow_rejected_factors', False))
    factor_violations = _validate_factors(resolved, allow_rejected=allow_from_config, registry_stage=registry_stage)
    if factor_violations:
        raise ValueError(
            f"Experiment '{resolved.get('name', path)}' has invalid factors:\n" +
            "\n".join(f"  - {v}" for v in factor_violations)
        )

    research_protocol_section = _resolve_research_protocol(resolved)
    protocol_violations = _validate_research_protocol(resolved, research_protocol_section)
    if protocol_violations:
        raise ValueError(
            f"Experiment '{resolved.get('name', path)}' violates research protocol:\n" +
            "\n".join(f"  - {v}" for v in protocol_violations)
        )
    multiple_testing_section = dict(resolved.get('multiple_testing', {}))
    risk_attribution_section = dict(resolved.get('risk_attribution', {}))
    overlay_section = _resolve_overlay(resolved)
    professional_control_violations = _validate_professional_controls(
        resolved=resolved,
        protocol=research_protocol_section,
        multiple_testing=multiple_testing_section,
        risk_attribution=risk_attribution_section,
        overlay=overlay_section,
    )
    if professional_control_violations:
        raise ValueError(
            f"Experiment '{resolved.get('name', path)}' violates professional research controls:\n" +
            "\n".join(f"  - {v}" for v in professional_control_violations)
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
        research_protocol=ResearchProtocolSpec(**research_protocol_section),
        multiple_testing=MultipleTestingSpec(**multiple_testing_section),
        risk_attribution=RiskAttributionSpec(**risk_attribution_section),
        overlay=OverlaySpec(**overlay_section),
        enhancer=EnhancerSpec(**enhancer_section) if enhancer_section else EnhancerSpec(),
    )
