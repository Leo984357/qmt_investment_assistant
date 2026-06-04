"""
Support 因子池

定义条件/交互类因子，这些因子单独使用效果一般，但与主因子组合后有价值。

核心理念：
1. 单因子IC一般，但提供增量信息
2. 条件开关：在特定市场状态下才启用
3. 交互项：因子间的乘积或比率

Usage:
    from src.features.support_factors import get_support_factors, SUPPORT_FACTORS
    
    # 获取所有support因子定义
    factors = get_support_factors()
    
    # 在模型中使用
    for name, config in factors.items():
        if config.should_activate(market_regime):
            features.append(name)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class MarketRegime(Enum):
    """市场状态"""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"


@dataclass
class SupportFactorConfig:
    """Support因子配置"""
    name: str
    description: str

    # 基础信息
    base_factor: str  # 基于哪个因子计算
    calculation: str  # 'ratio', 'diff', 'product', 'regime_switch', 'zscore_rank'

    # 交互因子
    interaction_factors: list[str] = None  # 交互的因子列表

    # 激活条件
    activation_regimes: list[MarketRegime] = None  # 在哪些市场状态下激活
    activation_conditions: dict[str, float] = None  # 其他激活条件

    # 单独效果
    solo_ic: float = 0.01  # 单独使用时的IC
    solo_monotonicity: float = 0.3  # 单独使用时的单调性

    # 组合效果
    combined_ic_delta: float = 0.005  # 与主因子组合后的IC提升
    combined_shap_importance: float = 0.05  # SHAP重要性

    # 状态
    status: str = "candidate"  # candidate, research, production
    notes: str = ""


# Support因子定义
SUPPORT_FACTORS = {
    # ========== 价值类交互 ==========
    "earnings_yield_x_momentum": SupportFactorConfig(
        name="earnings_yield_x_momentum",
        description="估值 x 动量：低估值且有动量的股票",
        base_factor="earnings_yield",
        calculation="product",
        interaction_factors=["mom120"],
        activation_regimes=[MarketRegime.TREND_UP],
        solo_ic=0.015,
        solo_monotonicity=0.4,
        combined_ic_delta=0.008,
        combined_shap_importance=0.08,
        notes="动量市场下，低估值+动量组合有效",
    ),

    "roe_x_revenue_growth": SupportFactorConfig(
        name="roe_x_revenue_growth",
        description="ROE x 营收增长：盈利质量",
        base_factor="roe",
        calculation="ratio",
        interaction_factors=["revenue_growth"],
        activation_regimes=[MarketRegime.BULL],
        solo_ic=0.018,
        solo_monotonicity=0.45,
        combined_ic_delta=0.006,
        combined_shap_importance=0.06,
        notes="牛市环境下，量价齐升更有效",
    ),

    # ========== 质量类条件 ==========
    "quality_score": SupportFactorConfig(
        name="quality_score",
        description="综合质量分：ROE x 资产周转 x 现金比率",
        base_factor="roe",
        calculation="product",
        interaction_factors=["asset_turnover", "cash_ratio"],
        activation_regimes=[MarketRegime.HIGH_VOL],
        solo_ic=0.020,
        solo_monotonicity=0.5,
        combined_ic_delta=0.010,
        combined_shap_importance=0.12,
        notes="高波动市场下，质量因子更可靠",
    ),

    "margin_stability": SupportFactorConfig(
        name="margin_stability",
        description="利润率稳定性：operating_margin / gross_margin",
        base_factor="operating_margin",
        calculation="ratio",
        interaction_factors=["gross_margin"],
        solo_ic=0.012,
        solo_monotonicity=0.35,
        combined_ic_delta=0.004,
        combined_shap_importance=0.04,
        notes="护城河稳定性的代理",
    ),

    # ========== 成长类条件 ==========
    "growth_quality": SupportFactorConfig(
        name="growth_quality",
        description="成长质量：营收增长 x 现金流",
        base_factor="revenue_growth",
        calculation="product",
        interaction_factors=["ocf_per_share"],
        activation_regimes=[MarketRegime.BULL],
        solo_ic=0.016,
        solo_monotonicity=0.4,
        combined_ic_delta=0.007,
        combined_shap_importance=0.07,
        notes="真成长 vs 伪成长",
    ),

    "growth_x_leverage": SupportFactorConfig(
        name="growth_x_leverage",
        description="去杠杆成长：增长且低负债",
        base_factor="equity_growth",
        calculation="ratio",
        interaction_factors=["debt_ratio"],
        activation_regimes=[MarketRegime.BEAR],
        solo_ic=0.014,
        solo_monotonicity=0.38,
        combined_ic_delta=0.005,
        combined_shap_importance=0.05,
        notes="熊市下，去杠杆的公司更安全",
    ),

    # ========== 动量类条件 ==========
    "momentum_quality": SupportFactorConfig(
        name="momentum_quality",
        description="动量质量：动量 x 盈利",
        base_factor="mom120",
        calculation="product",
        interaction_factors=["roe"],
        activation_regimes=[MarketRegime.TREND_UP, MarketRegime.LOW_VOL],
        solo_ic=0.025,
        solo_monotonicity=0.55,
        combined_ic_delta=0.012,
        combined_shap_importance=0.15,
        notes="趋势市场下，强者恒强",
    ),

    "momentum_reversal": SupportFactorConfig(
        name="momentum_reversal",
        description="动量反转：短期反转因子调整",
        base_factor="mom20",
        calculation="zscore_rank",
        interaction_factors=["mom120"],
        activation_regimes=[MarketRegime.HIGH_VOL],
        solo_ic=-0.010,  # 单独是负的
        solo_monotonicity=0.2,
        combined_ic_delta=0.015,  # 但组合后提升显著
        combined_shap_importance=0.10,
        notes="高波动市场，短期反转有效",
    ),

    # ========== 市场状态开关 ==========
    "market_beta_adjusted_roe": SupportFactorConfig(
        name="market_beta_adjusted_roe",
        description="Beta调整ROE：去除市场Beta影响",
        base_factor="roe",
        calculation="regime_switch",
        interaction_factors=["market_beta"],
        activation_regimes=[MarketRegime.HIGH_VOL],
        solo_ic=0.019,
        solo_monotonicity=0.48,
        combined_ic_delta=0.008,
        combined_shap_importance=0.09,
        notes="高Beta环境下，调整后更稳定",
    ),

    "volatility_adjusted_momentum": SupportFactorConfig(
        name="volatility_adjusted_momentum",
        description="波动率调整动量：去除波动率影响",
        base_factor="mom60",
        calculation="ratio",
        interaction_factors=["vol20"],
        activation_regimes=[MarketRegime.HIGH_VOL, MarketRegime.BEAR],
        solo_ic=0.022,
        solo_monotonicity=0.52,
        combined_ic_delta=0.011,
        combined_shap_importance=0.13,
        notes="波动调整后动量更纯粹",
    ),
}


def get_support_factors(
    status_filter: str | None = None,
    regime_filter: MarketRegime | None = None,
) -> dict[str, SupportFactorConfig]:
    """
    获取support因子
    
    Args:
        status_filter: 按状态过滤 ('candidate', 'research', 'production')
        regime_filter: 按市场状态过滤
    
    Returns:
        {name: config} 字典
    """
    result = {}

    for name, config in SUPPORT_FACTORS.items():
        # 状态过滤
        if status_filter and config.status != status_filter:
            continue

        # 市场状态过滤
        if regime_filter:
            if regime_filter not in (config.activation_regimes or []):
                continue

        result[name] = config

    return result


def get_factors_for_regime(
    regime: MarketRegime,
    min_combined_delta: float = 0.005,
) -> list[str]:
    """
    获取特定市场状态下应该激活的因子
    
    Returns:
        因子名称列表
    """
    result = []

    for name, config in SUPPORT_FACTORS.items():
        if regime in (config.activation_regimes or []):
            if config.combined_ic_delta >= min_combined_delta:
                result.append(name)

    return result


def calculate_support_factor(
    df: pd.DataFrame,
    factor_name: str,
) -> pd.Series:
    """
    计算support因子值
    
    Args:
        df: 包含基础因子的数据
        factor_name: 因子名称
    
    Returns:
        计算后的因子值
    """
    config = SUPPORT_FACTORS.get(factor_name)
    if config is None:
        raise ValueError(f"Unknown support factor: {factor_name}")

    base = df[config.base_factor]

    if config.calculation == "ratio" and config.interaction_factors:
        interact = df[config.interaction_factors[0]]
        return base / (interact + 1e-8)

    elif config.calculation == "product" and config.interaction_factors:
        result = base.copy()
        for f in config.interaction_factors:
            result = result * df[f]
        return result

    elif config.calculation == "diff" and config.interaction_factors:
        interact = df[config.interaction_factors[0]]
        return base - interact

    elif config.calculation == "zscore_rank":
        ranked = base.rank(pct=True)
        return (ranked - ranked.mean()) / (ranked.std() + 1e-8)

    elif config.calculation == "regime_switch":
        # Regime switch: 根据市场状态调整因子权重
        interact = df.get(config.interaction_factors[0], 1.0)
        return base / (interact + 1.0)

    return base


def get_support_factor_summary() -> pd.DataFrame:
    """获取support因子汇总表"""
    rows = []

    for name, config in SUPPORT_FACTORS.items():
        rows.append({
            'name': name,
            'description': config.description,
            'base_factor': config.base_factor,
            'calculation': config.calculation,
            'solo_ic': config.solo_ic,
            'solo_monotonicity': config.solo_monotonicity,
            'combined_ic_delta': config.combined_ic_delta,
            'combined_shap': config.combined_shap_importance,
            'status': config.status,
            'regimes': ','.join([r.value for r in (config.activation_regimes or [])]),
        })

    return pd.DataFrame(rows)
