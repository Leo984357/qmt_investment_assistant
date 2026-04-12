"""
宏观因子库 - 25个因子

覆盖: 利率敏感度、通胀敏感度、汇率敏感度、行业轮动等四大类
数据来源: akshare宏观数据、东方财富
"""

from dataclasses import dataclass
from typing import List


@dataclass
class MacroFactor:
    """宏观因子定义"""
    name: str
    category: str
    sub_category: str
    description: str
    economic_interpretation: str
    lookback: int
    data_requirement: List[str]
    formula: str
    ic_direction: str
    update_frequency: str = "daily"


MACRO_FACTORS: List[MacroFactor] = [

    # ========== 一、利率敏感度 (7个) ==========
    
    MacroFactor(
        name="rate_sensitivity",
        category="rate_sensitivity",
        sub_category="beta",
        description="利率Beta",
        economic_interpretation="对利率变化的敏感度",
        lookback=60,
        data_requirement=["stock_return", "rate_change"],
        formula="cov(stock, rate) / var(rate)",
        ic_direction="conditional",
        update_frequency="weekly",
    ),
    MacroFactor(
        name="duration_exposure",
        category="rate_sensitivity",
        sub_category="duration",
        description="久期暴露",
        economic_interpretation="长久期股票对利率敏感",
        lookback=60,
        data_requirement=["stock_return", "bond_yield"],
        formula="stock_return / bond_yield_change",
        ic_direction="negative",
        update_frequency="weekly",
    ),
    MacroFactor(
        name="rate_up_sentiment",
        category="rate_sensitivity",
        sub_category="regime",
        description="利率上升情绪",
        economic_interpretation="利率上升环境偏好",
        lookback=20,
        data_requirement=["rate_trend"],
        formula="is_rate_up_trend",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    MacroFactor(
        name="yield_curve_slope",
        category="rate_sensitivity",
        sub_category="curve",
        description="收益率曲线斜率",
        economic_interpretation="曲线陡峭vs平坦",
        lookback=1,
        data_requirement=["10y_bond", "2y_bond"],
        formula="10y_bond_yield - 2y_bond_yield",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    MacroFactor(
        name="credit_spread",
        category="rate_sensitivity",
        sub_category="credit",
        description="信用利差",
        economic_interpretation="企业债vs国债利差",
        lookback=1,
        data_requirement=["corporate_bond", "treasury"],
        formula="corporate_yield - treasury_yield",
        ic_direction="negative",
        update_frequency="daily",
    ),
    MacroFactor(
        name="real_rate_exposure",
        category="rate_sensitivity",
        sub_category="real",
        description="实际利率暴露",
        economic_interpretation="对实际利率的敏感度",
        lookback=60,
        data_requirement=["stock_return", "real_rate"],
        formula="corr(stock_return, real_rate_change)",
        ic_direction="conditional",
        update_frequency="weekly",
    ),
    MacroFactor(
        name="rate_change_momentum",
        category="rate_sensitivity",
        sub_category="momentum",
        description="利率变化动量",
        economic_interpretation="利率趋势持续性",
        lookback=20,
        data_requirement=["rate_change"],
        formula="rate_change_20d / rate_change_60d - 1",
        ic_direction="conditional",
        update_frequency="daily",
    ),

    # ========== 二、通胀敏感度 (5个) ==========
    
    MacroFactor(
        name="inflation_sensitivity",
        category="inflation_sensitivity",
        sub_category="beta",
        description="通胀Beta",
        economic_interpretation="对通胀的敏感度",
        lookback=60,
        data_requirement=["stock_return", "cpi"],
        formula="cov(stock, cpi) / var(cpi)",
        ic_direction="conditional",
        update_frequency="monthly",
    ),
    MacroFactor(
        name="commodity_exposure",
        category="inflation_sensitivity",
        sub_category="commodity",
        description="商品敞口",
        economic_interpretation="受益于通胀",
        lookback=20,
        data_requirement=["stock_return", "commodity_index"],
        formula="corr(stock_return, commodity)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MacroFactor(
        name="gold_correlation",
        category="inflation_sensitivity",
        sub_category="commodity",
        description="黄金相关性",
        economic_interpretation="避险情绪",
        lookback=20,
        data_requirement=["stock_return", "gold"],
        formula="corr(stock_return, gold_return)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    MacroFactor(
        name="input_cost_pressure",
        category="inflation_sensitivity",
        sub_category="cost",
        description="投入成本压力",
        economic_interpretation="原材料价格上涨影响",
        lookback=20,
        data_requirement=["ppi", "industry"],
        formula="ppi_change * industry_input_intensity",
        ic_direction="negative",
        update_frequency="weekly",
    ),
    MacroFactor(
        name="pricing_power",
        category="inflation_sensitivity",
        sub_category="pricing",
        description="定价权",
        economic_interpretation="能否转嫁成本",
        lookback=0,
        data_requirement=["gross_margin", "industry"],
        formula="gross_margin / industry_avg",
        ic_direction="positive",
        update_frequency="quarterly",
    ),

    # ========== 三、汇率敏感度 (5个) ==========
    
    MacroFactor(
        name="fx_sensitivity",
        category="fx_sensitivity",
        sub_category="beta",
        description="汇率Beta",
        economic_interpretation="对人民币汇率敏感度",
        lookback=60,
        data_requirement=["stock_return", "cny_usd"],
        formula="cov(stock, cny_change) / var(cny_change)",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    MacroFactor(
        name="export_exposure",
        category="fx_sensitivity",
        sub_category="export",
        description="出口敞口",
        economic_interpretation="海外收入占比",
        lookback=0,
        data_requirement=["overseas_revenue", "total_revenue"],
        formula="overseas_revenue / total_revenue",
        ic_direction="conditional",
        update_frequency="quarterly",
    ),
    MacroFactor(
        name="import_dependency",
        category="fx_sensitivity",
        sub_category="import",
        description="进口依赖度",
        economic_interpretation="进口成本占比",
        lookback=0,
        data_requirement=["import_cost", "total_cost"],
        formula="import_cost / total_cost",
        ic_direction="negative",
        update_frequency="quarterly",
    ),
    MacroFactor(
        name="cny_strength",
        category="fx_sensitivity",
        sub_category="regime",
        description="人民币强弱",
        economic_interpretation="人民币升值/贬值环境",
        lookback=20,
        data_requirement=["cny_usd_rate"],
        formula="cny_strength_20d",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    MacroFactor(
        name="hkd_correlation",
        category="fx_sensitivity",
        sub_category="hkd",
        description="港股联动性",
        economic_interpretation="A股vs港股联动",
        lookback=20,
        data_requirement=["a_stock", "h_stock"],
        formula="corr(a_stock_return, h_stock_return)",
        ic_direction="conditional",
        update_frequency="daily",
    ),

    # ========== 四、行业轮动 (8个) ==========
    
    MacroFactor(
        name="sector_momentum_20d",
        category="sector_rotation",
        sub_category="momentum",
        description="行业20日动量",
        economic_interpretation="行业短期趋势",
        lookback=20,
        data_requirement=["sector_return"],
        formula="sector_return_20d",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MacroFactor(
        name="sector_momentum_60d",
        category="sector_rotation",
        sub_category="momentum",
        description="行业60日动量",
        economic_interpretation="行业中长期趋势",
        lookback=60,
        data_requirement=["sector_return"],
        formula="sector_return_60d",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MacroFactor(
        name="sector_relative_strength",
        category="sector_rotation",
        sub_category="relative",
        description="行业相对强弱",
        economic_interpretation="相对市场的强弱",
        lookback=20,
        data_requirement=["sector_return", "market_return"],
        formula="sector_return - market_return",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MacroFactor(
        name="sector_rotation_signal",
        category="sector_rotation",
        sub_category="rotation",
        description="行业轮动信号",
        economic_interpretation="从防御到进攻",
        lookback=60,
        data_requirement=["sector_returns"],
        formula="offensive_return - defensive_return",
        ic_direction="conditional",
        update_frequency="weekly",
    ),
    MacroFactor(
        name="value_growth_rotation",
        category="sector_rotation",
        sub_category="style",
        description="价值-成长轮动",
        economic_interpretation="风格切换",
        lookback=20,
        data_requirement=["value_index", "growth_index"],
        formula="value_return - growth_return",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    MacroFactor(
        name="size_rotation",
        category="sector_rotation",
        sub_category="style",
        description="大小盘轮动",
        economic_interpretation="市值风格切换",
        lookback=20,
        data_requirement=["large_cap", "small_cap"],
        formula="large_cap_return - small_cap_return",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    MacroFactor(
        name="market_breadth",
        category="sector_rotation",
        sub_category="breadth",
        description="市场广度",
        economic_interpretation="上涨股票占比",
        lookback=1,
        data_requirement=["advance_count", "decline_count"],
        formula="advance / (advance + decline)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MacroFactor(
        name="sector_momentum_acceleration",
        category="sector_rotation",
        sub_category="acceleration",
        description="行业动量加速",
        economic_interpretation="趋势是否加速",
        lookback=20,
        data_requirement=["sector_return"],
        formula="momentum_20d - momentum_60d",
        ic_direction="positive",
        update_frequency="daily",
    ),
]


def get_macro_factors() -> List[MacroFactor]:
    """获取宏观因子列表"""
    return MACRO_FACTORS


def get_macro_factor_names() -> List[str]:
    """获取所有宏观因子名称"""
    return [f.name for f in MACRO_FACTORS]


def print_macro_factor_summary():
    """打印宏观因子汇总"""
    print("=" * 80)
    print("宏观因子库汇总")
    print("=" * 80)
    print(f"总计: {len(MACRO_FACTORS)}个因子")
    print()
    
    categories = {}
    for f in MACRO_FACTORS:
        if f.category not in categories:
            categories[f.category] = []
        categories[f.category].append(f)
    
    for cat, factors in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"【{cat}】{len(factors)}个")
        for f in factors[:3]:
            print(f"  - {f.name}: {f.description}")
        print()


if __name__ == "__main__":
    print_macro_factor_summary()
