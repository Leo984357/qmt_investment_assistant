"""
分析师预期因子库 - 50个因子

覆盖: 盈利预测、评级、预测修订、一致性预期、目标价等五大类
数据来源: akshare东方财富分析师预期数据
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AnalystFactor:
    """分析师预期因子定义"""
    name: str
    category: str
    sub_category: str
    description: str
    economic_interpretation: str
    lookback: int
    data_requirement: List[str]
    formula: str
    ic_direction: str
    update_frequency: str = "weekly"  # daily, weekly, monthly, quarterly


ANALYST_FACTORS: List[AnalystFactor] = [

    # ========== 一、盈利预测类 (12个) ==========
    
    # EPS预测
    AnalystFactor(
        name="eps_forecast_1y",
        category="earnings_forecast",
        sub_category="eps",
        description="1年期EPS预测",
        economic_interpretation="分析师预测的1年后EPS",
        lookback=0,
        data_requirement=["eps_forecast"],
        formula="分析师预测的EPS_1y",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="eps_forecast_2y",
        category="earnings_forecast",
        sub_category="eps",
        description="2年期EPS预测",
        economic_interpretation="分析师预测的2年后EPS",
        lookback=0,
        data_requirement=["eps_forecast"],
        formula="分析师预测的EPS_2y",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="eps_forecast_growth_1y",
        category="earnings_forecast",
        sub_category="eps",
        description="1年EPS预测增长率",
        economic_interpretation="分析师预测的盈利增速",
        lookback=0,
        data_requirement=["eps_forecast", "eps_actual"],
        formula="(eps_forecast_1y - eps_actual) / eps_actual",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="eps_forecast_growth_2y",
        category="earnings_forecast",
        sub_category="eps",
        description="2年EPS预测CAGR",
        economic_interpretation="分析师预测的2年复合增速",
        lookback=0,
        data_requirement=["eps_forecast"],
        formula="(eps_forecast_2y / eps_forecast_1y)^0.5 - 1",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    
    # 净利润预测
    AnalystFactor(
        name="profit_forecast_1y",
        category="earnings_forecast",
        sub_category="profit",
        description="1年净利润预测",
        economic_interpretation="分析师预测的1年后净利润",
        lookback=0,
        data_requirement=["profit_forecast"],
        formula="分析师预测的净利润_1y",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="profit_forecast_growth",
        category="earnings_forecast",
        sub_category="profit",
        description="净利润预测增长率",
        economic_interpretation="预测盈利增速",
        lookback=0,
        data_requirement=["profit_forecast", "profit_actual"],
        formula="(profit_forecast - profit_actual) / profit_actual",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    
    # 营收预测
    AnalystFactor(
        name="revenue_forecast_1y",
        category="earnings_forecast",
        sub_category="revenue",
        description="1年营收预测",
        economic_interpretation="分析师预测的1年后营收",
        lookback=0,
        data_requirement=["revenue_forecast"],
        formula="分析师预测的营收_1y",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="revenue_forecast_growth",
        category="earnings_forecast",
        sub_category="revenue",
        description="营收预测增长率",
        economic_interpretation="预测营收增速",
        lookback=0,
        data_requirement=["revenue_forecast"],
        formula="(revenue_forecast - revenue_actual) / revenue_actual",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    
    # 超预期
    AnalystFactor(
        name="eps_surprise",
        category="earnings_forecast",
        sub_category="surprise",
        description="EPS超预期程度",
        economic_interpretation="实际EPS/预测EPS - 1",
        lookback=0,
        data_requirement=["eps_actual", "eps_forecast"],
        formula="(eps_actual - eps_forecast) / abs(eps_forecast)",
        ic_direction="positive",
        update_frequency="quarterly",
    ),
    AnalystFactor(
        name="profit_surprise",
        category="earnings_forecast",
        sub_category="surprise",
        description="净利润超预期程度",
        economic_interpretation="实际净利润/预测净利润 - 1",
        lookback=0,
        data_requirement=["profit_actual", "profit_forecast"],
        formula="(profit_actual - profit_forecast) / abs(profit_forecast)",
        ic_direction="positive",
        update_frequency="quarterly",
    ),
    AnalystFactor(
        name="revenue_surprise",
        category="earnings_forecast",
        sub_category="surprise",
        description="营收超预期程度",
        economic_interpretation="实际营收/预测营收 - 1",
        lookback=0,
        data_requirement=["revenue_actual", "revenue_forecast"],
        formula="(revenue_actual - revenue_forecast) / abs(revenue_forecast)",
        ic_direction="positive",
        update_frequency="quarterly",
    ),
    AnalystFactor(
        name="consecutive_surprise",
        category="earnings_forecast",
        sub_category="surprise",
        description="连续超预期次数",
        economic_interpretation="最近N期持续超预期",
        lookback=250,
        data_requirement=["eps_actual", "eps_forecast"],
        formula="count(surprise > 0, last_4q)",
        ic_direction="positive",
        update_frequency="quarterly",
    ),

    # ========== 二、评级类 (10个) ==========
    
    AnalystFactor(
        name="rating_score",
        category="rating",
        sub_category="composite",
        description="综合评级分数",
        economic_interpretation="买入/增持/中性/减持/卖出",
        lookback=0,
        data_requirement=["rating"],
        formula="rating_score(rating)",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="rating_buy_pct",
        category="rating",
        sub_category="composition",
        description="买入评级占比",
        economic_interpretation="买入评级占总评级比例",
        lookback=0,
        data_requirement=["rating_details"],
        formula="count(rating=买入) / count(total)",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="rating_hold_pct",
        category="rating",
        sub_category="composition",
        description="中性评级占比",
        economic_interpretation="中性评级比例",
        lookback=0,
        data_requirement=["rating_details"],
        formula="count(rating=中性) / count(total)",
        ic_direction="negative",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="rating_sell_pct",
        category="rating",
        sub_category="composition",
        description="卖出评级占比",
        economic_interpretation="卖出评级比例",
        lookback=0,
        data_requirement=["rating_details"],
        formula="count(rating=卖出) / count(total)",
        ic_direction="negative",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="rating_upgrades_recent",
        category="rating",
        sub_category="changes",
        description="近期上调次数",
        economic_interpretation="最近30天上调次数",
        lookback=30,
        data_requirement=["rating_changes"],
        formula="count(rating_change=上调, last_30d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    AnalystFactor(
        name="rating_downgrades_recent",
        category="rating",
        sub_category="changes",
        description="近期下调次数",
        economic_interpretation="最近30天下调次数",
        lookback=30,
        data_requirement=["rating_changes"],
        formula="count(rating_change=下调, last_30d)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    AnalystFactor(
        name="rating_net_upgrades",
        category="rating",
        sub_category="changes",
        description="净上调次数",
        economic_interpretation="上调-下调",
        lookback=30,
        data_requirement=["rating_changes"],
        formula="upgrades - downgrades",
        ic_direction="positive",
        update_frequency="daily",
    ),
    AnalystFactor(
        name="rating_coverage",
        category="rating",
        sub_category="coverage",
        description="评级覆盖人数",
        economic_interpretation="多少分析师覆盖",
        lookback=0,
        data_requirement=["analyst_count"],
        formula="count(analysts)",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="rating_change_momentum",
        category="rating",
        sub_category="changes",
        description="评级变化动量",
        economic_interpretation="最近90天vs更早的净变化",
        lookback=90,
        data_requirement=["rating_changes"],
        formula="net_change_last_30d - net_change_60d_before",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="top_agency_rating",
        category="rating",
        sub_category="quality",
        description="头部机构评级",
        economic_interpretation="是否有大机构买入",
        lookback=0,
        data_requirement=["rating", "agency_rank"],
        formula="top_agency == 买入",
        ic_direction="positive",
        update_frequency="weekly",
    ),

    # ========== 三、预测修订类 (10个) ==========
    
    # EPS修订
    AnalystFactor(
        name="eps_revision_1m",
        category="forecast_revision",
        sub_category="eps",
        description="1月EPS修订幅度",
        economic_interpretation="分析师调整EPS预测",
        lookback=30,
        data_requirement=["eps_forecast"],
        formula="(eps_forecast_now - eps_forecast_1m_ago) / abs(eps_forecast_1m_ago)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    AnalystFactor(
        name="eps_revision_3m",
        category="forecast_revision",
        sub_category="eps",
        description="3月EPS修订幅度",
        economic_interpretation="分析师调整EPS预测",
        lookback=90,
        data_requirement=["eps_forecast"],
        formula="(eps_forecast_now - eps_forecast_3m_ago) / abs(eps_forecast_3m_ago)",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="eps_revision_direction",
        category="forecast_revision",
        sub_category="eps",
        description="EPS修订方向",
        economic_interpretation="上调vs下调次数",
        lookback=30,
        data_requirement=["eps_revision_history"],
        formula="count(up) - count(down)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    AnalystFactor(
        name="eps_revision_momentum",
        category="forecast_revision",
        sub_category="eps",
        description="EPS修订动量",
        economic_interpretation="修订是否加速",
        lookback=90,
        data_requirement=["eps_revision_history"],
        formula="(revision_1m - revision_3m) / abs(revision_3m)",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    
    # 净利润修订
    AnalystFactor(
        name="profit_revision_1m",
        category="forecast_revision",
        sub_category="profit",
        description="1月净利润修订幅度",
        economic_interpretation="分析师调整净利润预测",
        lookback=30,
        data_requirement=["profit_forecast"],
        formula="(profit_forecast_now - profit_forecast_1m_ago) / abs(profit_forecast_1m_ago)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    AnalystFactor(
        name="profit_revision_3m",
        category="forecast_revision",
        sub_category="profit",
        description="3月净利润修订幅度",
        economic_interpretation="分析师调整净利润预测",
        lookback=90,
        data_requirement=["profit_forecast"],
        formula="(profit_forecast_now - profit_forecast_3m_ago) / abs(profit_forecast_3m_ago)",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    
    # 营收修订
    AnalystFactor(
        name="revenue_revision_1m",
        category="forecast_revision",
        sub_category="revenue",
        description="1月营收修订幅度",
        economic_interpretation="分析师调整营收预测",
        lookback=30,
        data_requirement=["revenue_forecast"],
        formula="(revenue_forecast_now - revenue_forecast_1m_ago) / abs(revenue_forecast_1m_ago)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    
    # 目标价相关
    AnalystFactor(
        name="target_price_upside",
        category="forecast_revision",
        sub_category="target_price",
        description="目标价上涨空间",
        economic_interpretation="目标价/当前价 - 1",
        lookback=0,
        data_requirement=["target_price", "current_price"],
        formula="target_price / current_price - 1",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="target_price_revision_1m",
        category="forecast_revision",
        sub_category="target_price",
        description="1月目标价修订",
        economic_interpretation="目标价变化",
        lookback=30,
        data_requirement=["target_price"],
        formula="target_price_now / target_price_1m_ago - 1",
        ic_direction="positive",
        update_frequency="daily",
    ),
    AnalystFactor(
        name="target_price_convergence",
        category="forecast_revision",
        sub_category="target_price",
        description="目标价收敛度",
        economic_interpretation="分析师分歧是否减少",
        lookback=0,
        data_requirement=["target_price_std", "target_price_mean"],
        formula="1 - target_price_std / target_price_mean",
        ic_direction="positive",
        update_frequency="weekly",
    ),

    # ========== 四、一致性预期类 (10个) ==========
    
    # 共识度
    AnalystFactor(
        name="consensus_eps_std",
        category="consensus",
        sub_category="dispersion",
        description="EPS预测标准差",
        economic_interpretation="分析师分歧程度",
        lookback=0,
        data_requirement=["eps_forecast_all"],
        formula="std(eps_forecasts)",
        ic_direction="negative",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="consensus_eps_cv",
        category="consensus",
        sub_category="dispersion",
        description="EPS预测变异系数",
        economic_interpretation="相对分歧程度",
        lookback=0,
        data_requirement=["eps_forecast_all"],
        formula="std(eps) / mean(eps)",
        ic_direction="negative",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="consensus_strong",
        category="consensus",
        sub_category="dispersion",
        description="共识强度",
        economic_interpretation="1 - cv",
        lookback=0,
        data_requirement=["eps_forecast_all"],
        formula="1 - cv(eps)",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    
    # 预测期限结构
    AnalystFactor(
        name="forecast_curvature",
        category="consensus",
        sub_category="term_structure",
        description="预测曲线曲率",
        economic_interpretation="预测增长是否加速",
        lookback=0,
        data_requirement=["eps_forecast_series"],
        formula="(eps_2y - 2*eps_1y + eps_0) / eps_0",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="forecast_2y_vs_1y",
        category="consensus",
        sub_category="term_structure",
        description="2年vs1年预测比率",
        economic_interpretation="长期vs短期增长",
        lookback=0,
        data_requirement=["eps_forecast"],
        formula="eps_forecast_2y / eps_forecast_1y",
        ic_direction="conditional",
        update_frequency="weekly",
    ),
    
    # 高覆盖vs低覆盖
    AnalystFactor(
        name="high_cover_eps",
        category="consensus",
        sub_category="coverage_effect",
        description="高覆盖股票EPS",
        economic_interpretation="关注度高=流动性好",
        lookback=0,
        data_requirement=["eps_forecast", "analyst_count"],
        formula="eps_forecast if analyst_count >= 10",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="low_cover_eps",
        category="consensus",
        sub_category="coverage_effect",
        description="低覆盖股票EPS",
        economic_interpretation="关注度低=信息差大",
        lookback=0,
        data_requirement=["eps_forecast", "analyst_count"],
        formula="eps_forecast if analyst_count < 5",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    
    # 时间加权
    AnalystFactor(
        name="recent_eps_weighted",
        category="consensus",
        sub_category="time_weighted",
        description="近期预测加权EPS",
        economic_interpretation="近期预测权重更高",
        lookback=90,
        data_requirement=["eps_forecast_history"],
        formula="sum(eps * recency_weight) / sum(recency_weight)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    AnalystFactor(
        name="model_eps_weighted",
        category="consensus",
        sub_category="time_weighted",
        description="机构规模加权EPS",
        economic_interpretation="大机构预测权重更高",
        lookback=0,
        data_requirement=["eps_forecast", "institution_size"],
        formula="sum(eps * institution_size) / sum(institution_size)",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="accuracy_weighted_eps",
        category="consensus",
        sub_category="accuracy_weighted",
        description="准确度加权EPS",
        economic_interpretation="历史准确度加权",
        lookback=750,
        data_requirement=["eps_forecast_history", "eps_actual_history"],
        formula="sum(eps * accuracy_score) / sum(accuracy_score)",
        ic_direction="positive",
        update_frequency="weekly",
    ),

    # ========== 五、目标价与估值类 (8个) ==========
    
    AnalystFactor(
        name="target_price_to_52w_high",
        category="valuation_target",
        sub_category="target_price",
        description="目标价相对52周高点",
        economic_interpretation="相对历史高点的目标价",
        lookback=250,
        data_requirement=["target_price", "price_52w_high"],
        formula="target_price / price_52w_high",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="target_price_to_52w_low",
        category="valuation_target",
        sub_category="target_price",
        description="目标价相对52周低点",
        economic_interpretation="相对历史低点的目标价",
        lookback=250,
        data_requirement=["target_price", "price_52w_low"],
        formula="target_price / price_52w_low",
        ic_direction="conditional",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="pe_forward_to_pe_hist",
        category="valuation_target",
        sub_category="pe_comparison",
        description="前瞻PE相对历史PE",
        economic_interpretation="估值相对历史",
        lookback=1250,
        data_requirement=["pe_forward", "pe_hist"],
        formula="pe_forward / mean(pe_hist_5y)",
        ic_direction="negative",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="peg_ratio_adjusted",
        category="valuation_target",
        sub_category="peg",
        description="调整后PEG",
        economic_interpretation="考虑分歧的PEG",
        lookback=0,
        data_requirement=["pe", "growth", "cv"],
        formula="pe / (growth * (1 - cv))",
        ic_direction="negative",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="target_irr",
        category="valuation_target",
        sub_category="irr",
        description="目标IRR",
        economic_interpretation="隐含年化收益",
        lookback=0,
        data_requirement=["target_price", "current_price", "hold_period"],
        formula="(target_price / current_price)^(365/hold_days) - 1",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="upside_to_consensus",
        category="valuation_target",
        sub_category="upside",
        description="相对共识上涨空间",
        economic_interpretation="相对一致预期",
        lookback=0,
        data_requirement=["target_price", "consensus_target"],
        formula="(target_price - consensus_target) / consensus_target",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="price_target_ratio",
        category="valuation_target",
        sub_category="ratio",
        description="目标价离散度",
        economic_interpretation="分析师目标价差异",
        lookback=0,
        data_requirement=["target_prices"],
        formula="(max(target) - min(target)) / mean(target)",
        ic_direction="negative",
        update_frequency="weekly",
    ),
    AnalystFactor(
        name="analyst_price_divergence",
        category="valuation_target",
        sub_category="divergence",
        description="分析师价格分歧度",
        economic_interpretation="目标价vs当前价分歧",
        lookback=0,
        data_requirement=["target_price", "current_price"],
        formula="abs(target_price - current_price) / current_price",
        ic_direction="conditional",
        update_frequency="weekly",
    ),
]


def get_analyst_factors() -> List[AnalystFactor]:
    """获取分析师因子列表"""
    return ANALYST_FACTORS


def get_analyst_factors_by_category(category: str) -> List[AnalystFactor]:
    """按类别获取分析师因子"""
    return [f for f in ANALYST_FACTORS if f.category == category]


def get_analyst_factor_names() -> List[str]:
    """获取所有分析师因子名称"""
    return [f.name for f in ANALYST_FACTORS]


def print_analyst_factor_summary():
    """打印分析师因子汇总"""
    print("=" * 80)
    print("分析师预期因子库汇总")
    print("=" * 80)
    print(f"总计: {len(ANALYST_FACTORS)}个因子")
    print()
    
    categories = {}
    for f in ANALYST_FACTORS:
        if f.category not in categories:
            categories[f.category] = []
        categories[f.category].append(f)
    
    for cat, factors in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"【{cat}】{len(factors)}个")
        for f in factors[:3]:
            print(f"  - {f.name}: {f.description}")
        if len(factors) > 3:
            print(f"  ... 还有{len(factors)-3}个")
        print()


if __name__ == "__main__":
    print_analyst_factor_summary()
