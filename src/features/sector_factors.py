"""
行业轮动因子库 - 行业动量、相对强弱、资金流等

来源:
1. 行业动量
2. 行业相对强弱
3. 产业链动量
4. 主题/概念动量
5. 资金流因子
6. 宏观敏感度
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectorFactorSpec:
    name: str
    category: str
    description: str
    economic_interpretation: str
    lookback: int
    data_requirement: list[str]


SECTOR_FACTORS = [
    # ===== 行业动量因子 =====
    SectorFactorSpec(
        name="sector_mom_1m",
        category="momentum",
        description="行业1月动量",
        economic_interpretation="行业短期趋势延续",
        lookback=21,
        data_requirement=["sector_index_return"],
    ),
    SectorFactorSpec(
        name="sector_mom_3m",
        category="momentum",
        description="行业3月动量",
        economic_interpretation="行业中短期趋势",
        lookback=63,
        data_requirement=["sector_index_return"],
    ),
    SectorFactorSpec(
        name="sector_mom_6m",
        category="momentum",
        description="行业6月动量",
        economic_interpretation="行业中长期趋势",
        lookback=126,
        data_requirement=["sector_index_return"],
    ),
    SectorFactorSpec(
        name="sector_mom_12m",
        category="momentum",
        description="行业12月动量",
        economic_interpretation="行业长期趋势",
        lookback=252,
        data_requirement=["sector_index_return"],
    ),
    SectorFactorSpec(
        name="sector_mom_12_1",
        category="momentum",
        description="行业12-1月动量 (跳过最近1月)",
        economic_interpretation="中期趋势(避免短期反转)",
        lookback=273,
        data_requirement=["sector_index_return"],
    ),

    # ===== 行业相对强弱 =====
    SectorFactorSpec(
        name="sector_rs_1m",
        category="relative_strength",
        description="行业1月相对强弱",
        economic_interpretation="相对强弱(行业vs大盘)",
        lookback=21,
        data_requirement=["sector_return", "market_return"],
    ),
    SectorFactorSpec(
        name="sector_rs_3m",
        category="relative_strength",
        description="行业3月相对强弱",
        economic_interpretation="中期相对强弱",
        lookback=63,
        data_requirement=["sector_return", "market_return"],
    ),
    SectorFactorSpec(
        name="sector_rs_6m",
        category="relative_strength",
        description="行业6月相对强弱",
        economic_interpretation="中短期相对强弱",
        lookback=126,
        data_requirement=["sector_return", "market_return"],
    ),
    SectorFactorSpec(
        name="sector_rs_12m",
        category="relative_strength",
        description="行业12月相对强弱",
        economic_interpretation="长期相对强弱",
        lookback=252,
        data_requirement=["sector_return", "market_return"],
    ),

    # ===== 行业趋势强度 =====
    SectorFactorSpec(
        name="sector_trend_strength",
        category="momentum",
        description="行业趋势强度",
        economic_interpretation="趋势持续性",
        lookback=60,
        data_requirement=["daily_sector_return"],
    ),
    SectorFactorSpec(
        name="sector_momentum_acceleration",
        category="momentum",
        description="行业动量加速",
        economic_interpretation="动量正在增强",
        lookback=60,
        data_requirement=["daily_sector_return"],
    ),
    SectorFactorSpec(
        name="sector_momentum_reversal",
        category="reversal",
        description="行业动量反转",
        economic_interpretation="前期弱势行业可能反弹",
        lookback=252,
        data_requirement=["sector_return"],
    ),

    # ===== 行业轮动信号 =====
    SectorFactorSpec(
        name="sector_leading_lagging",
        category="rotation",
        description="行业领先/落后指标",
        economic_interpretation="领先行业预示经济周期",
        lookback=21,
        data_requirement=["sector_return"],
    ),
    SectorFactorSpec(
        name="sector_turnover_rate",
        category="rotation",
        description="行业换手率变化",
        economic_interpretation="资金轮动信号",
        lookback=21,
        data_requirement=["sector_volume", "sector_float_shares"],
    ),
    SectorFactorSpec(
        name="sector_new_high_ratio",
        category="rotation",
        description="行业创新高比例",
        economic_interpretation="行业强势股票占比",
        lookback=60,
        data_requirement=["stock_high", "sector_membership"],
    ),
    SectorFactorSpec(
        name="sector_new_low_ratio",
        category="rotation",
        description="行业创新低比例",
        economic_interpretation="行业弱势股票占比",
        lookback=60,
        data_requirement=["stock_low", "sector_membership"],
    ),

    # ===== 产业链动量 =====
    SectorFactorSpec(
        name="industry_chain_mom_upstream",
        category="industry_chain",
        description="上游行业动量",
        economic_interpretation="原材料行业领先",
        lookback=63,
        data_requirement=["upstream_sector_return"],
    ),
    SectorFactorSpec(
        name="industry_chain_mom_midstream",
        category="industry_chain",
        description="中游行业动量",
        economic_interpretation="制造行业",
        lookback=63,
        data_requirement=["midstream_sector_return"],
    ),
    SectorFactorSpec(
        name="industry_chain_mom_downstream",
        category="industry_chain",
        description="下游行业动量",
        economic_interpretation="消费行业",
        lookback=63,
        data_requirement=["downstream_sector_return"],
    ),
    SectorFactorSpec(
        name="industry_chain_spillover",
        category="industry_chain",
        description="产业链溢出效应",
        economic_interpretation="上游传导到下游",
        lookback=126,
        data_requirement=["upstream_return", "downstream_return"],
    ),

    # ===== 宏观敏感度 =====
    SectorFactorSpec(
        name="macro_beta_gdp",
        category="macro_sensitivity",
        description="GDP增长敏感度",
        economic_interpretation="经济周期敏感行业",
        lookback=252,
        data_requirement=["sector_return", "gdp_growth"],
    ),
    SectorFactorSpec(
        name="macro_beta_inflation",
        category="macro_sensitivity",
        description="通胀敏感度",
        economic_interpretation="通胀敏感行业",
        lookback=252,
        data_requirement=["sector_return", "inflation_rate"],
    ),
    SectorFactorSpec(
        name="macro_beta_interest",
        category="macro_sensitivity",
        description="利率敏感度",
        economic_interpretation="利率敏感行业(金融/地产)",
        lookback=252,
        data_requirement=["sector_return", "interest_rate"],
    ),
    SectorFactorSpec(
        name="macro_beta_credit",
        category="macro_sensitivity",
        description="信用敏感度",
        economic_interpretation="信用风险敏感行业",
        lookback=252,
        data_requirement=["sector_return", "credit_spread"],
    ),
    SectorFactorSpec(
        name="macro_beta_sentiment",
        category="macro_sensitivity",
        description="情绪敏感度",
        economic_interpretation="情绪敏感行业(消费/成长)",
        lookback=252,
        data_requirement=["sector_return", "sentiment_index"],
    ),

    # ===== 资金流因子 =====
    SectorFactorSpec(
        name="sector_flow_1w",
        category="money_flow",
        description="行业1周资金流",
        economic_interpretation="短期资金流入",
        lookback=5,
        data_requirement=["sector_trade_volume", "price_change"],
    ),
    SectorFactorSpec(
        name="sector_flow_1m",
        category="money_flow",
        description="行业1月资金流",
        economic_interpretation="月度资金净流入",
        lookback=21,
        data_requirement=["sector_trade_volume", "price_change"],
    ),
    SectorFactorSpec(
        name="sector_flow_3m",
        category="money_flow",
        description="行业3月资金流",
        economic_interpretation="季度资金趋势",
        lookback=63,
        data_requirement=["sector_trade_volume", "price_change"],
    ),
    SectorFactorSpec(
        name="sector_inflow_acceleration",
        category="money_flow",
        description="资金流入加速",
        economic_interpretation="流入速度加快",
        lookback=63,
        data_requirement=["sector_flow"],
    ),
    SectorFactorSpec(
        name="sector_margin_balance",
        category="money_flow",
        description="行业融资余额变化",
        economic_interpretation="杠杆资金偏好",
        lookback=21,
        data_requirement=["sector_margin_balance"],
    ),

    # ===== 行业估值 =====
    SectorFactorSpec(
        name="sector_pe",
        category="valuation",
        description="行业PE相对估值",
        economic_interpretation="行业估值吸引力",
        lookback=60,
        data_requirement=["sector_pe", "market_pe"],
    ),
    SectorFactorSpec(
        name="sector_pb",
        category="valuation",
        description="行业PB相对估值",
        economic_interpretation="行业账面价值吸引力",
        lookback=60,
        data_requirement=["sector_pb", "market_pb"],
    ),
    SectorFactorSpec(
        name="sector_pe_historical",
        category="valuation",
        description="行业PE历史分位",
        economic_interpretation="相对历史估值",
        lookback=252,
        data_requirement=["sector_pe"],
    ),
    SectorFactorSpec(
        name="sector_peg",
        category="valuation",
        description="行业PEG比率",
        economic_interpretation="成长性估值",
        lookback=60,
        data_requirement=["sector_pe", "sector_earning_growth"],
    ),

    # ===== 行业拥挤度 =====
    SectorFactorSpec(
        name="sector_crowding",
        category="crowding",
        description="行业拥挤度",
        economic_interpretation="资金过度集中风险",
        lookback=21,
        data_requirement=["sector_position", "sector_volume"],
    ),
    SectorFactorSpec(
        name="sector_short_interest",
        category="crowding",
        description="行业做空比例",
        economic_interpretation="看空情绪",
        lookback=60,
        data_requirement=["short_interest"],
    ),
    SectorFactorSpec(
        name="sector_beta_realized",
        category="crowding",
        description="行业实现波动率",
        economic_interpretation="波动异常信号",
        lookback=21,
        data_requirement=["daily_sector_return"],
    ),

    # ===== 分析师行业偏好 =====
    SectorFactorSpec(
        name="sector_analyst_rating",
        category="analyst",
        description="行业评级变化",
        economic_interpretation="分析师行业偏好",
        lookback=60,
        data_requirement=["analyst_rating_by_sector"],
    ),
    SectorFactorSpec(
        name="sector_forecast_revenue_growth",
        category="analyst",
        description="行业预测营收增长",
        economic_interpretation="行业成长预期",
        lookback=60,
        data_requirement=["consensus_revenue_growth_by_sector"],
    ),
    SectorFactorSpec(
        name="sector_forecast_earnings_revision",
        category="analyst",
        description="行业盈利预测调整",
        economic_interpretation="盈利预期变化方向",
        lookback=60,
        data_requirement=["earnings_revision_by_sector"],
    ),

    # ===== A股特有 =====
    SectorFactorSpec(
        name="sector_limit_up_count",
        category="china_specific",
        description="行业涨停家数",
        economic_interpretation="市场情绪热度",
        lookback=5,
        data_requirement=["limit_up_stocks_by_sector"],
    ),
    SectorFactorSpec(
        name="sector_limit_down_count",
        category="china_specific",
        description="行业跌停家数",
        economic_interpretation="市场恐慌程度",
        lookback=5,
        data_requirement=["limit_down_stocks_by_sector"],
    ),
    SectorFactorSpec(
        name="sector_state_owned_ratio",
        category="china_specific",
        description="行业国企占比",
        economic_interpretation="政策相关性",
        lookback=1,
        data_requirement=["stock_ownership_type"],
    ),
    SectorFactorSpec(
        name="sector_foreign_holding",
        category="china_specific",
        description="行业外资持股比例",
        economic_interpretation="外资偏好",
        lookback=60,
        data_requirement=["foreign_holding_by_sector"],
    ),
    SectorFactorSpec(
        name="sector_hsgt_flow",
        category="china_specific",
        description="沪股通/深股通资金流",
        economic_interpretation="北向资金偏好",
        lookback=5,
        data_requirement=["hsgt_flow_by_sector"],
    ),
    SectorFactorSpec(
        name="sector_insider_trading",
        category="china_specific",
        description="行业内部人交易",
        economic_interpretation="内部人信心",
        lookback=60,
        data_requirement=["insider_buysell_by_sector"],
    ),
]


def get_sector_factor_names() -> list[str]:
    """获取所有行业因子名称"""
    return [f.name for f in SECTOR_FACTORS]


def get_sector_factors_by_category(category: str) -> list[SectorFactorSpec]:
    """按类别获取行业因子"""
    return [f for f in SECTOR_FACTORS if f.category == category]


def print_sector_factor_summary():
    """打印行业因子库汇总"""
    print("=" * 100)
    print("行业轮动因子库汇总")
    print("=" * 100)

    categories = {}
    for f in SECTOR_FACTORS:
        if f.category not in categories:
            categories[f.category] = []
        categories[f.category].append(f)

    for cat, factors in sorted(categories.items()):
        print(f"\n【{cat.upper()}】{len(factors)}个因子")
        for f in factors:
            print(f"  {f.name:<40} {f.description}")

    print(f"\n总计: {len(SECTOR_FACTORS)}个行业轮动因子")


if __name__ == "__main__":
    print_sector_factor_summary()
