"""
学术因子库 - Fama-French, Carhart, q-factor, Novy-Marx等

来源:
1. Fama-French 3/5因子 (1993, 2015)
2. Carhart 4因子 (1997)
3. Novy-Marx盈利能力/投资因子 (2013)
4. Hou-Xue-Zhang q-factor模型 (2014)
5. Stambaugh-Yuan factors (2017)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class AcademicFeatureSpec:
    name: str
    category: str
    description: str
    economic_interpretation: str
    source_paper: str
    lookback: int
    computation: str


ACADEMIC_FACTORS = [
    # ===== Fama-French 3因子 =====
    AcademicFeatureSpec(
        name="mkt_rf",
        category="market",
        description="市场超额收益 (MKT)",
        economic_interpretation="市场系统性风险溢价",
        source_paper="Fama-French 1993",
        lookback=1,
        computation="(market_return - risk_free_rate)",
    ),
    AcademicFeatureSpec(
        name="smb",
        category="size",
        description="市值因子 (Small Minus Big)",
        economic_interpretation="小市值股票溢价",
        source_paper="Fama-French 1993",
        lookback=1,
        computation="long small caps, short large caps",
    ),
    AcademicFeatureSpec(
        name="hml",
        category="value",
        description="价值因子 (High Minus Low)",
        economic_interpretation="账面市值比溢价",
        source_paper="Fama-French 1993",
        lookback=1,
        computation="long high BM stocks, short low BM stocks",
    ),
    
    # ===== Fama-French 5因子 =====
    AcademicFeatureSpec(
        name="rmw",
        category="profitability",
        description="盈利能力因子 (Robust Minus Weak)",
        economic_interpretation="高盈利企业溢价",
        source_paper="Fama-French 2015",
        lookback=1,
        computation="long high operating profitability, short low OP",
    ),
    AcademicFeatureSpec(
        name="cma",
        category="investment",
        description="投资因子 (Conservative Minus Aggressive)",
        economic_interpretation="保守投资企业溢价",
        source_paper="Fama-French 2015",
        lookback=1,
        computation="long low asset growth, short high asset growth",
    ),
    
    # ===== Carhart 4因子 =====
    AcademicFeatureSpec(
        name="mom_12_1",
        category="momentum",
        description="动量因子 (12-1月收益率)",
        economic_interpretation="趋势延续效应",
        source_paper="Carhart 1997",
        lookback=13,
        computation="return_t-12 to return_t-2, skip most recent month",
    ),
    
    # ===== Novy-Marx因子 =====
    AcademicFeatureSpec(
        name="gross_profitability",
        category="profitability",
        description="毛利率 (Gross Profit / Assets)",
        economic_interpretation="盈利质量指标",
        source_paper="Novy-Marx 2013",
        lookback=120,
        computation="revenue - cost_of_goods_sold / total_assets",
    ),
    AcademicFeatureSpec(
        name="operating_profitability",
        category="profitability",
        description="营业利润率 (Operating Profit / Book Equity)",
        economic_interpretation="真实盈利能力",
        source_paper="Novy-Marx 2013",
        lookback=120,
        computation="(revenue - cogs - sga - interest) / book_equity",
    ),
    
    # ===== q-factor模型 (Hou-Xue-Zhang 2014) =====
    AcademicFeatureSpec(
        name="q_me",
        category="size",
        description="q-市值因子",
        economic_interpretation="企业规模溢价",
        source_paper="Hou-Xue-Zhang 2014",
        lookback=1,
        computation="market_equity as size measure",
    ),
    AcademicFeatureSpec(
        name="q_ia",
        category="investment",
        description="q-投资因子 (Investment / Assets)",
        economic_interpretation="投资水平与预期收益负相关",
        source_paper="Hou-Xue-Zhang 2014",
        lookback=120,
        computation="change in total assets / total assets (lagged)",
    ),
    AcademicFeatureSpec(
        name="q_roe",
        category="profitability",
        description="q-ROE因子 (Return on Equity)",
        economic_interpretation="高ROE企业溢价",
        source_paper="Hou-Xue-Zhang 2014",
        lookback=120,
        computation="net_income / book_equity",
    ),
    AcademicFeatureSpec(
        name="q_delta_roe",
        category="profitability",
        description="ROE变化因子",
        economic_interpretation="ROE改善企业溢价",
        source_paper="Hou-Xue-Zhang 2014",
        lookback=240,
        computation="change in ROE",
    ),
    
    # ===== Stambaugh-Yuan因子 (2017) =====
    AcademicFeatureSpec(
        name="mgmt",
        category="management",
        description="管理效率因子",
        economic_interpretation="管理良好的企业溢价",
        source_paper="Stambaugh-Yuan 2017",
        lookback=240,
        computation="基于费用率/员工数等管理指标",
    ),
    AcademicFeatureSpec(
        name="perf",
        category="performance",
        description="业绩效率因子",
        economic_interpretation="业绩好的企业溢价",
        source_paper="Stambaugh-Yuan 2017",
        lookback=240,
        computation="基于盈利/投资效率",
    ),
    
    # ===== Baugess-Danilov因子 =====
    AcademicFeatureSpec(
        name="bm_slope",
        category="value",
        description="BM斜率因子",
        economic_interpretation="成长性价值因子",
        source_paper="Baugess-Danilov 2022",
        lookback=120,
        computation="slope of book-to-market over time",
    ),
    
    # ===== 短期反转 =====
    AcademicFeatureSpec(
        name="short_term_reversal",
        category="reversal",
        description="短期反转 (周度)",
        economic_interpretation="微观结构噪声回归",
        source_paper="Lo-MacKinlay 1990",
        lookback=5,
        computation="-return_t-1 to t-5",
    ),
    AcademicFeatureSpec(
        name="intraday_reversal",
        category="reversal",
        description="日内反转",
        economic_interpretation="日内价格回归",
        source_paper="Heston et al. 2009",
        lookback=1,
        computation="close - VWAP deviation",
    ),
    
    # ===== 长期反转 =====
    AcademicFeatureSpec(
        name="long_term_reversal",
        category="reversal",
        description="长期反转 (3-5年)",
        economic_interpretation="V形反转效应",
        source_paper="DeBondt-Thaler 1985",
        lookback=750,
        computation="-return_t-36 to t-60",
    ),
    
    # ===== 彩票需求 =====
    AcademicFeatureSpec(
        name="max_daily_return",
        category="behavioral",
        description="极端收益因子 (MAX)",
        economic_interpretation="投资者彩票偏好导致低MAX高收益",
        source_paper="Bali et al. 2011",
        lookback=21,
        computation="max daily return in past month",
    ),
    AcademicFeatureSpec(
        name="idiosyncratic_vol",
        category="behavioral",
        description="特质波动率因子 (IVOL)",
        economic_interpretation="低IVOL股票高收益",
        source_paper="Ang et al. 2006",
        lookback=60,
        computation="residual volatility from market model",
    ),
    AcademicFeatureSpec(
        name="max5vol",
        category="behavioral",
        description="平均极端波动率",
        economic_interpretation="5天平均极端收益波动",
        source_paper="Bali et al. 2017",
        lookback=21,
        computation="average of top 5 daily returns",
    ),
    
    # ===== 盈利动量 =====
    AcademicFeatureSpec(
        name="earnings_momentum",
        category="earnings",
        description="盈利动量",
        economic_interpretation="盈利持续性",
        source_paper="Li et al. 2011",
        lookback=60,
        computation="quarterly earnings change",
    ),
    AcademicFeatureSpec(
        name="accruals",
        category="quality",
        description="应计项因子",
        economic_interpretation="低应计项高收益",
        source_paper="Sloan 1996",
        lookback=120,
        computation="(net income - operating cash flow) / total assets",
    ),
    
    # ===== 分析师因子 =====
    AcademicFeatureSpec(
        name="analyst_coverage",
        category="information",
        description="分析师覆盖度",
        economic_interpretation="信息扩散速度",
        source_paper="Hong et al. 2000",
        lookback=60,
        computation="number of analysts covering stock",
    ),
    AcademicFeatureSpec(
        name="forecast_dispersion",
        category="information",
        description="预测分歧度",
        economic_interpretation="不确定性",
        source_paper="Diether et al. 2002",
        lookback=60,
        computation="standard deviation of earnings forecasts",
    ),
    AcademicFeatureSpec(
        name="forecast_breadth",
        category="information",
        description="预测广度变化",
        economic_interpretation="信息趋势",
        source_paper="Fang-Suemmer 2015",
        lookback=60,
        computation="change in number of positive minus negative forecasts",
    ),
    
    # ===== 机构持仓 =====
    AcademicFeatureSpec(
        name="institutional_ownership",
        category="ownership",
        description="机构持仓比例",
        economic_interpretation="专业投资者认可",
        source_paper="Gompers-Metrick 2008",
        lookback=60,
        computation="institutional ownership percentage",
    ),
    AcademicFeatureSpec(
        name="inst_ownership_change",
        category="ownership",
        description="机构持仓变化",
        economic_interpretation="聪明钱信号",
        source_paper="Edmans et al. 2011",
        lookback=60,
        computation="change in institutional ownership",
    ),
    
    # ===== 宏观风险 =====
    AcademicFeatureSpec(
        name="beta",
        category="risk",
        description="市场Beta",
        economic_interpretation="系统性风险暴露",
        source_paper="Fama-MacBeth 1973",
        lookback=252,
        computation="cov(stock_return, market_return) / var(market_return)",
    ),
    AcademicFeatureSpec(
        name="idyncorr_mkt",
        category="risk",
        description="特质与市场相关性",
        economic_interpretation="特质信息与市场关联",
        source_paper="Kempf et al. 2017",
        lookback=252,
        computation="rolling correlation of residuals with market",
    ),
    
    # ===== 交易摩擦 =====
    AcademicFeatureSpec(
        name="amihud_illiquidity",
        category="liquidity",
        description="Amihud非流动性",
        economic_interpretation="流动性风险溢价",
        source_paper="Amihud 2002",
        lookback=21,
        computation="abs(return) / volume",
    ),
    AcademicFeatureSpec(
        name="bid_ask_spread",
        category="liquidity",
        description="买卖价差",
        economic_interpretation="交易成本代理",
        source_paper="Amihud-Mendelson 1986",
        lookback=21,
        computation="(ask - bid) / (ask + bid)",
    ),
    AcademicFeatureSpec(
        name="zero_trade_days",
        category="liquidity",
        description="零交易天数占比",
        economic_interpretation="交易活跃度",
        source_paper="Liu 2006",
        lookback=21,
        computation="percentage of zero trading days",
    ),
    
    # ===== 财务困境 =====
    AcademicFeatureSpec(
        name="bankruptcy_prob",
        category="distress",
        description="破产概率",
        economic_interpretation="违约风险溢价",
        source_paper="Campbell et al. 2008",
        lookback=120,
        computation="Merton distance to default",
    ),
    AcademicFeatureSpec(
        name="Altman_zscore",
        category="distress",
        description="Altman Z-Score",
        economic_interpretation="财务困境风险",
        source_paper="Altman 1968",
        lookback=120,
        computation="1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*ME/TL + 1.0*S/TA",
    ),
    
    # ===== 竞争因子 =====
    AcademicFeatureSpec(
        name="intangibles",
        category="value",
        description="无形资产比例",
        economic_interpretation="知识资本价值",
        source_paper="Petersen et al. 2022",
        lookback=120,
        computation="(total_assets - tangible_assets) / total_assets",
    ),
    AcademicFeatureSpec(
        name="asset_growth",
        category="investment",
        description="资产增长率",
        economic_interpretation="过度投资风险",
        source_paper="Cooper et al. 2008",
        lookback=120,
        computation="change in total assets / total assets (lagged)",
    ),
    AcademicFeatureSpec(
        name="capex_intensity",
        category="investment",
        description="资本支出强度",
        economic_interpretation="投资效率",
        source_paper="Biddle et al. 2009",
        lookback=120,
        computation="capex / total_assets",
    ),
    
    # ===== 盈利质量 =====
    AcademicFeatureSpec(
        name="roa",
        category="profitability",
        description="资产回报率",
        economic_interpretation="资产使用效率",
        source_paper="Balakrishnan et al. 2004",
        lookback=120,
        computation="net_income / total_assets",
    ),
    AcademicFeatureSpec(
        name="roe",
        category="profitability",
        description="股东权益回报率",
        economic_interpretation="股东价值创造",
        source_paper="Bartholdy-Peare 2005",
        lookback=120,
        computation="net_income / shareholders_equity",
    ),
    AcademicFeatureSpec(
        name="gross_margin",
        category="profitability",
        description="毛利率",
        economic_interpretation="定价权",
        source_paper="Novy-Marx 2013",
        lookback=120,
        computation="(revenue - cogs) / revenue",
    ),
    AcademicFeatureSpec(
        name="net_margin",
        category="profitability",
        description="净利率",
        economic_interpretation="最终盈利能力",
        source_paper="Sloan 1996",
        lookback=120,
        computation="net_income / revenue",
    ),
    AcademicFeatureSpec(
        name="asset_turnover",
        category="efficiency",
        description="资产周转率",
        economic_interpretation="运营效率",
        source_paper="Novy-Marx 2010",
        lookback=120,
        computation="revenue / total_assets",
    ),
    AcademicFeatureSpec(
        name="operating_leverage",
        category="risk",
        description="经营杠杆",
        economic_interpretation="固定成本比例",
        source_paper="Novy-Marx 2012",
        lookback=240,
        computation="change in gross profit / change in revenue",
    ),
    AcademicFeatureSpec(
        name="financial_leverage",
        category="risk",
        description="财务杠杆",
        economic_interpretation="债务风险",
        source_paper="Bhandari 1988",
        lookback=120,
        computation="debt / equity",
    ),
    
    # ===== 成长因子 =====
    AcademicFeatureSpec(
        name="revenue_growth",
        category="growth",
        description="营收增长率",
        economic_interpretation="成长性",
        source_paper="Lakonishok et al. 1994",
        lookback=120,
        computation="change in revenue / revenue (lagged)",
    ),
    AcademicFeatureSpec(
        name="earnings_growth",
        category="growth",
        description="盈利增长率",
        economic_interpretation="盈利成长性",
        source_paper="La Porta et al. 1997",
        lookback=120,
        computation="change in EPS / EPS (lagged)",
    ),
    AcademicFeatureSpec(
        name="book_equity_growth",
        category="growth",
        description="净资产增长率",
        economic_interpretation="内生成长",
        source_paper="Fairfield et al. 2003",
        lookback=120,
        computation="change in book_equity / book_equity (lagged)",
    ),
    
    # ===== 估值因子 =====
    AcademicFeatureSpec(
        name="pe_ratio",
        category="value",
        description="市盈率倒数 (E/P)",
        economic_interpretation="价值因子",
        source_paper="Basu 1977",
        lookback=60,
        computation="earnings / price",
    ),
    AcademicFeatureSpec(
        name="pb_ratio",
        category="value",
        description="市净率倒数 (B/P)",
        economic_interpretation="账面价值因子",
        source_paper="Fama-French 1993",
        lookback=60,
        computation="book_equity / market_cap",
    ),
    AcademicFeatureSpec(
        name="pcf_ratio",
        category="value",
        description="现金流比 (CF/P)",
        economic_interpretation="现金流价值",
        source_paper="Desai et al. 2004",
        lookback=60,
        computation="operating_cash_flow / market_cap",
    ),
    AcademicFeatureSpec(
        name="ps_ratio",
        category="value",
        description="市销率倒数 (S/P)",
        economic_interpretation="营收价值",
        source_paper="Hirshleifer et al. 2004",
        lookback=60,
        computation="revenue / market_cap",
    ),
    AcademicFeatureSpec(
        name="ev_ebitda",
        category="value",
        description="企业价值倍数",
        economic_interpretation="综合估值",
        source_paper="Biduong et al. 2013",
        lookback=60,
        computation="enterprise_value / ebitda",
    ),
    AcademicFeatureSpec(
        name="dividend_yield",
        category="value",
        description="股息率",
        economic_interpretation="现金流回报",
        source_paper="Litzenberger-Ramaswamy 1979",
        lookback=60,
        computation="dividends_per_share / price",
    ),
    
    # ===== 规模因子 =====
    AcademicFeatureSpec(
        name="ln_market_cap",
        category="size",
        description="对数市值",
        economic_interpretation="规模溢价",
        source_paper="Fama-French 1993",
        lookback=1,
        computation="log(market_capitalization)",
    ),
    AcademicFeatureSpec(
        name="ln_total_assets",
        category="size",
        description="对数总资产",
        economic_interpretation="企业规模",
        source_paper="Fama-French 1998",
        lookback=120,
        computation="log(total_assets)",
    ),
    
    # ===== 技术因子扩展 =====
    AcademicFeatureSpec(
        name="volume_price_correlation",
        category="technical",
        description="量价相关性",
        economic_interpretation="趋势确认",
        source_paper="Gervais et al. 2001",
        lookback=21,
        computation="correlation between volume and price changes",
    ),
    AcademicFeatureSpec(
        name="high_low_range",
        category="technical",
        description="日内振幅",
        economic_interpretation="波动性代理",
        source_paper="Lowes-Sinha 2014",
        lookback=21,
        computation="(high - low) / close",
    ),
    AcademicFeatureSpec(
        name="close_position",
        category="technical",
        description="收盘位置",
        economic_interpretation="日内趋势",
        source_paper="Lowes-Sinha 2014",
        lookback=21,
        computation="(close - low) / (high - low)",
    ),
    AcademicFeatureSpec(
        name="turnover_rate",
        category="technical",
        description="换手率",
        economic_interpretation="交易活跃度",
        source_paper="Datar et al. 1998",
        lookback=21,
        computation="volume / shares_outstanding",
    ),
    AcademicFeatureSpec(
        name="return_reversal_5d",
        category="reversal",
        description="5日反转",
        economic_interpretation="短期回归",
        source_paper="Jegadeesh 1990",
        lookback=6,
        computation="-return_t-1 to t-5",
    ),
    AcademicFeatureSpec(
        name="return_reversal_20d",
        category="reversal",
        description="20日反转",
        economic_interpretation="中期回归",
        source_paper="Jegadeesh-Titman 1995",
        lookback=21,
        computation="-return_t-1 to t-20",
    ),
    AcademicFeatureSpec(
        name="momentum_6m",
        category="momentum",
        description="6月动量",
        economic_interpretation="中期趋势",
        source_paper="Jegadeesh-Titman 1993",
        lookback=126,
        computation="return_t-6 to t-1",
    ),
    AcademicFeatureSpec(
        name="momentum_12m",
        category="momentum",
        description="12月动量",
        economic_interpretation="长期趋势",
        source_paper="Jegadeesh-Titman 1993",
        lookback=252,
        computation="return_t-12 to t-1",
    ),
    AcademicFeatureSpec(
        name="momentum_36m",
        category="momentum",
        description="36月动量",
        economic_interpretation="长期趋势",
        source_paper="Jegadeesh-Titman 1993",
        lookback=756,
        computation="return_t-36 to t-13",
    ),
    
    # ===== 行为因子 =====
    AcademicFeatureSpec(
        name="attention_grab",
        category="behavioral",
        description="关注度抓取",
        economic_interpretation="媒体效应",
        source_paper="Bartholdy et al. 2007",
        lookback=5,
        computation="abnormal volume + extreme return",
    ),
    AcademicFeatureSpec(
        name="investor_sentiment",
        category="behavioral",
        description="投资者情绪",
        economic_interpretation="市场情绪代理",
        source_paper="Baker-Wurgler 2006",
        lookback=21,
        computation="composite sentiment index",
    ),
    AcademicFeatureSpec(
        name="earnings_quality",
        category="quality",
        description="盈利质量",
        economic_interpretation="会计质量",
        source_paper="Dechow et al. 2010",
        lookback=120,
        computation="accruals quality measure",
    ),
    
    # ===== 供应链 =====
    AcademicFeatureSpec(
        name="supplier_concentration",
        category="business",
        description="供应商集中度",
        economic_interpretation="供应链风险",
        source_paper="Kale-Shahrur 2007",
        lookback=120,
        computation="percentage of purchases from top suppliers",
    ),
    AcademicFeatureSpec(
        name="customer_concentration",
        category="business",
        description="客户集中度",
        economic_interpretation="收入依赖风险",
        source_paper="Patatoukas 2012",
        lookback=120,
        computation="percentage of sales to top customers",
    ),
    
    # ===== ESG =====
    AcademicFeatureSpec(
        name="esg_score",
        category="sustainability",
        description="ESG综合评分",
        economic_interpretation="可持续经营",
        source_paper="Feng et al. 2017",
        lookback=60,
        computation="composite ESG rating",
    ),
    AcademicFeatureSpec(
        name="carbon_intensity",
        category="sustainability",
        description="碳强度",
        economic_interpretation="气候风险",
        source_paper="Bolton-Ketchum 2019",
        lookback=120,
        computation="carbon_emissions / revenue",
    ),
    
    # ===== 期权市场 =====
    AcademicFeatureSpec(
        name="option_skew",
        category="derivatives",
        description="期权偏度",
        economic_interpretation="尾部风险定价",
        source_paper="Cremers-Driessen 2011",
        lookback=21,
        computation="OTM put IV - OTM call IV",
    ),
    AcademicFeatureSpec(
        name="option_volume_ratio",
        category="derivatives",
        description="期权交易量比",
        economic_interpretation="投机活动",
        source_paper="Pan 2002",
        lookback=21,
        computation="put_volume / call_volume",
    ),
]


def get_academic_factor_names() -> list[str]:
    """获取所有学术因子名称"""
    return [f.name for f in ACADEMIC_FACTORS]


def get_academic_factors_by_category(category: str) -> list[AcademicFeatureSpec]:
    """按类别获取因子"""
    return [f for f in ACADEMIC_FACTORS if f.category == category]


def print_academic_factor_summary():
    """打印学术因子库汇总"""
    print("=" * 100)
    print("学术因子库汇总")
    print("=" * 100)
    
    categories = {}
    for f in ACADEMIC_FACTORS:
        if f.category not in categories:
            categories[f.category] = []
        categories[f.category].append(f)
    
    for cat, factors in sorted(categories.items()):
        print(f"\n【{cat.upper()}】{len(factors)}个因子")
        for f in factors:
            print(f"  {f.name:<30} {f.description:<40} 来源:{f.source_paper}")
    
    print(f"\n总计: {len(ACADEMIC_FACTORS)}个学术因子")


if __name__ == "__main__":
    print_academic_factor_summary()
