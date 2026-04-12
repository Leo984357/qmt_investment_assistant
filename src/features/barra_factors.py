"""
Barra风格因子库 - Barra Risk Model标准因子

来源: MSCI Barra Global Equity Model (GEM)

 Barra风格因子覆盖:
 1. Size - 规模
 2. Book-to-Price - 价值
 3. Earnings Yield - 盈利收益率
 4. Cash Flow Yield - 现金流收益率
 5. Dividend Yield - 股息率
 6. Growth - 成长
 7. Profitability - 盈利能力
 8. Leverage - 杠杆
 9. Liquidity - 流动性
 10. Volatility - 波动率
 11. Momentum - 动量
 12. Country/Industry - 国家/行业
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BarraStyleSpec:
    name: str
    barra_name: str
    category: str
    description: str
    formula: str
    long_lookback: int  # 月度
    short_lookback: int  # 日度
    data_requirement: list[str]


BARRA_STYLE_FACTORS = [
    # ===== Size因子 =====
    BarraStyleSpec(
        name="size",
        barra_name="SIZE",
        category="size",
        description="对数市值",
        formula="ln(market_equity)",
        long_lookback=1,
        short_lookback=1,
        data_requirement=["market_cap"],
    ),
    BarraStyleSpec(
        name="size_nonlinear",
        barra_name="SIZENL",
        category="size",
        description="非线性市值(捕捉规模溢价非线性)",
        formula="ln(market_equity)^2 or residual after linear size",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["market_cap"],
    ),
    BarraStyleSpec(
        name="log_assets",
        barra_name="LOGAT",
        category="size",
        description="对数总资产",
        formula="ln(total_assets)",
        long_lookback=3,
        short_lookback=1,
        data_requirement=["total_assets"],
    ),
    
    # ===== Value因子 =====
    BarraStyleSpec(
        name="book_to_price",
        barra_name="BTOP",
        category="value",
        description="账面市值比",
        formula="book_equity / market_equity",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["book_equity", "market_cap"],
    ),
    BarraStyleSpec(
        name="earnings_yield",
        barra_name="ETOP",
        category="value",
        description="盈利收益率 (E/P)",
        formula="earnings / market_equity",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["net_income", "market_cap"],
    ),
    BarraStyleSpec(
        name="cashflow_yield",
        barra_name="CFTOP",
        category="value",
        description="现金流收益率",
        formula="operating_cash_flow / market_equity",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["cash_flow_from_operations", "market_cap"],
    ),
    BarraStyleSpec(
        name="dividend_yield",
        barra_name="DY",
        category="value",
        description="股息率",
        formula="dividends / price",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["dividends", "price"],
    ),
    BarraStyleSpec(
        name="sales_to_price",
        barra_name="STOM",
        category="value",
        description="市销率倒数",
        formula="sales / market_equity",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["revenue", "market_cap"],
    ),
    BarraStyleSpec(
        name="forward_earnings_yield",
        barra_name="FETOP",
        category="value",
        description="前瞻盈利收益率",
        formula="forward_earnings / market_equity",
        long_lookback=3,
        short_lookback=1,
        data_requirement=["consensus_earnings", "market_cap"],
    ),
    
    # ===== Growth因子 =====
    BarraStyleSpec(
        name="earnings_growth",
        barra_name="EGRLF",
        category="growth",
        description="长期盈利增长率 (5年)",
        formula="CAGR of earnings over 5 years",
        long_lookback=60,
        short_lookback=1,
        data_requirement=["net_income"],
    ),
    BarraStyleSpec(
        name="revenue_growth",
        barra_name="SGRLF",
        category="growth",
        description="长期营收增长率 (5年)",
        formula="CAGR of revenue over 5 years",
        long_lookback=60,
        short_lookback=1,
        data_requirement=["revenue"],
    ),
    BarraStyleSpec(
        name="book_growth",
        barra_name="BGRLF",
        category="growth",
        description="长期净资产增长率",
        formula="CAGR of book equity over 5 years",
        long_lookback=60,
        short_lookback=1,
        data_requirement=["book_equity"],
    ),
    BarraStyleSpec(
        name="short_term_growth",
        barra_name="EGRSF",
        category="growth",
        description="短期盈利增长",
        formula="earnings change YoY",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["net_income"],
    ),
    BarraStyleSpec(
        name="forecast_growth",
        barra_name="FGE",
        category="growth",
        description="分析师预测增长",
        formula="mean analyst forecast for 1-2 year growth",
        long_lookback=3,
        short_lookback=1,
        data_requirement=["consensus_forecast"],
    ),
    
    # ===== Profitability因子 =====
    BarraStyleSpec(
        name="roe",
        barra_name="ROE",
        category="profitability",
        description="净资产收益率",
        formula="net_income / book_equity",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["net_income", "book_equity"],
    ),
    BarraStyleSpec(
        name="roa",
        barra_name="ROA",
        category="profitability",
        description="资产收益率",
        formula="net_income / total_assets",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["net_income", "total_assets"],
    ),
    BarraStyleSpec(
        name="gross_profitability",
        barra_name="GPA",
        category="profitability",
        description="毛利润/资产",
        formula="(revenue - cogs) / total_assets",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["revenue", "cogs", "total_assets"],
    ),
    BarraStyleSpec(
        name="operating_profitability",
        barra_name="OPA",
        category="profitability",
        description="营业利润率",
        formula="(revenue - cogs - sga) / total_assets",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["revenue", "cogs", "sga", "total_assets"],
    ),
    BarraStyleSpec(
        name="asset_turnover",
        barra_name="ATO",
        category="profitability",
        description="资产周转率",
        formula="revenue / total_assets",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["revenue", "total_assets"],
    ),
    BarraStyleSpec(
        name="cash_profitability",
        barra_name="CFOA",
        category="profitability",
        description="现金利润/资产",
        formula="operating_cash_flow / total_assets",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["cash_flow_from_operations", "total_assets"],
    ),
    BarraStyleSpec(
        name="accruals",
        barra_name="AC",
        category="profitability",
        description="应计项",
        formula="(net_income - cash_flow) / total_assets",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["net_income", "cash_flow", "total_assets"],
    ),
    BarraStyleSpec(
        name="ebitda_margin",
        barra_name="EBITDAM",
        category="profitability",
        description="EBITDA利润率",
        formula="EBITDA / revenue",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["ebitda", "revenue"],
    ),
    BarraStyleSpec(
        name="net_margin",
        barra_name="NPM",
        category="profitability",
        description="净利率",
        formula="net_income / revenue",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["net_income", "revenue"],
    ),
    
    # ===== Leverage因子 =====
    BarraStyleSpec(
        name="market_leverage",
        barra_name="MLEV",
        category="leverage",
        description="市场杠杆",
        formula="(market_equity + debt) / market_equity",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["market_cap", "total_debt"],
    ),
    BarraStyleSpec(
        name="book_leverage",
        barra_name="BLEV",
        category="leverage",
        description="账面杠杆",
        formula="(book_equity + debt) / book_equity",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["book_equity", "total_debt"],
    ),
    BarraStyleSpec(
        name="debt_to_assets",
        barra_name="DTOA",
        category="leverage",
        description="负债资产比",
        formula="total_debt / total_assets",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["total_debt", "total_assets"],
    ),
    BarraStyleSpec(
        name="debt_to_equity",
        barra_name="DTOE",
        category="leverage",
        description="负债权益比",
        formula="total_debt / book_equity",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["total_debt", "book_equity"],
    ),
    BarraStyleSpec(
        name="interest_coverage",
        barra_name="CR",
        category="leverage",
        description="利息覆盖倍数",
        formula="EBIT / interest_expense",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["ebit", "interest_expense"],
    ),
    BarraStyleSpec(
        name="net_debt_to_ebitda",
        barra_name="NDAR",
        category="leverage",
        description="净负债/EBITDA",
        formula="(total_debt - cash) / EBITDA",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["total_debt", "cash", "ebitda"],
    ),
    
    # ===== Liquidity因子 =====
    BarraStyleSpec(
        name="amihud_illiquidity",
        barra_name="AMIHUD",
        category="liquidity",
        description="Amihud非流动性",
        formula="avg(|return| / volume)",
        long_lookback=12,
        short_lookback=21,
        data_requirement=["returns", "volume", "price"],
    ),
    BarraStyleSpec(
        name="turnover_rate",
        barra_name="TURN",
        category="liquidity",
        description="换手率",
        formula="avg(volume / shares_outstanding) over 3M or 12M",
        long_lookback=12,
        short_lookback=21,
        data_requirement=["volume", "shares_outstanding"],
    ),
    BarraStyleSpec(
        name="log_turnover_rate",
        barra_name="LSTR",
        category="liquidity",
        description="对数换手率",
        formula="ln(turnover_rate)",
        long_lookback=12,
        short_lookback=21,
        data_requirement=["volume", "shares_outstanding"],
    ),
    BarraStyleSpec(
        name="zero_trade_ratio",
        barra_name="CHITOA",
        category="liquidity",
        description="零交易天数占比",
        formula="fraction of days with zero volume in past month",
        long_lookback=1,
        short_lookback=21,
        data_requirement=["volume"],
    ),
    BarraStyleSpec(
        name="bid_ask_spread",
        barra_name="BAS",
        category="liquidity",
        description="买卖价差",
        formula="(ask - bid) / (mid price)",
        long_lookback=1,
        short_lookback=5,
        data_requirement=["bid", "ask"],
    ),
    BarraStyleSpec(
        name="depth",
        barra_name="DEPTH",
        category="liquidity",
        description="订单深度",
        formula="avg(bid_size + ask_size)",
        long_lookback=1,
        short_lookback=5,
        data_requirement=["bid_size", "ask_size"],
    ),
    BarraStyleSpec(
        name="market_impact",
        barra_name="MIMP",
        category="liquidity",
        description="市场冲击",
        formula="price_change after trade / trade_size",
        long_lookback=12,
        short_lookback=21,
        data_requirement=["intraday_prices", "trade_data"],
    ),
    
    # ===== Volatility因子 =====
    BarraStyleSpec(
        name="residual_volatility",
        barra_name="RVOL",
        category="volatility",
        description="特质波动率",
        formula="std(residuals from market model) annualised",
        long_lookback=36,
        short_lookback=252,
        data_requirement=["daily_returns", "market_returns"],
    ),
    BarraStyleSpec(
        name="beta",
        barra_name="BETA",
        category="volatility",
        description="市场Beta",
        formula="cov(stock, market) / var(market)",
        long_lookback=36,
        short_lookback=252,
        data_requirement=["daily_returns", "market_returns"],
    ),
    BarraStyleSpec(
        name="total_volatility",
        barra_name="TVOL",
        category="volatility",
        description="总波动率",
        formula="std(daily returns) annualised",
        long_lookback=12,
        short_lookback=252,
        data_requirement=["daily_returns"],
    ),
    BarraStyleSpec(
        name="idiosyncratic_vol_60d",
        barra_name="IVOL60",
        category="volatility",
        description="60日特质波动率",
        formula="std(residuals) over 60 days",
        long_lookback=3,
        short_lookback=60,
        data_requirement=["daily_returns", "market_returns"],
    ),
    BarraStyleSpec(
        name="realized_range",
        barra_name="RANGE",
        category="volatility",
        description="已实现振幅",
        formula="avg(high-low) / close",
        long_lookback=12,
        short_lookback=21,
        data_requirement=["high", "low", "close"],
    ),
    BarraStyleSpec(
        name="downside_volatility",
        barra_name="DSV",
        category="volatility",
        description="下行波动率",
        formula="std(returns when return < 0)",
        long_lookback=12,
        short_lookback=252,
        data_requirement=["daily_returns"],
    ),
    
    # ===== Momentum因子 =====
    BarraStyleSpec(
        name="momentum_12_1",
        barra_name="MOM12",
        category="momentum",
        description="12-1月动量",
        formula="return_t-12 to t-2 (skip most recent month)",
        long_lookback=13,
        short_lookback=1,
        data_requirement=["daily_returns"],
    ),
    BarraStyleSpec(
        name="momentum_6_1",
        barra_name="MOM6",
        category="momentum",
        description="6-1月动量",
        formula="return_t-6 to t-2",
        long_lookback=7,
        short_lookback=1,
        data_requirement=["daily_returns"],
    ),
    BarraStyleSpec(
        name="momentum_3_1",
        barra_name="MOM3",
        category="momentum",
        description="3-1月动量",
        formula="return_t-3 to t-2",
        long_lookback=4,
        short_lookback=1,
        data_requirement=["daily_returns"],
    ),
    BarraStyleSpec(
        name="momentum_252_21",
        barra_name="MOM1Y",
        category="momentum",
        description="252日动量",
        formula="return_t-252 to t-22",
        long_lookback=253,
        short_lookback=21,
        data_requirement=["daily_returns"],
    ),
    BarraStyleSpec(
        name="momentum_reversal",
        barra_name="REVMOM",
        category="momentum",
        description="动量反转",
        formula="return_t-5 to t-2 (reversal of momentum)",
        long_lookback=6,
        short_lookback=1,
        data_requirement=["daily_returns"],
    ),
    
    # ===== Quality (Barra使用单独的Quality因子组) =====
    BarraStyleSpec(
        name="earnings_quality",
        barra_name="EQ",
        category="quality",
        description="盈利质量",
        formula="based on accruals and cash flow",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["net_income", "cash_flow", "assets"],
    ),
    BarraStyleSpec(
        name="balance_sheet_quality",
        barra_name="BQ",
        category="quality",
        description="资产负债表质量",
        formula="intangibles / total_assets",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["intangibles", "total_assets"],
    ),
    BarraStyleSpec(
        name="cash_generation",
        barra_name="CGE",
        category="quality",
        description="现金生成能力",
        formula="cash_flow / total_assets",
        long_lookback=36,
        short_lookback=1,
        data_requirement=["cash_flow", "total_assets"],
    ),
    
    # ===== Sentiment (Barra Extended) =====
    BarraStyleSpec(
        name="short_term_reversal",
        barra_name="STREV",
        category="reversal",
        description="短期反转",
        formula="-return_t-1",
        long_lookback=1,
        short_lookback=1,
        data_requirement=["daily_returns"],
    ),
    BarraStyleSpec(
        name="medium_reversal",
        barra_name="MDREV",
        category="reversal",
        description="中期反转",
        formula="-return_t-20 to t-5",
        long_lookback=21,
        short_lookback=5,
        data_requirement=["daily_returns"],
    ),
    BarraStyleSpec(
        name="analyst_sentiment",
        barra_name="AS",
        category="sentiment",
        description="分析师情绪",
        formula="upgrade - downgrade / total",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["analyst_ratings"],
    ),
    BarraStyleSpec(
        name="earnings_surprise",
        barra_name="ES",
        category="sentiment",
        description="盈利惊喜",
        formula="(actual_earnings - forecast) / price",
        long_lookback=3,
        short_lookback=1,
        data_requirement=["actual_earnings", "consensus_forecast", "price"],
    ),
    
    # ===== Country/ESG =====
    BarraStyleSpec(
        name="esg_combined",
        barra_name="ESG",
        category="esg",
        description="ESG综合得分",
        formula="weighted avg of E, S, G scores",
        long_lookback=6,
        short_lookback=1,
        data_requirement=["environmental", "social", "governance_scores"],
    ),
    BarraStyleSpec(
        name="carbon_risk",
        barra_name="CARB",
        category="esg",
        description="碳风险",
        formula="carbon_emissions / revenue or assets",
        long_lookback=12,
        short_lookback=1,
        data_requirement=["carbon_emissions", "revenue"],
    ),
]


def get_barra_factor_names() -> list[str]:
    """获取所有Barra因子名称"""
    return [f.name for f in BARRA_STYLE_FACTORS]


def get_barra_factors_by_category(category: str) -> list[BarraStyleSpec]:
    """按类别获取Barra因子"""
    return [f for f in BARRA_STYLE_FACTORS if f.category == category]


def print_barra_factor_summary():
    """打印Barra因子库汇总"""
    print("=" * 100)
    print("Barra风格因子库汇总")
    print("=" * 100)
    
    categories = {}
    for f in BARRA_STYLE_FACTORS:
        if f.category not in categories:
            categories[f.category] = []
        categories[f.category].append(f)
    
    for cat, factors in sorted(categories.items()):
        print(f"\n【{cat.upper()}】{len(factors)}个因子")
        for f in factors:
            print(f"  {f.name:<30} ({f.barra_name})")
            print(f"    公式: {f.formula}")
    
    print(f"\n总计: {len(BARRA_STYLE_FACTORS)}个Barra风格因子")


if __name__ == "__main__":
    print_barra_factor_summary()
