"""
统一因子库 - 整合学术、Barra、WorldQuant、行业、形态学因子

⚠️ 警告: 这是描述池，不是可计算池！
- 667个条目，637个唯一名，25个重复名
- 不能直接用于实验，只能作为因子探索的参考描述
- Production实验必须使用 src/features/simple_definitions.py 中的因子

使用方式:
from src.features.factor_pool import get_all_factors, get_factors_by_source, print_factor_pool_summary

因子真相源优先级:
1. simple_definitions.py (203因子) - 可计算主因子库
2. factor_catalog.py (144因子) - 有研究状态
3. factor_pool.py (667条目) - 仅描述参考
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class UnifiedFactor:
    """统一因子定义"""
    name: str
    source: str
    category: str
    sub_category: str
    description: str
    economic_interpretation: str
    lookback: int
    data_requirement: list[str]
    formula: str | None = None
    paper_reference: str | None = None


def build_unified_factor_pool() -> list[UnifiedFactor]:
    """构建统一因子池"""
    factors = []

    # ===== 学术因子 (Academic Factors) =====
    from src.features.academic_factors import ACADEMIC_FACTORS

    for f in ACADEMIC_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="academic",
            category="academic",
            sub_category=f.category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=[],  # 需要根据具体实现确定
            formula=f.computation,
            paper_reference=f.source_paper,
        ))

    # ===== Barra因子 =====
    from src.features.barra_factors import BARRA_STYLE_FACTORS

    for f in BARRA_STYLE_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="barra",
            category="barra_style",
            sub_category=f.category,
            description=f.description,
            economic_interpretation=f.formula,
            lookback=f.long_lookback,
            data_requirement=f.data_requirement,
            paper_reference="MSCI Barra GEM",
        ))

    # ===== WorldQuant Alpha =====
    from src.features.worldquant_alphas import WORLDQUANT_ALPHAS

    for a in WORLDQUANT_ALPHAS:
        factors.append(UnifiedFactor(
            name=a.name,
            source="worldquant",
            category="worldquant",
            sub_category=a.category,
            description=a.description,
            economic_interpretation=a.formula,
            lookback=a.lookback,
            data_requirement=a.data_requirement,
            paper_reference="Kakushadze 2016",
        ))

    # ===== 行业因子 =====
    from src.features.sector_factors import SECTOR_FACTORS

    for f in SECTOR_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="sector",
            category="sector",
            sub_category=f.category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
        ))

    # ===== 形态学因子 =====
    from src.features.pattern_factors import PATTERN_FACTORS

    for f in PATTERN_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="pattern",
            category="pattern",
            sub_category=f.category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
        ))

    # ===== 扩展财务因子 =====
    from src.features.extended_financial_factors import EXTENDED_FINANCIAL_FACTORS

    for f in EXTENDED_FINANCIAL_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="extended_financial",
            category="extended_financial",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.computation,
            paper_reference=f.paper_reference,
        ))

    # ===== 扩展技术因子 =====
    from src.features.extended_technical_factors import EXTENDED_TECHNICAL_FACTORS

    for f in EXTENDED_TECHNICAL_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="extended_technical",
            category="extended_technical",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.formula,
        ))

    # ===== 分析师预期因子 =====
    from src.features.analyst_expectation_factors import ANALYST_FACTORS

    for f in ANALYST_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="analyst",
            category="analyst",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.formula,
        ))

    # ===== 资金流因子 =====
    from src.features.money_flow_factors import MONEY_FLOW_FACTORS

    for f in MONEY_FLOW_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="money_flow",
            category="money_flow",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.formula,
        ))

    # ===== 情绪因子 =====
    from src.features.sentiment_factors import SENTIMENT_FACTORS

    for f in SENTIMENT_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="sentiment",
            category="sentiment",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.formula,
        ))

    # ===== 宏观因子 =====
    from src.features.macro_factors import MACRO_FACTORS

    for f in MACRO_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="macro",
            category="macro",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.formula,
        ))

    return factors


def get_all_factors() -> list[UnifiedFactor]:
    """获取所有因子"""
    return build_unified_factor_pool()


def get_factors_by_source(source: str) -> list[UnifiedFactor]:
    """按来源获取因子"""
    return [f for f in build_unified_factor_pool() if f.source == source]


def get_factors_by_category(category: str) -> list[UnifiedFactor]:
    """按类别获取因子"""
    return [f for f in build_unified_factor_pool() if f.category == category]


def inventory() -> pd.DataFrame:
    """导出因子清单"""
    factors = build_unified_factor_pool()
    rows = []
    for f in factors:
        rows.append({
            'name': f.name,
            'source': f.source,
            'category': f.category,
            'sub_category': f.sub_category,
            'description': f.description,
            'lookback': f.lookback,
            'data_requirement': ', '.join(f.data_requirement),
            'formula': f.formula or '',
            'paper': f.paper_reference or '',
        })
    return pd.DataFrame(rows)


def print_factor_pool_summary():
    """打印因子池汇总"""
    factors = build_unified_factor_pool()

    print("=" * 100)
    print("统一因子库汇总")
    print("=" * 100)

    # 按来源统计
    sources = {}
    for f in factors:
        if f.source not in sources:
            sources[f.source] = []
        sources[f.source].append(f)

    print("\n【一、按来源分类】")
    for source, f_list in sorted(sources.items(), key=lambda x: -len(x[1])):
        print(f"  {source:<15}: {len(f_list)}个因子")

    # 按大类统计
    categories = {}
    for f in factors:
        if f.category not in categories:
            categories[f.category] = []
        categories[f.category].append(f)

    print("\n【二、按大类分类】")
    for cat, f_list in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<25}: {len(f_list)}个因子")

    # 详细列表
    print("\n【三、详细因子列表】")
    for source in ['academic', 'barra', 'worldquant', 'sector', 'pattern']:
        if source in sources:
            print(f"\n  === {source.upper()} ===")
            f_list = sources[source]

            # 按子类别分组
            sub_cats = {}
            for f in f_list:
                if f.sub_category not in sub_cats:
                    sub_cats[f.sub_category] = []
                sub_cats[f.sub_category].append(f)

            for sub, s_list in sorted(sub_cats.items()):
                print(f"    [{sub}] {len(s_list)}个")
                for f in s_list[:3]:  # 只显示前3个
                    print(f"      - {f.name}: {f.description[:50]}")
                if len(s_list) > 3:
                    print(f"      ... 还有{len(s_list)-3}个")

    print("\n" + "=" * 100)
    print(f"总计: {len(factors)}个因子")
    print("=" * 100)


def get_data_requirement_summary() -> pd.DataFrame:
    """数据需求汇总"""
    factors = build_unified_factor_pool()

    all_requirements = {}
    for f in factors:
        for req in f.data_requirement:
            if req not in all_requirements:
                all_requirements[req] = 0
            all_requirements[req] += 1

    df = pd.DataFrame([
        {'data_requirement': k, 'factor_count': v}
        for k, v in sorted(all_requirements.items(), key=lambda x: -x[1])
    ])
    return df


if __name__ == "__main__":
    print_factor_pool_summary()

    print("\n")
    df = get_data_requirement_summary()
    print("【数据需求汇总】")
    print(df.to_string(index=False))


# =============================================================================
# 因子去重映射
# =============================================================================

DUPLICATE_RESOLUTION: dict[str, str] = {
    # 学术 + Barra 重复
    'accruals': 'academic',
    'amihud_illiquidity': 'extended_technical',  # extended_technical更符合A股
    'asset_turnover': 'extended_financial',
    'beta': 'academic',
    'bid_ask_spread': 'academic',
    'capex_intensity': 'extended_financial',
    'debt_to_equity': 'extended_financial',
    'dividend_yield': 'extended_financial',
    'earnings_growth': 'academic',
    'earnings_quality': 'academic',
    'earnings_yield': 'extended_financial',  # 与simple_definitions一致
    'ebitda_margin': 'extended_financial',
    'ev_ebitda': 'extended_financial',
    'financial_leverage': 'extended_financial',
    'gross_profitability': 'academic',
    'gross_margin': 'extended_financial',  # 与simple_definitions一致
    'investments': 'extended_financial',
    'leverage': 'academic',
    'momentum': 'extended_technical',  # extended_technical更符合A股
    'net_issuance': 'academic',
    'operating_leverage': 'extended_financial',
    'price_impact': 'extended_technical',
    'roe': 'extended_financial',  # 与simple_definitions一致
    'roa': 'extended_financial',  # 与simple_definitions一致
    'size': 'barra',  # Barra的size因子最标准
    'total_volatility': 'extended_technical',
    'value': 'academic',
    'volatility': 'extended_technical',
}

# 667池名称 → simple_definitions 名称映射。
#
# 这张表用于把“描述池”落到可计算实现。部分名称是严格同义映射，部分是
# proxy 映射；proxy 只能用于候选因子扩展和探索，正式研究仍需单因子验证。
POOL_TO_SIMPLE_MAPPING: dict[str, str] = {
    # 估值类 (Value)
    'earnings_yield': 'earnings_yield',
    'book_to_price': 'book_to_price',
    'dividend_yield': 'dividend_yield',
    'cashflow_yield': 'cashflow_yield',
    'ev_ebitda': 'ev_ebitda',
    'pe_ratio': 'earnings_yield',
    'pb_ratio': 'book_to_price',
    'pcf_ratio': 'cashflow_yield',
    'ps_ratio': 'sales_to_price',
    'forward_earnings_yield': 'earnings_yield',

    # 盈利类 (Profitability)
    'roe': 'roe',
    'roa': 'roa',
    'gross_margin': 'gross_margin',
    'net_margin': 'net_margin',
    'operating_margin': 'operating_margin',
    'roe_weighted': 'roe_weighted',
    'total_roa': 'total_roa',
    'gross_profitability': 'gross_margin',
    'operating_profitability': 'operating_margin',
    'ebitda_margin': 'ebitda_margin',
    'cash_profitability': 'cash_profitability',

    # 杠杆类 (Leverage)
    'debt_ratio': 'debt_ratio',
    'current_ratio': 'current_ratio',
    'quick_ratio': 'quick_ratio',
    'cash_ratio': 'cash_ratio',
    'market_leverage': 'market_leverage',
    'book_leverage': 'book_leverage',
    'financial_leverage': 'financial_leverage',
    'operating_leverage': 'operating_leverage',

    # 成长类 (Growth)
    'revenue_growth': 'revenue_growth',
    'profit_growth': 'profit_growth',
    'equity_growth': 'equity_growth',
    'earnings_growth': 'profit_growth',
    'asset_growth': 'asset_growth',
    'book_growth': 'equity_growth',
    'short_term_growth': 'profit_growth',
    'forecast_growth': 'forecast_growth',
    'book_equity_growth': 'equity_growth',

    # 动量类 (Momentum)
    'momentum': 'mom250',
    'mom_20': 'mom20',
    'mom_60': 'mom60',
    'mom_120': 'mom120',
    'mom_250': 'mom250',
    'mom12_1': 'mom_12_1',
    'momentum_6m': 'mom120',
    'momentum_12m': 'mom250',
    'momentum_36m': 'mom250',

    # 反转类 (Reversal)
    'short_term_reversal': 'rev5',
    'return_reversal_5d': 'rev5',
    'return_reversal_20d': 'rev20',
    'long_term_reversal': 'mom250',

    # 波动率类 (Volatility)
    'volatility': 'vol_20',
    'total_volatility': 'vol_120',
    'idiosyncratic_vol': 'idio_vol',
    'max5vol': 'max_daily_return',

    # 流动性类 (Liquidity)
    'amihud_illiquidity': 'amihud_illiq_20d',
    'turnover_rate': 'turnover_rate',
    'bid_ask_spread': 'bid_ask_spread',
    'zero_trade_days': 'zero_days_ratio',

    # 技术指标类
    'rsi': 'rsi14',
    'rsi_6': 'rsi6',
    'rsi_12': 'rsi12',
    'rsi_24': 'rsi24',
    'macd': 'macd',
    'macd_hist': 'macd_hist',
    'cci': 'cci14',
    'kdj_k': 'kdj_k9',
    'adx14': 'adx14',
    'adx28': 'adx28',
    'adx_slope': 'adx14',
    'close_position': 'close_position_20d',
    'high_low_range': 'high_low_pos20',

    # 规模类 (Size)
    'size': 'ln_market_cap',
    'ln_market_cap': 'ln_market_cap',
    'log_assets': 'ln_market_cap',
    'ln_total_assets': 'ln_total_assets',
    'size_nonlinear': 'size_nonlinear',

    # 质量类 (Quality)
    'accruals': 'accruals',
    'accrual_ratio': 'accruals',
    'Altman_zscore': 'altman_zscore',
    'earnings_quality': 'earnings_quality',
    'intangibles': 'intangibles_ratio',

    # 市场类 (Market)
    'beta': 'market_beta',
    'idyncorr_mkt': 'idio_mkt_corr',
    'max_daily_return': 'max_daily_return',

    # 分析师类 (Analyst)
    'analyst_coverage': 'analyst_coverage',
    'forecast_dispersion': 'forecast_dispersion',
    'forecast_breadth': 'forecast_breadth',
    'analyst_performance_rank': 'analyst_performance_rank',
    'avg_analyst_return': 'avg_analyst_return',
    'institution_coverage': 'institution_coverage',
    'institutional_ownership': 'inst_ownership',
    'inst_ownership_change': 'inst_ownership',

    # 资金流类
    'main_flow_rank': 'main_flow_rank',
    'institutional_intensity': 'institutional_intensity',
    'super_flow_mean': 'super_flow_mean',
    'money_flow_intensity': 'money_flow_intensity',
    'sector_inflow_rank': 'sector_inflow_rank',

    # 研报类
    'research_report_count': 'research_report_count',
    'avg_pe_2026': 'avg_pe_2026',

    # 行业类
    'sector_mom_20d': 'sector_mom_20d',
    'sector_mom_60d': 'sector_mom_60d',
    'sector_rs_20d': 'sector_rs_20d',
    'sector_regime': 'sector_regime',

    # WorldQuant Alpha
    'alpha_001': 'alpha_001',
    'alpha_002': 'alpha_002',
    'alpha_003': 'alpha_003',
    'alpha_004': 'alpha_004',
    'alpha_005': 'alpha_005',
    'alpha_007': 'alpha_007',
    'alpha_011': 'alpha_011',
    'alpha_015': 'alpha_015',
    'alpha_017': 'alpha_017',
    'alpha_019': 'alpha_019',
    'alpha_021': 'alpha_021',
    'alpha_024': 'alpha_024',
    'alpha_025': 'alpha_025',
    'alpha_030': 'alpha_030',
    'alpha_035': 'alpha_035',
    'alpha_037': 'alpha_037',
    'alpha_038': 'alpha_038',
    'alpha_040': 'alpha_040',
    'alpha_041': 'alpha_041',
    'alpha_045': 'alpha_045',
    'alpha_048': 'alpha_048',
    'alpha_049': 'alpha_049',
    'alpha_050': 'alpha_050',

    # Extended Technical
    'vol_ratio_20_60': 'vol_ratio_20_60',
    'vol_ratio_5_60': 'vol_ratio_5_60',
    'vol_ratio_10_60': 'vol_ratio_10_60',
    'close_to_high60': 'close_to_high60',
    'high_low_pos20': 'high_low_pos20',
    'high_low_pos60': 'high_low_pos60',
    'ma_diff_10_20': 'ma_diff_10_20',
    'ma_diff_5_60': 'ma_diff_5_60',
    'amount_growth5': 'amount_growth5',
    'amount_growth20': 'amount_growth20',
    'amount_growth60': 'amount_growth60',
    'vol_growth5': 'vol_growth5',
    'vol_growth10': 'vol_growth10',
    'vol_growth20': 'vol_growth20',
    'mom90': 'mom90',
    'rev5': 'rev5',
    'rev10': 'rev10',
    'candle_body_ratio': 'candle_body_ratio',
    'candle_upper_shadow': 'candle_upper_shadow',
    'candle_lower_shadow': 'candle_lower_shadow',
    'candle_doji': 'candle_doji',
    'volume_trend_20d': 'volume_trend_20d',
    'volume_momentum_5d': 'volume_momentum_5d',
    'volume_momentum_20d': 'volume_momentum_20d',
    'close_position_20d': 'close_position_20d',
    'close_position_60d': 'close_position_60d',
    'close_position_120d': 'close_position_120d',
    'trend_strength_20d': 'trend_strength_20d',
    'trend_strength_60d': 'trend_strength_60d',
    'gap_size': 'gap_size',
    'momentum_divergence_20d': 'momentum_divergence_20d',
    'new_high_20d': 'new_high_20d',
    'new_low_20d': 'new_low_20d',
    'up_day_ratio_20d': 'up_day_ratio_20d',
    'consecutive_up': 'consecutive_up',
    'consecutive_down': 'consecutive_down',
}

# 扩展 proxy 映射：优先复用已有稳定实现，避免为了凑数复制低质量公式。
POOL_TO_SIMPLE_MAPPING.update({
    # Academic / Barra aliases
    'q_me': 'ln_market_cap',
    'q_ia': 'asset_growth',
    'q_roe': 'roe',
    'q_delta_roe': 'roe_change',
    'mgmt': 'earnings_momentum',
    'perf': 'earnings_momentum',
    'bm_slope': 'book_to_price',
    'bankruptcy_prob': 'altman_zscore',
    'Altman_zscore': 'altman_zscore',
    'attention_grab': 'volume_spike_2x',
    'investor_sentiment': 'volume_price_correlation',
    'debt_to_assets': 'debt_ratio',
    'debt_to_equity': 'debt_equity',
    'net_debt_to_ebitda': 'financial_leverage',
    'log_turnover_rate': 'turnover_rate',
    'zero_trade_ratio': 'zero_days_ratio',
    'depth': 'lotus_liquidity',
    'market_impact': 'price_impact_20d',
    'residual_volatility': 'idio_vol',
    'idiosyncratic_vol_60d': 'idio_vol',
    'realized_range': 'price_range_20',
    'downside_volatility': 'vol_realized_20',
    'momentum_12_1': 'mom_12_1',
    'momentum_6_1': 'mom120',
    'momentum_3_1': 'mom60',
    'momentum_252_21': 'ts_momentum_250',
    'momentum_reversal': 'momentum_reversal_ratio',
    'balance_sheet_quality': 'debt_ratio',
    'cash_generation': 'cash_profitability',
    'medium_reversal': 'rev20',
    'analyst_sentiment': 'rating_change',
    'earnings_surprise': 'earning_surprise_proxy',

    # Extended financial aliases
    'roe_qoq': 'roe_change',
    'roe_yoy': 'roe_change',
    'roe_stability': 'earnings_quality',
    'roe_weighted_3y': 'roe_weighted',
    'roa_ttm': 'roa',
    'roa_qoq': 'roa',
    'roce': 'total_roa',
    'roic': 'total_roa',
    'roe_excellent': 'roe',
    'gross_margin_ttm': 'gross_margin',
    'operating_margin_ttm': 'operating_margin',
    'net_margin_ttm': 'net_margin',
    'ebit_margin': 'operating_margin',
    'pre_tax_margin': 'operating_margin',
    'margin_expansion': 'margin_change',
    'margin_cv': 'earnings_quality',
    'cash_roe': 'cash_profitability',
    'asset_turnover_ttm': 'asset_turnover',
    'revenue_growth_qoq': 'revenue_growth',
    'revenue_growth_yoy_ttm': 'revenue_growth',
    'revenue_cagr_3y': 'revenue_growth',
    'profit_growth_qoq': 'profit_growth',
    'profit_growth_yoy_ttm': 'profit_growth',
    'profit_cagr_3y': 'profit_growth',
    'operating_profit_growth_yoy': 'profit_growth',
    'ebit_growth_yoy': 'profit_growth',
    'ocf_growth_yoy': 'ocf_per_share',
    'asset_growth_yoy': 'asset_growth',
    'equity_growth_yoy': 'equity_growth',
    'bvps_growth_yoy': 'book_growth',
    'eps_growth_yoy': 'profit_growth',
    'revenue_acceleration': 'revenue_growth',
    'profit_acceleration': 'profit_growth',
    'pe_ttm': 'earnings_yield',
    'pe_forward': 'earnings_yield',
    'pe_g': 'earnings_yield',
    'pe_historical_low': 'earnings_yield',
    'pb_median': 'book_to_price',
    'ps': 'sales_to_price',
    'pcf': 'cashflow_yield',
    'ev_sales': 'sales_to_price',
    'ev_ebit': 'ev_ebitda',
    'payout_ratio': 'dividend_payout_ratio',
    'cf_yield': 'cashflow_yield',
    'book_yield': 'book_to_price',
    'revenue_ocf_ratio': 'ocf_per_share',
    'pe_5y_avg': 'earnings_yield',
    'pb_5y_avg': 'book_to_price',
    'ev_ebitda_industry_adj': 'ev_ebitda',
    'inventory_turnover': 'inv_turnover',
    'receivable_turnover': 'ar_turnover',
    'payable_turnover': 'asset_turnover',
    'cash_turnover': 'asset_turnover',
    'working_capital_turnover': 'working_capital_ratio',
    'equity_turnover': 'asset_turnover',
    'fixed_asset_turnover': 'fixasset_turnover',
    'operating_cycle': 'ar_turnover_days',
    'cash_cycle': 'ar_turnover_days',
    'long_term_debt_ratio': 'longterm_debt_ratio',
    'debt_service_coverage': 'interest_coverage_ratio',
    'equity_multiplier': 'financial_leverage',
    'debt_growth_vs_asset_growth': 'debt_ratio',
    'fcf_per_share': 'ocf_per_share',
    'ocf_to_debt': 'interest_coverage_ratio',
    'fcf_to_debt': 'interest_coverage_ratio',
    'cash_conversion_ratio': 'cash_ratio',
    'investing_cash_flow_ratio': 'ocf_per_share',
    'financing_cash_flow_ratio': 'debt_ratio',
    'ocf_growth_3y': 'ocf_per_share',
    'cash_dividend_coverage': 'dividend_payout_ratio',
    'asset_quality': 'asset_turnover',
    'receivable_to_revenue': 'ar_turnover',
    'inventory_to_revenue': 'inv_turnover',
    'big_bath_indicator': 'earnings_quality',
    'deferred_revenue_growth': 'revenue_growth',
    'tax_burden': 'net_margin',
    'minority_interest_ratio': 'equity_ratio',
    'goodwill_ratio': 'intangibles_ratio',
    'intangible_ratio': 'intangibles_ratio',
    'roe_consistency': 'roe_weighted',
    'earnings_volatility': 'earnings_quality',

    # Extended technical aliases
    'ma5': 'price_to_ma20',
    'ma10': 'price_to_ma20',
    'ma20': 'price_to_ma20',
    'ma60': 'price_to_ma60',
    'ma120': 'price_to_ma120',
    'ma250': 'close_to_high250',
    'ma5_10_golden_cross': 'ma_diff_5_20',
    'ma5_20_death_cross': 'ma_diff_5_20',
    'ma20_60_golden_cross': 'ma_diff_20_60',
    'ma20_60_death_cross': 'ma_diff_20_60',
    'ma_bull_alignment': 'trend_strength_20d',
    'ma_bear_alignment': 'trend_strength_20d',
    'ema12': 'macd_diff',
    'ema26': 'macd_diff',
    'ema9': 'macd_dea',
    'wma5': 'price_to_ma20',
    'wma20': 'price_to_ma20',
    'ma5_bias': 'price_to_ma20',
    'ma20_bias': 'price_to_ma20',
    'ma250_bias': 'close_to_high250',
    'macd_signal': 'macd_dea',
    'macd_histogram': 'macd_hist',
    'kdj_d': 'kdj_d9',
    'kdj_j': 'kdj_k9',
    'kdj_golden_cross': 'kdj_k9',
    'kdj_death_cross': 'kdj_k9',
    'wr14': 'williams_r14',
    'wr28': 'williams_r28',
    'roc5': 'mom5',
    'roc20': 'mom20',
    'roc60': 'mom60',
    'cci28': 'cci20',
    'trix12': 'trix_15',
    'momentum_10': 'mom10',
    'atr28': 'atr20',
    'atr_ratio': 'atr20',
    'bb_upper': 'pct_b_20',
    'bb_lower': 'pct_b_20',
    'bb_width': 'price_range_20',
    'bb_position': 'pct_b_20',
    'hv20': 'vol20',
    'hv60': 'vol60',
    'hv120': 'vol120',
    'vol_change': 'vol_term_structure',
    'vol_rank': 'vol_realized_20',
    'daily_range_pct': 'price_range_20',
    'close_to_high': 'close_to_high60',
    'gap_pct': 'overnight_gap',
    'upper_shadow_pct': 'candle_upper_shadow',
    'volume_price_trend': 'volume_price_correlation',
    'obv_slope': 'obv_momentum_10',
    'obv_ma5_cross': 'obv_momentum_10',
    'vr14': 'volume_turnover_20',
    'vr28': 'volume_turnover_20',
    'volume_ratio': 'relative_volume_20d',
    'volume_ratio_20': 'relative_volume_20d',
    'turnover_rate_5d': 'turnover_rate',
    'price_volume_divergence': 'momentum_divergence_20d',
    'volume_momentum': 'volume_momentum_20d',
    'plus_di': 'adx14',
    'minus_di': 'adx14',
    'trend_strength': 'trend_strength_20d',
    'trend_persistence': 'trend_strength_60d',
    'donchian_high': 'donchian_position',
    'donchian_low': 'donchian_position',

    # Pattern aliases
    'candle_marubozu': 'candle_body_ratio',
    'candle_engulf_bullish': 'candle_engulfing',
    'candle_engulf_bearish': 'candle_engulfing',
    'candle_harami_bullish': 'candle_engulfing',
    'candle_harami_bearish': 'candle_engulfing',
    'candle_piercing': 'candle_hammer',
    'candle_dark_cloud': 'candle_shooting_star',
    'candle_evening_star': 'candle_morning_star',
    'pattern_breakout_strength': 'breakout_20d',
    'pattern_support_resistance_strength': 'donchian_position',
    'pattern_volume_breakout': 'volume_spike_2x',
    'pattern_close_breakout': 'breakout_20d',
    'pattern_trendline_slope': 'trend_strength_20d',
    'pattern_trendline_angle': 'trend_strength_20d',
    'pattern_channel_width': 'price_range_20',
    'pattern_price_channel_position': 'close_position_20d',
    'pattern_triangle_convergence': 'vol_cone_5_20',
    'pattern_consolidation_range': 'price_range_20',
    'pattern_rectangle_stability': 'vol_realized_20',
    'pattern_flag_pole_height': 'mom20',
    'pattern_head_shoulders_score': 'head_shoulders',
    'pattern_double_top_bottom_score': 'head_shoulders',
    'pattern_triple_top_bottom_score': 'head_shoulders',
    'pattern_v_reversal_strength': 'rev20',
    'ma_alignment_short': 'ma_diff_5_20',
    'ma_alignment_long': 'ma_diff_20_60',
    'ma_golden_cross': 'ma_diff_5_20',
    'ma_death_cross': 'ma_diff_5_20',
    'ma_bands_width': 'price_range_20',
    'ma_bands_position': 'pct_b_20',
    'volatility_breakout': 'vol_jump_20',
    'volatility_squeeze': 'vol_cone_5_20',
    'volatility_asymmetry': 'tail_ratio_20',
    'gap_fill_ratio': 'gap_size',
    'momentum_divergence': 'momentum_divergence_20d',
    'momentum_acceleration': 'acceleration_20',
    'momentum_exhaustion': 'momentum_reversal_ratio',
    'momentum_sequencing': 'relative_momentum_20_60',
    'volume_price_divergence': 'momentum_divergence_20d',
    'volume_climax': 'volume_spike_3x',
    'volume_accumulation': 'money_flow_ratio',
    'volume_on_up_days': 'up_day_ratio_20d',
    'breadth_momentum': 'sector_adj_up_day_ratio',
    'breadth_divergence': 'sector_adj_volume_momentum',
    'breadth_thrust': 'sector_adj_up_day_ratio',
    'new_high_low_ratio': 'new_high_20d',
    'limit_up_consecutive': 'consecutive_up',
    'turnover_rate_spike': 'volume_spike_2x',
    'investor_tracking': 'volume_price_correlation',
    'st_speculation': 'volume_spike_5x',
    'small_cap_speculation': 'ln_market_cap',
    'board_leading': 'sector_adj_trend_strength',

    # Sector / macro aliases
    'sector_mom_1m': 'sector_mom_20d',
    'sector_mom_3m': 'sector_mom_60d',
    'sector_mom_6m': 'sector_mom_60d',
    'sector_mom_12m': 'sector_mom_60d',
    'sector_mom_12_1': 'sector_mom_60d',
    'sector_rs_1m': 'sector_rs_20d',
    'sector_rs_3m': 'sector_rs_20d',
    'sector_rs_6m': 'sector_rs_20d',
    'sector_rs_12m': 'sector_rs_20d',
    'sector_trend_strength': 'sector_adj_trend_strength',
    'sector_momentum_acceleration': 'sector_mom_20d',
    'sector_momentum_reversal': 'sector_rs_20d',
    'sector_leading_lagging': 'sector_rs_20d',
    'sector_turnover_rate': 'sector_volume_trend',
    'sector_new_high_ratio': 'new_high_20d',
    'sector_new_low_ratio': 'new_low_20d',
    'industry_chain_mom_upstream': 'sector_mom_20d',
    'industry_chain_mom_midstream': 'sector_mom_20d',
    'industry_chain_mom_downstream': 'sector_mom_20d',
    'industry_chain_spillover': 'sector_correlation',
    'macro_beta_gdp': 'sector_beta',
    'macro_beta_inflation': 'sector_beta',
    'macro_beta_interest': 'sector_beta',
    'macro_beta_credit': 'sector_beta',
    'macro_beta_sentiment': 'sector_beta',
    'sector_flow_1w': 'sector_inflow_rank',
    'sector_flow_1m': 'sector_inflow_rank',
    'sector_flow_3m': 'sector_inflow_rank',
    'sector_inflow_acceleration': 'sector_inflow_rank',
    'sector_margin_balance': 'sector_volume_trend',
    'sector_pe': 'earnings_yield',
    'sector_pb': 'book_to_price',
    'sector_pe_historical': 'earnings_yield',
    'sector_peg': 'earnings_yield',
    'sector_crowding': 'sector_volume_trend',
    'sector_short_interest': 'sector_volume_trend',
    'sector_beta_realized': 'sector_beta',
    'sector_analyst_rating': 'sector_analyst_breadth',
    'sector_forecast_revenue_growth': 'forecast_growth',
    'sector_forecast_earnings_revision': 'rating_change',
    'sector_limit_up_count': 'consecutive_up',
    'sector_limit_down_count': 'consecutive_down',
    'sector_hsgt_flow': 'sector_inflow_rank',
    'sector_momentum_20d': 'sector_mom_20d',
    'sector_momentum_60d': 'sector_mom_60d',
    'sector_relative_strength': 'sector_rs_20d',
    'sector_rotation_signal': 'sector_regime',
    'value_growth_rotation': 'hml',
    'size_rotation': 'smb',
    'market_breadth': 'sector_adj_up_day_ratio',
    'rate_sensitivity': 'sector_beta',
    'duration_exposure': 'sector_beta',
    'rate_change_momentum': 'sector_mom_20d',
    'inflation_sensitivity': 'sector_beta',
    'commodity_exposure': 'sector_beta',
    'input_cost_pressure': 'net_margin',
    'pricing_power': 'gross_margin',
    'fx_sensitivity': 'sector_beta',
    'export_exposure': 'sector_beta',
    'import_dependency': 'sector_beta',
    'cny_strength': 'sector_mom_20d',

    # Money flow / analyst / sentiment aliases
    'super_large_net_flow': 'money_flow_20d',
    'super_large_net_ratio': 'money_flow_ratio',
    'super_large_buy_pct': 'large_flow_ratio',
    'super_large_5d_net_flow': 'net_flow_5d',
    'super_large_20d_net_flow': 'money_flow_20d',
    'large_net_flow': 'money_flow_20d',
    'large_net_ratio': 'large_flow_ratio',
    'large_5d_net_flow': 'net_flow_5d',
    'large_20d_net_flow': 'money_flow_20d',
    'big_order_net_flow': 'money_flow_20d',
    'big_order_5d_net_flow': 'net_flow_5d',
    'big_order_20d_net_flow': 'money_flow_20d',
    'medium_net_flow': 'money_flow_20d',
    'medium_5d_net_flow': 'net_flow_5d',
    'small_net_flow': 'money_flow_20d',
    'small_5d_net_flow': 'net_flow_5d',
    'small_20d_net_flow': 'money_flow_20d',
    'inflow_ratio': 'money_flow_ratio',
    'inflow_5d_ratio': 'net_flow_5d',
    'inflow_20d_ratio': 'money_flow_ratio',
    'institutional_buy_ratio': 'institutional_intensity',
    'retail_sell_ratio': 'small_order_imbalance',
    'main_force_net_flow': 'money_flow_20d',
    'main_force_5d_net_flow': 'net_flow_5d',
    'main_force_10d_net_flow': 'flow_momentum',
    'main_force_20d_net_flow': 'money_flow_20d',
    'main_force_pct': 'money_flow_ratio',
    'flow_strength': 'money_flow_ratio',
    'flow_acceleration': 'flow_momentum',
    'flow_consistency': 'money_flow_ratio',
    'net_flow_rank': 'main_flow_rank',
    'buy_sell_imbalance': 'money_flow_ratio',
    'large_vs_small_flow': 'large_flow_ratio',
    'institutional_pressure': 'institutional_intensity',
    'retail_activity': 'small_order_imbalance',
    'consecutive_inflow_days': 'consecutive_up',
    'consecutive_outflow_days': 'consecutive_down',
    'flow_reversal': 'flow_momentum',
    'flow_divergence': 'momentum_divergence_20d',
    'eps_forecast_1y': 'consensus_estimate',
    'eps_forecast_2y': 'consensus_estimate',
    'eps_forecast_growth_1y': 'forecast_growth',
    'eps_forecast_growth_2y': 'forecast_growth',
    'profit_forecast_1y': 'consensus_estimate',
    'profit_forecast_growth': 'forecast_growth',
    'revenue_forecast_1y': 'consensus_estimate',
    'revenue_forecast_growth': 'forecast_growth',
    'eps_surprise': 'earning_surprise_proxy',
    'profit_surprise': 'earning_surprise_proxy',
    'revenue_surprise': 'earning_surprise_proxy',
    'consecutive_surprise': 'earning_surprise_proxy',
    'rating_score': 'rating_change',
    'rating_buy_pct': 'rating_change',
    'rating_hold_pct': 'rating_change',
    'rating_sell_pct': 'rating_change',
    'rating_upgrades_recent': 'rating_change',
    'rating_downgrades_recent': 'rating_change',
    'rating_net_upgrades': 'rating_change',
    'rating_coverage': 'analyst_coverage',
    'rating_change_momentum': 'rating_change',
    'top_agency_rating': 'rating_change',
    'eps_revision_1m': 'rating_change',
    'eps_revision_3m': 'rating_change',
    'eps_revision_direction': 'rating_change',
    'eps_revision_momentum': 'rating_change',
    'profit_revision_1m': 'rating_change',
    'profit_revision_3m': 'rating_change',
    'revenue_revision_1m': 'rating_change',
    'target_price_upside': 'target_price_ratio',
    'target_price_revision_1m': 'target_price_ratio',
    'target_price_convergence': 'target_price_ratio',
    'consensus_eps_std': 'estimate_dispersion',
    'consensus_eps_cv': 'estimate_dispersion',
    'consensus_strong': 'consensus_estimate',
    'forecast_curvature': 'forecast_growth',
    'forecast_2y_vs_1y': 'forecast_growth',
    'high_cover_eps': 'consensus_estimate',
    'low_cover_eps': 'consensus_estimate',
    'recent_eps_weighted': 'consensus_estimate',
    'model_eps_weighted': 'consensus_estimate',
    'accuracy_weighted_eps': 'analyst_performance_rank',
    'target_price_to_52w_high': 'target_price_ratio',
    'target_price_to_52w_low': 'target_price_ratio',
    'pe_forward_to_pe_hist': 'earnings_yield',
    'peg_ratio_adjusted': 'earnings_yield',
    'target_irr': 'target_price_ratio',
    'upside_to_consensus': 'target_price_ratio',
    'price_target_ratio': 'target_price_ratio',
    'analyst_price_divergence': 'target_price_ratio',
    'news_sentiment': 'volume_price_correlation',
    'news_sentiment_3d': 'volume_price_correlation',
    'news_sentiment_7d': 'volume_price_correlation',
    'news_count_7d': 'volume_spike_2x',
    'news_acceleration': 'volume_acceleration',
    'negative_news_ratio': 'tail_ratio_20',
    'news_sentiment_momentum': 'momentum_20d',
    'analyst_report_sentiment': 'rating_change',
    'social_media_buzz': 'volume_spike_2x',
    'search_index_trend': 'volume_trend_20d',
    'attention_rank': 'volume_spike_2x',
    'hot_stock_rank': 'volume_spike_2x',
    'hot_stock_change': 'volume_acceleration',
    'stock_mention_frequency': 'volume_spike_2x',
    'institutional_follow_count': 'institution_coverage',
    'public_follower_count': 'volume_spike_2x',
    'discussion_intensity': 'volume_price_correlation',
    'media_coverage_change': 'volume_acceleration',
    'volatility_sentiment': 'vol_realized_20',
    'tail_risk_sentiment': 'tail_ratio_20',
    'drawdown_sentiment': 'rev20',
    'recovery_sentiment': 'mom20',
    'volatility_clustering': 'vol_realized_20',
    'volume_spike': 'volume_spike_2x',
    'price_spike': 'price_acceleration',
    'limit_up_count': 'consecutive_up',
    'limit_down_count': 'consecutive_down',
    'abnormal_volume_ratio': 'relative_volume_20d',
    'order_imbalance': 'order_flow_imbalance',
})

REJECTED_POOL_NAMES: set[str] = {
    'roe_ttm',  # 与simple_definitions的roe_ttm定义不同
    'pe',  # 应该用earnings_yield (1/PE)
    'pb',  # 应该用book_to_price (1/PB)
}


def get_pool_duplicates() -> dict[str, list[str]]:
    """返回重复因子名和它们的来源"""
    factors = build_unified_factor_pool()
    name_to_sources = {}
    for f in factors:
        if f.name not in name_to_sources:
            name_to_sources[f.name] = []
        name_to_sources[f.name].append(f.source)

    return {k: v for k, v in name_to_sources.items() if len(v) > 1}


def check_pool_name(name: str) -> dict:
    """
    检查667池因子名是否可用
    
    Returns:
        dict with keys: is_duplicate, is_rejected, has_simple_mapping, canonical_source
    """
    factors = build_unified_factor_pool()
    sources = [f.source for f in factors if f.name == name]

    result = {
        'name': name,
        'is_duplicate': len(set(sources)) > 1 if sources else False,
        'sources': list(set(sources)) if sources else [],
        'is_rejected': name in REJECTED_POOL_NAMES,
        'canonical_source': DUPLICATE_RESOLUTION.get(name),
        'simple_mapping': POOL_TO_SIMPLE_MAPPING.get(name),
    }

    if not sources:
        result['warning'] = f"因子 '{name}' 不在667池中"

    return result


def get_simple_factor_names() -> set[str]:
    """返回 simple_definitions 当前可计算因子名。"""
    from src.features.simple_definitions import simple_factor_registry

    inventory = simple_factor_registry().inventory()
    if inventory.empty:
        return set()
    return set(inventory['feature_name'])


def audit_pool_computability() -> dict:
    """
    审计描述池到可计算池的覆盖率。

    注意：mapped_count 包含同义和 proxy 映射，不能替代正式单因子验证。
    """
    factors = build_unified_factor_pool()
    simple_names = get_simple_factor_names()
    pool_names = [factor.name for factor in factors]
    pool_unique = set(pool_names)

    direct = pool_unique & simple_names
    valid_mappings = {
        pool_name: simple_name
        for pool_name, simple_name in POOL_TO_SIMPLE_MAPPING.items()
        if simple_name in simple_names
    }
    invalid_mappings = {
        pool_name: simple_name
        for pool_name, simple_name in POOL_TO_SIMPLE_MAPPING.items()
        if simple_name not in simple_names
    }
    computable_unique = direct | (set(valid_mappings) & pool_unique)
    unmapped_unique = pool_unique - computable_unique - REJECTED_POOL_NAMES

    return {
        'pool_entries': len(pool_names),
        'pool_unique': len(pool_unique),
        'simple_count': len(simple_names),
        'direct_count': len(direct),
        'mapped_count': len(set(valid_mappings) & pool_unique),
        'computable_unique': len(computable_unique),
        'computable_rate': len(computable_unique) / len(pool_unique) if pool_unique else 0.0,
        'invalid_mapping_count': len(invalid_mappings),
        'invalid_mappings': invalid_mappings,
        'rejected_count': len(REJECTED_POOL_NAMES & pool_unique),
        'unmapped_count': len(unmapped_unique),
        'unmapped_names': sorted(unmapped_unique),
    }
