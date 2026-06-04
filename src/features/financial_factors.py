"""
财务因子库 - Barra/Academic财务因子

这些因子需要从akshare获取财务数据，季度更新
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .registry import FeatureRegistry, FeatureSpec

# 财务因子缓存
_financial_factor_cache: pd.DataFrame | None = None


def load_financial_factor_cache() -> pd.DataFrame:
    """加载财务因子缓存"""
    global _financial_factor_cache

    if _financial_factor_cache is None:
        from src.ops.paths import RAW_DATA_DIR
        cache_path = RAW_DATA_DIR / 'financial_factors.parquet'

        if cache_path.exists():
            _financial_factor_cache = pd.read_parquet(cache_path)
            print(f'Loaded financial factor cache: {len(_financial_factor_cache)} rows')
        else:
            _financial_factor_cache = pd.DataFrame()
            print('No financial factor cache found')

    return _financial_factor_cache


def get_financial_factor_value(
    bars: pd.DataFrame,
    factor_name: str,
    symbol_col: str = 'symbol',
    date_col: str = 'trade_date',
) -> pd.Series:
    """获取财务因子值
    
    Args:
        bars: 日线数据
        factor_name: 因子名称
        symbol_col: 股票代码列名
        date_col: 日期列名
    
    Returns:
        因子值Series，索引与bars相同
    """
    cache = load_financial_factor_cache()

    if cache.empty:
        return pd.Series(np.nan, index=bars.index)

    # 筛选指定因子
    factor_data = cache[cache['factor_name'] == factor_name].copy()
    if factor_data.empty:
        return pd.Series(np.nan, index=bars.index)

    # 获取每只股票最新的财务数据日期
    factor_data['trade_date'] = pd.to_datetime(factor_data['trade_date'])
    bars_date = pd.to_datetime(bars[date_col])

    # 创建映射: (symbol, trade_date) -> value
    factor_data = factor_data.rename(columns={'symbol': symbol_col})

    # 为每个交易日匹配最新的财务数据
    result = pd.Series(np.nan, index=bars.index)

    # 转换为日期用于匹配
    bars['date_for_merge'] = bars_date

    # 获取每只股票最新的可用财务数据
    latest_factors = factor_data.sort_values('trade_date').groupby(symbol_col).last()

    # 匹配到bars
    symbol_to_factor = bars[symbol_col].map(latest_factors.set_index(symbol_col)['value'])
    result = symbol_to_factor

    return result


@dataclass
class FinancialFactorSpec:
    """财务因子规格"""
    name: str
    akshare_column: str
    barra_name: str
    category: str
    description: str
    direction: int  # 1 = 高值好, -1 = 低值好


# 财务因子映射 (akshare列名 -> 因子名)
FINANCIAL_FACTOR_SPECS = [
    # Value因子
    FinancialFactorSpec(
        name='roe',
        akshare_column='净资产收益率(%)',
        barra_name='ROE',
        category='value',
        description='净资产收益率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='roa',
        akshare_column='资产报酬率(%)',
        barra_name='ROA',
        category='value',
        description='资产报酬率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='gross_margin',
        akshare_column='销售毛利率(%)',
        barra_name='GPM',
        category='value',
        description='销售毛利率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='operating_margin',
        akshare_column='营业利润率(%)',
        barra_name='OPM',
        category='value',
        description='营业利润率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='net_margin',
        akshare_column='销售净利率(%)',
        barra_name='NPM',
        category='value',
        description='销售净利率',
        direction=1,
    ),

    # Profitability因子
    FinancialFactorSpec(
        name='roe_weighted',
        akshare_column='加权净资产收益率(%)',
        barra_name='ROE_W',
        category='profitability',
        description='加权净资产收益率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='total_roa',
        akshare_column='总资产利润率(%)',
        barra_name='ROA_T',
        category='profitability',
        description='总资产利润率',
        direction=1,
    ),

    # Growth因子
    FinancialFactorSpec(
        name='revenue_growth',
        akshare_column='主营业务收入增长率(%)',
        barra_name='REV_G',
        category='growth',
        description='营收增长率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='profit_growth',
        akshare_column='净利润增长率(%)',
        barra_name='PROF_G',
        category='growth',
        description='净利润增长率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='equity_growth',
        akshare_column='净资产增长率(%)',
        barra_name='EQUITY_G',
        category='growth',
        description='净资产增长率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='asset_growth',
        akshare_column='总资产增长率(%)',
        barra_name='ASSET_G',
        category='growth',
        description='总资产增长率',
        direction=-1,  # 资产增长通常是负向信号
    ),

    # Leverage因子
    FinancialFactorSpec(
        name='debt_ratio',
        akshare_column='资产负债率(%)',
        barra_name='DTE',
        category='leverage',
        description='资产负债率',
        direction=-1,
    ),
    FinancialFactorSpec(
        name='current_ratio',
        akshare_column='流动比率',
        barra_name='CR',
        category='leverage',
        description='流动比率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='quick_ratio',
        akshare_column='速动比率',
        barra_name='QR',
        category='leverage',
        description='速动比率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='cash_ratio',
        akshare_column='现金比率(%)',
        barra_name='CASH_R',
        category='leverage',
        description='现金比率',
        direction=1,
    ),

    # Efficiency因子
    FinancialFactorSpec(
        name='asset_turnover',
        akshare_column='总资产周转率(次)',
        barra_name='ATO',
        category='efficiency',
        description='总资产周转率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='inv_turnover',
        akshare_column='存货周转率(次)',
        barra_name='INV_TO',
        category='efficiency',
        description='存货周转率',
        direction=1,
    ),
    FinancialFactorSpec(
        name='ar_turnover',
        akshare_column='应收账款周转率(次)',
        barra_name='AR_TO',
        category='efficiency',
        description='应收账款周转率',
        direction=1,
    ),
]


def create_financial_factor_func(spec: FinancialFactorSpec):
    """创建财务因子计算函数"""
    def compute_fn(bars: pd.DataFrame) -> pd.Series:
        cache = load_financial_factor_cache()

        if cache.empty:
            return pd.Series(np.nan, index=bars.index)

        # 筛选指定因子
        factor_data = cache[cache['factor_name'] == spec.name].copy()
        if factor_data.empty:
            return pd.Series(np.nan, index=bars.index)

        # 获取每只股票最新的财务数据
        latest = factor_data.sort_values('trade_date').groupby('symbol').last()

        # 映射到bars
        result = bars['symbol'].map(latest['value'])

        # 应用方向
        if spec.direction < 0:
            result = -result

        return result

    return compute_fn


def financial_factor_registry() -> FeatureRegistry:
    """注册所有财务因子"""
    registry = FeatureRegistry()

    for spec in FINANCIAL_FACTOR_SPECS:
        compute_fn = create_financial_factor_func(spec)

        registry.register(FeatureSpec(
            name=spec.name,
            inputs=(),
            lookback=1,
            description=f'{spec.description} (barra: {spec.barra_name})',
            compute=compute_fn,
            category='financial',
            preprocessing=('winsorize', 'cross_sectional_scale'),
            economic_meaning=f'Direction: {spec.direction}',
        ))

    return registry


def get_all_financial_factor_names() -> list[str]:
    """获取所有财务因子名称"""
    return [spec.name for spec in FINANCIAL_FACTOR_SPECS]


if __name__ == '__main__':
    # 测试
    reg = financial_factor_registry()
    print('=== 财务因子注册 ===')
    print(f'注册因子数: {len(reg.features)}')
    for name in get_all_financial_factor_names():
        print(f'  - {name}')
