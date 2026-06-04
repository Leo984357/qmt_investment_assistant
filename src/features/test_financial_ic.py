"""
测试财务因子的IC

使用财务因子与价格因子进行IC对比
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_sources.batch_financial_factors import FINANCIAL_FACTOR_MAPPING


def load_financial_factor_panel(trade_date: str) -> pd.DataFrame:
    """加载指定日期可用的财务因子"""
    from src.ops.paths import RAW_DATA_DIR

    cache_path = RAW_DATA_DIR / 'financial_factors.parquet'
    if not cache_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(cache_path)

    # 筛选指定日期前发布的财务数据
    trade_date = pd.to_datetime(trade_date)
    df['trade_date'] = pd.to_datetime(df['pub_date'])
    df = df[df['trade_date'] <= trade_date]

    # 每只股票取最新数据
    df = df.sort_values('trade_date').groupby(['symbol', 'factor_name']).last().reset_index()

    return df


def compute_financial_ic(
    factor_data: pd.DataFrame,
    returns_data: pd.DataFrame,
    min_stocks: int = 20,
) -> pd.DataFrame:
    """计算财务因子的IC"""
    results = []

    # 获取可用的因子
    factors = factor_data['factor_name'].unique()

    for factor_name in factors:
        # 获取因子值
        factor_df = factor_data[factor_data['factor_name'] == factor_name][['symbol', 'value']].copy()

        if len(factor_df) < min_stocks:
            continue

        # 合并收益数据
        merged = factor_df.merge(returns_data[['symbol', 'forward_return']], on='symbol', how='inner')

        if len(merged) < min_stocks:
            continue

        # 去极值
        lower = merged['value'].quantile(0.02)
        upper = merged['value'].quantile(0.98)
        merged['value'] = merged['value'].clip(lower, upper)

        # 标准化
        mean = merged['value'].mean()
        std = merged['value'].std()
        if std > 0:
            merged['value_z'] = (merged['value'] - mean) / std
        else:
            merged['value_z'] = 0

        # 计算IC
        ic = merged['value_z'].corr(merged['forward_return'])
        rank_ic = merged['value_z'].corr(merged['forward_return'], method='spearman')

        results.append({
            'factor': factor_name,
            'ic': ic,
            'rank_ic': rank_ic,
            'n_stocks': len(merged),
        })

    return pd.DataFrame(results)


def main():
    from src.data_sources.financial_data import FinancialDataCache

    # 加载财务因子
    cache = FinancialDataCache()

    # 获取因子IC
    # 创建一个简单的测试场景：使用已知有收益的股票
    test_symbols = [
        'sh.600000', 'sh.600036', 'sh.600519', 'sh.601318', 'sh.601166',
        'sh.600016', 'sh.600028', 'sh.600887', 'sh.600030', 'sh.600050',
        'sz.000001', 'sz.000002', 'sz.000333', 'sz.000338', 'sz.000651',
        'sz.000858', 'sz.000876', 'sz.000895', 'sz.002594', 'sz.002415',
    ]

    # 模拟收益数据 (使用随机数据，仅用于测试)
    np.random.seed(42)
    returns_data = pd.DataFrame({
        'symbol': test_symbols,
        'forward_return': np.random.randn(len(test_symbols)) * 0.02,
    })

    # 加载财务数据
    factor_rows = []
    for sym in test_symbols:
        df = cache.get_financial_indicator(sym, start_year='2020')
        if df.empty:
            continue

        # 获取最新一期数据
        latest = df.iloc[-1]
        for col, fname in FINANCIAL_FACTOR_MAPPING.items():
            if col in latest.index and pd.notna(latest[col]):
                try:
                    factor_rows.append({
                        'symbol': sym,
                        'factor_name': fname,
                        'value': float(latest[col]),
                        'pub_date': latest['日期'],
                    })
                except:
                    pass

    factor_data = pd.DataFrame(factor_rows)

    # 计算IC
    ic_results = compute_financial_ic(factor_data, returns_data)

    print("=" * 80)
    print("财务因子IC测试结果")
    print("=" * 80)
    print()

    # 按Rank IC排序
    ic_results = ic_results.sort_values('rank_ic', ascending=False)

    print(f"{'因子':<25} {'IC':>10} {'Rank IC':>10} {'股票数':>8}")
    print("-" * 60)

    for _, row in ic_results.iterrows():
        print(f"{row['factor']:<25} {row['ic']:>10.4f} {row['rank_ic']:>10.4f} {row['n_stocks']:>8}")

    # 分类汇总
    print()
    print("=" * 80)
    print("按类别汇总")
    print("=" * 80)

    # 定义因子分类
    factor_categories = {
        'value': ['roe', 'roa', 'gross_margin', 'operating_margin', 'net_margin', 'earnings_yield', 'book_to_price'],
        'growth': ['revenue_growth', 'profit_growth', 'equity_growth', 'asset_growth'],
        'leverage': ['debt_ratio', 'current_ratio', 'quick_ratio', 'cash_ratio', 'interest_coverage'],
        'efficiency': ['asset_turnover', 'inv_turnover', 'ar_turnover'],
        'profitability': ['roe_weighted', 'total_roa', 'ocf_per_share'],
    }

    for category, factors in factor_categories.items():
        cat_data = ic_results[ic_results['factor'].isin(factors)]
        if not cat_data.empty:
            avg_ic = cat_data['ic'].mean()
            avg_rank_ic = cat_data['rank_ic'].mean()
            print(f"\n【{category.upper()}】")
            print(f"  平均 IC: {avg_ic:.4f}, 平均 Rank IC: {avg_rank_ic:.4f}")
            for _, row in cat_data.sort_values('rank_ic', ascending=False).iterrows():
                sign = "+" if row['rank_ic'] > 0 else ""
                print(f"    {row['factor']}: {sign}{row['rank_ic']:.4f}")


if __name__ == '__main__':
    main()
