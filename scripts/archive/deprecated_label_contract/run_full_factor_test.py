"""
完整因子测试流程 - 测试所有667个因子
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from src.features.factor_pool import get_all_factors, inventory
from src.data_sources.complete_factor_factory import CompleteFactorDataFactory, calculate_ic
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('data/artifacts')
OUTPUT_DIR.mkdir(exist_ok=True)


def run_full_factor_test():
    """运行完整因子测试"""
    print("=" * 70)
    print("完整因子测试流程")
    print("=" * 70)
    
    # Step 1: 获取所有因子
    print("\n[1/5] 获取因子池...")
    all_factors = get_all_factors()
    print(f"  总计: {len(all_factors)} 个因子")
    
    # 分析因子来源
    sources = {}
    for f in all_factors:
        src = f.source
        sources[src] = sources.get(src, 0) + 1
    
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count}")
    
    # Step 2: 初始化数据工厂
    print("\n[2/5] 初始化数据工厂...")
    factory = CompleteFactorDataFactory()
    
    # Step 3: 定义可计算的因子
    print("\n[3/5] 定义可计算的因子...")
    
    # 技术因子定义 (从factor_pool的data_requirement映射)
    technical_factors = []
    for f in all_factors:
        if f.source in ['worldquant', 'academic', 'pattern']:
            # 这些需要价格数据
            technical_factors.append(f.name)
    
    # 财务因子
    financial_mapping = {
        'roe': 'roe',
        'roe_weighted': 'roe_weighted',
        'eps': 'eps',
        'bvps': 'bvps',
        'ocf_per_share': 'ocf_per_share',
        'gross_margin': 'gross_margin',
        'operating_margin': 'operating_margin',
        'net_margin': 'net_margin',
        'asset_turnover': 'asset_turnover',
        'current_ratio': 'current_ratio',
        'quick_ratio': 'quick_ratio',
        'debt_ratio': 'debt_ratio',
        'revenue_growth': 'revenue_growth',
        'profit_growth': 'profit_growth',
        'equity_growth': 'equity_growth',
        'total_asset_growth': 'total_asset_growth',
    }
    
    # 扩展技术因子
    extended_tech = []
    for window in [5, 10, 20, 60, 120, 250]:
        extended_tech.append(f'mom{window}')
        extended_tech.append(f'rev{window}')
        extended_tech.append(f'vol_{window}d')
        extended_tech.append(f'close_to_high_{window}')
        extended_tech.append(f'high_low_pos_{window}')
        extended_tech.append(f'close_to_ma_{window}')
        extended_tech.append(f'ma_{window}')
    
    # WorldQuant Alpha
    worldquant_alphas = [f'alpha_{str(i).zfill(3)}' for i in range(1, 102)]
    
    # 合并所有可计算因子
    calculable_factors = list(set(extended_tech + list(financial_mapping.keys()) + worldquant_alphas))
    print(f"  可计算因子: {len(calculable_factors)}")
    
    # Step 4: 批量计算IC
    print("\n[4/5] 计算因子IC...")
    
    # 获取价格面板
    price_panel = factory.price[['trade_date', 'symbol', 'fwd_return_10d', 'fwd_return_20d']].copy()
    
    # 计算财务因子面板 (需要对齐到交易日)
    print("  计算财务因子...")
    financial_panel = factory.financial.copy()
    financial_panel['trade_date'] = financial_panel['pub_date']
    
    # 合并财务因子到价格面板
    ff_cols = {
        'roe': '净资产收益率(%)',
        'roe_weighted': '加权净资产收益率(%)',
        'eps': '摊薄每股收益(元)',
        'ocf_per_share': '每股经营性现金流(元)',
        'gross_margin': '销售毛利率(%)',
        'operating_margin': '营业利润率(%)',
        'net_margin': '销售净利率(%)',
        'asset_turnover': '总资产周转率(次)',
        'current_ratio': '流动比率',
        'quick_ratio': '速动比率',
        'debt_ratio': '资产负债率(%)',
    }
    
    # Forward fill financial data
    financial_merged = price_panel.copy()
    for fname, col in ff_cols.items():
        if col in factory.financial.columns:
            ff_df = factory.financial[['pub_date', 'symbol', col]].copy()
            ff_df = ff_df.rename(columns={'pub_date': 'trade_date'})
            ff_df[col] = pd.to_numeric(ff_df[col], errors='coerce')
            financial_merged = financial_merged.merge(ff_df, on=['trade_date', 'symbol'], how='left')
    
    # 计算IC
    print("  计算IC (这可能需要几分钟)...")
    ic_results = []
    
    # 技术因子
    tech_cols = [c for c in price_panel.columns if any(c.startswith(p) for p in ['mom', 'rev', 'vol_', 'close_to_', 'high_low_pos', 'ma_'])]
    print(f"    技术因子列: {len(tech_cols)}")
    
    for col in tech_cols:
        if col in price_panel.columns:
            ic = calculate_ic(price_panel, col, 'fwd_return_20d')
            ic_results.append({
                'factor': col,
                'source': 'technical',
                'ic': ic['ic'],
                'rank_ic': ic['rank_ic'],
                'ic_ir': ic.get('ic_ir', np.nan),
            })
    
    # 财务因子
    for fname, col in ff_cols.items():
        if col in financial_merged.columns:
            ic = calculate_ic(financial_merged, col, 'fwd_return_20d')
            ic_results.append({
                'factor': fname,
                'source': 'financial',
                'ic': ic['ic'],
                'rank_ic': ic['rank_ic'],
                'ic_ir': ic.get('ic_ir', np.nan),
            })
    
    # Step 5: 保存结果
    print("\n[5/5] 保存结果...")
    results_df = pd.DataFrame(ic_results)
    results_df = results_df.sort_values('ic', ascending=False)
    
    output_path = OUTPUT_DIR / 'full_factor_ic_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"  结果已保存: {output_path}")
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("IC 测试结果摘要")
    print("=" * 70)
    
    print(f"\n总因子数: {len(results_df)}")
    print(f"IC > 0: {len(results_df[results_df['ic'] > 0])}")
    print(f"IC > 0.02: {len(results_df[results_df['ic'] > 0.02])}")
    print(f"IC > 0.05: {len(results_df[results_df['ic'] > 0.05])}")
    print(f"IC < -0.02: {len(results_df[results_df['ic'] < -0.02])}")
    
    print("\n--- Top 20 正向因子 ---")
    top_positive = results_df[results_df['ic'] > 0].head(20)
    print(top_positive.to_string(index=False))
    
    print("\n--- Top 20 负向因子 (可做空) ---")
    top_negative = results_df[results_df['ic'] < 0].head(20)
    print(top_negative.to_string(index=False))
    
    return results_df


if __name__ == '__main__':
    results = run_full_factor_test()
