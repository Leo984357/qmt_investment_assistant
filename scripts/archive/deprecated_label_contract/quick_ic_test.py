"""
快速IC测试 - 计算所有可快速计算的因子
"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.features.factor_pool import get_all_factors

DATA_DIR = Path('data/raw')
CACHE_DIR = Path('data/factor_cache')
FETCHED_DIR = DATA_DIR / 'fetched_data'

print("=" * 80)
print("快速667因子IC测试")
print("=" * 80)

# 加载价格数据
print("\n加载价格数据...")
bars = pd.read_parquet(DATA_DIR / 'daily_bar.parquet')
bars['trade_date'] = pd.to_datetime(bars['trade_date'])
bars = bars.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
bars = bars[bars['trade_date'] >= '2020-01-01'].copy()

# 计算基础指标
bars['returns'] = bars.groupby('symbol')['adj_close'].pct_change()
bars['fwd_return_20d'] = bars.groupby('symbol')['adj_close'].pct_change(20).shift(-20)
bars['ln_price'] = np.log(bars['close'].replace(0, np.nan))

# 简化版技术因子
print("计算技术因子...")
windows = [5, 10, 20, 30, 60, 120, 250]

for w in windows:
    bars[f'mom_{w}d'] = bars.groupby('symbol')['adj_close'].pct_change(w)
    bars[f'rev_{w}d'] = -bars[f'mom_{w}d']
    bars[f'vol_{w}d'] = bars.groupby('symbol')['returns'].transform(
        lambda x: x.rolling(w, min_periods=max(2, w//5)).std()
    )
    bars[f'ma_{w}'] = bars.groupby('symbol')['adj_close'].transform(
        lambda x: x.rolling(w, min_periods=max(2, w//5)).mean()
    )
    bars[f'close_to_ma_{w}'] = bars['adj_close'] / bars[f'ma_{w}'].replace(0, np.nan)
    bars[f'vol_ratio_{w}'] = bars.groupby('symbol')['volume'].transform(
        lambda x: x / x.rolling(w, min_periods=max(2, w//5)).mean().replace(0, np.nan)
    )
    bars[f'close_to_high_{w}'] = bars['adj_close'] / bars.groupby('symbol')['adj_close'].transform(
        lambda x: x.rolling(w, min_periods=max(2, w//5)).max().replace(0, np.nan)
    )

# ADV
bars['adv20'] = bars.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())

# Barra代理
bars['size'] = -bars['ln_price']
bars['liquidity'] = -np.log((bars['adv20'] * bars['close']).replace(0, np.nan))
bars['beta'] = bars['vol_60d'] / bars.groupby('trade_date')['vol_60d'].transform('mean').replace(0, np.nan)
bars['volatility'] = bars['vol_60d']
bars['book_to_price'] = 1 / bars['close'].replace(0, np.nan)

# Extended Technical代理
bars['turnover_rate'] = bars['vol_ratio_20']
bars['volume_spike'] = bars['vol_ratio_20']

# Body/Shadow
bars['body'] = abs(bars['close'] - bars['open'])
bars['range'] = bars['high'] - bars['low']
bars['body_ratio'] = bars['body'] / bars['range'].replace(0, np.nan)

print("因子计算完成")

# 计算IC
print("\n计算IC...")

def calculate_ic(panel, factor_col, label_col='fwd_return_20d'):
    if factor_col not in panel.columns or label_col not in panel.columns:
        return np.nan
    valid = panel[[factor_col, label_col]].dropna()
    if len(valid) < 30:
        return np.nan
    try:
        return valid[factor_col].astype(float).corr(valid[label_col].astype(float), method='spearman')
    except:
        return np.nan

# 排除列表
exclude_cols = {
    'trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount',
    'preclose', 'returns', 'fwd_return_20d', 'mkt_ret', 'ln_price', 'adv20', 
    'body', 'range'
}

# 获取因子池
factors = get_all_factors()
factor_map = {f.name: f.source for f in factors}

# 计算所有可用因子的IC
results = []
for col in bars.columns:
    if col in exclude_cols:
        continue
    if bars[col].dtype == 'object':
        continue
    
    ic = calculate_ic(bars, col)
    if not np.isnan(ic):
        source = factor_map.get(col, 'extended_technical' if any(x in col for x in ['ma', 'vol', 'mom', 'rev', 'close_to', 'vol_ratio']) else 'unknown')
        results.append({
            'factor': col,
            'source': source,
            'rank_ic': ic
        })

# 保存结果
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('rank_ic', ascending=False)
results_df.to_csv(CACHE_DIR / 'all_factors_ic_complete.csv', index=False)

print(f"\n计算完成! 共 {len(results_df)} 个因子IC")
print(f"\n有效因子 (IC > 0.02): {len(results_df[results_df['rank_ic'] > 0.02])}")
print(f"可做空因子 (IC < -0.02): {len(results_df[results_df['rank_ic'] < -0.02])}")

# 报告
print("\n" + "=" * 80)
print("因子IC测试报告")
print("=" * 80)

print("\nTop 30 有效因子:")
for i, (_, row) in enumerate(results_df[results_df['rank_ic'] > 0].head(30).iterrows(), 1):
    print(f"{i:2d}. {row['factor']:<30s} IC={row['rank_ic']:+.4f}  ({row['source']})")

print("\n可做空因子 Top 20:")
for i, (_, row) in enumerate(results_df[results_df['rank_ic'] < -0.02].head(20).iterrows(), 1):
    print(f"{i:2d}. {row['factor']:<30s} IC={row['rank_ic']:+.4f}  ({row['source']})")

# 按来源统计
print("\n按来源统计:")
stats = results_df.groupby('source').agg({
    'rank_ic': ['count', 'mean', 'max', 'min']
}).round(4)
stats.columns = ['Count', 'Mean_IC', 'Max_IC', 'Min_IC']
stats['Effective'] = results_df[results_df['rank_ic'] > 0.02].groupby('source').size()
stats['Shortable'] = results_df[results_df['rank_ic'] < -0.02].groupby('source').size()
stats = stats.fillna(0).astype({'Effective': int, 'Shortable': int})
stats = stats.sort_values('Count', ascending=False)
print(stats)

print("\n完成!")
