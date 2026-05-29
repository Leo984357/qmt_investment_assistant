"""
第二次策略发现流程 - 基于完整IC测试结果
使用新发现的alpha_004和alpha_017因子

关键发现:
- alpha_004: IC=0.144, IR=0.832 (最强因子!)
- alpha_017: IC=0.094, IR=0.588
- 财务因子: earnings_yield, roe仍稳定
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/data/silver")

print("="*70)
print("第二次策略发现流程 - 基于完整IC测试")
print("="*70)

# 1. 加载数据
print("\n[1/6] 加载数据...")
bars = pd.read_parquet(DATA_DIR / "daily_bar.parquet")
run_dir = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs/hs300_ridge_with_support_20260412_000435_8bad53b6")
financial = pd.read_parquet(run_dir / "datasets/model_dataset.parquet")
df = bars.merge(financial[['trade_date', 'symbol', 'roe', 'earnings_yield', 'operating_margin', 
                           'equity_growth', 'ocf_per_share', 'revenue_growth', 'asset_turnover',
                           'gross_margin', 'cash_ratio', 'mom120', 'fwd_return_20d', 
                           'is_tradable']], 
                on=['trade_date', 'symbol'], how='left')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
print(f"数据量: {len(df):,}, 日期: {df['trade_date'].min()} ~ {df['trade_date'].max()}")

# 2. 定义标签
print("\n[2/6] 计算标签...")
df['fwd_return_20d'] = df.groupby('symbol')['close'].pct_change(20).shift(-20)

# 3. 定义因子 - 基于IC测试结果
print("\n[3/6] 定义候选因子...")
candidate_factors = [
    # Top Alpha (新发现!)
    'alpha_004', 'alpha_017', 'alpha_006', 'alpha_003', 'alpha_002',
    
    # 财务因子
    'earnings_yield', 'roe', 'operating_margin', 'ocf_per_share',
    'asset_turnover', 'equity_growth', 'gross_margin', 'profit_growth',
    
    # 动量
    'mom250', 'mom120',
]

# 计算Alpha因子
print("\n[4/6] 计算Alpha因子...")

# Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close-open)/open)), 6))
df['alpha_002'] = df.groupby('symbol').apply(
    lambda x: -1 * x['open'].rolling(6, min_periods=3).corr(x['volume'].rank())
).reset_index(level=0, drop=True)

# Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
df['alpha_003'] = df.groupby('symbol').apply(
    lambda x: -1 * x['open'].rank().rolling(10, min_periods=5).corr(x['volume'].rank())
).reset_index(level=0, drop=True)

# Alpha#4: (-1 * Ts_Rank(rank(low), 9)) - 简化版
df['alpha_004'] = df.groupby('symbol').apply(
    lambda x: -1 * x['low'].rank().rolling(9, min_periods=5).mean()
).reset_index(level=0, drop=True)

# Alpha#6: (-1 * correlation(open, volume, 10))
df['alpha_006'] = df.groupby('symbol').apply(
    lambda x: -1 * x['open'].rolling(10, min_periods=5).corr(x['volume'])
).reset_index(level=0, drop=True)

# Alpha#17: ((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)))
df['alpha_017'] = (-1 * df.groupby('symbol')['close'].transform(
    lambda x: x.rank().rolling(10, min_periods=5).mean()
).rank()) * df.groupby('symbol')['close'].diff().diff().rank()

# 动量因子
for window in [5, 20, 60, 120, 250]:
    df[f'mom{window}'] = df.groupby('symbol')['close'].pct_change(window)

# 4. 筛选有效因子
print("\n[5/6] 筛选有效因子...")
valid_factors = [f for f in candidate_factors if f in df.columns]
print(f"有效候选因子: {len(valid_factors)}个")

# 计算每个因子的IC
ic_results = []
for factor in valid_factors:
    valid_df = df.dropna(subset=[factor, 'fwd_return_20d'])
    valid_df = valid_df.copy()
    valid_df[factor] = pd.to_numeric(valid_df[factor], errors='coerce')
    valid_df = valid_df.dropna(subset=[factor])
    
    if len(valid_df) < 1000:
        continue
    
    daily_ics = []
    for date in valid_df['trade_date'].unique():
        group = valid_df[valid_df['trade_date'] == date]
        if len(group) < 30:
            continue
        if group[factor].std() < 1e-8 or group['fwd_return_20d'].std() < 1e-8:
            continue
        
        score_rank = group[factor].rank(pct=True)
        label_rank = group['fwd_return_20d'].rank(pct=True)
        ic = score_rank.corr(label_rank)
        daily_ics.append(ic)
    
    if len(daily_ics) >= 60:
        ic_mean = np.mean(daily_ics)
        ic_std = np.std(daily_ics)
        ic_results.append({
            'factor': factor,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_mean / max(ic_std, 0.001),
            'positive_rate': np.mean([x > 0 for x in daily_ics])
        })

ic_df = pd.DataFrame(ic_results).sort_values('ic_ir', ascending=False)
print("\nIC排名:")
print(ic_df.head(15).to_string(index=False))

# 5. 选择因子
print("\n[6/6] 选择最优因子组合...")
# 选择IC IR > 0.15的因子
selected = ic_df[ic_df['ic_ir'] > 0.15]['factor'].tolist()
if len(selected) < 3:
    selected = ic_df.head(5)['factor'].tolist()

print(f"\n选中因子 ({len(selected)}个): {selected}")

# 去冗余
print("\n因子去冗余...")
corr_matrix = df[selected].corr()
high_corr_pairs = []
for i in range(len(selected)):
    for j in range(i+1, len(selected)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            f1, f2 = selected[i], selected[j]
            ir1 = ic_df[ic_df['factor'] == f1]['ic_ir'].values[0]
            ir2 = ic_df[ic_df['factor'] == f2]['ic_ir'].values[0]
            keep = f1 if ir1 > ir2 else f2
            remove = f2 if ir1 > ir2 else f1
            high_corr_pairs.append((f1, f2, corr_matrix.iloc[i, j], keep, remove))

for f1, f2, corr, keep, remove in high_corr_pairs:
    if remove in selected:
        selected.remove(remove)
        print(f"  剔除{remove} (与{keep}相关性{corr:.2f})")

print(f"\n最终因子 ({len(selected)}个): {selected}")

# 保存结果
import json
result = {
    'selected_factors': selected,
    'ic_results': ic_df.to_dict('records'),
    'high_corr_pairs': [(f1, f2, corr, keep, remove) for f1, f2, corr, keep, remove in high_corr_pairs]
}
with open('artifacts/strategy_discovery_v2/selected_factors.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*70)
print("第二次策略发现完成!")
print("="*70)
