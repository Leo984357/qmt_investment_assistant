"""
alpha_004 原版 vs 简化版 IC对比验证

原版公式: (-1 * Ts_Rank(rank(low), 9))
- rank(low): 截面排名
- Ts_Rank(..., 9): 过去9天的排名滚动值

简化版问题: 用rolling mean代替Ts_Rank，概念不同
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("alpha_004 原版 vs 简化版 IC对比")
print("="*70)

# ==================== 配置 ====================
DATA_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/data/silver")
RUN_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs/hs300_ridge_with_support_20260412_000435_8bad53b6")

# ==================== 数据加载 ====================
print("\n[1/4] 加载数据...")
bars = pd.read_parquet(DATA_DIR / "daily_bar.parquet")
financial = pd.read_parquet(RUN_DIR / "datasets/model_dataset.parquet")

df = bars.merge(financial[['trade_date', 'symbol', 'fwd_return_20d']], 
                on=['trade_date', 'symbol'], how='left')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
print(f"  数据量: {len(df):,}")

# ==================== 原版公式 ====================
print("\n[2/4] 计算原版 alpha_004...")

def alpha_004_original(s: pd.Series, low_col: pd.Series) -> pd.Series:
    """
    原版Ts_Rank实现
    
    Ts_Rank(X, N):
    - 对过去N个周期，计算X的排名
    - 返回当前时刻的ts_rank值
    
    实现方式:
    1. 计算当前值在历史窗口中的排名百分位
    2. 滚动计算
    """
    result = pd.Series(index=s.index, dtype=float)
    
    for i in range(len(s)):
        if i < 8:  # 不足9天用已有数据
            window = s.iloc[max(0, i-8):i+1]
        else:
            window = s.iloc[i-8:i+1]
        
        current_val = s.iloc[i]
        if len(window) > 0:
            # 当前值在窗口中的排名
            rank = (window <= current_val).sum() / len(window)
            result.iloc[i] = -rank  # 取负值
    
    return result

# 优化：向量化实现
def alpha_004_original_vectorized(df: pd.DataFrame) -> pd.Series:
    """原版 alpha_004 向量化实现"""
    low = df['low']
    n = 9
    
    result = pd.Series(index=df.index, dtype=float)
    
    # 按symbol分组
    for sym, group in df.groupby('symbol'):
        low_vals = group['low'].values
        ranks = np.full(len(low_vals), np.nan)
        
        for i in range(len(low_vals)):
            start_idx = max(0, i - n + 1)
            window = low_vals[start_idx:i+1]
            if len(window) > 0:
                current = low_vals[i]
                ranks[i] = -np.mean(window <= current)  # 排名百分位
        
        result.loc[group.index] = ranks
    
    return result

print("  计算中（向量化版本）...")
df['alpha_004_original'] = alpha_004_original_vectorized(df)

# ==================== 简化版公式 ====================
print("\n[3/4] 计算简化版 alpha_004...")

def alpha_004_simplified(df: pd.DataFrame) -> pd.Series:
    """简化版 alpha_004"""
    def calc(s):
        return -s['low'].rank().rolling(9, min_periods=5).mean()
    return df.groupby('symbol', group_keys=False).apply(calc)

df['alpha_004_simplified'] = alpha_004_simplified(df)

# ==================== IC对比 ====================
print("\n[4/4] IC对比...")

def calc_rank_ic(df: pd.DataFrame, factor_col: str, label_col: str = 'fwd_return_20d') -> dict:
    """计算横截面RankIC"""
    valid_df = df.dropna(subset=[factor_col, label_col]).copy()
    valid_df[factor_col] = pd.to_numeric(valid_df[factor_col], errors='coerce')
    valid_df = valid_df.dropna(subset=[factor_col])
    
    if len(valid_df) < 100:
        return {}
    
    daily_ics = []
    for date in valid_df['trade_date'].unique():
        group = valid_df[valid_df['trade_date'] == date]
        if len(group) < 30:
            continue
        
        score_rank = group[factor_col].rank(pct=True)
        label_rank = group[label_col].rank(pct=True)
        
        ic = score_rank.corr(label_rank)
        if not np.isnan(ic):
            daily_ics.append(ic)
    
    if len(daily_ics) < 60:
        return {}
    
    return {
        'ic_mean': np.mean(daily_ics),
        'ic_std': np.std(daily_ics),
        'ic_ir': np.mean(daily_ics) / max(np.std(daily_ics), 0.001),
        'positive_rate': np.mean([x > 0 for x in daily_ics]),
        'n_days': len(daily_ics),
    }

results_original = calc_rank_ic(df, 'alpha_004_original')
results_simplified = calc_rank_ic(df, 'alpha_004_simplified')

print("\n" + "="*60)
print("IC对比结果")
print("="*60)
print(f"\n{'版本':<20} {'IC均值':>10} {'IC IR':>10} {'正率':>10} {'天数':>8}")
print("-"*60)
print(f"{'原版(Ts_Rank)':<20} {results_original.get('ic_mean', 0):>10.4f} {results_original.get('ic_ir', 0):>10.4f} {results_original.get('positive_rate', 0):>9.1%} {results_original.get('n_days', 0):>8}")
print(f"{'简化版(rolling)':<20} {results_simplified.get('ic_mean', 0):>10.4f} {results_simplified.get('ic_ir', 0):>10.4f} {results_simplified.get('positive_rate', 0):>9.1%} {results_simplified.get('n_days', 0):>8}")

# 差异分析
if results_original and results_simplified:
    ic_diff = results_original['ic_mean'] - results_simplified['ic_mean']
    ir_diff = results_original['ic_ir'] - results_simplified['ic_ir']
    
    print(f"\n{'差异':<20} {ic_diff:>+10.4f} {ir_diff:>+10.4f}")
    
    print("\n" + "="*60)
    print("结论")
    print("="*60)
    
    if abs(ic_diff) < 0.005:
        print("✓ 两版本IC差异小于0.5%，简化版可接受")
    elif ic_diff > 0.005:
        print("⚠ 原版IC更高，简化版可能高估约{:.1f}%".format(ic_diff/results_original['ic_mean']*100))
    else:
        print("⚠ 简化版IC更高，可能存在偏差")
    
    # 判断哪个版本更稳定
    if results_original['ic_ir'] > results_simplified['ic_ir']:
        print("✓ 原版IR更高，稳定性更好")
    else:
        print("⚠ 简化版IR更高，需进一步验证")

# 保存结果
import json
output = {
    'original': results_original,
    'simplified': results_simplified,
    'comparison': {
        'ic_diff': results_original.get('ic_mean', 0) - results_simplified.get('ic_mean', 0),
        'ir_diff': results_original.get('ic_ir', 0) - results_simplified.get('ic_ir', 0),
    }
}
with open('artifacts/true_oos_validation/alpha_004_comparison.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n结果已保存")
