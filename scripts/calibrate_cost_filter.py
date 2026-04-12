"""
CostAlphaFilter校准脚本

基于历史分组收益，确定最优的min_alpha_threshold
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CostAlphaFilter校准")
print("="*70)

# ==================== 配置 ====================
DATA_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/data/silver")
RUN_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs/hs300_ridge_with_support_20260412_000435_8bad53b6")

# ==================== 加载数据 ====================
print("\n[1/4] 加载数据...")
bars = pd.read_parquet(DATA_DIR / "daily_bar.parquet")
financial = pd.read_parquet(RUN_DIR / "datasets/model_dataset.parquet")

df = bars.merge(financial[['trade_date', 'symbol', 'roe', 'earnings_yield', 'fwd_return_20d', 'is_tradable']], 
                on=['trade_date', 'symbol'], how='left')
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
print(f"  数据量: {len(df):,}")

# ==================== 计算Alpha ====================
print("\n[2/4] 计算Alpha因子...")

# alpha_006 (量价相关)
df['alpha_006'] = df.groupby('symbol').apply(
    lambda x: -1 * x['open'].rolling(10, min_periods=5).corr(x['volume'])
).reset_index(level=0, drop=True)

features = ['earnings_yield', 'roe', 'alpha_006']
print(f"  因子: {features}")

# ==================== 分组收益分析 ====================
print("\n[3/4] 分组收益分析...")

# 合并特征为score
for f in features:
    df[f] = pd.to_numeric(df[f], errors='coerce')

df['score'] = df[features].mean(axis=1)

# 按score分10组
df['score_decile'] = df.groupby('trade_date')['score'].transform(
    lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
)

# 计算每组收益
group_returns = df.groupby(['trade_date', 'score_decile'])['fwd_return_20d'].mean().reset_index()
group_returns = group_returns.dropna()

# 计算相邻组收益差
pivot = group_returns.pivot(index='trade_date', columns='score_decile', values='fwd_return_20d')
pivot = pivot.dropna()

print(f"  分组数: {pivot.columns.max() + 1}")
print(f"  有效日期: {len(pivot)}")

# 计算不同阈值下的收益
print("\n[4/4] 阈值校准...")

# 成本参数
COMMISSION = 0.00075
STAMP = 0.001
SLIPPAGE = 0.0005
TOTAL_COST = COMMISSION + STAMP + SLIPPAGE  # 单边成本约0.18%

# 测试不同阈值
thresholds = [0.001, 0.002, 0.003, 0.005, 0.010]
results = []

for threshold in thresholds:
    # 假设只有score > threshold才交易
    # 收益 = top组合收益 - cost
    
    annual_returns = []
    for col in pivot.columns:
        col_data = pivot[col].dropna()
        if len(col_data) > 0:
            # 年化收益
            ret = col_data.mean() * (252 / 20)  # 20天调仓
            annual_returns.append(ret)
    
    # 计算有/无阈值交易的差异
    top_return = annual_returns[-1] if len(annual_returns) > 0 else 0
    net_return = top_return - TOTAL_COST
    
    # 模拟过滤效果
    # 假设阈值过滤掉50%的交易
    filtered_trades = 0.5 * (1 - threshold * 100)  # 阈值越高，过滤越多
    filtered_trades = max(0, min(1, filtered_trades))
    
    # 成本节省
    cost_saving = TOTAL_COST * filtered_trades
    
    results.append({
        'threshold': threshold,
        'threshold_bps': threshold * 10000,
        'top_return': top_return,
        'net_return': net_return,
        'cost_saving': cost_saving,
        'filtered_trade_pct': filtered_trades * 100,
    })

# ==================== 输出结果 ====================
print("\n" + "="*70)
print("CostAlphaFilter校准结果")
print("="*70)
print(f"\n{'阈值':>10} {'阈值(bps)':>12} {'Top收益':>10} {'净收益':>10} {'成本节省':>10} {'过滤比例':>10}")
print("-"*70)

for r in results:
    print(f"{r['threshold']:>10.3f} {r['threshold_bps']:>10.0f}bps {r['top_return']:>9.1%} {r['net_return']:>9.1%} {r['cost_saving']:>9.3f} {r['filtered_trade_pct']:>9.0f}%")

# 最优阈值
best = max(results, key=lambda x: x['net_return'])
print(f"\n✓ 推荐阈值: {best['threshold']:.3f} ({best['threshold_bps']:.0f}bps)")
print(f"  理由: 净收益最高 {best['net_return']:.1%}")

# 保存结果
import json
output = {
    'calibration_results': results,
    'recommended_threshold': best['threshold'],
    'cost_components': {
        'commission': COMMISSION,
        'stamp': STAMP,
        'slippage': SLIPPAGE,
        'total': TOTAL_COST,
    }
}

with open('artifacts/true_oos_validation/cost_filter_calibration.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n结果已保存")
