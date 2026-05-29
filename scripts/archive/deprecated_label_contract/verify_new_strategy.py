"""
新策略发现验证 - 使用Top 10因子
基于第二次策略发现结果
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("新策略发现验证 - Top 10因子")
print("="*70)

# 配置
DATA_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/data/silver")
RUN_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs/hs300_ridge_with_support_20260412_000435_8bad53b6")
OUTPUT_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/strategy_discovery_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 选中因子
SELECTED_FACTORS = [
    'alpha_004', 'alpha_017', 'earnings_yield', 'roe', 'alpha_006',
    'alpha_002', 'ocf_per_share', 'operating_margin', 'asset_turnover', 'equity_growth'
]

# 1. 加载数据
print("\n[1/5] 加载数据...")
bars = pd.read_parquet(DATA_DIR / "daily_bar.parquet")
financial = pd.read_parquet(RUN_DIR / "datasets/model_dataset.parquet")
df = bars.merge(financial[['trade_date', 'symbol', 'roe', 'earnings_yield', 'operating_margin', 
                           'equity_growth', 'ocf_per_share', 'revenue_growth', 'asset_turnover',
                           'gross_margin', 'fwd_return_20d', 'is_tradable']], 
                on=['trade_date', 'symbol'], how='left')
print(f"数据量: {len(df):,}")

# 2. 计算Alpha因子
print("\n[2/5] 计算Alpha因子...")

# Alpha#2
df['alpha_002'] = df.groupby('symbol').apply(
    lambda x: -1 * x['open'].rolling(6, min_periods=3).corr(x['volume'].rank())
).reset_index(level=0, drop=True)

# Alpha#4
df['alpha_004'] = df.groupby('symbol').apply(
    lambda x: -1 * x['low'].rank().rolling(9, min_periods=5).mean()
).reset_index(level=0, drop=True)

# Alpha#6
df['alpha_006'] = df.groupby('symbol').apply(
    lambda x: -1 * x['open'].rolling(10, min_periods=5).corr(x['volume'])
).reset_index(level=0, drop=True)

# Alpha#17
df['alpha_017'] = (-1 * df.groupby('symbol')['close'].transform(
    lambda x: x.rank().rolling(10, min_periods=5).mean()
).rank()) * df.groupby('symbol')['close'].diff().diff().rank()

# 3. Walk-Forward回测
print("\n[3/5] Walk-Forward回测...")

train_window = 500
test_window = 60
step = 30
start_date = df['trade_date'].min() + pd.Timedelta(days=train_window)
end_date = df['trade_date'].max() - pd.Timedelta(days=test_window)

results = []
dates = pd.date_range(start=start_date, end=end_date, freq=f'{step}d')

for i, train_end in enumerate(dates):
    train_start = train_end - pd.Timedelta(days=train_window)
    test_start = train_end
    test_end = test_start + pd.Timedelta(days=test_window)
    
    train_df = df[(df['trade_date'] >= train_start) & (df['trade_date'] < train_start + pd.Timedelta(days=train_window))].copy()
    test_df = df[(df['trade_date'] >= test_start) & (df['trade_date'] < test_end)].copy()
    
    if len(train_df) < 1000 or len(test_df) < 100:
        continue
    
    features = [f for f in SELECTED_FACTORS if f in train_df.columns]
    
    train_valid = train_df.dropna(subset=features + ['fwd_return_20d']).copy()
    if len(train_valid) < 100:
        continue
    
    for f in features:
        train_valid[f] = train_valid[f].replace([np.inf, -np.inf], np.nan)
    train_valid['fwd_return_20d'] = train_valid['fwd_return_20d'].replace([np.inf, -np.inf], np.nan)
    train_valid = train_valid.dropna(subset=features + ['fwd_return_20d'])
    
    if len(train_valid) < 100:
        continue
    
    X_train = train_valid[features].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_valid['fwd_return_20d'].fillna(0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    test_valid = test_df.dropna(subset=features + ['fwd_return_20d']).copy()
    if len(test_valid) < 50:
        continue
    
    for f in features:
        test_valid[f] = test_valid[f].replace([np.inf, -np.inf], np.nan)
    test_valid['fwd_return_20d'] = test_valid['fwd_return_20d'].replace([np.inf, -np.inf], np.nan)
    test_valid = test_valid.dropna(subset=features + ['fwd_return_20d'])
    
    if len(test_valid) < 50:
        continue
    
    X_test = test_valid[features].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_valid['fwd_return_20d'].fillna(0)
    
    X_test_scaled = scaler.transform(X_test)
    pred = model.predict(X_test_scaled)
    
    test_valid = test_valid.copy()
    test_valid['pred'] = pred
    test_valid['label'] = y_test
    
    top_pct = test_valid.nlargest(int(len(test_valid)*0.2), 'pred')
    bottom_pct = test_valid.nsmallest(int(len(test_valid)*0.2), 'pred')
    
    long_return = top_pct['label'].mean()
    short_return = bottom_pct['label'].mean()
    spread_return = long_return - short_return
    
    results.append({
        'train_end': train_end,
        'test_start': test_start,
        'long_return': long_return,
        'short_return': short_return,
        'spread_return': spread_return,
        'n_stocks': len(test_valid)
    })

results_df = pd.DataFrame(results)
print(f"窗口数: {len(results_df)}")

# 4. 计算统计指标
print("\n[4/5] 计算统计指标...")

n_periods_per_year = 252 / test_window
mean_return = results_df['spread_return'].mean() * n_periods_per_year
std_return = results_df['spread_return'].std() * np.sqrt(n_periods_per_year)
sharpe = mean_return / std_return if std_return > 0 else 0

win_rate = (results_df['spread_return'] > 0).mean()

print(f"\n{'='*50}")
print(f"新策略发现验证结果 (Top 10因子)")
print(f"{'='*50}")
print(f"窗口数: {len(results_df)}")
print(f"年化收益: {mean_return*100:.2f}%")
print(f"年化波动: {std_return*100:.2f}%")
print(f"Sharpe比率: {sharpe:.3f}")
print(f"胜率: {win_rate*100:.1f}%")
print(f"平均spread: {results_df['spread_return'].mean()*100:.2f}%")

# 5. 对比历史基线
print("\n[5/5] 对比历史基线...")

# 加载之前的基线结果
baseline_sharpe = 0.150  # Ridge基线OOS
baseline_return = 0.012  # Ridge基线年化收益

print(f"\n对比Ridge基线 (2026-04-12):")
print(f"  OOS Sharpe: {baseline_sharpe:.3f}")
print(f"  OOS Annual Return: {baseline_return*100:.2f}%")
print(f"\n新策略 (Top 10因子):")
print(f"  OOS Sharpe: {sharpe:.3f}")
print(f"  OOS Annual Return: {mean_return*100:.2f}%")

if sharpe > baseline_sharpe * 1.2:
    print(f"\n✓ 新策略显著优于基线 (+{(sharpe/baseline_sharpe-1)*100:.0f}%)")
elif sharpe > baseline_sharpe:
    print(f"\n△ 新策略略优于基线 (+{(sharpe/baseline_sharpe-1)*100:.0f}%)")
else:
    print(f"\n✗ 新策略劣于基线 ({(sharpe/baseline_sharpe-1)*100:.0f}%)")

# 保存结果
results_df.to_csv(OUTPUT_DIR / 'walk_forward_results.csv', index=False)
print(f"\n结果已保存至: {OUTPUT_DIR}")

print("\n" + "="*70)
print("验证完成!")
print("="*70)
