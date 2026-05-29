"""
DEPRECATED / DO NOT USE FOR FORMAL RESEARCH
===========================================
此脚本使用已废弃的标签口径和禁止的因子(IC结果不可用于正式结论)。

禁止使用的因子: alpha_017, alpha_006, alpha_004, alpha_002
原因: 使用旧标签口径 pct_change(20).shift(-20)

正式研究入口: python -m src.cli experiment --config configs/experiments/<name>.yaml

真实的Walk-Forward OOS验证 - 简化版

每个窗口：
1. 用过去500天数据训练Ridge模型
2. 在未来60天进行预测
3. 选出top 15，计算等权组合收益
4. 扣除估算交易成本
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("真实Walk-Forward OOS验证 (简化版)")
print("="*70)

# ==================== 配置 ====================
DATA_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/data/silver")
RUN_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs/hs300_ridge_with_support_20260412_000435_8bad53b6")
OUTPUT_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/true_oos_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 验证因子
VALIDATION_FACTORS = [
    'alpha_004', 'alpha_017', 'earnings_yield', 'roe', 'alpha_006',
    'alpha_002', 'ocf_per_share', 'operating_margin', 'asset_turnover', 'equity_growth'
]

# Walk-Forward配置
TRAIN_WINDOW = 500
TEST_WINDOW = 60
STEP = 30
TOP_N = 15

# 成本配置
COMMISSION = 0.00075
STAMP = 0.001
SLIPPAGE = 0.0005

# ==================== 数据加载 ====================
print("\n[1/5] 加载数据...")

bars = pd.read_parquet(DATA_DIR / "daily_bar.parquet")
financial = pd.read_parquet(RUN_DIR / "datasets/model_dataset.parquet")

df = bars.merge(financial[['trade_date', 'symbol', 'roe', 'earnings_yield', 'operating_margin', 
                           'equity_growth', 'ocf_per_share', 'revenue_growth', 'asset_turnover',
                           'gross_margin', 'fwd_return_20d', 'is_tradable']], 
                on=['trade_date', 'symbol'], how='left')

df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

# ==================== 计算Alpha因子 ====================
print("\n[2/5] 计算Alpha因子...")

df['alpha_002'] = df.groupby('symbol').apply(
    lambda x: -1 * x['open'].rolling(6, min_periods=3).corr(x['volume'].rank())
).reset_index(level=0, drop=True)

df['alpha_004'] = df.groupby('symbol').apply(
    lambda x: -1 * x['low'].rank().rolling(9, min_periods=5).mean()
).reset_index(level=0, drop=True)

df['alpha_006'] = df.groupby('symbol').apply(
    lambda x: -1 * x['open'].rolling(10, min_periods=5).corr(x['volume'])
).reset_index(level=0, drop=True)

df['alpha_017'] = (-1 * df.groupby('symbol')['close'].transform(
    lambda x: x.rank().rolling(10, min_periods=5).mean()
).rank()) * df.groupby('symbol')['close'].diff().diff().rank()

print(f"  数据范围: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")

# ==================== Walk-Forward ====================
print("\n[3/5] 执行Walk-Forward验证...")

dates = sorted(df['trade_date'].unique())
features = [f for f in VALIDATION_FACTORS if f in df.columns]

results = []
for i in range(TRAIN_WINDOW, len(dates) - TEST_WINDOW, STEP):
    train_end = dates[i]
    test_start = dates[i + 1]
    test_end = dates[min(i + TEST_WINDOW, len(dates) - 1)]
    
    # 训练数据
    train_df = df[(df['trade_date'] <= train_end)].copy()
    train_valid = train_df.dropna(subset=features + ['fwd_return_20d']).copy()
    train_valid = train_valid[train_valid['is_tradable'] == True]
    
    for f in features:
        train_valid[f] = pd.to_numeric(train_valid[f], errors='coerce')
    train_valid = train_valid.dropna(subset=features)
    
    if len(train_valid) < 200:
        continue
    
    # 训练模型
    X_train = train_valid[features].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_valid['fwd_return_20d'].fillna(0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # 测试数据 - 只看调仓日
    test_df = df[(df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)].copy()
    
    # 获取调仓日的预测
    rebalance_dates = sorted(test_df['trade_date'].unique())[::10]  # 每10天调仓
    
    portfolio_returns = []
    total_cost = 0
    prev_selected = None
    
    for rdate in rebalance_dates:
        day_df = test_df[test_df['trade_date'] == rdate].dropna(subset=features + ['close']).copy()
        
        for f in features:
            day_df[f] = pd.to_numeric(day_df[f], errors='coerce')
        day_df = day_df.dropna(subset=features)
        
        if len(day_df) < 50:
            continue
        
        # 预测
        X_test = day_df[features].fillna(0).replace([np.inf, -np.inf], 0)
        X_test_scaled = scaler.transform(X_test)
        day_df['pred'] = model.predict(X_test_scaled)
        
        # 选择top N
        selected = day_df.nlargest(TOP_N, 'pred')
        current_selected = set(selected['symbol'].values)
        
        # 计算换手率和成本
        if prev_selected is not None:
            changed = len(prev_selected - current_selected) + len(current_selected - prev_selected)
            turnover = changed / TOP_N
            cost_rate = (COMMISSION + STAMP + SLIPPAGE) * turnover / 2  # 平均买卖成本
            total_cost += cost_rate
        
        prev_selected = current_selected
        
        # 获取这些股票在测试窗口结束时的价格
        end_df = test_df[(test_df['trade_date'] == test_end)]
        
        stock_returns = []
        for _, row in selected.iterrows():
            sym = row['symbol']
            buy_price = row['close']
            
            end_row = end_df[end_df['symbol'] == sym]
            if len(end_row) > 0 and buy_price > 0:
                sell_price = end_row['close'].values[0]
                ret = (sell_price / buy_price) - 1
                stock_returns.append(ret)
        
        if stock_returns:
            period_return = np.mean(stock_returns)
            portfolio_returns.append(period_return)
    
    if portfolio_returns:
        # 组合收益 = 各期收益的平均 - 成本
        avg_return = np.mean(portfolio_returns)
        window_return = avg_return - total_cost
        
        # 计算IC
        first_date = rebalance_dates[0]
        first_day_df = test_df[test_df['trade_date'] == first_date].dropna(subset=features + ['fwd_return_20d']).copy()
        for f in features:
            first_day_df[f] = pd.to_numeric(first_day_df[f], errors='coerce')
        first_day_df = first_day_df.dropna(subset=features)
        
        ic = 0
        if len(first_day_df) >= 30:
            X_first = first_day_df[features].fillna(0).replace([np.inf, -np.inf], 0)
            X_first_scaled = scaler.transform(X_first)
            pred = pd.Series(model.predict(X_first_scaled))
            ic = pred.rank().corr(pd.Series(first_day_df['fwd_return_20d'].fillna(0).values).rank())
        
        results.append({
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'portfolio_return': window_return,
            'avg_period_return': avg_return,
            'total_cost': total_cost,
            'n_periods': len(portfolio_returns),
            'test_ic': ic,
        })

results_df = pd.DataFrame(results)
print(f"  有效窗口数: {len(results_df)}")

# ==================== 统计分析 ====================
print("\n[4/5] 统计分析...")

# 过滤异常值
returns = results_df['portfolio_return'].values
returns = returns[np.isfinite(returns)]

# 过滤极端值 (> 3倍标准差)
mean_ret = np.mean(returns)
std_ret = np.std(returns)
mask = np.abs(returns - mean_ret) < 3 * std_ret
valid_returns = returns[mask]

mean_return = np.mean(valid_returns)
std_return = np.std(valid_returns)
n_periods = len(valid_returns)
sharpe = mean_return / std_return * np.sqrt(n_periods / 12) if std_return > 0 and n_periods > 0 else 0

win_rate = np.mean(valid_returns > 0)
worst = np.min(valid_returns)
best = np.max(valid_returns)

print(f"\n{'='*60}")
print("真实Walk-Forward OOS结果 (过滤异常后)")
print('='*60)
print(f"有效窗口数: {len(valid_returns)} (原始{len(returns)}, 过滤{len(returns)-len(valid_returns)})")
print()
print("【收益统计】")
print(f"  平均单期收益: {mean_return*100:.2f}%")
print(f"  收益标准差: {std_return*100:.2f}%")
print(f"  年化收益: {mean_return * (252/TEST_WINDOW) * 100:.2f}%")
print(f"  Sharpe比率: {sharpe:.3f}")
print()
print("【稳定性】")
print(f"  胜率: {win_rate*100:.1f}%")
print(f"  最差窗口: {worst*100:.2f}%")
print(f"  最好窗口: {best*100:.2f}%")
print(f"  亏损窗口: {np.sum(valid_returns < 0)}/{len(valid_returns)}")

# IC统计
print("\n【IC统计】")
print(f"  平均IC: {results_df['test_ic'].mean():.4f}")
print(f"  IC IR: {results_df['test_ic'].mean() / results_df['test_ic'].std():.3f}" if results_df['test_ic'].std() > 0 else "  IC IR: N/A")

# 按年份
print("\n【按年份分解】")
results_df['year'] = pd.to_datetime(results_df['test_start']).dt.year
for year, group in results_df.groupby('year'):
    if len(group) > 0:
        yr_returns = group['portfolio_return'].values
        yr_returns = yr_returns[np.isfinite(yr_returns)]
        if len(yr_returns) > 0:
            yr_mean = np.mean(yr_returns)
            yr_std = np.std(yr_returns)
            yr_sharpe = yr_mean / yr_std * np.sqrt(len(yr_returns) / 12) if yr_std > 0 else 0
            print(f"  {year}: 收益={yr_mean*100:+.1f}%, Sharpe={yr_sharpe:.2f}, n={len(yr_returns)}")

# ==================== 对比基线 ====================
print("\n[5/5] 对比基线...")

ridge_sharpe = 0.150
ridge_return = 0.0103

print(f"\n{'='*60}")
print("与Ridge基线对比 (真实OOS)")
print('='*60)
print(f"{'指标':<20} {'Ridge基线':>12} {'新策略(Top10)':>12}")
print('-'*60)
print(f"{'Sharpe':<20} {ridge_sharpe:>12.3f} {sharpe:>12.3f}")
print(f"{'年化收益':<20} {ridge_return*100:>11.1f}% {(mean_return * (252/TEST_WINDOW))*100:>11.1f}%")
print(f"{'胜率':<20} {'59.3%':>12} {win_rate*100:>11.1f}%")

# ==================== 保存 ====================
results_df.to_csv(OUTPUT_DIR / 'window_results.csv', index=False)

import json
summary = {
    'validation_factors': VALIDATION_FACTORS,
    'n_windows': len(valid_returns),
    'portfolio_return': {
        'mean': float(mean_return),
        'std': float(std_return),
        'sharpe': float(sharpe),
        'annualized': float(mean_return * (252/TEST_WINDOW)),
        'win_rate': float(win_rate),
        'worst': float(worst),
        'best': float(best),
    },
    'ic_stats': {
        'mean': float(results_df['test_ic'].mean()),
        'ir': float(results_df['test_ic'].mean() / results_df['test_ic'].std()) if results_df['test_ic'].std() > 0 else 0,
    },
    'baseline_comparison': {
        'ridge_sharpe': ridge_sharpe,
        'new_sharpe': float(sharpe),
        'improvement_pct': float((sharpe - ridge_sharpe) / ridge_sharpe * 100) if ridge_sharpe > 0 else 0,
    }
}

with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存至: {OUTPUT_DIR}")

print("\n" + "="*70)
print("验证完成")
print("="*70)
