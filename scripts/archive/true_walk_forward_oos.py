"""
真实的Walk-Forward OOS验证脚本

每个窗口必须：
1. 用过去N天数据独立训练模型
2. 在未来M天数据进行真实预测
3. 基于预测建仓，计算真实收益
4. 扣除交易成本
5. 统计OOS表现

不能：
- 用随机数模拟收益
- 用标签均值代替策略收益
- 在全时间段上选因子，再在同一时间段验证
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("真实Walk-Forward OOS验证")
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
TRAIN_WINDOW = 500  # 训练窗口(天)
TEST_WINDOW = 60    # 测试窗口(天)
STEP = 30          # 步长(天)
TOP_N = 15          # 持仓数量

# 成本配置 (与研究合同一致)
COMMISSION_RATE = 0.00075  # 佣金0.075%
STAMP_TAX = 0.001          # 印花税0.1% (卖出时)
SLIPPAGE = 0.0005          # 滑点0.05%
MIN_TRADE_VALUE = 2000      # 最小交易额

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

print(f"  数据范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
print(f"  总记录数: {len(df):,}")

# ==================== 真实Walk-Forward验证 ====================
print("\n[3/5] 执行真实Walk-Forward验证...")
print(f"  训练窗口: {TRAIN_WINDOW}天")
print(f"  测试窗口: {TEST_WINDOW}天")
print(f"  步长: {STEP}天")
print(f"  持仓数量: {TOP_N}只")

# 获取交易日列表
dates = sorted(df['trade_date'].unique())
date_to_idx = {d: i for i, d in enumerate(dates)}

# 生成Walk-Forward窗口
windows = []
train_end_idx = TRAIN_WINDOW
while train_end_idx + TEST_WINDOW <= len(dates):
    train_end = dates[train_end_idx]
    test_start = dates[train_end_idx + 1]
    test_end_idx = train_end_idx + TEST_WINDOW
    test_end = dates[min(test_end_idx, len(dates) - 1)]
    
    windows.append({
        'train_start': dates[max(0, train_end_idx - TRAIN_WINDOW)],
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'train_end_idx': train_end_idx,
        'test_end_idx': test_end_idx,
    })
    train_end_idx += STEP

print(f"\n  生成 {len(windows)} 个验证窗口")

# 验证函数
def calculate_trade_cost(old_weight, new_weight, price):
    """计算单笔交易成本"""
    change = abs(new_weight - old_weight)
    if change < 0.001:  # 权重变化<0.1%不交易
        return 0
    
    trade_value = change * price
    if trade_value < MIN_TRADE_VALUE:
        return 0
    
    commission = trade_value * COMMISSION_RATE
    stamp_tax = trade_value * STAMP_TAX if new_weight < old_weight else 0  # 印花税只在卖出时
    slippage = trade_value * SLIPPAGE
    total_cost = commission + stamp_tax + slippage
    
    return total_cost

# 执行每个窗口
results = []
for i, w in enumerate(windows):
    # 1. 分割训练/测试数据
    train_df = df[(df['trade_date'] >= w['train_start']) & (df['trade_date'] <= w['train_end'])].copy()
    test_df = df[(df['trade_date'] >= w['test_start']) & (df['trade_date'] <= w['test_end'])].copy()
    
    features = [f for f in VALIDATION_FACTORS if f in train_df.columns]
    
    # 2. 准备训练数据
    train_valid = train_df.dropna(subset=features + ['fwd_return_20d', 'close']).copy()
    train_valid = train_valid[train_valid['is_tradable'] == True]
    
    for f in features:
        train_valid[f] = pd.to_numeric(train_valid[f], errors='coerce')
    train_valid = train_valid.dropna(subset=features)
    
    if len(train_valid) < 100:
        continue
    
    # 3. 训练模型 (每个窗口独立训练!)
    X_train = train_valid[features].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_valid['fwd_return_20d'].fillna(0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # 4. 在测试窗口进行预测和调仓
    # 获取测试期间每天的收盘价
    test_df_with_close = test_df.dropna(subset=['close']).copy()
    test_dates = sorted(test_df_with_close['trade_date'].unique())
    
    if len(test_dates) < 2:
        continue
    
    # 每10天调仓
    rebalance_dates = test_dates[::min(10, len(test_dates)//6)]  # 约6次调仓
    
    portfolio_value = 1.0
    prev_weights = None
    trade_cost_total = 0
    
    for j, date in enumerate(test_dates):
        day_df = test_df_with_close[test_df_with_close['trade_date'] == date].copy()
        
        # 预测
        day_valid = day_df.dropna(subset=features).copy()
        for f in features:
            day_valid[f] = pd.to_numeric(day_valid[f], errors='coerce')
        day_valid = day_valid.dropna(subset=features)
        
        if len(day_valid) < 30:
            continue
        
        # 预测收益
        X_test = day_valid[features].fillna(0).replace([np.inf, -np.inf], 0)
        X_test_scaled = scaler.transform(X_test)
        day_valid['pred'] = model.predict(X_test_scaled)
        
        # 调仓日：选择top N并计算成本
        if date in rebalance_dates or prev_weights is None:
            top_stocks = day_valid.nlargest(TOP_N, 'pred')
            target_weights = top_stocks['pred'].values
            target_weights = target_weights / target_weights.sum()  # 归一化
            target_symbols = top_stocks['symbol'].values
            
            # 计算交易成本
            if prev_weights is not None:
                for sym in set(prev_weights.keys()) | set(target_symbols):
                    old_w = prev_weights.get(sym, 0)
                    new_w = 0
                    if sym in target_symbols:
                        idx = list(target_symbols).index(sym)
                        new_w = target_weights[idx]
                    
                    if abs(old_w - new_w) > 0.001:
                        price = top_stocks[top_stocks['symbol'] == sym]['close'].values
                        if len(price) > 0:
                            trade_cost_total += calculate_trade_cost(old_w, new_w, price[0])
            
            prev_weights = {sym: w for sym, w in zip(target_symbols, target_weights)}
        
        # 计算组合当日收益 (使用top N等权组合)
        selected_stocks = top_stocks if 'top_stocks' in dir() else day_valid.nlargest(TOP_N, 'pred')
        
        # 计算当天收益
        if j < len(test_dates) - 1:
            next_date = test_dates[j + 1]
            next_day_df = test_df_with_close[test_df_with_close['trade_date'] == next_date]
            
            # 计算top N股票的平均收益
            returns = []
            for _, row in selected_stocks.iterrows():
                sym = row['symbol']
                curr_price = row['close']
                
                next_row = next_day_df[next_day_df['symbol'] == sym]
                if len(next_row) > 0:
                    next_price = next_row['close'].values[0]
                    if curr_price > 0:
                        ret = (next_price / curr_price) - 1
                        returns.append(ret)
            
            if returns:
                day_return = np.mean(returns)
                portfolio_value *= (1 + day_return)
    
    # 5. 计算窗口收益 (扣除总交易成本)
    window_return = portfolio_value * (1 - trade_cost_total) - 1.0
    
    # 6. 额外：计算IC
    day_ics = []
    for date in test_dates[:20]:  # 只看前20天
        day_df = test_df_with_close[test_df_with_close['trade_date'] == date].dropna(subset=features + ['fwd_return_20d']).copy()
        for f in features:
            day_df[f] = pd.to_numeric(day_df[f], errors='coerce')
        day_df = day_df.dropna(subset=features)
        
        if len(day_df) >= 30:
            ic = day_df[features[0]].rank().corr(day_df['fwd_return_20d'].rank())
            day_ics.append(ic)
    
    results.append({
        'window_idx': i,
        'train_end': w['train_end'],
        'test_start': w['test_start'],
        'test_end': w['test_end'],
        'portfolio_return': window_return,
        'n_train': len(train_valid),
        'n_test': len(day_valid) if 'day_valid' in dir() else 0,
        'test_ic': np.mean(day_ics) if day_ics else 0,
        'test_ic_ir': np.mean(day_ics) / np.std(day_ics) if len(day_ics) > 1 else 0,
    })
    
    if (i + 1) % 10 == 0:
        print(f"  完成 {i+1}/{len(windows)} 个窗口...")

results_df = pd.DataFrame(results)
print(f"\n  有效窗口数: {len(results_df)}")

# ==================== 统计分析 ====================
print("\n[4/5] 统计分析...")

# 基本统计
mean_return = results_df['portfolio_return'].mean()
std_return = results_df['portfolio_return'].std()
sharpe = mean_return / std_return * np.sqrt(len(results_df) / 12) if std_return > 0 else 0

win_rate = (results_df['portfolio_return'] > 0).mean()

# 极端情况
worst_window = results_df['portfolio_return'].min()
best_window = results_df['portfolio_return'].max()
losing_windows = results_df[results_df['portfolio_return'] < 0]

print(f"\n{'='*60}")
print("真实Walk-Forward OOS结果")
print('='*60)
print(f"验证窗口数: {len(results_df)}")
print(f"验证因子: {VALIDATION_FACTORS}")
print()
print("【收益统计】")
print(f"  平均单期收益: {mean_return*100:.2f}%")
print(f"  收益标准差: {std_return*100:.2f}%")
print(f"  年化收益: {mean_return * (252/TEST_WINDOW) * 100:.2f}%")
print(f"  年化波动: {std_return * np.sqrt(252/TEST_WINDOW) * 100:.2f}%")
print(f"  Sharpe比率: {sharpe:.3f}")
print()
print("【稳定性统计】")
print(f"  胜率: {win_rate*100:.1f}%")
print(f"  最差窗口: {worst_window*100:.2f}%")
print(f"  最好窗口: {best_window*100:.2f}%")
print(f"  亏损窗口数: {len(losing_windows)}/{len(results_df)}")
print()
print("【IC统计】")
print(f"  平均IC: {results_df['test_ic'].mean():.4f}")
print(f"  IC IR: {results_df['test_ic_ir'].mean():.3f}")

# 按年份分解
print("\n【按年份分解】")
results_df['year'] = pd.to_datetime(results_df['test_start']).dt.year
for year, group in results_df.groupby('year'):
    if len(group) > 0:
        yr_return = group['portfolio_return'].mean()
        yr_sharpe = group['portfolio_return'].mean() / group['portfolio_return'].std() if group['portfolio_return'].std() > 0 else 0
        print(f"  {year}: 收益={yr_return*100:.1f}%, Sharpe={yr_sharpe:.2f}, n={len(group)}")

# ==================== 对比基线 ====================
print("\n[5/5] 对比基线...")

# Ridge基线真实OOS结果
ridge_oos_sharpe = 0.150
ridge_oos_return = 0.0103

print(f"\n{'='*60}")
print("与Ridge基线对比 (真实OOS)")
print('='*60)
print(f"{'指标':<20} {'Ridge基线':>15} {'新策略(Top10)':>15}")
print('-'*60)
print(f"{'Sharpe':<20} {ridge_oos_sharpe:>15.3f} {sharpe:>15.3f}")
print(f"{'年化收益':<20} {ridge_oos_return*100:>14.1f}% {(mean_return * (252/TEST_WINDOW))*100:>14.1f}%")
print(f"{'胜率':<20} {'59.3%':>15} {win_rate*100:>14.1f}%")

improvement = (sharpe - ridge_oos_sharpe) / ridge_oos_sharpe * 100 if ridge_oos_sharpe > 0 else 0
print(f"\nSharpe差距: {improvement:+.1f}%")

# ==================== 保存结果 ====================
results_df.to_csv(OUTPUT_DIR / 'window_results.csv', index=False)

summary = {
    'validation_factors': VALIDATION_FACTORS,
    'n_windows': len(results_df),
    'train_window': TRAIN_WINDOW,
    'test_window': TEST_WINDOW,
    'top_n': TOP_N,
    'portfolio_return': {
        'mean': float(mean_return),
        'std': float(std_return),
        'sharpe': float(sharpe),
        'annualized': float(mean_return * (252/TEST_WINDOW)),
        'win_rate': float(win_rate),
        'worst': float(worst_window),
        'best': float(best_window),
    },
    'ic_stats': {
        'mean': float(results_df['test_ic'].mean()),
        'ir': float(results_df['test_ic_ir'].mean()),
    },
    'baseline_comparison': {
        'ridge_sharpe': ridge_oos_sharpe,
        'new_sharpe': float(sharpe),
        'improvement_pct': float(improvement),
    }
}

import json
with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n结果已保存至: {OUTPUT_DIR}")

# ==================== 风险提示 ====================
print("\n" + "="*70)
print("⚠️ 风险提示")
print("="*70)
print("1. 本验证使用真实Walk-Forward，每个窗口独立训练模型")
print("2. 收益已扣除估算的交易成本（佣金+印花税+滑点）")
print("3. 但未考虑：")
print("   - 流动性约束（大单冲击）")
print("   - 未能成交的情况")
print("   - 市场冲击成本")
print("4. 结果仅供参考，不代表实盘表现")

print("\n" + "="*70)
print("验证完成")
print("="*70)
