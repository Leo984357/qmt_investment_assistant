"""
Ridge回归 + 9因子 策略回测 (修正版)

使用真实价格计算收益，修正了之前的bug：
1. 成本计算错误（portfolio_value * turnover * COST）
2. 成本被重复累加
3. Walk-Forward窗口不连续
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Ridge回归 + 9因子 策略回测 (修正版)")
print("="*70)

# ==================== 配置 ====================
DATA_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/data/silver")
RUN_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs/hs300_ridge_with_support_20260412_000435_8bad53b6")
OUTPUT_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/backtests/best_strategy_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'factors': [
        'alpha_017', 'earnings_yield', 'roe', 'alpha_006',
        'alpha_002', 'ocf_per_share', 'operating_margin',
        'asset_turnover', 'equity_growth'
    ],
    'top_n': 15,
    'rebalance_days': 10,
}

# 成本配置 (单边)
COMMISSION = 0.00075  # 佣金0.075%
STAMP_TAX = 0.001     # 印花税0.1% (仅卖出)
SLIPPAGE = 0.0005     # 滑点0.05%
TOTAL_COST_BUY = COMMISSION + SLIPPAGE       # 买入总成本
TOTAL_COST_SELL = COMMISSION + STAMP_TAX + SLIPPAGE  # 卖出总成本

# ==================== 加载数据 ====================
print("\n[1/4] 加载数据...")

bars = pd.read_parquet(DATA_DIR / "daily_bar.parquet")
financial = pd.read_parquet(RUN_DIR / "datasets/model_dataset.parquet")

df = bars.merge(financial[['trade_date', 'symbol', 'roe', 'earnings_yield', 'operating_margin', 
                           'equity_growth', 'ocf_per_share', 'revenue_growth', 'asset_turnover',
                           'gross_margin', 'fwd_return_20d', 'is_tradable']], 
                on=['trade_date', 'symbol'], how='left')

df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

# 计算Alpha
df['alpha_002'] = df.groupby('symbol').apply(
    lambda x: -1 * x['open'].rolling(6, min_periods=3).corr(x['volume'].rank())
).reset_index(level=0, drop=True)

df['alpha_006'] = df.groupby('symbol').apply(
    lambda x: -1 * x['open'].rolling(10, min_periods=5).corr(x['volume'])
).reset_index(level=0, drop=True)

df['alpha_017'] = (-1 * df.groupby('symbol')['close'].transform(
    lambda x: x.rank().rolling(10, min_periods=5).mean()
).rank()) * df.groupby('symbol')['close'].diff().diff().rank()

features = [f for f in CONFIG['factors'] if f in df.columns]
print(f"  数据量: {len(df):,}")

# ==================== Walk-Forward回测 ====================
print("\n[2/4] Walk-Forward回测...")

TRAIN_WINDOW = 500

dates = sorted(df['trade_date'].unique())

initial_capital = 1000000
nav_list = []
trades_list = []
current_holdings = {}  # symbol -> shares
portfolio_value = initial_capital
cash = initial_capital

# 找到第一个有足够训练数据的日期
train_end_idx = TRAIN_WINDOW
signal_dates = []

while train_end_idx < len(dates):
    train_end = dates[train_end_idx]
    train_df = df[df['trade_date'] <= train_end].copy()
    train_valid = train_df.dropna(subset=features + ['fwd_return_20d'])
    train_valid = train_valid[train_valid['is_tradable'] == True]
    
    for f in features:
        train_valid[f] = pd.to_numeric(train_valid[f], errors='coerce')
    train_valid = train_valid.dropna(subset=features)
    
    if len(train_valid) >= 200:
        signal_dates.append((train_end_idx, train_end))
    
    train_end_idx += CONFIG['rebalance_days']

print(f"  找到 {len(signal_dates)} 个调仓日")

# 连续Walk-Forward
for idx, (train_end_idx, train_end) in enumerate(signal_dates):
    # 训练
    train_df = df[df['trade_date'] <= train_end].copy()
    train_valid = train_df.dropna(subset=features + ['fwd_return_20d'])
    train_valid = train_valid[train_valid['is_tradable'] == True]
    
    for f in features:
        train_valid[f] = pd.to_numeric(train_valid[f], errors='coerce')
    train_valid = train_valid.dropna(subset=features)
    
    if len(train_valid) < 200:
        continue
    
    X_train = train_valid[features].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_valid['fwd_return_20d'].fillna(0)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # 信号日 = 训练窗口结束后第一个交易日
    signal_idx = train_end_idx + 1
    if signal_idx >= len(dates):
        break
    signal_date = dates[signal_idx]
    
    # 执行日 = 信号日后一天
    exec_idx = signal_idx + 1
    if exec_idx >= len(dates):
        break
    exec_date = dates[exec_idx]
    
    # 持有期直到下一个调仓日
    if idx + 1 < len(signal_dates):
        next_exec_idx = signal_dates[idx + 1][0] + 1
    else:
        next_exec_idx = len(dates) - 1
    next_exec_idx = min(next_exec_idx, len(dates) - 1)
    next_exec_date = dates[next_exec_idx]
    
    # 获取执行日价格
    exec_day = df[df['trade_date'] == exec_date]
    exec_day = exec_day.dropna(subset=features)
    for f in features:
        exec_day[f] = pd.to_numeric(exec_day[f], errors='coerce')
    exec_day = exec_day.dropna(subset=features)
    
    if len(exec_day) < CONFIG['top_n']:
        continue
    
    # 预测
    X_exec = exec_day[features].fillna(0).replace([np.inf, -np.inf], 0)
    X_exec_scaled = scaler.transform(X_exec)
    exec_day['pred'] = model.predict(X_exec_scaled)
    
    # 选择top N
    selected = exec_day.nlargest(CONFIG['top_n'], 'pred')
    target_symbols = set(selected['symbol'].tolist())
    
    # 获取收盘价作为执行价
    exec_prices = dict(zip(exec_day['symbol'], exec_day['close']))
    
    # 计算目标持仓
    target_shares = {}
    total_equity = cash + sum(
        shares * exec_prices.get(sym, 0)
        for sym, shares in current_holdings.items()
    )
    per_stock_value = total_equity / CONFIG['top_n']
    
    for sym in target_symbols:
        price = exec_prices.get(sym, 0)
        if price > 0:
            shares = int(per_stock_value / price / 100) * 100
            if shares * price >= 2000:  # 最小交易额
                target_shares[sym] = shares
            else:
                target_shares[sym] = 0
    
    # 执行交易
    trade_cost = 0
    new_holdings = {}
    
    # 卖出不在目标中的
    for sym in list(current_holdings.keys()):
        current_shares = current_holdings.get(sym, 0)
        target_shares_n = target_shares.get(sym, 0)
        
        if current_shares > 0 and target_shares_n == 0:
            price = exec_prices.get(sym, 0)
            if price > 0:
                sell_value = current_shares * price
                cost = sell_value * TOTAL_COST_SELL
                cash += sell_value - cost
                trade_cost += cost
    
    # 买入目标中的
    for sym, shares in target_shares.items():
        if shares <= 0:
            continue
        price = exec_prices.get(sym, 0)
        if price > 0:
            buy_value = shares * price
            cost = buy_value * TOTAL_COST_BUY
            if cash >= buy_value + cost:
                cash -= buy_value + cost
                trade_cost += cost
                current_holdings[sym] = current_holdings.get(sym, 0) + shares
            else:
                affordable_shares = int(cash / (price * (1 + TOTAL_COST_BUY)) / 100) * 100
                if affordable_shares > 0:
                    buy_value = affordable_shares * price
                    cost = buy_value * TOTAL_COST_BUY
                    cash -= buy_value + cost
                    trade_cost += cost
                    current_holdings[sym] = current_holdings.get(sym, 0) + affordable_shares
    
    # 更新持仓
    current_holdings = {sym: shares for sym, shares in current_holdings.items() if target_shares.get(sym, 0) > 0}
    
    # 计算持有期收益 (到下一个执行日)
    next_day = df[df['trade_date'] == next_exec_date]
    next_prices = dict(zip(next_day['symbol'], next_day['close']))
    
    period_pnl = 0
    holdings_value_start = 0
    holdings_value_end = 0
    
    for sym, shares in list(current_holdings.items()):
        curr_price = exec_prices.get(sym, 0)
        next_price = next_prices.get(sym, 0)
        if curr_price > 0:
            holdings_value_start += shares * curr_price
        if next_price > 0:
            holdings_value_end += shares * next_price
            period_pnl += shares * (next_price - curr_price)
    
    # 更新净值
    portfolio_value = cash + holdings_value_end
    
    trades_list.append({
        'date': exec_date,
        'signal_date': signal_date,
        'n_holdings': len(current_holdings),
        'cash': cash,
        'holdings_value': holdings_value_end,
        'trade_cost': trade_cost,
        'period_pnl': period_pnl,
        'nav': portfolio_value,
    })
    
    nav_list.append({
        'date': next_exec_date,
        'nav': portfolio_value,
        'cash': cash,
        'holdings_value': holdings_value_end,
    })
    


print(f"  完成: {len(trades_list)} 次调仓")

# ==================== 统计 ====================
print("\n[3/4] 计算指标...")

nav_df = pd.DataFrame(nav_list)
trades_df = pd.DataFrame(trades_list)

if len(nav_df) < 2:
    print("  错误: 数据不足")
    exit(1)

nav_df = nav_df.sort_values('date').reset_index(drop=True)
nav_df['daily_return'] = nav_df['nav'].pct_change()

# 计算全部期间指标
total_return_all = (nav_df['nav'].iloc[-1] / initial_capital - 1) * 100
n_days_all = (nav_df['date'].iloc[-1] - nav_df['date'].iloc[0]).days
annual_return_all = total_return_all / (n_days_all / 365) if n_days_all > 0 else 0

# 筛选与基线相同时间段的回测 (2022-09-01 ~ 2026-04-09)
baseline_start = pd.Timestamp('2022-09-01')
baseline_end = pd.Timestamp('2026-04-09')
baseline_mask = (nav_df['date'] >= baseline_start) & (nav_df['date'] <= baseline_end)
nav_baseline = nav_df[baseline_mask].copy()

if len(nav_baseline) > 1:
    initial_baseline = nav_baseline['nav'].iloc[0]
    total_return = (nav_baseline['nav'].iloc[-1] / initial_baseline - 1) * 100
    n_days = (nav_baseline['date'].iloc[-1] - nav_baseline['date'].iloc[0]).days
    annual_return = total_return / (n_days / 365) if n_days > 0 else 0
    
    daily_returns = nav_baseline['daily_return'].dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    nav_baseline['peak'] = nav_baseline['nav'].cummax()
    nav_baseline['drawdown'] = (nav_baseline['nav'] - nav_baseline['peak']) / nav_baseline['peak'] * 100
    max_dd = nav_baseline['drawdown'].min()
else:
    total_return = total_return_all
    n_days = n_days_all
    annual_return = annual_return_all
    daily_returns = nav_df['daily_return'].dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe = annual_return / volatility if volatility > 0 else 0
    nav_df['peak'] = nav_df['nav'].cummax()
    nav_df['drawdown'] = (nav_df['nav'] - nav_df['peak']) / nav_df['nav'] / 100
    max_dd = nav_df['drawdown'].min()

# 成本
total_cost = trades_df['trade_cost'].sum() if 'trade_cost' in trades_df.columns else 0
win_rate = (trades_df['period_pnl'] > 0).mean() if 'period_pnl' in trades_df.columns else 0

print("\n" + "="*70)
print("回测结果 (修正版)")
print("="*70)
print(f"\n【全部期间】")
print(f"  初始: {initial_capital:,.0f}")
print(f"  最终: {nav_df['nav'].iloc[-1]:,.0f}")
print(f"  总收益: {total_return_all:.1f}%")
print(f"  天数: {n_days_all}")

print(f"\n【与基线同期 (2022-09 ~ 2026-04)】")
print(f"  初始: {initial_baseline if len(nav_baseline) > 1 else initial_capital:,.0f}")
print(f"  最终: {nav_baseline['nav'].iloc[-1] if len(nav_baseline) > 1 else nav_df['nav'].iloc[-1]:,.0f}")
print(f"  总收益: {total_return:.1f}%")
print(f"  年化: {annual_return:.1f}%")
print(f"  Sharpe: {sharpe:.3f}")
print(f"\n【风险】")
print(f"  波动: {volatility:.1f}%")
print(f"  最大回撤: {max_dd:.1f}%")
print(f"  胜率: {win_rate:.1%}")
print(f"\n【交易】")
print(f"  调仓次数: {len(trades_df)}")
print(f"  总成本: {total_cost:,.0f}元")

# ==================== 保存 ====================
print("\n[4/4] 保存...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
nav_df.to_parquet(OUTPUT_DIR / "nav.parquet", index=False)
trades_df.to_parquet(OUTPUT_DIR / "trades.parquet", index=False)

summary = {
    'strategy': 'Ridge + 9因子 (修正版)',
    'total_return': float(total_return),
    'annual_return': float(annual_return),
    'sharpe': float(sharpe),
    'max_drawdown': float(max_dd),
    'win_rate': float(win_rate),
    'total_cost': float(total_cost),
    'volatility': float(volatility),
    'n_rebalances': len(trades_df),
    'start_date': str(nav_df['date'].iloc[0].date()),
    'end_date': str(nav_df['date'].iloc[-1].date()),
}

with open(OUTPUT_DIR / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

# 对比
print("\n" + "="*70)
print("与基线对比")
print("="*70)
print(f"\n{'指标':<15} {'基线':>10} {'本策略':>10}")
print("-"*40)
print(f"{'总收益':<15} {'21.2%':>10} {total_return:>9.1f}%")
print(f"{'Sharpe':<15} {'0.594':>10} {sharpe:>10.3f}")
print(f"{'最大回撤':<15} {'-18.2%':>10} {max_dd:>9.1f}%")
print(f"{'年化收益':<15} {'5.1%':>10} {annual_return:>9.1f}%")
print(f"{'总成本':<15} {'1,092':>10} {total_cost:>10,.0f}")
