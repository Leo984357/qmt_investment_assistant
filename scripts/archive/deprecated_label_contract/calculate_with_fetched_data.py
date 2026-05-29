"""
用已爬取的数据计算因子IC
"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from src.features.factor_pool import get_all_factors

DATA_DIR = Path('data/raw')
FETCHED_DIR = Path('data/raw/fetched_data')
CACHE_DIR = Path('data/factor_cache')

print('=' * 60)
print('用已爬取数据计算因子IC')
print('=' * 60)

# Load price data
print('\nLoading price data...')
bars = pd.read_parquet(DATA_DIR / 'daily_bar.parquet')
bars['trade_date'] = pd.to_datetime(bars['trade_date'])
bars = bars.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
bars = bars[bars['trade_date'] >= '2020-01-01']
bars['returns'] = bars.groupby('symbol')['adj_close'].pct_change()
bars['fwd_return_20d'] = bars.groupby('symbol')['adj_close'].pct_change(20).shift(-20)

# Standardize symbol
bars['code'] = bars['symbol'].str.replace('.SH', '').str.replace('.SZ', '')

def calculate_ic(panel, factor_col, label_col='fwd_return_20d'):
    if factor_col not in panel.columns or label_col not in panel.columns:
        return np.nan
    valid = panel[[factor_col, label_col]].dropna()
    if len(valid) < 30:
        return np.nan
    return valid[factor_col].corr(valid[label_col], method='spearman')

results = []
factor_map = {f.name: f for f in get_all_factors()}

# ===== 1. 融资融券因子 =====
print('\n1. 融资融券因子...')
try:
    margin = pd.read_parquet(FETCHED_DIR / 'margin_summary.parquet')
    margin['日期'] = pd.to_datetime(margin['日期']) if '日期' in margin.columns else pd.Timestamp.now()
    margin['code'] = margin['证券代码'].astype(str).str.zfill(6)
    
    # Merge with bars
    margin_merged = bars.merge(margin, left_on=['trade_date', 'code'], right_on=['日期', 'code'], how='left')
    
    # Calculate margin factors
    if '融资余额' in margin.columns:
        margin_merged['margin_balance'] = margin_merged['融资余额']
        margin_merged['short_balance'] = margin_merged['融券余额']
        margin_merged['margin_ratio'] = margin_merged['融资余额'] / margin_merged['融券余额'].replace(0, np.nan)
        margin_merged['margin_chg'] = margin_merged.groupby('code')['融资余额'].pct_change()
        
        for f in ['margin_balance', 'short_balance', 'margin_ratio', 'margin_chg']:
            if f in margin_merged.columns:
                ic = calculate_ic(margin_merged, f)
                results.append({'factor': f, 'source': 'money_flow', 'rank_ic': ic})
                print(f'  {f}: IC = {ic:.4f}')
except Exception as e:
    print(f'  Error: {e}')

# ===== 2. 北向资金因子 =====
print('\n2. 北向资金因子...')
try:
    hsgt = pd.read_parquet(FETCHED_DIR / 'hsgt_hist.parquet')
    hsgt['日期'] = pd.to_datetime(hsgt['日期'])
    
    # Merge
    hsgt_merged = bars.merge(hsgt, left_on='trade_date', right_on='日期', how='left')
    
    for col in ['当日成交净买额', '买入成交额', '持股市值']:
        if col in hsgt_merged.columns:
            ic = calculate_ic(hsgt_merged, col)
            results.append({'factor': f'hsgt_{col}', 'source': 'money_flow', 'rank_ic': ic})
            print(f'  hsgt_{col}: IC = {ic:.4f}')
except Exception as e:
    print(f'  Error: {e}')

# ===== 3. 龙虎榜因子 =====
print('\n3. 龙虎榜因子...')
try:
    lhb = pd.read_parquet(FETCHED_DIR / 'lhb_full.parquet')
    lhb['上榜日'] = pd.to_datetime(lhb['上榜日'])
    lhb['code'] = lhb['代码'].astype(str).str.zfill(6)
    
    # Calculate LHB frequency
    lhb_count = lhb.groupby('code').size().reset_index(name='lhb_count')
    lhb_merged = bars.merge(lhb_count, on='code', how='left')
    lhb_merged['lhb_count'] = lhb_merged['lhb_count'].fillna(0)
    
    ic = calculate_ic(lhb_merged, 'lhb_count')
    results.append({'factor': 'lhb_count', 'source': 'sentiment', 'rank_ic': ic})
    print(f'  lhb_count: IC = {ic:.4f}')
    
    # Net buy amount
    lhb_net = lhb.groupby('code')['龙虎榜净买额'].mean().reset_index(name='lhb_net_buy')
    lhb_merged2 = bars.merge(lhb_net, on='code', how='left')
    
    ic = calculate_ic(lhb_merged2, 'lhb_net_buy')
    results.append({'factor': 'lhb_net_buy', 'source': 'sentiment', 'rank_ic': ic})
    print(f'  lhb_net_buy: IC = {ic:.4f}')
    
    # After LHB performance
    if '上榜后5日' in lhb.columns:
        lhb_after = lhb.groupby('code')['上榜后5日'].mean().reset_index(name='lhb_after5')
        lhb_merged3 = bars.merge(lhb_after, on='code', how='left')
        ic = calculate_ic(lhb_merged3, 'lhb_after5')
        results.append({'factor': 'lhb_after5_perf', 'source': 'sentiment', 'rank_ic': ic})
        print(f'  lhb_after5_perf: IC = {ic:.4f}')
        
except Exception as e:
    print(f'  Error: {e}')

# ===== 4. 分析师预期因子 =====
print('\n4. 分析师预期因子...')
try:
    forecast = pd.read_parquet(FETCHED_DIR / 'stock_profit_forecast_all.parquet')
    forecast['code'] = forecast['代码'].astype(str).str.zfill(6)
    
    # Merge
    forecast_merged = bars.merge(forecast[['code', '研报数', '2024预测每股收益', '2025预测每股收益']], on='code', how='left')
    
    for col in ['研报数', '2024预测每股收益', '2025预测每股收益']:
        if col in forecast_merged.columns:
            ic = calculate_ic(forecast_merged, col)
            results.append({'factor': f'forecast_{col}', 'source': 'analyst', 'rank_ic': ic})
            print(f'  forecast_{col}: IC = {ic:.4f}')
            
except Exception as e:
    print(f'  Error: {e}')

# ===== 5. 涨跌停因子 =====
print('\n5. 涨跌停因子...')
try:
    zt = pd.read_parquet(FETCHED_DIR / 'zt_pool_strong.parquet')
    zt['code'] = zt['代码'].astype(str).str.zfill(6)
    
    # ZT frequency
    zt_count = zt.groupby('code').size().reset_index(name='zt_count')
    zt_merged = bars.merge(zt_count, on='code', how='left')
    zt_merged['zt_count'] = zt_merged['zt_count'].fillna(0)
    
    ic = calculate_ic(zt_merged, 'zt_count')
    results.append({'factor': 'zt_count', 'source': 'sentiment', 'rank_ic': ic})
    print(f'  zt_count: IC = {ic:.4f}')
    
    # 强势股
    strong_count = zt.groupby('code').size().reset_index(name='strong_count')
    strong_merged = bars.merge(strong_count, on='code', how='left')
    strong_merged['strong_count'] = strong_merged['strong_count'].fillna(0)
    
    ic = calculate_ic(strong_merged, 'strong_count')
    results.append({'factor': 'strong_count', 'source': 'sentiment', 'rank_ic': ic})
    print(f'  strong_count: IC = {ic:.4f}')
    
except Exception as e:
    print(f'  Error: {e}')

# ===== 6. 宏观因子 =====
print('\n6. 宏观因子...')
try:
    # Bond yield
    bond = pd.read_parquet(FETCHED_DIR / 'bond_yield.parquet')
    if '日期' in bond.columns:
        bond['日期'] = pd.to_datetime(bond['日期'])
        bond_merged = bars.merge(bond, left_on='trade_date', right_on='日期', how='left', suffixes=('', '_macro'))
        for col in ['1年', '3年', '5年', '10年']:
            if col in bond_merged.columns:
                ic = calculate_ic(bond_merged, col)
                results.append({'factor': f'bond_yield_{col}', 'source': 'macro', 'rank_ic': ic})
                print(f'  bond_yield_{col}: IC = {ic:.4f}')
                
except Exception as e:
    print(f'  Error: {e}')

# ===== 7. 计算更多因子 =====
print('\n7. 计算注册的其他因子...')

# Load financial data
financial = pd.read_parquet(CACHE_DIR / 'financial_factors.parquet')
financial['pub_date'] = pd.to_datetime(financial['pub_date'])
financial = financial.rename(columns={'pub_date': 'trade_date'})

for col in financial.columns:
    if col not in ['trade_date', 'symbol'] and col not in bars.columns:
        temp = financial[['trade_date', 'symbol', col]].copy()
        bars = bars.merge(temp, on=['trade_date', 'symbol'], how='left')
        bars[col] = bars.groupby('symbol')[col].ffill()

# Calculate technical factors
windows = [1, 5, 10, 20, 30, 60, 90, 120]
for w in windows:
    bars[f'mom_{w}d'] = bars.groupby('symbol')['adj_close'].pct_change(w)
    bars[f'rev_{w}d'] = -bars.groupby('symbol')['adj_close'].pct_change(w)

# Calculate all registered factors
factors = get_all_factors()
tested = set()

for f in factors:
    name = f.name
    source = f.source
    
    if name in tested:
        continue
    
    # Try to calculate
    val = np.nan
    
    try:
        if name in bars.columns:
            val = bars[name]
        elif name.startswith('mom_'):
            w = name.replace('mom_', '').replace('d', '')
            if w.isdigit() and int(w) in windows:
                val = bars[name]
        elif name.startswith('rev_'):
            w = name.replace('rev_', '').replace('d', '')
            if w.isdigit() and int(w) in windows:
                val = bars[name]
        elif name in ['roe', 'profit_growth', 'revenue_growth', 'earnings_yield', 'asset_turnover']:
            if name in bars.columns:
                val = bars[name]
        elif name == 'size':
            val = -np.log(bars['close'].replace(0, np.nan))
        elif name == 'liquidity':
            vol_ma = bars.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
            val = -np.log((vol_ma * bars['close']).replace(0, np.nan))
        elif name in ['margin_balance', 'short_balance', 'margin_ratio']:
            pass  # Already calculated
        elif name == 'lhb_count':
            pass  # Already calculated
        elif name.startswith('forecast_') or name.startswith('bond_') or name.startswith('hsgt_'):
            pass  # Already calculated
        elif name in ['zt_count', 'strong_count']:
            pass  # Already calculated
            
        if isinstance(val, pd.Series) and name not in tested:
            ic = calculate_ic(bars, name)
            if not np.isnan(ic):
                results.append({'factor': name, 'source': source, 'rank_ic': ic})
                tested.add(name)
                
    except:
        pass

# Save results
print('\n' + '=' * 60)
print('Saving results...')
print('=' * 60)

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    results_df = results_df.drop_duplicates(subset=['factor'])
    results_df.to_csv(CACHE_DIR / 'fetched_data_factors_ic.csv', index=False)
    print(f'Saved {len(results_df)} factors')

# Combine with existing
all_ic = pd.read_csv(CACHE_DIR / 'all_factors_ic_complete.csv')
all_ic = pd.concat([all_ic, results_df], ignore_index=True)
all_ic = all_ic.dropna()
all_ic = all_ic.drop_duplicates(subset=['factor'], keep='last')
all_ic = all_ic.sort_values('rank_ic', ascending=False)
all_ic.to_csv(CACHE_DIR / 'all_factors_ic_complete.csv', index=False)

# Summary
print('\n' + '=' * 60)
print('Final Summary')
print('=' * 60)
effective = all_ic[all_ic['rank_ic'] > 0.02]
print(f'Total tested: {len(all_ic)} / 667')
print(f'Effective (IC > 0.02): {len(effective)}')

print('\nTop 30:')
for i, (_, row) in enumerate(all_ic.head(30).iterrows(), 1):
    print(f'{i:2d}. {str(row[\"factor\"])[:35]:35s} IC={row[\"rank_ic\"]:+.4f}')

# Check remaining
factors = get_all_factors()
tested = set(all_ic['factor'].tolist())
untested = [f.name for f in factors if f.name not in tested]
print(f'\nRemaining untested: {len(untested)} factors')
