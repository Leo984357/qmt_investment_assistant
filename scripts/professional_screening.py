#!/usr/bin/env python
"""Professional factor screening with multi-metrics"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from src.data_store.catalog import LocalResearchCatalog
from src.features.simple_definitions import simple_factor_registry

print('Loading data...', flush=True)
catalog = LocalResearchCatalog()
daily_bar = catalog.read_table('daily_bar')
universe = catalog.read_table('universe_membership')
universe_symbols = universe[['trade_date', 'symbol']].drop_duplicates()
daily_bar = daily_bar.merge(universe_symbols, on=['trade_date', 'symbol'], how='inner')
daily_bar = daily_bar.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

# Label
label_df = daily_bar[['trade_date', 'symbol', 'close']].copy()
label_df['label'] = daily_bar.groupby('symbol')['close'].transform(lambda x: x.shift(-20) / x.shift(-1) - 1)
label_df = label_df.dropna(subset=['label'])

# Get all factors
registry = simple_factor_registry()
inv = registry.inventory()
all_factors = sorted(inv['feature_name'].tolist())

results = []
for i, f in enumerate(all_factors):
    if i % 100 == 0:
        print(f'  {i}/{len(all_factors)}', flush=True)
    
    try:
        spec = registry.get(f)
        factor_vals = spec.compute(daily_bar.copy())
        
        df = daily_bar[['trade_date', 'symbol']].copy()
        df['factor'] = factor_vals.values
        df = df.merge(label_df[['trade_date', 'symbol', 'label']], on=['trade_date', 'symbol'], how='inner')
        df = df.dropna(subset=['factor', 'label'])
        
        if len(df) < 100:
            continue
            
        # IC by year
        df['year'] = df['trade_date'].dt.year
        yearly_ic = []
        for year in df['year'].unique():
            yeardf = df[df['year'] == year]
            if len(yeardf) > 20:
                ic, _ = spearmanr(yeardf['factor'], yeardf['label'])
                yearly_ic.append(ic)
        
        # Metrics
        ic_mean = np.nanmean(yearly_ic)
        ic_std = np.nanstd(yearly_ic)
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_pos_rate = np.nanmean([1 if x > 0 else 0 for x in yearly_ic])
        years_pos = sum(1 for x in yearly_ic if x > 0)
        
        results.append({
            'factor': f,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_win_rate': ic_pos_rate,
            'years_positive': years_pos,
            'n_years': len(yearly_ic),
            'n_samples': len(df)
        })
    except:
        pass

df_res = pd.DataFrame(results)

# Filters (professional criteria)
print('\n=== FILTERS APPLIED ===')
print('1. IC IR > 0.10 (stability)')
print('2. IC win rate > 55%')
print('3. At least 3 positive years')

# Apply filters
df_filtered = df_res[
    (df_res['ic_ir'] > 0.10) & 
    (df_res['ic_win_rate'] > 0.55) &
    (df_res['years_positive'] >= 3)
].sort_values('ic_ir', ascending=False)

print(f'\n=== FILTERED: {len(df_filtered)} / {len(df_res)} ===')
print('\n=== TOP 30 (PROFESSIONAL SCREEN) ===')
print(df_filtered.head(30).to_string(index=False))

# Save both
df_res.to_csv('/Users/leolee/Desktop/qmt_investment_assistant/artifacts/all_factors_ic_full.csv', index=False)
df_filtered.to_csv('/Users/leolee/Desktop/qmt_investment_assistant/artifacts/all_factors_professional.csv', index=False)

# Also show by factor family for diversity
print('\n=== FACTOR DIVERSITY ===')
# Group by similar prefixes
def get_family(f):
    if f.startswith('alpha_'):
        return 'WorldQuant'
    elif f.startswith('mom') or 'reversal' in f:
        return 'Momentum'
    elif any(x in f for x in ['rsi', 'macd', 'kdj', 'adx', 'cci']):
        return 'Technical'
    elif any(x in f for x in ['vol', 'atr', 'volatility']):
        return 'Volatility'
    elif any(x in f for x in ['roe', 'margin', 'turnover', 'growth']):
        return 'Financial'
    elif any(x in f for x in ['ps_ratio', 'pe_', 'pb_', 'ev_', 'yield']):
        return 'Value'
    elif any(x in f for x in ['volume', 'turnover', 'amihud']):
        return 'Liquidity'
    else:
        return 'Other'

df_filtered['family'] = df_filtered['factor'].apply(get_family)
print(df_filtered.groupby('family').size().to_string())