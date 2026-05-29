#!/usr/bin/env python
"""Full IC test - all simple factors - continued"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from src.data_store.catalog import LocalResearchCatalog
from src.features.simple_definitions import simple_factor_registry

print("Loading data...")
catalog = LocalResearchCatalog()
daily_bar = catalog.read_table('daily_bar')
universe = catalog.read_table('universe_membership')

universe_symbols = universe[['trade_date', 'symbol']].drop_duplicates()
daily_bar = daily_bar.merge(universe_symbols, on=['trade_date', 'symbol'], how='inner')
daily_bar = daily_bar.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

label_df = daily_bar[['trade_date', 'symbol']].copy()
label_df['label'] = daily_bar.groupby('symbol')['close'].transform(lambda x: x.shift(-20) / x.shift(-1) - 1)
label_df = label_df.dropna(subset=['label'])

# More factors
factors = [
    'rsi24', 'rsi12', 'cci20', 'cci28', 'kdj_k', 'kdj_d', 'kdj_j',
    'ma5', 'ma20', 'ma60', 'ma120', 'ma250',
    'ma5_bias', 'ma20_bias', 'ma60_bias',
    'ma5_10_golden_cross', 'ma5_20_death_cross', 'ma20_60_golden_cross',
    'ar_turnover', 'ar_turnover_days', 'asset_turnover', 'inventory_turnover',
    'price_to_ma20', 'price_to_ma60', 'price_to_ma120',
    'obv', 'mfi_14', 'atr14', 'atr20',
    'volume_ratio', 'volume_spike', 'volume_momentum',
    'bid_ask_spread', 'depth'
]

registry = simple_factor_registry()
results = []
print("\n=== More factors ===")
for f in factors:
    try:
        spec = registry.get(f)
        factor_vals = spec.compute(daily_bar.copy())
        
        df_temp = daily_bar[['trade_date', 'symbol']].copy()
        df_temp['factor'] = factor_vals.values
        df_temp = df_temp.merge(label_df, on=['trade_date', 'symbol'], how='inner')
        df_temp = df_temp.dropna(subset=['factor', 'label'])
        
        if len(df_temp) >= 30:
            ic, _ = spearmanr(df_temp['factor'], df_temp['label'])
            print(f"  {f}: IC={ic:.4f}")
            results.append({'factor': f, 'ic': ic})
        else:
            results.append({'factor': f, 'ic': np.nan})
    except Exception as e:
        print(f"  {f}: ERROR {str(e)[:30]}")
        results.append({'factor': f, 'ic': np.nan})

df = pd.DataFrame(results).sort_values('ic', ascending=False)
old = pd.read_csv('/Users/leolee/Desktop/qmt_investment_assistant/artifacts/all_factors_ic.csv')
combined = pd.concat([old, df]).sort_values('ic', ascending=False)
combined.to_csv('/Users/leolee/Desktop/qmt_investment_assistant/artifacts/all_factors_ic.csv', index=False)
print("\n=== COMBINED TOP 20 ===")
print(combined.head(20).to_string(index=False))