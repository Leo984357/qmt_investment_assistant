#!/usr/bin/env python
"""Batch IC test - runs in background"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from src.data_store.catalog import LocalResearchCatalog
from src.features.simple_definitions import simple_factor_registry

print('Loading...', flush=True)
catalog = LocalResearchCatalog()
daily_bar = catalog.read_table('daily_bar')
universe = catalog.read_table('universe_membership')
universe_symbols = universe[['trade_date', 'symbol']].drop_duplicates()
daily_bar = daily_bar.merge(universe_symbols, on=['trade_date', 'symbol'], how='inner')
daily_bar = daily_bar.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

label_df = daily_bar[['trade_date', 'symbol']].copy()
label_df['label'] = daily_bar.groupby('symbol')['close'].transform(lambda x: x.shift(-20) / x.shift(-1) - 1)
label_df = label_df.dropna(subset=['label'])

registry = simple_factor_registry()
inv = registry.inventory()
all_factors = sorted(inv['feature_name'].tolist())

done = pd.read_csv('/Users/leolee/Desktop/qmt_investment_assistant/artifacts/all_factors_ic.csv')
done_factors = set(done['factor'].dropna().tolist())
remaining = [f for f in all_factors if f not in done_factors]

print(f'Remaining: {len(remaining)}', flush=True)

results = []
count = 0
batch_size = 100

for batch_start in range(0, len(remaining), batch_size):
    batch = remaining[batch_start:batch_start + batch_size]
    for f in batch:
        try:
            spec = registry.get(f)
            factor_vals = spec.compute(daily_bar.copy())
            df_temp = daily_bar[['trade_date', 'symbol']].copy()
            df_temp['factor'] = factor_vals.values
            df_temp = df_temp.merge(label_df, on=['trade_date', 'symbol'], how='inner')
            df_temp = df_temp.dropna(subset=['factor', 'label'])
            if len(df_temp) >= 30:
                ic, _ = spearmanr(df_temp['factor'], df_temp['label'])
                results.append({'factor': f, 'ic': ic})
            else:
                results.append({'factor': f, 'ic': np.nan})
        except:
            results.append({'factor': f, 'ic': np.nan})
    
    count += len(batch)
    print(f'Done: {count}/{len(remaining)}', flush=True)
    
    # Save intermediate
    df = pd.DataFrame(results)
    old = pd.read_csv('/Users/leolee/Desktop/qmt_investment_assistant/artifacts/all_factors_ic.csv')
    combined = pd.concat([old, df]).sort_values('ic', ascending=False).drop_duplicates(subset=['factor'])
    combined.to_csv('/Users/leolee/Desktop/qmt_investment_assistant/artifacts/all_factors_ic.csv', index=False)

print(f'Total tested: {len(combined)}', flush=True)
print('\\n=== TOP 30 ===')
print(combined.head(30).to_string(index=False), flush=True)