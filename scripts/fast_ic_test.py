#!/usr/bin/env python
"""Full IC test - all simple factors"""
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

# All simple tech/momentum factors - split into batches
batches = [
    # Batch 1: Momentum
    ['mom250', 'mom120', 'mom60', 'mom90', 'mom20', 'momentum_6m', 'acceleration_20'],
    # Batch 2: Reversal
    ['rev20', 'rev10', 'rev5', 'short_term_reversal', 'medium_reversal'],
    # Batch 3: Technical
    ['rsi14', 'rsi6', 'macd', 'adx14', 'cci14', 'williams_r14'],
    # Batch 4: Volatility
    ['vol20', 'vol60', 'vol_realized_20', 'vol_realized_60', 'downside_volatility'],
    # Batch 5: Price position  
    ['close_to_high250', 'close_to_high120', 'close_to_high60', 'close_to_low250', 'close_to_low120'],
    # Batch 6: Liquidity
    ['turnover_rate', 'amihud_illiq_20d', 'relative_volume_20d'],
    # Batch 7: WorldQuant
    ['alpha_001', 'alpha_002', 'alpha_003', 'alpha_006', 'alpha_014', 'alpha_016'],
]

registry = simple_factor_registry()
all_results = []
for batch in batches:
    print(f"\n=== Batch ===")
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
                print(f"  {f}: IC={ic:.4f}")
                all_results.append({'factor': f, 'ic': ic})
            else:
                all_results.append({'factor': f, 'ic': np.nan})
        except Exception as e:
            print(f"  {f}: ERROR {str(e)[:30]}")
            all_results.append({'factor': f, 'ic': np.nan})

df = pd.DataFrame(all_results).sort_values('ic', ascending=False)
df.to_csv('/Users/leolee/Desktop/qmt_investment_assistant/artifacts/all_factors_ic.csv', index=False)
print("\n=== ALL RESULTS ===")
print(df.to_string(index=False))