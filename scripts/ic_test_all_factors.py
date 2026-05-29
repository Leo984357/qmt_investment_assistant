#!/usr/bin/env python
"""IC test for all available factors - bypasses config validation"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.simple_definitions import simple_factor_registry
from src.data_store.catalog import LocalResearchCatalog


def calculate_factor_ic(
    feature_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    factor_name: str,
) -> dict:
    """Calculate IC (Spearman correlation) for a single factor"""
    merged = feature_panel[['trade_date', 'symbol', factor_name]].merge(
        label_panel[['trade_date', 'symbol', 'label']],
        on=['trade_date', 'symbol'],
        how='inner'
    )
    
    if len(merged) < 30:
        return {'factor': factor_name, 'ic': np.nan, 'rank_ic': np.nan, 'n': len(merged)}
    
    valid = merged.dropna(subset=[factor_name, 'label'])
    if len(valid) < 30:
        return {'factor': factor_name, 'ic': np.nan, 'rank_ic': np.nan, 'n': len(valid)}
    
    ic, _ = spearmanr(valid[factor_name], valid['label'])
    rank_ic = ic  # Spearman is rank IC
    
    return {
        'factor': factor_name,
        'ic': ic,
        'rank_ic': rank_ic,
        'n': len(valid)
    }


def main():
    print("="*60)
    print("IC Test for All 905 Factors")
    print("="*60)
    
    # Get factor registry
    registry = simple_factor_registry()
    inv = registry.inventory()
    all_factors = sorted(inv['feature_name'].tolist())
    print(f"Total factors in registry: {len(all_factors)}")
    
    # Load data
    print("\nLoading data...")
    catalog = LocalResearchCatalog()
    
    # Check what data is available
    try:
        daily_bar = catalog.read_table('daily_bar')
        universe_membership = catalog.read_table('universe_membership')
        print(f"Loaded daily_bar: {len(daily_bar)} rows")
        print(f"Loaded universe_membership: {len(universe_membership)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Try bootstrapping data first...")
        from src.data_sources.factory import build_data_source
        from src.experiment.spec import build_source_data_spec_with_warmup
        
        spec_dict = {
            'data': {
                'source': 'baostock_ashare',
                'snapshot_id': 'hs300_baostock_snapshot_v1',
                'start_date': '2022-09-01',
                'end_date': '2026-03-28',
                'universe_name': 'HS300',
            }
        }
        source_spec = build_source_data_spec_with_warmup(spec_dict)
        source = build_data_source(source_spec)
        catalog.bootstrap(source)
        
        daily_bar = catalog.read_table('daily_bar')
        universe_membership = catalog.read_table('universe_membership')
    
    # Build universe frame
    universe = universe_membership[['trade_date', 'symbol', 'universe_weight']].copy()
    
    # Compute label (fwd_return_20d)
    print("\nComputing label panel...")
    daily_bar = daily_bar.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    label_panel = daily_bar[['trade_date', 'symbol']].copy()
    label_panel['label'] = (
        daily_bar.groupby('symbol')['close']
        .transform(lambda x: x.shift(-20) / x.shift(-1) - 1)
    )
    label_panel = label_panel.dropna(subset=['label'])
    
    # Filter to HS300 universe
    label_panel = label_panel.merge(universe[['trade_date', 'symbol']], on=['trade_date', 'symbol'], how='inner')
    print(f"Label panel: {len(label_panel)} rows")
    
    # Calculate IC for each factor
    print("\nCalculating IC for each factor...")
    results = []
    batch_size = 50
    
    for i, factor_name in enumerate(all_factors):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(all_factors)}")
        
        try:
            # Compute factor panel
            bars = daily_bar.copy()
            feature_registry = simple_factor_registry()
            spec = feature_registry.get(factor_name)
            factor_values = spec.compute(bars)
            
            # Create factor panel
            factor_panel = bars[['trade_date', 'symbol']].copy()
            factor_panel[factor_name] = factor_values
            
            # Calculate IC
            result = calculate_factor_ic(factor_panel, label_panel, factor_name)
            results.append(result)
            
        except Exception as e:
            results.append({
                'factor': factor_name,
                'ic': np.nan,
                'rank_ic': np.nan,
                'n': 0,
                'error': str(e)
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by IC
    results_df = results_df.sort_values('ic', ascending=False, na_position='last')
    
    # Save results
    output_file = Path(__file__).parent.parent / 'artifacts' / 'all_factors_ic.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("TOP 30 FACTORS BY IC")
    print("="*60)
    top30 = results_df.head(30)
    print(top30.to_string(index=False))
    
    print("\n" + "="*60)
    print("BOTTOM 30 FACTORS BY IC")
    print("="*60)
    bottom30 = results_df.tail(30)
    print(bottom30.to_string(index=False))
    
    print(f"\nResults saved to: {output_file}")
    
    return results_df


if __name__ == '__main__':
    results_df = main()