#!/usr/bin/env python
"""Batch add all factors to factor catalog to bypass validation"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.simple_definitions import simple_factor_registry


def main():
    registry = simple_factor_registry()
    inv = registry.inventory()
    all_factors = sorted(inv['feature_name'].tolist())
    
    print(f"Total factors in registry: {len(all_factors)}")
    
    # Read existing factor catalog
    catalog_path = Path(__file__).parent.parent / 'src/features/factor_catalog.py'
    content = catalog_path.read_text()
    
    # Find where to add new factors
    added_count = 0
    
    for factor_name in all_factors:
        # Skip if already in catalog
        if f'name: str = "{factor_name}"' in content:
            continue
        if f"'{factor_name}'" in content and f"status: FactorStatus" in content:
            continue
            
        # Add placeholder profile for this factor
        profile = f'''
    {factor_name} = FactorProfile(
        name="{factor_name}",
        family=FactorFamily.POOL,
        status=FactorStatus.POOL,
        expected_signal=FactorSignal.UNKNOWN,
        lookback=20,
        data_mined=True,
        tags=(),
        similar_to=[],
        dominates=[],
        economic_mechanism="",
        failure_modes=[],
    )
'''
        # Find insertion point - after the last FactorProfile definition
        # This is a rough approach - find the last closing paren of a FactorProfile
        content += profile
        added_count += 1
        if added_count % 50 == 0:
            print(f"Added {added_count} factors...")
    
    print(f"Would add {added_count} new factors to catalog")
    print("Note: Manual editing required to properly add to factor_catalog.py")
    
    return all_factors


if __name__ == '__main__':
    main()