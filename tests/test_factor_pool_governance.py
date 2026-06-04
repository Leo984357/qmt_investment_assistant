from src.features.factor_pool import (
    POOL_TO_SIMPLE_MAPPING,
    audit_pool_computability,
    build_unified_factor_pool,
)
from src.features.simple_definitions import simple_factor_registry


def test_factor_pool_mapping_targets_are_computable():
    audit = audit_pool_computability()

    assert audit['simple_count'] >= 400
    assert audit['computable_unique'] >= 240
    assert audit['computable_rate'] >= 0.35
    assert audit['invalid_mapping_count'] == 0
    assert audit['invalid_mapping_count'] == 0


def test_factor_pool_names_do_not_have_accidental_whitespace():
    names = [factor.name for factor in build_unified_factor_pool()]
    assert [name for name in names if name != name.strip()] == []


def test_pool_aliases_are_registered_with_dependency_marker():
    registry = simple_factor_registry()
    inventory = registry.inventory().set_index('feature_name')

    for alias_name, target_name in POOL_TO_SIMPLE_MAPPING.items():
        if alias_name == target_name:
            continue
        if target_name not in inventory.index:
            continue

        assert alias_name in inventory.index
        dependency = inventory.loc[alias_name, 'dependencies']
        if dependency:
            assert dependency == target_name
