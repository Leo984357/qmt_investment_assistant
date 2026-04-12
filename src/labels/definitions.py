from __future__ import annotations

from .registry import LabelRegistry, LabelSpec


def default_label_registry() -> LabelRegistry:
    registry = LabelRegistry()
    registry.register(
        LabelSpec(
            name='fwd_return_20d',
            horizon=20,
            description='Forward 20-day total return based on adjusted close.',
            compute=lambda bars: bars.groupby('symbol')['adj_close'].transform(lambda series: series.shift(-20) / series - 1.0),
            category='forward_return',
        )
    )
    return registry
