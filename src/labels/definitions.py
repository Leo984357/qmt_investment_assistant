from __future__ import annotations

from .registry import LabelRegistry, LabelSpec


def default_label_registry() -> LabelRegistry:
    registry = LabelRegistry()
    registry.register(
        LabelSpec(
            name='fwd_return_20d',
            horizon=20,
            description='Forward 20-day total return from T+1 execution to T+20. '
                        'Label = price[T+20] / price[T+1] - 1 (not signal-day-based).',
            compute=lambda bars: bars.groupby('symbol')['adj_close'].transform(
                lambda series: series.shift(-20) / series.shift(-1) - 1.0
            ),
            category='forward_return',
        )
    )
    return registry
