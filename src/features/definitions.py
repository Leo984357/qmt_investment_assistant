from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureRegistry, FeatureSpec


def _group_pct_change(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda series: series.pct_change(window))


def _group_rolling_vol(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda series: series.pct_change().rolling(window).std())


def _group_rolling_mean(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda series: series.rolling(window).mean())


def _group_liquidity_ratio(frame: pd.DataFrame, column: str, short_window: int, long_window: int) -> pd.Series:
    grouped = frame.groupby('symbol')[column]
    short_mean = grouped.transform(lambda series: series.rolling(short_window).mean())
    long_mean = grouped.transform(lambda series: series.rolling(long_window).mean())
    return short_mean / long_mean.replace(0.0, np.nan)


def default_feature_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    registry.register(
        FeatureSpec(
            name='mom20',
            inputs=('adj_close',),
            lookback=20,
            description='20-day momentum.',
            compute=lambda bars: _group_pct_change(bars, 'adj_close', 20),
            category='momentum',
            preprocessing=('winsorize', 'cross_sectional_scale'),
            economic_meaning='中短期趋势延续，反映资金对近期强势股票的持续追逐。',
            logic='如果价格在过去20日持续走强，通常意味着短期资金行为和基本面预期在同向强化。',
            failure_modes='震荡市或政策突发反转阶段容易失效，且可能和高 beta 暴露混在一起。',
        )
    )
    registry.register(
        FeatureSpec(
            name='mom60',
            inputs=('adj_close',),
            lookback=60,
            description='60-day momentum.',
            compute=lambda bars: _group_pct_change(bars, 'adj_close', 60),
            category='momentum',
            preprocessing=('winsorize', 'cross_sectional_scale'),
            economic_meaning='中期趋势强度，反映更稳态的资金偏好和景气延续。',
            logic='比20日更慢，能过滤部分短期噪声，更接近原策略里给 mom60 较高权重的核心趋势信号。',
            failure_modes='风格急切换时反应偏慢，可能在高换手反转环境下滞后。',
        )
    )
    registry.register(
        FeatureSpec(
            name='mom120',
            inputs=('adj_close',),
            lookback=120,
            description='120-day momentum.',
            compute=lambda bars: _group_pct_change(bars, 'adj_close', 120),
            category='momentum',
            preprocessing=('winsorize', 'cross_sectional_scale'),
            economic_meaning='长一点的趋势持续性，反映中长期资金共识。',
            logic='为模型提供更平滑的趋势背景，帮助区分短期脉冲和持续强势。',
            failure_modes='长期单边之后容易拥挤，在估值修正或风格切换时回撤更明显。',
        )
    )
    registry.register(
        FeatureSpec(
            name='rev5',
            inputs=('adj_close',),
            lookback=5,
            description='5-day reversal, higher means sharper recent pullback.',
            compute=lambda bars: -_group_pct_change(bars, 'adj_close', 5),
            category='reversal',
            preprocessing=('winsorize', 'cross_sectional_scale'),
            economic_meaning='短期超跌反弹或情绪过冲修复。',
            logic='在中期趋势仍在时，短线回撤后的修复常提供更优入场点。',
            failure_modes='如果下跌是基本面恶化而非短期扰动，反转因子会变成接飞刀。',
        )
    )
    registry.register(
        FeatureSpec(
            name='vol20',
            inputs=('adj_close',),
            lookback=20,
            description='20-day return volatility.',
            compute=lambda bars: _group_rolling_vol(bars, 'adj_close', 20),
            category='risk',
            preprocessing=('winsorize', 'cross_sectional_scale'),
            economic_meaning='短期波动风险和不确定性溢价。',
            logic='高波动股票往往需要更高预期收益才能被持有，也常伴随情绪交易。',
            failure_modes='在风险偏好极强阶段，高波动反而可能带来更高弹性，符号会时变。',
        )
    )
    registry.register(
        FeatureSpec(
            name='vol60',
            inputs=('adj_close',),
            lookback=60,
            description='60-day return volatility.',
            compute=lambda bars: _group_rolling_vol(bars, 'adj_close', 60),
            category='risk',
            preprocessing=('winsorize', 'cross_sectional_scale'),
            economic_meaning='中期风险水平，过滤纯短噪声波动。',
            logic='比 vol20 更稳，用于刻画一个股票过去一段时间是否处于高风险状态。',
            failure_modes='在波动率 regime 突变时，历史波动率对未来风险的解释力会下降。',
        )
    )
    registry.register(
        FeatureSpec(
            name='liq20',
            inputs=('volume',),
            lookback=60,
            description='20-day average volume divided by 60-day average volume.',
            compute=lambda bars: _group_liquidity_ratio(bars, 'volume', 20, 60),
            category='liquidity',
            preprocessing=('winsorize', 'cross_sectional_scale'),
            economic_meaning='近期成交活跃度相对抬升，反映资金关注度变化。',
            logic='当20日均量明显高于60日均量，通常意味着增量资金或交易拥挤度正在上升。',
            failure_modes='消息驱动和主题炒作会让量能异常放大，未必对应可持续 alpha。',
        )
    )
    return registry
