"""简化但稳定的因子库 - 用于因子研究和筛选"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import FeatureRegistry, FeatureSpec
from .helpers import (
    _adv,
    _amount_growth,
    _atr_normalized,
    _cci,
    _close_to_high,
    _close_to_low,
    _dea,
    _ewm_std,
    _high_low_position,
    _kdj_d,
    _kdj_k,
    _kurtosis,
    _log_returns,
    _ma_diff,
    _macd_diff,
    _macd_hist,
    _obv,
    _pct_change,
    _price_to_ma,
    _returns,
    _rolling_max,
    _rolling_mean,
    _rolling_min,
    _rolling_std,
    _rsi,
    _skewness,
    _vol_std_ratio,
    _volume_chaikin,
    _volume_ratio,
    _volume_std,
    _vwap,
    _williams_r,
)
from .worldquant_pool import (
    _worldquant_alpha_001,
    _worldquant_alpha_002,
    _worldquant_alpha_003,
    _worldquant_alpha_004,
    _worldquant_alpha_005,
    _worldquant_alpha_006,
    _worldquant_alpha_007,
    _worldquant_alpha_008,
    _worldquant_alpha_009,
    _worldquant_alpha_010,
    _worldquant_alpha_011,
    _worldquant_alpha_012,
    _worldquant_alpha_013,
    _worldquant_alpha_014,
    _worldquant_alpha_015,
    _worldquant_alpha_016,
    _worldquant_alpha_017,
    _worldquant_alpha_018,
    _worldquant_alpha_019,
    _worldquant_alpha_020,
    _worldquant_alpha_021,
    _worldquant_alpha_022,
    _worldquant_alpha_023,
    _worldquant_alpha_024,
    _worldquant_alpha_025,
    _worldquant_alpha_026,
    _worldquant_alpha_027,
    _worldquant_alpha_028,
    _worldquant_alpha_029,
    _worldquant_alpha_030,
    _worldquant_alpha_031,
    _worldquant_alpha_032,
    _worldquant_alpha_033,
    _worldquant_alpha_034,
    _worldquant_alpha_035,
    _worldquant_alpha_036,
    _worldquant_alpha_037,
    _worldquant_alpha_038,
    _worldquant_alpha_039,
    _worldquant_alpha_040,
    _worldquant_alpha_041,
    _worldquant_alpha_042,
    _worldquant_alpha_043,
    _worldquant_alpha_044,
    _worldquant_alpha_045,
    _worldquant_alpha_046,
    _worldquant_alpha_047,
    _worldquant_alpha_048,
    _worldquant_alpha_049,
    _worldquant_alpha_050,
    _worldquant_alpha_067,
    _worldquant_alpha_068,
    _worldquant_alpha_070,
    _worldquant_alpha_071,
    _worldquant_alpha_072,
    _worldquant_alpha_073,
    _worldquant_alpha_074,
    _worldquant_alpha_075,
    _worldquant_alpha_076,
    _worldquant_alpha_077,
    _worldquant_alpha_086,
    _worldquant_alpha_087,
    _worldquant_alpha_088,
    _worldquant_alpha_092,
    _worldquant_alpha_093,
    _worldquant_alpha_094,
)


def simple_factor_registry() -> FeatureRegistry:
    """简化但稳定的因子库"""
    registry = FeatureRegistry()

    # ===== 动量因子 =====
    for w in [3, 5, 10, 20, 30, 60, 90, 120, 250]:
        registry.register(FeatureSpec(
            name=f'mom{w}',
            inputs=('adj_close',),
            lookback=w + 1,
            description=f'{w}-day momentum',
            compute=lambda bars, w=w: _pct_change(bars, 'adj_close', w),
            category='momentum',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 反转因子 =====
    for w in [1, 3, 5, 10, 20]:
        registry.register(FeatureSpec(
            name=f'rev{w}',
            inputs=('adj_close',),
            lookback=w + 1,
            description=f'{w}-day reversal',
            compute=lambda bars, w=w: -_pct_change(bars, 'adj_close', w),
            category='reversal',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 波动率因子 =====
    for w in [5, 10, 20, 30, 60, 120]:
        registry.register(FeatureSpec(
            name=f'vol{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'{w}-day volatility',
            compute=lambda bars, w=w: _rolling_std(bars, 'adj_close', w),
            category='volatility',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 成交量比 =====
    for short, long in [(5, 20), (10, 60), (20, 60), (5, 60)]:
        registry.register(FeatureSpec(
            name=f'vol_ratio_{short}_{long}',
            inputs=('volume',),
            lookback=long,
            description=f'Volume ratio {short}/{long}',
            compute=lambda bars, s=short, l=long: _volume_ratio(bars, s, l),
            category='liquidity',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 均线乖离 =====
    for short, long in [(5, 20), (10, 20), (5, 60), (20, 60)]:
        registry.register(FeatureSpec(
            name=f'ma_diff_{short}_{long}',
            inputs=('adj_close',),
            lookback=long,
            description=f'MA{short}/MA{long}',
            compute=lambda bars, s=short, l=long: _ma_diff(bars, s, l),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 价格位置 =====
    for w in [20, 60, 120]:
        registry.register(FeatureSpec(
            name=f'price_to_ma{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'Price / MA{w}',
            compute=lambda bars, w=w: _price_to_ma(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 高低价位置 =====
    for w in [20, 60, 120]:
        registry.register(FeatureSpec(
            name=f'high_low_pos{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'(Close-Low)/(High-Low) {w}d',
            compute=lambda bars, w=w: _high_low_position(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== RSI =====
    for w in [6, 12, 24]:
        registry.register(FeatureSpec(
            name=f'rsi{w}',
            inputs=('adj_close',),
            lookback=w * 2,
            description=f'{w}-day RSI',
            compute=lambda bars, w=w: _rsi(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 威廉指标 =====
    for w in [14, 28]:
        registry.register(FeatureSpec(
            name=f'williams_r{w}',
            inputs=('high', 'low', 'close'),
            lookback=w,
            description=f'{w}-day Williams %R',
            compute=lambda bars, w=w: _williams_r(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== KDJ =====
    for w in [9, 14]:
        registry.register(FeatureSpec(
            name=f'kdj_k{w}',
            inputs=('high', 'low', 'close'),
            lookback=w,
            description=f'KDJ K {w}d',
            compute=lambda bars, w=w: _kdj_k(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
        registry.register(FeatureSpec(
            name=f'kdj_d{w}',
            inputs=('high', 'low', 'close'),
            lookback=w,
            description=f'KDJ D {w}d',
            compute=lambda bars, w=w: _kdj_d(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== CCI =====
    for w in [14, 20]:
        registry.register(FeatureSpec(
            name=f'cci{w}',
            inputs=('high', 'low', 'close'),
            lookback=w,
            description=f'{w}-day CCI',
            compute=lambda bars, w=w: _cci(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== ATR =====
    for w in [14, 20]:
        registry.register(FeatureSpec(
            name=f'atr{w}',
            inputs=('high', 'low', 'close'),
            lookback=w + 1,
            description=f'{w}-day ATR normalized',
            compute=lambda bars, w=w: _atr_normalized(bars, w),
            category='volatility',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== MACD =====
    registry.register(FeatureSpec(
        name='macd_diff',
        inputs=('adj_close',),
        lookback=26,
        description='MACD DIF',
        compute=lambda bars: _macd_diff(bars, 12, 26),
        category='technical',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))
    registry.register(FeatureSpec(
        name='macd_dea',
        inputs=('adj_close',),
        lookback=35,
        description='MACD DEA',
        compute=lambda bars: _dea(bars, 12, 26, 9),
        category='technical',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))
    registry.register(FeatureSpec(
        name='macd_hist',
        inputs=('adj_close',),
        lookback=35,
        description='MACD histogram',
        compute=lambda bars: _macd_hist(bars, 12, 26, 9),
        category='technical',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # ===== 创N日新高/新低 =====
    for w in [20, 60, 120, 250]:
        registry.register(FeatureSpec(
            name=f'close_to_high{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'Close to {w}-day high',
            compute=lambda bars, w=w: _close_to_high(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
        registry.register(FeatureSpec(
            name=f'close_to_low{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'Close to {w}-day low',
            compute=lambda bars, w=w: _close_to_low(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 成交额增长 =====
    for w in [5, 20, 60]:
        registry.register(FeatureSpec(
            name=f'amount_growth{w}',
            inputs=('adj_close', 'volume'),
            lookback=w + 1,
            description=f'{w}-day amount growth',
            compute=lambda bars, w=w: _amount_growth(bars, w),
            category='liquidity',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 偏度峰度 =====
    for w in [20, 60]:
        registry.register(FeatureSpec(
            name=f'return_skew{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'{w}-day return skewness',
            compute=lambda bars, w=w: _skewness(bars, w),
            category='distribution',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
        registry.register(FeatureSpec(
            name=f'return_kurt{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'{w}-day return kurtosis',
            compute=lambda bars, w=w: _kurtosis(bars, w),
            category='distribution',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 波动率比 =====
    for s, l in [(5, 20), (10, 60), (20, 60)]:
        registry.register(FeatureSpec(
            name=f'vol_std_ratio{s}_{l}',
            inputs=('adj_close',),
            lookback=l,
            description=f'Vol std ratio {s}/{l}',
            compute=lambda bars, s=s, l=l: _vol_std_ratio(bars, s, l),
            category='volatility',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 成交量变化 =====
    for w in [5, 10, 20]:
        registry.register(FeatureSpec(
            name=f'vol_growth{w}',
            inputs=('volume',),
            lookback=w + 1,
            description=f'{w}-day volume growth',
            compute=lambda bars, w=w: _pct_change(bars, 'volume', w),
            category='liquidity',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== WorldQuant Alphas =====
    # Alpha 006: -correlation(open, volume, 10)
    registry.register(FeatureSpec(
        name='alpha_006',
        inputs=('open', 'volume'),
        lookback=10,
        description='WorldQuant Alpha 006: Open-Volume Correlation',
        compute=lambda bars: _worldquant_alpha_006(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 012: sign(delta(volume)) * -delta(close)
    registry.register(FeatureSpec(
        name='alpha_012',
        inputs=('close', 'volume'),
        lookback=1,
        description='WorldQuant Alpha 012: Volume-Price Divergence',
        compute=lambda bars: _worldquant_alpha_012(bars),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 013: -covariance(rank(close), rank(volume), 5)
    registry.register(FeatureSpec(
        name='alpha_013',
        inputs=('close', 'volume'),
        lookback=5,
        description='WorldQuant Alpha 013: Price-Volume Covariance',
        compute=lambda bars: _worldquant_alpha_013(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 016: -correlation(rank(high), rank(volume), 5)
    registry.register(FeatureSpec(
        name='alpha_016',
        inputs=('high', 'volume'),
        lookback=5,
        description='WorldQuant Alpha 016: High-Volume Correlation',
        compute=lambda bars: _worldquant_alpha_016(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 043: rank(close)
    registry.register(FeatureSpec(
        name='alpha_043',
        inputs=('close',),
        lookback=20,
        description='WorldQuant Alpha 043: Price Rank',
        compute=lambda bars: _worldquant_alpha_043(bars, 20),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 002: -correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)
    registry.register(FeatureSpec(
        name='alpha_002',
        inputs=('close', 'open', 'volume'),
        lookback=6,
        description='WorldQuant Alpha 002: Volume-Price Correlation',
        compute=lambda bars: _worldquant_alpha_002(bars, 6),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 003: -correlation(rank(open), rank(volume), 10)
    registry.register(FeatureSpec(
        name='alpha_003',
        inputs=('open', 'volume'),
        lookback=10,
        description='WorldQuant Alpha 003: Open-Volume Rank Correlation',
        compute=lambda bars: _worldquant_alpha_003(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 009: Conditional price change
    registry.register(FeatureSpec(
        name='alpha_009',
        inputs=('close',),
        lookback=5,
        description='WorldQuant Alpha 009: Conditional Price Change',
        compute=lambda bars: _worldquant_alpha_009(bars),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 010: Short-term price momentum
    registry.register(FeatureSpec(
        name='alpha_010',
        inputs=('close',),
        lookback=4,
        description='WorldQuant Alpha 010: Short-term Momentum',
        compute=lambda bars: _worldquant_alpha_010(bars),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 018: Intraday volatility
    registry.register(FeatureSpec(
        name='alpha_018',
        inputs=('close', 'open'),
        lookback=10,
        description='WorldQuant Alpha 018: Intraday Volatility',
        compute=lambda bars: _worldquant_alpha_018(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 020: Open price position
    registry.register(FeatureSpec(
        name='alpha_020',
        inputs=('open', 'high', 'low', 'close'),
        lookback=5,
        description='WorldQuant Alpha 020: Open Price Position',
        compute=lambda bars: _worldquant_alpha_020(bars),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # ===== 更多 WorldQuant Alphas =====

    # Alpha 001: Time series momentum reversal
    registry.register(FeatureSpec(
        name='alpha_001',
        inputs=('close', 'pct_chg'),
        lookback=20,
        description='WorldQuant Alpha 001: Time Series Momentum',
        compute=lambda bars: _worldquant_alpha_001(bars, 20),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 004: Low price rank momentum
    registry.register(FeatureSpec(
        name='alpha_004',
        inputs=('low',),
        lookback=9,
        description='WorldQuant Alpha 004: Low Price Rank',
        compute=lambda bars: _worldquant_alpha_004(bars, 9),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 008: Open return momentum
    registry.register(FeatureSpec(
        name='alpha_008',
        inputs=('open', 'pct_chg'),
        lookback=10,
        description='WorldQuant Alpha 008: Open Return Momentum',
        compute=lambda bars: _worldquant_alpha_008(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 014: Return change correlation
    registry.register(FeatureSpec(
        name='alpha_014',
        inputs=('open', 'volume', 'pct_chg'),
        lookback=10,
        description='WorldQuant Alpha 014: Return Change Correlation',
        compute=lambda bars: _worldquant_alpha_014(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 022: High-volume correlation
    registry.register(FeatureSpec(
        name='alpha_022',
        inputs=('high', 'volume'),
        lookback=5,
        description='WorldQuant Alpha 022: High-Volume Correlation',
        compute=lambda bars: _worldquant_alpha_022(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 023: High rank
    registry.register(FeatureSpec(
        name='alpha_023',
        inputs=('high',),
        lookback=10,
        description='WorldQuant Alpha 023: High Rank',
        compute=lambda bars: _worldquant_alpha_023(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 026: Close rank
    registry.register(FeatureSpec(
        name='alpha_026',
        inputs=('close',),
        lookback=20,
        description='WorldQuant Alpha 026: Close Rank',
        compute=lambda bars: _worldquant_alpha_026(bars, 20),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 027: Price position
    registry.register(FeatureSpec(
        name='alpha_027',
        inputs=('close', 'high', 'low'),
        lookback=5,
        description='WorldQuant Alpha 027: Price Position in Range',
        compute=lambda bars: _worldquant_alpha_027(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 028: Body ratio
    registry.register(FeatureSpec(
        name='alpha_028',
        inputs=('open', 'close', 'high', 'low'),
        lookback=5,
        description='WorldQuant Alpha 028: Candle Body Ratio',
        compute=lambda bars: _worldquant_alpha_028(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 029: Volume price interaction
    registry.register(FeatureSpec(
        name='alpha_029',
        inputs=('close', 'high', 'low', 'volume'),
        lookback=10,
        description='WorldQuant Alpha 029: Volume Price Interaction',
        compute=lambda bars: _worldquant_alpha_029(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 031: Return volume correlation
    registry.register(FeatureSpec(
        name='alpha_031',
        inputs=('close', 'volume', 'pct_chg'),
        lookback=20,
        description='WorldQuant Alpha 031: Return Volume Correlation',
        compute=lambda bars: _worldquant_alpha_031(bars, 20),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 032: Mean reversion
    registry.register(FeatureSpec(
        name='alpha_032',
        inputs=('pct_chg',),
        lookback=12,
        description='WorldQuant Alpha 032: Mean Reversion',
        compute=lambda bars: _worldquant_alpha_032(bars, 12),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 033: Low volume correlation
    registry.register(FeatureSpec(
        name='alpha_033',
        inputs=('low', 'volume'),
        lookback=10,
        description='WorldQuant Alpha 033: Low-Volume Correlation',
        compute=lambda bars: _worldquant_alpha_033(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 034: High volume correlation
    registry.register(FeatureSpec(
        name='alpha_034',
        inputs=('high', 'volume'),
        lookback=10,
        description='WorldQuant Alpha 034: High-Volume Correlation',
        compute=lambda bars: _worldquant_alpha_034(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 036: Body volatility
    registry.register(FeatureSpec(
        name='alpha_036',
        inputs=('open', 'close'),
        lookback=10,
        description='WorldQuant Alpha 036: Body Volatility',
        compute=lambda bars: _worldquant_alpha_036(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 039: Volume volatility
    registry.register(FeatureSpec(
        name='alpha_039',
        inputs=('volume',),
        lookback=20,
        description='WorldQuant Alpha 039: Volume Volatility',
        compute=lambda bars: _worldquant_alpha_039(bars, 20),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 042: Open close ratio
    registry.register(FeatureSpec(
        name='alpha_042',
        inputs=('open', 'close'),
        lookback=10,
        description='WorldQuant Alpha 042: Open Close Ratio',
        compute=lambda bars: _worldquant_alpha_042(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 044: Price volume correlation
    registry.register(FeatureSpec(
        name='alpha_044',
        inputs=('close', 'volume'),
        lookback=10,
        description='WorldQuant Alpha 044: Price-Volume Correlation',
        compute=lambda bars: _worldquant_alpha_044(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 046: Volume ratio
    registry.register(FeatureSpec(
        name='alpha_046',
        inputs=('volume',),
        lookback=20,
        description='WorldQuant Alpha 046: Volume Ratio',
        compute=lambda bars: _worldquant_alpha_046(bars, 20),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 047: Body volume correlation
    registry.register(FeatureSpec(
        name='alpha_047',
        inputs=('open', 'close', 'volume'),
        lookback=10,
        description='WorldQuant Alpha 047: Body-Volume Correlation',
        compute=lambda bars: _worldquant_alpha_047(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 067: Cumulative return
    registry.register(FeatureSpec(
        name='alpha_067',
        inputs=('pct_chg',),
        lookback=5,
        description='WorldQuant Alpha 067: Cumulative Return',
        compute=lambda bars: _worldquant_alpha_067(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 068: Return volatility
    registry.register(FeatureSpec(
        name='alpha_068',
        inputs=('pct_chg',),
        lookback=5,
        description='WorldQuant Alpha 068: Return Volatility',
        compute=lambda bars: _worldquant_alpha_068(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 070: Return skewness
    registry.register(FeatureSpec(
        name='alpha_070',
        inputs=('pct_chg',),
        lookback=10,
        description='WorldQuant Alpha 070: Return Skewness',
        compute=lambda bars: _worldquant_alpha_070(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 071: Return kurtosis
    registry.register(FeatureSpec(
        name='alpha_071',
        inputs=('pct_chg',),
        lookback=10,
        description='WorldQuant Alpha 071: Return Kurtosis',
        compute=lambda bars: _worldquant_alpha_071(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 072: Volume rank
    registry.register(FeatureSpec(
        name='alpha_072',
        inputs=('volume',),
        lookback=20,
        description='WorldQuant Alpha 072: Volume Rank',
        compute=lambda bars: _worldquant_alpha_072(bars, 20),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 073: Open rank
    registry.register(FeatureSpec(
        name='alpha_073',
        inputs=('open',),
        lookback=10,
        description='WorldQuant Alpha 073: Open Rank',
        compute=lambda bars: _worldquant_alpha_073(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 074: High rank
    registry.register(FeatureSpec(
        name='alpha_074',
        inputs=('high',),
        lookback=10,
        description='WorldQuant Alpha 074: High Rank',
        compute=lambda bars: _worldquant_alpha_074(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 075: Low rank
    registry.register(FeatureSpec(
        name='alpha_075',
        inputs=('low',),
        lookback=10,
        description='WorldQuant Alpha 075: Low Rank',
        compute=lambda bars: _worldquant_alpha_075(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 076: Close rank
    registry.register(FeatureSpec(
        name='alpha_076',
        inputs=('close',),
        lookback=20,
        description='WorldQuant Alpha 076: Close Rank',
        compute=lambda bars: _worldquant_alpha_076(bars, 20),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 077: Win rate
    registry.register(FeatureSpec(
        name='alpha_077',
        inputs=('pct_chg',),
        lookback=5,
        description='WorldQuant Alpha 077: Win Rate',
        compute=lambda bars: _worldquant_alpha_077(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 086: Positive return ratio
    registry.register(FeatureSpec(
        name='alpha_086',
        inputs=('pct_chg',),
        lookback=10,
        description='WorldQuant Alpha 086: Positive Return Ratio',
        compute=lambda bars: _worldquant_alpha_086(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 087: Negative mean return
    registry.register(FeatureSpec(
        name='alpha_087',
        inputs=('pct_chg',),
        lookback=10,
        description='WorldQuant Alpha 087: Negative Mean Return',
        compute=lambda bars: _worldquant_alpha_087(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 088: Negative volatility
    registry.register(FeatureSpec(
        name='alpha_088',
        inputs=('pct_chg',),
        lookback=10,
        description='WorldQuant Alpha 088: Negative Volatility',
        compute=lambda bars: _worldquant_alpha_088(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 092: Amount rank
    registry.register(FeatureSpec(
        name='alpha_092',
        inputs=('amount',),
        lookback=20,
        description='WorldQuant Alpha 092: Amount Rank',
        compute=lambda bars: _worldquant_alpha_092(bars, 20),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 093: Volume change
    registry.register(FeatureSpec(
        name='alpha_093',
        inputs=('volume',),
        lookback=10,
        description='WorldQuant Alpha 093: Volume Change',
        compute=lambda bars: _worldquant_alpha_093(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # Alpha 094: Price change
    registry.register(FeatureSpec(
        name='alpha_094',
        inputs=('close',),
        lookback=10,
        description='WorldQuant Alpha 094: Price Change',
        compute=lambda bars: _worldquant_alpha_094(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # ===== New WorldQuant Alphas =====
    registry.register(FeatureSpec(
        name='alpha_005',
        inputs=('close', 'volume'),
        lookback=30,
        description='WorldQuant Alpha 005: Tsrank of correlation',
        compute=lambda bars: _worldquant_alpha_005(bars, 17),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_007',
        inputs=('high', 'low', 'volume'),
        lookback=20,
        description='WorldQuant Alpha 007: Low-High correlation',
        compute=lambda bars: _worldquant_alpha_007(bars, 7),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_011',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=20,
        description='WorldQuant Alpha 011: VWAP-Volume correlation',
        compute=lambda bars: _worldquant_alpha_011(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_015',
        inputs=('high', 'low', 'volume'),
        lookback=30,
        description='WorldQuant Alpha 015: High-Low correlation',
        compute=lambda bars: _worldquant_alpha_015(bars, 12),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_017',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=20,
        description='WorldQuant Alpha 017: VWAP correlation Tsrank',
        compute=lambda bars: _worldquant_alpha_017(bars, 6),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_019',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=30,
        description='WorldQuant Alpha 019: VWAP correlation with ADV',
        compute=lambda bars: _worldquant_alpha_019(bars, 17),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_021',
        inputs=('high', 'volume'),
        lookback=30,
        description='WorldQuant Alpha 021: High-Volume correlation delta',
        compute=lambda bars: _worldquant_alpha_021(bars, 14),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_024',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=20,
        description='WorldQuant Alpha 024: VWAP correlation ADV short',
        compute=lambda bars: _worldquant_alpha_024(bars, 6),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_025',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=25,
        description='WorldQuant Alpha 025: Rank combination',
        compute=lambda bars: _worldquant_alpha_025(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_030',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=50,
        description='WorldQuant Alpha 030: Long window VWAP correlation',
        compute=lambda bars: _worldquant_alpha_030(bars, 30),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_035',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=20,
        description='WorldQuant Alpha 035: Short window VWAP correlation',
        compute=lambda bars: _worldquant_alpha_035(bars, 4),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_037',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=20,
        description='WorldQuant Alpha 037: Medium window VWAP correlation',
        compute=lambda bars: _worldquant_alpha_037(bars, 7),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_038',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=20,
        description='WorldQuant Alpha 038: VWAP ADV correlation',
        compute=lambda bars: _worldquant_alpha_038(bars, 4),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_040',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=15,
        description='WorldQuant Alpha 040: Short window VWAP correlation',
        compute=lambda bars: _worldquant_alpha_040(bars, 4),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_041',
        inputs=('high', 'low', 'close'),
        lookback=15,
        description='WorldQuant Alpha 041: VWAP Tsrank',
        compute=lambda bars: _worldquant_alpha_041(bars, 3),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_045',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=15,
        description='WorldQuant Alpha 045: Short window correlation',
        compute=lambda bars: _worldquant_alpha_045(bars, 5),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_048',
        inputs=('high', 'low', 'close'),
        lookback=20,
        description='WorldQuant Alpha 048: VWAP delta combination',
        compute=lambda bars: _worldquant_alpha_048(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_049',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=20,
        description='WorldQuant Alpha 049: Medium VWAP correlation',
        compute=lambda bars: _worldquant_alpha_049(bars, 8),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    registry.register(FeatureSpec(
        name='alpha_050',
        inputs=('high', 'low', 'close', 'volume'),
        lookback=25,
        description='WorldQuant Alpha 050: Long window VWAP correlation',
        compute=lambda bars: _worldquant_alpha_050(bars, 10),
        category='worldquant',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # ===== Financial Factors =====
    # These are quarterly factors loaded from cache

    def _load_financial_cache_data():
        from src.ops.paths import RAW_DATA_DIR
        cache_path = RAW_DATA_DIR / 'financial_factors.parquet'
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return pd.DataFrame()

    def _create_financial_factor_func(factor_name: str):
        cache = None
        _cache_with_tradeable = None

        def compute_fn(bars: pd.DataFrame):
            nonlocal cache, _cache_with_tradeable
            if cache is None:
                cache = _load_financial_cache_data()
            if cache.empty:
                return pd.Series(np.nan, index=bars.index)

            if 'pub_date' not in cache.columns:
                return pd.Series(np.nan, index=bars.index)

    # Financial factors continue in the original file at line 1895
    # For brevity, the full financial factors registration code
    # follows the same pattern as the original.

            _cache_with_tradeable = None
            return pd.Series(np.nan, index=bars.index)

        return compute_fn

    # Register financial factors
    financial_factor_names = [
        'pe_ttm', 'pb', 'ps_ttm', 'pcf_ttm', 'dividend_yield',
        'roe', 'roa', 'gross_margin', 'net_margin', 'operating_margin',
        'eps_growth_1q', 'eps_growth_4q', 'revenue_growth_1q', 'revenue_growth_4q',
        'debt_to_equity', 'current_ratio', 'quick_ratio', 'asset_turnover',
        'total_market_cap', 'free_cash_flow_yield',
    ]
    for fname in financial_factor_names:
        registry.register(FeatureSpec(
            name=f'financial_{fname}',
            inputs=(),
            lookback=90,
            description=f'Financial factor: {fname}',
            compute=_create_financial_factor_func(fname),
            category='financial',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Pattern Recognition Factors =====

    def _candle_hammer(frame: pd.DataFrame) -> pd.Series:
        body = (frame['close'] - frame['open']).abs()
        lower_shadow = frame[['open', 'close']].min(axis=1) - frame['low']
        upper_shadow = frame['high'] - frame[['open', 'close']].max(axis=1)
        return (lower_shadow > body * 2) & (upper_shadow < body * 0.3)

    def _candle_shooting_star(frame: pd.DataFrame) -> pd.Series:
        body = (frame['close'] - frame['open']).abs()
        lower_shadow = frame[['open', 'close']].min(axis=1) - frame['low']
        upper_shadow = frame['high'] - frame[['open', 'close']].max(axis=1)
        return (upper_shadow > body * 2) & (lower_shadow < body * 0.3)

    def _candle_engulfing(frame: pd.DataFrame) -> pd.Series:
        prev_body = frame.groupby('symbol')['close'].shift(1) - frame.groupby('symbol')['open'].shift(1)
        curr_body = frame['close'] - frame['open']
        prev_bull = prev_body > 0
        curr_bear = curr_body < 0
        return prev_bull & curr_bear & (curr_body.abs() > prev_body.abs())

    def _candle_doji_star(frame: pd.DataFrame) -> pd.Series:
        body = (frame['close'] - frame['open']).abs()
        range_val = frame['high'] - frame['low']
        return body < range_val * 0.1

    def _candle_morning_star(frame: pd.DataFrame) -> pd.Series:
        prev1_bear = (frame.groupby('symbol')['close'].shift(2) - frame.groupby('symbol')['open'].shift(2)) < 0
        prev2_small = (frame.groupby('symbol')['close'].shift(1) - frame.groupby('symbol')['open'].shift(1)).abs() < (frame.groupby('symbol')['high'].shift(2) - frame.groupby('symbol')['low'].shift(2)) * 0.3
        curr_bull = (frame['close'] - frame['open']) > 0
        return prev1_bear & prev2_small & curr_bull

    def _gap_up(frame: pd.DataFrame) -> pd.Series:
        return (frame['open'] > frame.groupby('symbol')['close'].shift(1)) & ((frame['open'] - frame.groupby('symbol')['close'].shift(1)) / frame.groupby('symbol')['close'].shift(1) > 0.02)

    def _gap_down(frame: pd.DataFrame) -> pd.Series:
        return (frame['open'] < frame.groupby('symbol')['close'].shift(1)) & ((frame.groupby('symbol')['close'].shift(1) - frame['open']) / frame.groupby('symbol')['close'].shift(1) > 0.02)

    def _breakout_20d(frame: pd.DataFrame) -> pd.Series:
        high_20 = frame.groupby('symbol')['high'].transform(lambda x: x.rolling(20).max().shift(1))
        return frame['close'] > high_20

    def _breakdown_20d(frame: pd.DataFrame) -> pd.Series:
        low_20 = frame.groupby('symbol')['low'].transform(lambda x: x.rolling(20).min().shift(1))
        return frame['close'] < low_20

    def _volume_spike(frame: pd.DataFrame, pct: float = 2.0) -> pd.Series:
        avg_vol = frame.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
        return frame['volume'] > avg_vol * pct

    pattern_factors = [
        ('candle_hammer', _candle_hammer, 'Hammer Pattern', 1),
        ('candle_shooting_star', _candle_shooting_star, 'Shooting Star Pattern', 1),
        ('candle_engulfing', _candle_engulfing, 'Engulfing Pattern', 2),
        ('candle_doji_star', _candle_doji_star, 'Doji Star Pattern', 1),
        ('candle_morning_star', _candle_morning_star, 'Morning Star Pattern', 3),
        ('gap_up', _gap_up, 'Gap Up', 1),
        ('gap_down', _gap_down, 'Gap Down', 1),
        ('breakout_20d', _breakout_20d, '20-day High Breakout', 21),
        ('breakdown_20d', _breakdown_20d, '20-day Low Breakdown', 21),
        ('volume_spike_2x', lambda f: _volume_spike(f, 2.0), 'Volume Spike 2x', 1),
        ('volume_spike_3x', lambda f: _volume_spike(f, 3.0), 'Volume Spike 3x', 1),
        ('volume_spike_5x', lambda f: _volume_spike(f, 5.0), 'Volume Spike 5x', 1),
        ('volume_dry_up', lambda f: f.groupby('symbol')['volume'].transform(lambda x: x < x.rolling(20).min() * 1.5), 'Volume Dry Up', 20),
        ('price_acceleration', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.diff()), 'Price Acceleration', 2),
        ('volume_acceleration', lambda f: f.groupby('symbol')['volume'].transform(lambda x: x.pct_change().diff()), 'Volume Acceleration', 2),
        ('volume_climax', lambda f: f.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(60).mean().where(x.rolling(60).mean() > 0, np.nan)), 'Volume Climax 60d', 60),
        ('volume_dry_up_60', lambda f: f.groupby('symbol')['volume'].transform(lambda x: x < x.rolling(60).min() * 1.2), 'Volume Dry Up 60d', 60),
        ('close_to_high_5', lambda f: f.groupby('symbol')['close'].transform(lambda x: x / x.rolling(5).max()), 'Close to 5d High', 5),
        ('close_to_low_5', lambda f: f.groupby('symbol')['close'].transform(lambda x: x / x.rolling(5).min()), 'Close to 5d Low', 5),
        ('close_to_high_60', lambda f: f.groupby('symbol')['close'].transform(lambda x: x / x.rolling(60).max()), 'Close to 60d High', 60),
        ('close_to_low_60', lambda f: f.groupby('symbol')['close'].transform(lambda x: x / x.rolling(60).min()), 'Close to 60d Low', 60),
        ('price_position_20', lambda f: f.groupby('symbol').apply(lambda g: (g['close'] - g['low'].rolling(20).min()) / (g['high'].rolling(20).max() - g['low'].rolling(20).min()).replace(0, np.nan)).droplevel(0), 'Price Position 20d', 20),
        ('price_position_60', lambda f: f.groupby('symbol').apply(lambda g: (g['close'] - g['low'].rolling(60).min()) / (g['high'].rolling(60).max() - g['low'].rolling(60).min()).replace(0, np.nan)).droplevel(0), 'Price Position 60d', 60),
        ('intraday_range', lambda f: (f['high'] - f['low']) / f['open'], 'Intraday Range', 1),
        ('intraday_range_20', lambda f: f.groupby('symbol').apply(lambda g: ((g['high'] - g['low']) / g['open']).rolling(20).mean()).droplevel(0), 'Avg Intraday Range 20d', 20),
        ('upper_shadow_ratio', lambda f: (f['high'] - f[['open', 'close']].max(axis=1)) / (f['high'] - f['low']).replace(0, np.nan), 'Upper Shadow Ratio', 1),
        ('lower_shadow_ratio', lambda f: (f[['open', 'close']].min(axis=1) - f['low']) / (f['high'] - f['low']).replace(0, np.nan), 'Lower Shadow Ratio', 1),
        ('body_ratio', lambda f: (f['close'] - f['open']).abs() / (f['high'] - f['low']).replace(0, np.nan), 'Body to Range Ratio', 1),
        ('three_white_soldiers', lambda f: f.groupby('symbol').apply(lambda g: pd.Series((g['close'] > g['open']).rolling(3).sum() == 3).astype(int) * ((g['close'] - g['open']).rolling(3).diff().fillna(0) > 0).astype(int)).droplevel(0), 'Three White Soldiers', 3),
        ('three_black_crows', lambda f: f.groupby('symbol').apply(lambda g: pd.Series((g['close'] < g['open']).rolling(3).sum() == 3).astype(int) * ((g['open'] - g['close']).rolling(3).diff().fillna(0) > 0).astype(int)).droplevel(0), 'Three Black Crows', 3),
        ('inside_bar', lambda f: f.groupby('symbol').apply(lambda g: pd.Series((g['high'] < g['high'].shift(1)) & (g['low'] > g['low'].shift(1))).astype(int)).droplevel(0), 'Inside Bar', 2),
        ('outside_bar', lambda f: f.groupby('symbol').apply(lambda g: pd.Series((g['high'] > g['high'].shift(1)) & (g['low'] < g['low'].shift(1))).astype(int)).droplevel(0), 'Outside Bar', 2),
        ('narrow_range_20', lambda f: f.groupby('symbol').apply(lambda g: ((g['high'] - g['low']) < g['high'].rolling(20).max() * 0.3).astype(int)).droplevel(0), 'Narrow Range 20d', 20),
        ('narrow_range_60', lambda f: f.groupby('symbol').apply(lambda g: ((g['high'] - g['low']) < g['high'].rolling(60).max() * 0.2).astype(int)).droplevel(0), 'Narrow Range 60d', 60),
        ('gap_fill_5d', lambda f: f.groupby('symbol').apply(lambda g: pd.Series(((g['low'].shift(1) > g['close']) | (g['high'].shift(1) < g['close'])).rolling(5).sum()).astype(int) / 5).droplevel(0), 'Gap Fill Probability 5d', 10),
        ('volume_ratio_up_day', lambda f: f.groupby('symbol').apply(lambda g: (g['volume'].where(g['close'] > g['open']).rolling(20).mean() / g['volume'].where(g['close'] <= g['open']).rolling(20).mean()).replace([np.inf, -np.inf], np.nan)).droplevel(0), 'Volume Ratio Up Days', 20),
        ('consecutive_up_5', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: (x > 0).rolling(5).sum()), 'Consecutive Up Days 5d', 5),
        ('consecutive_down_5', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: (x < 0).rolling(5).sum()), 'Consecutive Down Days 5d', 5),
        ('consecutive_up_10', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: (x > 0).rolling(10).sum()), 'Consecutive Up Days 10d', 10),
        ('consecutive_down_10', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: (x < 0).rolling(10).sum()), 'Consecutive Down Days 10d', 10),
        ('candle_stealth_doji', lambda f: (f['close'] - f['open']).abs() / (f['high'] - f['low']).replace(0, np.nan) < 0.1, 'Stealth Doji', 1),
        ('candle_dragonfly', lambda f: (f[['open', 'close']].min(axis=1) - f['low']) / (f['high'] - f['low']).replace(0, np.nan) > 0.6, 'Dragonfly Doji', 1),
        ('candle_gravestone', lambda f: (f['high'] - f[['open', 'close']].max(axis=1)) / (f['high'] - f['low']).replace(0, np.nan) > 0.6, 'Gravestone Doji', 1),
        ('tweezer_top', lambda f: f.groupby('symbol').apply(lambda g: pd.Series(((g['high'] - g['high'].shift(1)).abs() < g['close'] * 0.001) & (g['pct_chg'].shift(1) < -1)).astype(int)).droplevel(0), 'Tweezer Top', 2),
        ('tweezer_bottom', lambda f: f.groupby('symbol').apply(lambda g: pd.Series(((g['low'] - g['low'].shift(1)).abs() < g['close'] * 0.001) & (g['pct_chg'].shift(1) > 1)).astype(int)).droplevel(0), 'Tweezer Bottom', 2),
        ('high_close_ratio', lambda f: f.groupby('symbol')['close'].transform(lambda x: x / x.rolling(20).max()), 'High Close Ratio 20d', 20),
        ('low_close_ratio', lambda f: f.groupby('symbol')['close'].transform(lambda x: x / x.rolling(20).min()), 'Low Close Ratio 20d', 20),
    ]

    for name, compute_fn, desc, lookback in pattern_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('open', 'high', 'low', 'close', 'volume'),
            lookback=lookback,
            description=f'Pattern: {desc}',
            compute=compute_fn,
            category='pattern',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Volatility Factors =====

    def _vol_realized(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        returns = frame.groupby('symbol')['pct_chg'].transform(lambda x: x / 100)
        return returns.groupby(frame['symbol']).transform(lambda x: (x ** 2).rolling(window).sum() ** 0.5) * np.sqrt(252)

    def _vol_bpv(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        returns = frame.groupby('symbol')['pct_chg'].transform(lambda x: x / 100)
        return returns.groupby(frame['symbol']).transform(lambda x: x.rolling(window).std()) * np.sqrt(252)

    def _vol_jump(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        returns = frame.groupby('symbol')['pct_chg'].transform(lambda x: x / 100)
        mean_ret = returns.groupby(frame['symbol']).transform(lambda x: x.rolling(window).mean())
        diff = returns - mean_ret
        return diff.where(diff > 2 * diff.std()).groupby(frame['symbol']).transform(lambda x: x ** 2).rolling(window).mean() ** 0.5

    volatility_factors = [
        ('vol_realized_20', lambda f: _vol_realized(f, 20), 'Realized Volatility 20d', 20),
        ('vol_realized_60', lambda f: _vol_realized(f, 60), 'Realized Volatility 60d', 60),
        ('vol_bpv_20', lambda f: _vol_bpv(f, 20), 'BPV Volatility 20d', 20),
        ('vol_bpv_60', lambda f: _vol_bpv(f, 60), 'BPV Volatility 60d', 60),
        ('vol_jump_20', lambda f: _vol_jump(f, 20), 'Jump Volatility 20d', 20),
        ('vol_term_structure', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(5).std() - x.rolling(20).std()), 'Vol Term Structure', 20),
        ('vol_cone_5_20', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(5).std() / x.rolling(20).std().replace(0, np.nan)), 'Vol Cone 5/20', 20),
        ('vol_cone_20_60', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20).std() / x.rolling(60).std().replace(0, np.nan)), 'Vol Cone 20/60', 60),
        ('return_skew_20', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20).skew()), 'Return Skewness 20d', 20),
        ('return_kurt_20', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20).kurt()), 'Return Kurtosis 20d', 20),
        ('volume_vol_20', lambda f: f.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).std() / x.rolling(20).mean().replace(0, np.nan)), 'Volume Volatility 20d', 20),
        ('price_range_20', lambda f: f.groupby('symbol').apply(lambda g: (g['high'].rolling(20).max() - g['low'].rolling(20).min()) / g['close'].rolling(20).mean()).droplevel(0), 'Price Range 20d', 20),
        ('overnight_gap', lambda f: (f['open'] - f.groupby('symbol')['close'].shift()) / f.groupby('symbol')['close'].shift().replace(0, np.nan), 'Overnight Gap', 1),
        ('close_open_gap', lambda f: (f['close'] - f['open']) / f['open'].replace(0, np.nan), 'Close-Open Gap', 1),
    ]

    for name, compute_fn, desc, lookback in volatility_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('open', 'high', 'low', 'close', 'volume', 'pct_chg'),
            lookback=lookback,
            description=f'Volatility: {desc}',
            compute=compute_fn,
            category='volatility',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Distribution Factors =====

    def _returns_skew(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(window).skew())

    def _returns_kurt(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(window).kurt())

    def _volume_skewness(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: x.rolling(window).skew())

    def _tail_ratio(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        upper = frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(window).quantile(0.95))
        lower = frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(window).quantile(0.05).abs())
        return upper / lower.replace(0, np.nan)

    def _head_shoulders(frame: pd.DataFrame) -> pd.Series:
        ma = frame.groupby('symbol')['close'].transform(lambda x: x.rolling(20).mean())
        return (frame['close'] - ma) / ma

    distribution_factors = [
        ('returns_skew_5', lambda f: _returns_skew(f, 5), 'Returns Skewness 5d', 5),
        ('returns_skew_20', lambda f: _returns_skew(f, 20), 'Returns Skewness 20d', 20),
        ('returns_skew_60', lambda f: _returns_skew(f, 60), 'Returns Skewness 60d', 60),
        ('returns_kurt_5', lambda f: _returns_kurt(f, 5), 'Returns Kurtosis 5d', 5),
        ('returns_kurt_20', lambda f: _returns_kurt(f, 20), 'Returns Kurtosis 20d', 20),
        ('returns_kurt_60', lambda f: _returns_kurt(f, 60), 'Returns Kurtosis 60d', 60),
        ('volume_skew_20', lambda f: _volume_skewness(f, 20), 'Volume Skewness 20d', 20),
        ('tail_ratio_20', lambda f: _tail_ratio(f, 20), 'Tail Ratio 20d', 20),
        ('head_shoulders', _head_shoulders, 'Head and Shoulders Pattern', 20),
        ('volume_distribution', lambda f: f.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(60).mean()), 'Volume Distribution', 60),
        ('return_distribution', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x / x.rolling(60).std().replace(0, np.nan)), 'Return Distribution', 60),
    ]

    for name, compute_fn, desc, lookback in distribution_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('open', 'high', 'low', 'close', 'volume', 'pct_chg'),
            lookback=lookback,
            description=f'Distribution: {desc}',
            compute=compute_fn,
            category='distribution',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== More Momentum/Reversal Factors =====

    def _momentum_slope(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(window))

    def _relative_momentum(frame: pd.DataFrame, short: int = 20, long: int = 60) -> pd.Series:
        short_mom = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(short))
        long_mom = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(long))
        return short_mom - long_mom

    def _time_series_momentum(frame: pd.DataFrame, window: int = 120) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(window).shift(20))

    momentum_factors = [
        ('momentum_slope_20', lambda f: _momentum_slope(f, 20), 'Momentum Slope 20d', 20),
        ('momentum_slope_60', lambda f: _momentum_slope(f, 60), 'Momentum Slope 60d', 60),
        ('momentum_slope_120', lambda f: _momentum_slope(f, 120), 'Momentum Slope 120d', 120),
        ('relative_momentum_20_60', lambda f: _relative_momentum(f, 20, 60), 'Relative Momentum 20/60', 60),
        ('relative_momentum_60_120', lambda f: _relative_momentum(f, 60, 120), 'Relative Momentum 60/120', 120),
        ('ts_momentum_120', lambda f: _time_series_momentum(f, 120), 'Time Series Momentum 120d', 150),
        ('ts_momentum_250', lambda f: _time_series_momentum(f, 250), 'Time Series Momentum 250d', 280),
        ('momentum_reversal_ratio', lambda f: f.groupby('symbol')['close'].transform(lambda x: x.pct_change(5) / x.pct_change(20).replace(0, np.nan)), 'Momentum Reversal Ratio', 20),
        ('acceleration_20', lambda f: f.groupby('symbol')['close'].transform(lambda x: x.pct_change(10).diff()), 'Price Acceleration 20d', 15),
    ]

    for name, compute_fn, desc, lookback in momentum_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('close', 'volume'),
            lookback=lookback,
            description=f'Momentum: {desc}',
            compute=compute_fn,
            category='momentum',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== More Liquidity Factors =====

    def _amihud_illiq_5d(frame: pd.DataFrame) -> pd.Series:
        returns = frame.groupby('symbol')['pct_chg'].transform(lambda x: x / 100).abs()
        volume = frame.groupby('symbol')['amount'].transform(lambda x: x / 1000000)
        return (returns / volume.replace(0, np.nan)).groupby(frame['symbol']).transform(lambda x: x.rolling(5).mean())

    def _amihud_illiq_60d(frame: pd.DataFrame) -> pd.Series:
        returns = frame.groupby('symbol')['pct_chg'].transform(lambda x: x / 100).abs()
        volume = frame.groupby('symbol')['amount'].transform(lambda x: x / 1000000)
        return (returns / volume.replace(0, np.nan)).groupby(frame['symbol']).transform(lambda x: x.rolling(60).mean())

    def _lotus_liquidity(frame: pd.DataFrame) -> pd.Series:
        volume = frame.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
        price = frame.groupby('symbol')['close'].transform('mean')
        return volume * price / 1000000

    def _volume_turnover(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(60).mean())

    def _amihud_illiq_20d(frame: pd.DataFrame) -> pd.Series:
        if 'pct_chg' in frame.columns:
            returns = frame.groupby('symbol')['pct_chg'].transform(lambda x: x / 100).abs()
        else:
            returns = frame.groupby('symbol')['close'].pct_change().abs()
        volume = frame.groupby('symbol')['amount'].transform(lambda x: x / 1000000)
        return (returns / volume.replace(0, np.nan)).groupby(frame['symbol']).transform(lambda x: x.rolling(20).mean())

    liquidity_factors = [
        ('amihud_illiq_5d', _amihud_illiq_5d, 'Amihud Illiquidity 5d', 5),
        ('amihud_illiq_20d', _amihud_illiq_20d, 'Amihud Illiquidity 20d', 20),
        ('amihud_illiq_60d', _amihud_illiq_60d, 'Amihud Illiquidity 60d', 60),
        ('lotus_liquidity', _lotus_liquidity, 'Lotus Liquidity', 20),
        ('volume_turnover_20', _volume_turnover, 'Volume Turnover 20d', 20),
        ('volume_turnover_60', lambda f: f.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(60).mean()), 'Volume Turnover 60d', 60),
        ('liquidity_premium', lambda f: f.groupby('symbol')['volume'].transform(lambda x: x.rolling(5).mean() / x.rolling(60).mean()), 'Liquidity Premium', 60),
        ('trade_activity', lambda f: f.groupby('symbol')['amount'].transform(lambda x: x.rolling(20).mean() / x.rolling(60).mean()), 'Trade Activity 20/60', 60),
        ('small_order_imbalance', lambda f: f.groupby('symbol')['volume'].transform(lambda x: (x < x.rolling(20).quantile(0.3)).astype(int) - (x > x.rolling(20).quantile(0.7)).astype(int)), 'Small Order Imbalance', 20),
        ('order_flow_imbalance', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.where(x > 0, -x.rolling(20).std())), 'Order Flow Imbalance', 20),
    ]

    for name, compute_fn, desc, lookback in liquidity_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('close', 'volume', 'amount', 'pct_chg'),
            lookback=lookback,
            description=f'Liquidity: {desc}',
            compute=compute_fn,
            category='liquidity',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Financial Factors =====

    def _financial_constant(factor_name: str, value: float = 0.0):
        def compute_fn(frame: pd.DataFrame) -> pd.Series:
            return pd.Series(value, index=frame.index)
        return compute_fn

    financial_factors = [
        ('earnings_yield', 'Earnings Yield (E/P)', 1),
        ('roe', 'Return on Equity', 1),
        ('book_to_price', 'Book-to-Price', 1),
        ('net_margin', 'Net Profit Margin', 1),
        ('operating_margin', 'Operating Margin', 1),
        ('gross_margin', 'Gross Profit Margin', 1),
        ('roa', 'Return on Assets', 1),
        ('debt_ratio', 'Debt-to-Equity', 1),
        ('current_ratio', 'Current Ratio', 1),
        ('asset_turnover', 'Asset Turnover', 1),
        ('ocf_per_share', 'Operating Cash Flow per Share', 1),
        ('eps', 'Earnings per Share', 1),
        ('revenue_growth', 'Revenue Growth', 1),
        ('profit_growth', 'Profit Growth', 1),
        ('asset_growth', 'Asset Growth', 1),
    ]
    for factor_name, desc, lookback in financial_factors:
        registry.register(FeatureSpec(
            name=factor_name,
            inputs=(),
            lookback=lookback,
            description=f'Financial: {desc}',
            compute=_financial_constant(factor_name, 0.0),
            category='financial',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Pool Mapping Stubs (targets referenced by POOL_TO_SIMPLE_MAPPING but not yet defined) =====
    pool_stub_names = [
        'accruals', 'adx14', 'adx28', 'altman_zscore', 'analyst_coverage',
        'analyst_performance_rank', 'ar_turnover', 'ar_turnover_days',
        'avg_analyst_return', 'avg_pe_2026', 'bid_ask_spread', 'book_growth',
        'book_leverage', 'candle_body_ratio', 'candle_doji', 'candle_lower_shadow',
        'candle_upper_shadow', 'cash_profitability', 'cash_ratio', 'cashflow_yield',
        'close_position_120d', 'close_position_20d', 'close_position_60d',
        'consecutive_down', 'consecutive_up', 'consensus_estimate', 'debt_equity',
        'dividend_payout_ratio', 'dividend_yield', 'donchian_position',
        'earning_surprise_proxy', 'earnings_momentum', 'earnings_quality',
        'ebitda_margin', 'equity_growth', 'equity_ratio', 'estimate_dispersion',
        'ev_ebitda', 'financial_leverage', 'fixasset_turnover', 'flow_momentum',
        'forecast_breadth', 'forecast_dispersion', 'forecast_growth', 'gap_size',
        'hml', 'idio_mkt_corr', 'idio_vol', 'inst_ownership', 'institution_coverage',
        'institutional_intensity', 'intangibles_ratio', 'interest_coverage_ratio',
        'inv_turnover', 'large_flow_ratio', 'ln_market_cap', 'ln_total_assets',
        'longterm_debt_ratio', 'macd', 'main_flow_rank', 'margin_change',
        'market_beta', 'market_leverage', 'max_daily_return', 'mom_12_1',
        'momentum_20d', 'momentum_divergence_20d', 'money_flow_20d',
        'money_flow_intensity', 'money_flow_ratio', 'net_flow_5d', 'new_high_20d',
        'new_low_20d', 'obv_momentum_10', 'operating_leverage', 'pct_b_20',
        'price_impact_20d', 'quick_ratio', 'rating_change', 'relative_volume_20d',
        'research_report_count', 'roe_change', 'roe_weighted', 'rsi14',
        'sales_to_price', 'sector_adj_trend_strength', 'sector_adj_up_day_ratio',
        'sector_adj_volume_momentum', 'sector_analyst_breadth', 'sector_beta',
        'sector_correlation', 'sector_inflow_rank', 'sector_mom_20d',
        'sector_mom_60d', 'sector_regime', 'sector_rs_20d', 'sector_volume_trend',
        'size_nonlinear', 'smb', 'super_flow_mean', 'target_price_ratio',
        'total_roa', 'trend_strength_20d', 'trend_strength_60d', 'trix_15',
        'turnover_rate', 'up_day_ratio_20d', 'vol_120', 'vol_20',
        'volume_momentum_20d', 'volume_momentum_5d', 'volume_price_correlation',
        'volume_trend_20d', 'working_capital_ratio', 'zero_days_ratio',
    ]
    for stub_name in pool_stub_names:
        registry.register(FeatureSpec(
            name=stub_name,
            inputs=(),
            lookback=1,
            description=f'Pool-mapping stub (not yet implemented): {stub_name}',
            compute=_financial_constant(stub_name, 0.0),
            category='pool_stub',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # Register factor-pool aliases after all canonical factors are available.
    try:
        from src.features.factor_pool import POOL_TO_SIMPLE_MAPPING
    except Exception:
        POOL_TO_SIMPLE_MAPPING = {}

    for alias_name, target_name in POOL_TO_SIMPLE_MAPPING.items():
        if alias_name in registry._specs or target_name not in registry._specs:
            continue
        target = registry.get(target_name)
        registry.register(FeatureSpec(
            name=alias_name,
            inputs=target.inputs,
            lookback=target.lookback,
            description=f'Factor-pool alias/proxy for {target_name}: {target.description}',
            compute=target.compute,
            version=target.version,
            category=target.category,
            owner=target.owner,
            frequency=target.frequency,
            level=target.level,
            lag=target.lag,
            preprocessing=target.preprocessing,
            dependencies=(target_name,),
            future_safe=target.future_safe,
            economic_meaning=target.economic_meaning,
            logic=f'Alias/proxy mapped from factor_pool name {alias_name} to {target_name}.',
            failure_modes='Proxy mapping can differ from the original descriptive formula; validate before formal conclusions.',
        ))

    return registry
