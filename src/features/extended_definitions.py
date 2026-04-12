from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureRegistry, FeatureSpec


def _pct_change(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda s: s.pct_change(window))


def _rolling_std(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda s: s.pct_change().rolling(window).std())


def _rolling_mean(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda s: s.rolling(window).mean())


def _rolling_max(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda s: s.rolling(window).max())


def _rolling_min(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda s: s.rolling(window).min())


def _rolling_skew(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda s: s.pct_change().rolling(window).skew())


def _rolling_kurt(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda s: s.pct_change().rolling(window).kurt())


def _rolling_sum(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda s: s.rolling(window).sum())


def _rolling_corr(frame: pd.DataFrame, col1: str, col2: str, window: int) -> pd.Series:
    return frame.groupby('symbol').apply(
        lambda g: g[col1].rolling(window).corr(g[col2])
    ).droplevel(0)


def _volume_ratio(frame: pd.DataFrame, short: int, long: int) -> pd.Series:
    short_mean = frame.groupby('symbol')['volume'].transform(lambda s: s.rolling(short).mean())
    long_mean = frame.groupby('symbol')['volume'].transform(lambda s: s.rolling(long).mean())
    return short_mean / long_mean.replace(0, np.nan)


def _price_position(frame: pd.DataFrame, window: int) -> pd.Series:
    price = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).mean())
    current = frame['adj_close']
    return current / price.replace(0, np.nan)


def _high_low_position(frame: pd.DataFrame, window: int) -> pd.Series:
    high = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).max())
    low = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).min())
    current = frame['adj_close']
    return (current - low) / (high - low).replace(0, np.nan)


def _volume_weighted_price(frame: pd.DataFrame, window: int) -> pd.Series:
    grouped = frame.groupby('symbol')
    typical = (frame['high'] + frame['low'] + frame['close']) / 3
    vwap = (typical * frame['volume']).groupby(frame['symbol']).transform(lambda s: s.rolling(window).sum())
    vol_sum = frame.groupby('symbol')['volume'].transform(lambda s: s.rolling(window).sum())
    return vwap / vol_sum.replace(0, np.nan) / grouped['adj_close'].transform(lambda s: s.rolling(window).mean())


def _amihud_illiquidity(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = frame.groupby('symbol')['adj_close'].transform(lambda s: s.pct_change().abs())
    vol = frame.groupby('symbol')['volume'].transform(lambda s: s.rolling(window).mean())
    illiq = ret / vol.replace(0, np.nan)
    return frame.groupby('symbol')['volume'].transform(lambda s: (1 / s.replace(0, np.nan)).rolling(window).mean())


def _rsi(frame: pd.DataFrame, window: int) -> pd.Series:
    delta = frame.groupby('symbol')['adj_close'].transform(lambda s: s.diff())
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = frame.groupby('symbol')['volume'].transform(lambda s: gain.rolling(window).mean())
    avg_loss = frame.groupby('symbol')['volume'].transform(lambda s: loss.rolling(window).mean())
    avg_gain = frame.groupby('symbol').apply(lambda g: gain.loc[g.index].rolling(window).mean()).droplevel(0)
    avg_loss = frame.groupby('symbol').apply(lambda g: loss.loc[g.index].rolling(window).mean()).droplevel(0)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_signal(frame: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.Series:
    ema_fast = frame.groupby('symbol')['adj_close'].transform(lambda s: s.ewm(span=fast, adjust=False).mean())
    ema_slow = frame.groupby('symbol')['adj_close'].transform(lambda s: s.ewm(span=slow, adjust=False).mean())
    macd = ema_fast - ema_slow
    signal_line = macd.groupby(frame['symbol']).transform(lambda s: s.ewm(span=signal, adjust=False).mean())
    return macd - signal_line


def _bollinger_position(frame: pd.DataFrame, window: int, num_std: float) -> pd.Series:
    mean = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).mean())
    std = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).std())
    upper = mean + num_std * std
    lower = mean - num_std * std
    return (frame['adj_close'] - lower) / (upper - lower).replace(0, np.nan)


def _volatility_ratio(frame: pd.DataFrame, short: int, long: int) -> pd.Series:
    vol_short = _rolling_std(frame, 'adj_close', short)
    vol_long = _rolling_std(frame, 'adj_close', long)
    return vol_short / vol_long.replace(0, np.nan)


def _turnover_rate(frame: pd.DataFrame, window: int) -> pd.Series:
    vol = frame.groupby('symbol')['volume'].transform(lambda s: s.rolling(window).sum())
    shares = frame.groupby('symbol')['volume'].transform(lambda s: s.rolling(window).mean()) * 0  # placeholder
    return vol * 0  # 需要总股本数据


def _money_flow(frame: pd.DataFrame, window: int) -> pd.Series:
    typical = (frame['high'] + frame['low'] + frame['close']) / 3
    money = typical * frame['volume']
    mf = money.groupby(frame['symbol']).transform(lambda s: s.rolling(window).sum())
    return mf / frame.groupby('symbol')['volume'].transform(lambda s: s.rolling(window).sum()).replace(0, np.nan)


def _close_to_high(frame: pd.DataFrame, window: int) -> pd.Series:
    high = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).max())
    return frame['adj_close'] / high.replace(0, np.nan)


def _close_to_low(frame: pd.DataFrame, window: int) -> pd.Series:
    low = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).min())
    return frame['adj_close'] / low.replace(0, np.nan)


def _log_return(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame.groupby('symbol')['adj_close'].transform(lambda s: np.log(s / s.shift(window)))


def _volume_growth(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame.groupby('symbol')['volume'].transform(lambda s: s.pct_change(window))


def _price_growth_accel(frame: pd.DataFrame, short: int, long: int) -> pd.Series:
    mom_short = _pct_change(frame, 'adj_close', short)
    mom_long = _pct_change(frame, 'adj_close', long)
    return mom_short - mom_long


def _volume_momentum(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame.groupby('symbol')['volume'].transform(lambda s: s.pct_change(window))


def _amount_growth(frame: pd.DataFrame, window: int) -> pd.Series:
    amount = frame['adj_close'] * frame['volume']
    return frame.groupby('symbol').transform(lambda s: amount.loc[s.index].pct_change(window)) if 'amount' not in frame.columns else _pct_change(frame, 'amount', window)


def extended_feature_registry() -> FeatureRegistry:
    registry = FeatureRegistry()

    # ===== 动量因子 (Momentum) =====
    for w in [5, 10, 20, 30, 60, 90, 120, 250]:
        registry.register(FeatureSpec(
            name=f'mom{w}',
            inputs=('adj_close',),
            lookback=w + 1,
            description=f'{w}-day momentum.',
            compute=lambda bars, w=w: _pct_change(bars, 'adj_close', w),
            category='momentum',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 反转因子 (Reversal) =====
    for w in [1, 3, 5, 10, 20]:
        registry.register(FeatureSpec(
            name=f'rev{w}',
            inputs=('adj_close',),
            lookback=w + 1,
            description=f'{w}-day reversal.',
            compute=lambda bars, w=w: -_pct_change(bars, 'adj_close', w),
            category='reversal',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 波动率因子 (Volatility) =====
    for w in [5, 10, 20, 30, 60, 120]:
        registry.register(FeatureSpec(
            name=f'vol{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'{w}-day volatility.',
            compute=lambda bars, w=w: _rolling_std(bars, 'adj_close', w),
            category='volatility',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 流动性因子 (Liquidity) =====
    for short, long in [(5, 20), (10, 60), (20, 60), (5, 60), (20, 120)]:
        registry.register(FeatureSpec(
            name=f'vol_ratio_{short}_{long}',
            inputs=('adj_close',),
            lookback=long,
            description=f'Volume ratio {short}/{long}.',
            compute=lambda bars, s=short, l=long: _volume_ratio(bars, s, l),
            category='liquidity',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 换手率/成交额因子 =====
    for w in [5, 10, 20, 60]:
        registry.register(FeatureSpec(
            name=f'vol_growth{w}',
            inputs=('volume',),
            lookback=w + 1,
            description=f'{w}-day volume growth.',
            compute=lambda bars, w=w: _volume_growth(bars, w),
            category='liquidity',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 资金流向因子 =====
    for w in [5, 10, 20]:
        registry.register(FeatureSpec(
            name=f'money_flow{w}',
            inputs=('high', 'low', 'close', 'volume'),
            lookback=w,
            description=f'{w}-day money flow.',
            compute=lambda bars, w=w: _money_flow(bars, w),
            category='money_flow',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 价格位置因子 =====
    for w in [20, 60, 120, 250]:
        registry.register(FeatureSpec(
            name=f'price_position{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'Close / {w}-day MA.',
            compute=lambda bars, w=w: _price_position(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 高低价位置 =====
    for w in [20, 60, 120]:
        registry.register(FeatureSpec(
            name=f'high_low_pos{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'(Close-Low)/(High-Low) {w}d.',
            compute=lambda bars, w=w: _high_low_position(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 布林带位置 =====
    for w in [20, 60]:
        registry.register(FeatureSpec(
            name=f'bb_position{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'Bollinger band position {w}d.',
            compute=lambda bars, w=w: _bollinger_position(bars, w, 2),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 波动率比 =====
    for s, l in [(5, 20), (10, 60), (20, 60)]:
        registry.register(FeatureSpec(
            name=f'vol_ratio{s}_{l}',
            inputs=('adj_close',),
            lookback=l,
            description=f'Volatility ratio {s}/{l}.',
            compute=lambda bars, s=s, l=l: _volatility_ratio(bars, s, l),
            category='volatility',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== MACD信号 =====
    registry.register(FeatureSpec(
        name='macd_signal',
        inputs=('adj_close',),
        lookback=26,
        description='MACD signal line.',
        compute=lambda bars: _macd_signal(bars, 12, 26, 9),
        category='technical',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # ===== 动量加速度 =====
    registry.register(FeatureSpec(
        name='mom_accel_5_20',
        inputs=('adj_close',),
        lookback=21,
        description='Momentum acceleration 5-20d.',
        compute=lambda bars: _price_growth_accel(bars, 5, 20),
        category='momentum',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))
    registry.register(FeatureSpec(
        name='mom_accel_20_60',
        inputs=('adj_close',),
        lookback=61,
        description='Momentum acceleration 20-60d.',
        compute=lambda bars: _price_growth_accel(bars, 20, 60),
        category='momentum',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # ===== 偏度和峰度 =====
    for w in [20, 60]:
        registry.register(FeatureSpec(
            name=f'return_skew{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'{w}-day return skewness.',
            compute=lambda bars, w=w: _rolling_skew(bars, 'adj_close', w),
            category='distribution',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
        registry.register(FeatureSpec(
            name=f'return_kurt{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'{w}-day return kurtosis.',
            compute=lambda bars, w=w: _rolling_kurt(bars, 'adj_close', w),
            category='distribution',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 创N日新高/新低 =====
    for w in [20, 60, 120, 250]:
        registry.register(FeatureSpec(
            name=f'close_to_high{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'Close to {w}-day high.',
            compute=lambda bars, w=w: _close_to_high(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
        registry.register(FeatureSpec(
            name=f'close_to_low{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'Close to {w}-day low.',
            compute=lambda bars, w=w: _close_to_low(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 对数收益 =====
    for w in [5, 20, 60]:
        registry.register(FeatureSpec(
            name=f'log_ret{w}',
            inputs=('adj_close',),
            lookback=w + 1,
            description=f'{w}-day log return.',
            compute=lambda bars, w=w: _log_return(bars, w),
            category='momentum',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 均线多头排列 =====
    for w in [5, 10, 20, 60]:
        registry.register(FeatureSpec(
            name=f'ma{w}_ratio',
            inputs=('adj_close',),
            lookback=w * 2,
            description=f'Price / {w}-day MA.',
            compute=lambda bars, w=w: bars.groupby('symbol')['adj_close'].transform(lambda s: s / s.rolling(w).mean()),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 成交额增长 =====
    for w in [5, 20, 60]:
        registry.register(FeatureSpec(
            name=f'amount_growth{w}',
            inputs=('adj_close', 'volume'),
            lookback=w + 1,
            description=f'{w}-day amount growth.',
            compute=lambda bars, w=w: bars.groupby('symbol').apply(
                lambda g: (g['adj_close'] * g['volume']).pct_change(w)
            ).droplevel(0),
            category='liquidity',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 量价相关性 =====
    for w in [20, 60]:
        registry.register(FeatureSpec(
            name=f'pv_corr{w}',
            inputs=('adj_close', 'volume'),
            lookback=w,
            description=f'Price-volume corr {w}d.',
            compute=lambda bars, w=w: _rolling_corr(bars, 'adj_close', 'volume', w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 收益率离散度 =====
    for w in [5, 20]:
        registry.register(FeatureSpec(
            name=f'ret_discrete{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'Return std/mean {w}d.',
            compute=lambda bars, w=w: bars.groupby('symbol')['adj_close'].transform(
                lambda s: (s.pct_change().rolling(w).std() / (s.pct_change().rolling(w).mean().abs() + 1e-8))
            ),
            category='volatility',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 均线金叉/死叉强度 =====
    registry.register(FeatureSpec(
        name='ma_cross_strength',
        inputs=('adj_close',),
        lookback=60,
        description='MA5/MA20 - MA20/MA60.',
        compute=lambda bars: (
            bars.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(5).mean() / s.rolling(20).mean()) -
            bars.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(20).mean() / s.rolling(60).mean())
        ),
        category='technical',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # ===== ATR (Average True Range) =====
    for w in [14, 20]:
        registry.register(FeatureSpec(
            name=f'atr{w}',
            inputs=('high', 'low', 'close'),
            lookback=w + 1,
            description=f'{w}-day ATR.',
            compute=lambda bars, w=w: _atr_normalized(bars, w),
            category='volatility',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Beta因子 =====
    for w in [20, 60, 120]:
        registry.register(FeatureSpec(
            name=f'beta{w}',
            inputs=('adj_close',),
            lookback=w + 1,
            description=f'{w}-day beta (vs market).',
            compute=lambda bars, w=w: _rolling_beta(bars, w),
            category='risk',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 下行波动率 =====
    for w in [20, 60]:
        registry.register(FeatureSpec(
            name=f'downside_vol{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'{w}-day downside volatility.',
            compute=lambda bars, w=w: _downside_vol(bars, w),
            category='risk',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 上行波动率 =====
    for w in [20, 60]:
        registry.register(FeatureSpec(
            name=f'upside_vol{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'{w}-day upside volatility.',
            compute=lambda bars, w=w: _upside_vol(bars, w),
            category='risk',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 涨跌不对称 =====
    for w in [20, 60]:
        registry.register(FeatureSpec(
            name=f'asymmetry{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'Up/down vol asymmetry {w}d.',
            compute=lambda bars, w=w: _upside_vol(bars, w) / (_downside_vol(bars, w) + 1e-8),
            category='risk',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== RSI =====
    for w in [6, 12, 24]:
        registry.register(FeatureSpec(
            name=f'rsi{w}',
            inputs=('adj_close',),
            lookback=w * 2,
            description=f'{w}-day RSI.',
            compute=lambda bars, w=w: _compute_rsi(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 威廉指标 =====
    for w in [14, 28]:
        registry.register(FeatureSpec(
            name=f'williams_r{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'{w}-day Williams %R.',
            compute=lambda bars, w=w: _williams_r(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== CCI ( Commodity Channel Index) =====
    for w in [14, 20]:
        registry.register(FeatureSpec(
            name=f'cci{w}',
            inputs=('high', 'low', 'close'),
            lookback=w,
            description=f'{w}-day CCI.',
            compute=lambda bars, w=w: _compute_cci(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== KDJ随机指标 =====
    registry.register(FeatureSpec(
        name='kdj_j',
        inputs=('high', 'low', 'close'),
        lookback=9,
        description='KDJ J indicator.',
        compute=lambda bars: _kdj_j(bars),
        category='technical',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # ===== OBV (On Balance Volume) =====
    registry.register(FeatureSpec(
        name='obv_slope20',
        inputs=('adj_close', 'volume'),
        lookback=21,
        description='OBV 20d slope.',
        compute=lambda bars: _obv_slope_simple(bars, 20),
        category='money_flow',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # ===== 趋势强度 =====
    for w in [20, 60]:
        registry.register(FeatureSpec(
            name=f'trend_strength{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'Trend strength {w}d.',
            compute=lambda bars, w=w: _trend_strength(bars, w),
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 均线收敛度 =====
    registry.register(FeatureSpec(
        name='ma_convergence',
        inputs=('adj_close',),
        lookback=60,
        description='MA5-MA20-MA60 convergence.',
        compute=lambda bars: _ma_convergence(bars),
        category='technical',
        preprocessing=('winsorize', 'cross_sectional_scale'),
    ))

    # ===== 收益率分位数 =====
    for w in [20, 60]:
        registry.register(FeatureSpec(
            name=f'ret_quantile{w}',
            inputs=('adj_close',),
            lookback=w,
            description=f'Return quantile {w}d.',
            compute=lambda bars, w=w: _ret_quantile(bars, w),
            category='distribution',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 成交额/市值 =====
    for w in [20, 60]:
        registry.register(FeatureSpec(
            name=f'turnover_mc{w}',
            inputs=('adj_close', 'volume'),
            lookback=w + 1,
            description=f'Turnover/market cap {w}d.',
            compute=lambda bars, w=w: _turnover_mc(bars, w),
            category='liquidity',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== 涨跌家数比 =====
    for w in [5, 20]:
        registry.register(FeatureSpec(
            name=f'updown_ratio{w}',
            inputs=('adj_close',),
            lookback=w + 1,
            description=f'Up/down count ratio {w}d.',
            compute=lambda bars, w=w: _updown_ratio(bars, w),
            category='market',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    return registry


def _atr_normalized(frame: pd.DataFrame, window: int) -> pd.Series:
    high = frame['high']
    low = frame['low']
    close = frame['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr / close.replace(0, np.nan)


def _obv_slope_simple(frame: pd.DataFrame, window: int) -> pd.Series:
    diff = frame.groupby('symbol')['adj_close'].diff()
    sign = diff.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    obv = (sign * frame['volume']).groupby(frame['symbol']).cumsum()
    return obv.groupby(frame['symbol']).transform(lambda s: s.pct_change(window))


def _rolling_beta(frame: pd.DataFrame, window: int) -> pd.Series:
    market = frame.groupby('symbol')['adj_close'].transform(lambda s: s.pct_change())
    stock = frame['adj_close'].pct_change()
    cov = stock.rolling(window).cov(market.groupby(frame['symbol']).transform(lambda s: s.pct_change()))
    var = market.groupby(frame['symbol']).transform(lambda s: s.pct_change()).rolling(window).var()
    return cov / var.replace(0, np.nan)


def _downside_vol(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = frame.groupby('symbol')['adj_close'].transform(lambda s: s.pct_change())
    downside = ret.clip(upper=0)
    return downside.rolling(window).std() * np.sqrt(252)


def _upside_vol(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = frame.groupby('symbol')['adj_close'].transform(lambda s: s.pct_change())
    upside = ret.clip(lower=0)
    return upside.rolling(window).std() * np.sqrt(252)


def _compute_rsi(frame: pd.DataFrame, window: int) -> pd.Series:
    delta = frame.groupby('symbol')['adj_close'].transform(lambda s: s.diff())
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(frame['symbol']).transform(lambda s: s.ewm(span=window, adjust=False).mean())
    avg_loss = loss.groupby(frame['symbol']).transform(lambda s: s.ewm(span=window, adjust=False).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _williams_r(frame: pd.DataFrame, window: int) -> pd.Series:
    high = frame.groupby('symbol')['high'].transform(lambda s: s.rolling(window).max())
    low = frame.groupby('symbol')['low'].transform(lambda s: s.rolling(window).min())
    close = frame['adj_close']
    return -100 * (high - close) / (high - low).replace(0, np.nan)


def _compute_cci(frame: pd.DataFrame, window: int) -> pd.Series:
    typical = (frame['high'] + frame['low'] + frame['close']) / 3
    tp_mean = typical.groupby(frame['symbol']).transform(lambda s: s.rolling(window).mean())
    mad = typical.groupby(frame['symbol']).transform(lambda s: (s - s.rolling(window).mean()).abs().rolling(window).mean())
    return (typical - tp_mean) / (0.015 * mad.replace(0, np.nan))


def _kdj_j(frame: pd.DataFrame, window: int = 9) -> pd.Series:
    low_n = frame.groupby('symbol')['low'].transform(lambda s: s.rolling(window).min())
    high_n = frame.groupby('symbol')['high'].transform(lambda s: s.rolling(window).max())
    rsv = (frame['adj_close'] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.groupby(frame['symbol']).transform(lambda s: s.ewm(span=3, adjust=False).mean())
    d = k.groupby(frame['symbol']).transform(lambda s: s.ewm(span=3, adjust=False).mean())
    return 3 * k - 2 * d


def _trend_strength(frame: pd.DataFrame, window: int) -> pd.Series:
    ma = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).mean())
    std = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).std())
    return (ma / frame['adj_close'] - 1) / (std / ma.replace(0, np.nan))


def _ma_convergence(frame: pd.DataFrame) -> pd.Series:
    ma5 = frame.groupby('symbol')['adj_close'].transform(lambda s: s / s.rolling(5).mean())
    ma20 = frame.groupby('symbol')['adj_close'].transform(lambda s: s / s.rolling(20).mean())
    ma60 = frame.groupby('symbol')['adj_close'].transform(lambda s: s / s.rolling(60).mean())
    return (ma5 - ma20).abs() + (ma20 - ma60).abs()


def _ret_quantile(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = frame.groupby('symbol')['adj_close'].transform(lambda s: s.pct_change(window))
    return ret.groupby(frame['symbol']).transform(lambda s: s.rank(pct=True))


def _turnover_mc(frame: pd.DataFrame, window: int) -> pd.Series:
    amount = frame['adj_close'] * frame['volume']
    market_cap = frame['adj_close'] * frame['volume']  # placeholder, need outstanding shares
    return amount.groupby(frame['symbol']).transform(lambda s: s.rolling(window).mean()) / frame['adj_close'].replace(0, np.nan)


def _updown_ratio(frame: pd.DataFrame, window: int) -> pd.Series:
    ret = frame.groupby('symbol')['adj_close'].transform(lambda s: s.pct_change())
    up = (ret > 0).groupby(frame['symbol']).transform(lambda s: s.rolling(window).sum())
    down = (ret < 0).groupby(frame['symbol']).transform(lambda s: s.rolling(window).sum())
    return up / (up + down.replace(0, np.nan))


def full_factor_registry() -> FeatureRegistry:
    """包含基础+扩展因子的完整因子库"""
    from .definitions import default_feature_registry
    reg = default_feature_registry()
    ext = extended_feature_registry()
    for name in ext._specs:
        if name not in reg._specs:
            reg.register(ext.get(name))
    return reg
