"""Basic helper functions and technical indicators for factor computation."""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def _vwap(frame: pd.DataFrame, window: int = 1) -> pd.Series:
    if 'amount' in frame.columns:
        return (frame['amount'] / frame['volume'].replace(0, np.nan)).groupby(frame['symbol']).transform(
            lambda x: x.rolling(window).mean() if window > 1 else x
        )
    return (frame['high'] + frame['low'] + frame['close']) / 3


def _adv(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    return frame.groupby('symbol')['volume'].transform(lambda s: s.rolling(window).mean())


def _returns(frame: pd.DataFrame) -> pd.Series:
    if 'pct_chg' in frame.columns:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x / 100)
    return frame.groupby('symbol')['close'].pct_change()


def _log_returns(frame: pd.DataFrame) -> pd.Series:
    return np.log(frame.groupby('symbol')['close'].pct_change() + 1)


def _volume_ratio(frame: pd.DataFrame, short: int, long: int) -> pd.Series:
    short_mean = frame.groupby('symbol')['volume'].transform(lambda s: s.rolling(short).mean())
    long_mean = frame.groupby('symbol')['volume'].transform(lambda s: s.rolling(long).mean())
    return short_mean / long_mean.replace(0, np.nan)


def _ewm_std(frame: pd.DataFrame, column: str, span: int) -> pd.Series:
    return frame.groupby('symbol')[column].transform(lambda s: s.pct_change().ewm(span=span).std())


def _rsi(frame: pd.DataFrame, window: int) -> pd.Series:
    delta = frame.groupby('symbol')['adj_close'].diff()
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


def _kdj_k(frame: pd.DataFrame, window: int) -> pd.Series:
    low_n = frame.groupby('symbol')['low'].transform(lambda s: s.rolling(window).min())
    high_n = frame.groupby('symbol')['high'].transform(lambda s: s.rolling(window).max())
    rsv = (frame['adj_close'] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    return rsv.groupby(frame['symbol']).transform(lambda s: s.ewm(span=3, adjust=False).mean())


def _kdj_d(frame: pd.DataFrame, window: int) -> pd.Series:
    k = _kdj_k(frame, window)
    return k.groupby(frame['symbol']).transform(lambda s: s.ewm(span=3, adjust=False).mean())


def _cci(frame: pd.DataFrame, window: int) -> pd.Series:
    typical = (frame['high'] + frame['low'] + frame['close']) / 3
    tp_mean = typical.groupby(frame['symbol']).transform(lambda s: s.rolling(window).mean())
    mad = typical.groupby(frame['symbol']).transform(lambda s: (s - s.rolling(window).mean()).abs().rolling(window).mean())
    return (typical - tp_mean) / (0.015 * mad.replace(0, np.nan))


def _obv(frame: pd.DataFrame) -> pd.Series:
    diff = frame.groupby('symbol')['adj_close'].diff()
    sign = diff.clip(lower=-np.inf, upper=0).fillna(0).replace(0, -1) + diff.clip(lower=0).fillna(0)
    return (sign * frame['volume']).groupby(frame['symbol']).cumsum()


def _volume_chaikin(frame: pd.DataFrame, window: int) -> pd.Series:
    mf = ((frame['close'] - frame['low']) - (frame['high'] - frame['close'])) / (frame['high'] - frame['low']).replace(0, np.nan)
    cumflow = (mf * frame['volume']).groupby(frame['symbol']).cumsum()
    return cumflow.groupby(frame['symbol']).transform(lambda s: s.rolling(window).mean())


def _ma_diff(frame: pd.DataFrame, short: int, long: int) -> pd.Series:
    ma_short = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(short).mean())
    ma_long = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(long).mean())
    return ma_short / ma_long.replace(0, np.nan)


def _price_to_ma(frame: pd.DataFrame, window: int) -> pd.Series:
    ma = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).mean())
    return frame['adj_close'] / ma.replace(0, np.nan)


def _high_low_position(frame: pd.DataFrame, window: int) -> pd.Series:
    high = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).max())
    low = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).min())
    return (frame['adj_close'] - low) / (high - low).replace(0, np.nan)


def _vol_std_ratio(frame: pd.DataFrame, short: int, long: int) -> pd.Series:
    std_short = frame.groupby('symbol')['adj_close'].transform(lambda s: s.pct_change().rolling(short).std())
    std_long = frame.groupby('symbol')['adj_close'].transform(lambda s: s.pct_change().rolling(long).std())
    return std_short / std_long.replace(0, np.nan)


def _volume_std(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame.groupby('symbol')['volume'].transform(lambda s: s.rolling(window).std())


def _skewness(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame.groupby('symbol')['adj_close'].transform(lambda s: s.pct_change().rolling(window).skew())


def _kurtosis(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame.groupby('symbol')['adj_close'].transform(lambda s: s.pct_change().rolling(window).kurt())


def _amount_growth(frame: pd.DataFrame, window: int) -> pd.Series:
    amount = frame['adj_close'] * frame['volume']
    return amount.groupby(frame['symbol']).transform(lambda s: s.pct_change(window))


def _close_to_high(frame: pd.DataFrame, window: int) -> pd.Series:
    high = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).max())
    return frame['adj_close'] / high.replace(0, np.nan)


def _close_to_low(frame: pd.DataFrame, window: int) -> pd.Series:
    low = frame.groupby('symbol')['adj_close'].transform(lambda s: s.rolling(window).min())
    return frame['adj_close'] / low.replace(0, np.nan)


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


def _macd_diff(frame: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    ema_fast = frame.groupby('symbol')['adj_close'].transform(lambda s: s.ewm(span=fast, adjust=False).mean())
    ema_slow = frame.groupby('symbol')['adj_close'].transform(lambda s: s.ewm(span=slow, adjust=False).mean())
    return ema_fast - ema_slow


def _dea(frame: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.Series:
    macd = _macd_diff(frame, fast, slow)
    return macd.groupby(frame['symbol']).transform(lambda s: s.ewm(span=signal, adjust=False).mean())


def _macd_hist(frame: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.Series:
    macd = _macd_diff(frame, fast, slow)
    dea = _dea(frame, fast, slow, signal)
    return 2 * (macd - dea)
