"""简化但稳定的因子库 - 用于因子研究和筛选"""
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


def _worldquant_alpha_001(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        neg_mask = ret < 0
        stddev_neg = ret.where(neg_mask).rolling(window).std().fillna(0)
        signed_power = pd.Series(np.where(neg_mask, stddev_neg ** 2, grp['close'].values), index=grp.index)
        if len(grp) >= 15:
            ts_argmax = signed_power.rolling(15).apply(lambda x: np.argmax(x), raw=True)
            rank_val = (ts_argmax - 0.5)
            result.loc[grp.index] = rank_val
    return result


def _worldquant_alpha_004(frame: pd.DataFrame, window: int = 9) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        low_rank = grp['low'].rolling(window).apply(lambda x: pd.Series(x).rank().iloc[-1], raw=False)
        result.loc[grp.index] = -low_rank.rolling(window).apply(lambda x: pd.Series(x).rank().iloc[-1], raw=False)
    return result


def _worldquant_alpha_008(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        sum_open = grp['open'].rolling(window).sum()
        sum_ret = ret.rolling(window).sum()
        product = sum_open * sum_ret
        delayed = product.shift(10)
        if len(grp) >= 10:
            result.loc[grp.index] = -(product - delayed).rank(pct=True)
    return result


def _worldquant_alpha_014(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        delta_ret = ret.diff(3)
        vol_corr = grp['open'].rolling(window).corr(grp['volume'])
        if len(grp) >= 10:
            result.loc[grp.index] = -delta_ret.rank(pct=True) * vol_corr
    return result


def _worldquant_alpha_022(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        vol_ma = grp['volume'].rolling(window).mean()
        mask = vol_ma > 0
        if mask.sum() >= window:
            corr = grp['high'][mask].rolling(window).corr(grp['volume'][mask])
            result.loc[grp.index] = corr
    return result


def _worldquant_alpha_023(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        result.loc[grp.index] = grp['high'].rolling(window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    return result


def _worldquant_alpha_026(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        result.loc[grp.index] = grp['close'].rolling(window).rank(pct=True)
    return result


def _worldquant_alpha_027(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        high_max = grp['high'].rolling(window).max()
        low_min = grp['low'].rolling(window).min()
        pos = (grp['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
        result.loc[grp.index] = pos
    return result


def _worldquant_alpha_028(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        high_max = grp['high'].rolling(window).max()
        low_min = grp['low'].rolling(window).min()
        range_val = high_max - low_min
        body = (grp['close'] - grp['open']) / range_val.replace(0, np.nan)
        result.loc[grp.index] = body
    return result


def _worldquant_alpha_029(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        high_max = grp['high'].rolling(window).max()
        low_min = grp['low'].rolling(window).min()
        range_val = high_max - low_min
        vol_norm = (grp['volume'] - grp['volume'].rolling(window).min()) / (grp['volume'].rolling(window).max() - grp['volume'].rolling(window).min()).replace(0, np.nan)
        price_pos = (grp['close'] - low_min) / range_val.replace(0, np.nan)
        result.loc[grp.index] = vol_norm * price_pos
    return result


def _worldquant_alpha_031(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        vol_ret_corr = ret.rolling(window).corr(grp['volume'])
        if len(grp) >= 20:
            result.loc[grp.index] = vol_ret_corr
    return result


def _worldquant_alpha_032(frame: pd.DataFrame, window: int = 12) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        result.loc[grp.index] = -ret.rolling(window).mean()
    return result


def _worldquant_alpha_033(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        vol_ma = grp['volume'].rolling(window).mean()
        mask = vol_ma > 0
        if mask.sum() >= window:
            corr = grp['low'][mask].rolling(window).corr(grp['volume'][mask])
            result.loc[grp.index] = corr
    return result


def _worldquant_alpha_034(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        vol_ma = grp['volume'].rolling(window).mean()
        mask = vol_ma > 0
        if mask.sum() >= window:
            corr = grp['high'][mask].rolling(window).corr(grp['volume'][mask])
            result.loc[grp.index] = corr
    return result


def _worldquant_alpha_036(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        body = (grp['close'] - grp['open']).abs()
        body_std = body.rolling(window).std()
        corr = body.rolling(window).corr(grp['close'])
        if len(grp) >= window:
            result.loc[grp.index] = -corr
    return result


def _worldquant_alpha_039(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        vol_std = grp['volume'].rolling(window).std()
        result.loc[grp.index] = vol_std / grp['volume'].rolling(window).mean().replace(0, np.nan)
    return result


def _worldquant_alpha_042(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        open_pos = (grp['open'] - grp['close']) / (grp['close'] - grp['open']).replace(0, np.nan)
        result.loc[grp.index] = -open_pos.rolling(window).mean()
    return result


def _worldquant_alpha_044(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        vol_ma = grp['volume'].rolling(window).mean()
        mask = vol_ma > 0
        if mask.sum() >= window:
            corr = grp['close'][mask].rolling(window).corr(grp['volume'][mask])
            result.loc[grp.index] = corr
    return result


def _worldquant_alpha_046(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        vol_ratio = grp['volume'] / grp['volume'].rolling(window).mean().replace(0, np.nan)
        result.loc[grp.index] = vol_ratio
    return result


def _worldquant_alpha_047(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        vol_ma = grp['volume'].rolling(window).mean()
        mask = vol_ma > 0
        if mask.sum() >= window:
            corr = (grp['close'] - grp['open'])[mask].rolling(window).corr(grp['volume'][mask])
            result.loc[grp.index] = corr
    return result


def _worldquant_alpha_067(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        result.loc[grp.index] = ret.rolling(window).sum()
    return result


def _worldquant_alpha_068(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        result.loc[grp.index] = ret.rolling(window).std()
    return result


def _worldquant_alpha_070(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        result.loc[grp.index] = ret.rolling(window).skew()
    return result


def _worldquant_alpha_071(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        result.loc[grp.index] = ret.rolling(window).kurt()
    return result


def _worldquant_alpha_072(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        result.loc[grp.index] = grp['volume'].rolling(window).rank(pct=True)
    return result


def _worldquant_alpha_073(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        result.loc[grp.index] = grp['open'].rolling(window).rank(pct=True)
    return result


def _worldquant_alpha_074(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        result.loc[grp.index] = grp['high'].rolling(window).rank(pct=True)
    return result


def _worldquant_alpha_075(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        result.loc[grp.index] = grp['low'].rolling(window).rank(pct=True)
    return result


def _worldquant_alpha_076(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        result.loc[grp.index] = grp['close'].rolling(window).rank(pct=True)
    return result


def _worldquant_alpha_077(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        result.loc[grp.index] = (ret > 0).rolling(window).sum() / window
    return result


def _worldquant_alpha_086(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        pos_ret = ret.where(ret > 0, 0)
        result.loc[grp.index] = pos_ret.rolling(window).sum() / ret.abs().rolling(window).sum()
    return result


def _worldquant_alpha_087(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        result.loc[grp.index] = -ret.rolling(window).mean()
    return result


def _worldquant_alpha_088(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        ret = grp['pct_chg'] / 100 if 'pct_chg' in grp.columns else grp['close'].pct_change()
        result.loc[grp.index] = -ret.rolling(window).std()
    return result


def _worldquant_alpha_092(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        result.loc[grp.index] = grp['amount'].rolling(window).rank(pct=True) if 'amount' in grp.columns else grp['volume'].rolling(window).rank(pct=True)
    return result


def _worldquant_alpha_093(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        vol_ma = grp['volume'].rolling(window).mean()
        result.loc[grp.index] = grp['volume'] / vol_ma.replace(0, np.nan) - 1
    return result


def _worldquant_alpha_094(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        close_ma = grp['close'].rolling(window).mean()
        result.loc[grp.index] = grp['close'] / close_ma.replace(0, np.nan) - 1
    return result


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


def _worldquant_alpha_006(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        vol_ma = grp['volume'].rolling(window).mean()
        mask = vol_ma > 0
        corr = grp['open'][mask].rolling(window).corr(grp['volume'][mask])
        result.loc[grp.index] = -corr
    return result


def _worldquant_alpha_012(frame: pd.DataFrame) -> pd.Series:
    delta_vol = frame.groupby('symbol')['volume'].diff()
    delta_close = frame.groupby('symbol')['close'].diff()
    return np.sign(delta_vol) * (-delta_close)


def _worldquant_alpha_016(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        high_ma = grp['high'].rolling(window).mean()
        vol_ma = grp['volume'].rolling(window).mean()
        mask = (high_ma > 0) & (vol_ma > 0)
        if mask.sum() > window:
            corr = grp['high'][mask].rolling(window).corr(grp['volume'][mask])
            result.loc[grp.index] = -corr
    return result


def _worldquant_alpha_043(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    return frame.groupby('symbol')['close'].transform(lambda s: s.rolling(window).rank(pct=True))


def _worldquant_alpha_013(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        vol_ma = grp['volume'].rolling(window).mean()
        mask = vol_ma > 0
        if mask.sum() > window:
            close_rank = grp['close'][mask].rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            vol_rank = grp['volume'][mask].rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            cov = close_rank * vol_rank
            result.loc[grp.index] = -cov.rolling(window).mean()
    return result


def _worldquant_alpha_002(frame: pd.DataFrame, window: int = 6) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        log_vol_diff = np.log(grp['volume']).diff(2)
        ret = (grp['close'] - grp['open']) / grp['open'].replace(0, np.nan)
        if len(grp) >= window:
            corr = log_vol_diff.rolling(window).corr(ret)
            result.loc[grp.index] = -corr
    return result


def _worldquant_alpha_003(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) >= window:
            open_rank = grp['open'].rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            vol_rank = grp['volume'].rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            corr = open_rank.rolling(window).corr(vol_rank)
            result.loc[grp.index] = -corr
    return result


def _worldquant_alpha_009(frame: pd.DataFrame) -> pd.Series:
    delta_close = frame.groupby('symbol')['close'].diff()
    ts_min = delta_close.groupby(frame['symbol']).transform(lambda x: x.rolling(5).min())
    ts_max = delta_close.groupby(frame['symbol']).transform(lambda x: x.rolling(5).max())
    
    result = delta_close.copy()
    mask_pos = ts_min > 0
    mask_neg = ts_max < 0
    
    result = np.where(mask_pos, delta_close, 
                     np.where(mask_neg, delta_close, -delta_close))
    return pd.Series(result, index=frame.index)


def _worldquant_alpha_010(frame: pd.DataFrame) -> pd.Series:
    delta_close = frame.groupby('symbol')['close'].diff()
    ts_min = delta_close.groupby(frame['symbol']).transform(lambda x: x.rolling(4).min())
    ts_max = delta_close.groupby(frame['symbol']).transform(lambda x: x.rolling(4).max())
    
    result = delta_close.copy()
    mask_pos = ts_min > 0
    mask_neg = ts_max < 0
    
    result = np.where(mask_pos, delta_close,
                     np.where(mask_neg, delta_close, -delta_close))
    return pd.Series(result, index=frame.index)


def _worldquant_alpha_018(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        body_std = (grp['close'] - grp['open']).abs().rolling(window).std()
        open_close_diff = grp['close'] - grp['open']
        if len(grp) >= 10:
            corr = open_close_diff.rolling(10).corr(body_std)
            result.loc[grp.index] = -corr
    return result


def _worldquant_alpha_020(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        range_5d = grp['high'].rolling(5).max() - grp['low'].rolling(5).min()
        open_pos = (grp['open'] - grp['low'].rolling(5).min()) / range_5d.replace(0, np.nan)
        result.loc[grp.index] = open_pos
    return result


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
            
            if factor_name not in cache.columns:
                return pd.Series(np.nan, index=bars.index)
            
            if 'trade_date' in bars.columns:
                bars_symbols = bars['symbol'].unique()
            else:
                return pd.Series(np.nan, index=bars.index)
            
            if _cache_with_tradeable is None:
                _cache_with_tradeable = cache.copy()
                # pub_date 现在是从 akshare stock_report_disclosure 获取的真实披露日期
                # tradeable_date 直接使用 pub_date，不再做规则化转换
                _cache_with_tradeable['tradeable_date'] = pd.to_datetime(_cache_with_tradeable['pub_date'])
            
            result_list = []
            trade_dates = bars['trade_date'].unique()
            
            for td in trade_dates:
                td_ts = pd.Timestamp(td)
                mask = bars['trade_date'] == td_ts
                symbols = bars.loc[mask, 'symbol'].unique()
                
                avail_cache = _cache_with_tradeable[
                    (_cache_with_tradeable['tradeable_date'] <= td_ts) &
                    (_cache_with_tradeable['symbol'].isin(symbols))
                ]
                
                if avail_cache.empty:
                    df = pd.DataFrame({
                        'symbol': symbols,
                        factor_name: np.nan
                    })
                else:
                    latest = avail_cache.sort_values('tradeable_date').groupby('symbol').last().reset_index()
                    df = latest[['symbol', factor_name]].copy()
                
                df['trade_date'] = td_ts
                result_list.append(df)
            
            if not result_list:
                return pd.Series(np.nan, index=bars.index)
            
            result = pd.concat(result_list, ignore_index=True)
            
            result_dict = {}
            for td in trade_dates:
                td_ts = pd.Timestamp(td)
                td_result = result[result['trade_date'] == td_ts] if 'trade_date' in result.columns else pd.DataFrame()
                for _, row in td_result.iterrows():
                    result_dict[(row['symbol'], td_ts)] = row[factor_name]
            
            result_series = []
            for idx, row in bars.iterrows():
                sym = row['symbol']
                td = pd.Timestamp(row['trade_date'])
                result_series.append(result_dict.get((sym, td), np.nan))
            
            return pd.Series(result_series, index=bars.index)
        return compute_fn
    
    for factor_name, barra_name, desc in [
        ('earnings_yield', 'EP', 'Earnings Yield (E/P)'),
        ('roe', 'ROE', 'Return on Equity'),
        ('book_to_price', 'BTOP', 'Book-to-Price'),
        ('roe_weighted', 'ROE_W', 'Weighted ROE'),
        ('net_margin', 'NPM', 'Net Profit Margin'),
        ('operating_margin', 'OPM', 'Operating Margin'),
        ('gross_margin', 'GPM', 'Gross Profit Margin'),
        ('roa', 'ROA', 'Return on Assets'),
        ('total_roa', 'ROA_T', 'Total ROA'),
        ('equity_growth', 'EQ_G', 'Equity Growth'),
        ('revenue_growth', 'REV_G', 'Revenue Growth'),
        ('profit_growth', 'PROF_G', 'Profit Growth'),
        ('asset_growth', 'ASSET_G', 'Asset Growth'),
        ('debt_ratio', 'DTE', 'Debt-to-Equity'),
        ('current_ratio', 'CR', 'Current Ratio'),
        ('quick_ratio', 'QR', 'Quick Ratio'),
        ('cash_ratio', 'CASH_R', 'Cash Ratio'),
        ('asset_turnover', 'ATO', 'Asset Turnover'),
        ('inv_turnover', 'INV_TO', 'Inventory Turnover'),
        ('ar_turnover', 'AR_TO', 'Accounts Receivable Turnover'),
        ('ocf_per_share', 'OCF_PS', 'Operating Cash Flow per Share'),
        # 每股指标
        ('retained_earnings_per_share', 'REPS', 'Retained Earnings per Share'),
        ('capital_reserve_per_share', 'CRPS', 'Capital Reserve per Share'),
        # Efficiency扩展
        ('fixasset_turnover', 'FATO', 'Fixed Asset Turnover'),
        ('inv_turnover_days', 'INV_D', 'Inventory Turnover Days'),
        ('ar_turnover_days', 'AR_D', 'Accounts Receivable Turnover Days'),
        ('asset_turnover_days', 'ATO_D', 'Asset Turnover Days'),
        # Leverage扩展
        ('equity_ratio', 'EQ_R', 'Equity Ratio'),
        ('equity_ratio_total', 'EQ_RT', 'Total Equity Ratio'),
        ('longterm_debt_ratio', 'LTDR', 'Long-term Debt Ratio'),
        ('capital_fix_ratio', 'CFR', 'Capital Fixation Ratio'),
        ('liquidation_ratio', 'LQ_R', 'Liquidation Ratio'),
        ('interest_coverage', 'ICV', 'Interest Coverage'),
        # Profitability扩展
        ('roe_assets', 'ROEA', 'ROE from Assets'),
        ('total_roa_net', 'ROA_N', 'Total Net ROA'),
        ('mainbiz_profit_ratio', 'MPRO', 'Main Business Profit Ratio'),
        # 其他
        ('cost_profit_margin', 'CPM', 'Cost-to-Profit Margin'),
        ('mainbiz_profit_margin', 'MPM', 'Main Business Profit Margin'),
        ('three_expense_ratio', 'TER', 'Three Expense Ratio'),
        ('dividend_payout_ratio', 'DPR', 'Dividend Payout Ratio'),
        ('investment_return_ratio', 'IRR', 'Investment Return Ratio'),
        ('earnings_yield_weighted', 'EPW', 'Weighted Earnings Yield'),
        ('earnings_yield_ex', 'EPX', 'Earnings Yield Excluding Non-recurring'),
    ]:
        registry.register(FeatureSpec(
            name=factor_name,
            inputs=(),
            lookback=1,
            description=f'Financial: {desc} ({barra_name})',
            compute=_create_financial_factor_func(factor_name),
            category='financial',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Pattern Factors =====
    # Candlestick patterns
    
    def _candle_body_ratio(frame: pd.DataFrame) -> pd.Series:
        body = (frame['close'] - frame['open']).abs()
        range_val = frame['high'] - frame['low']
        return body / range_val.replace(0, np.nan)
    
    def _candle_upper_shadow(frame: pd.DataFrame) -> pd.Series:
        upper = frame['high'] - frame[['open', 'close']].max(axis=1)
        range_val = frame['high'] - frame['low']
        return upper / range_val.replace(0, np.nan)
    
    def _candle_lower_shadow(frame: pd.DataFrame) -> pd.Series:
        lower = frame[['open', 'close']].min(axis=1) - frame['low']
        range_val = frame['high'] - frame['low']
        return lower / range_val.replace(0, np.nan)
    
    def _candle_doji(frame: pd.DataFrame) -> pd.Series:
        body = (frame['close'] - frame['open']).abs()
        range_val = frame['high'] - frame['low']
        return body / range_val.replace(0, np.nan)
    
    # Volume patterns
    def _volume_trend(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: x.pct_change(window))
    
    def _volume_momentum(frame: pd.DataFrame, window: int = 5) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: x.rolling(window).mean() / x.rolling(window*2).mean() - 1)
    
    # Price position patterns
    def _close_position(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        high = frame.groupby('symbol')['close'].transform(lambda x: x.rolling(window).max())
        low = frame.groupby('symbol')['close'].transform(lambda x: x.rolling(window).min())
        return (frame['close'] - low) / (high - low).replace(0, np.nan)
    
    # Trend strength
    def _trend_strength(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        def calc_strength(x):
            if len(x) < 5:
                return np.nan
            return np.polyfit(np.arange(len(x)), x, 1)[0] / x.mean() if x.mean() != 0 else 0
        
        return frame.groupby('symbol')['close'].transform(lambda x: x.rolling(window).apply(calc_strength, raw=True))
    
    # Volatility patterns
    def _volatility_ratio(frame: pd.DataFrame, short: int, long: int) -> pd.Series:
        short_vol = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change().rolling(short).std())
        long_vol = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change().rolling(long).std())
        return short_vol / long_vol.replace(0, np.nan)
    
    # Gap patterns
    def _gap_size(frame: pd.DataFrame) -> pd.Series:
        prev_close = frame.groupby('symbol')['close'].shift(1)
        return (frame['open'] - prev_close) / prev_close.replace(0, np.nan)
    
    # Price momentum patterns
    def _momentum_divergence(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        price_mom = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(window))
        vol_mom = frame.groupby('symbol')['volume'].transform(lambda x: x.pct_change(window))
        return price_mom - vol_mom
    
    # ===== Liquidity Factors =====
    def _amihud_illiquidity(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """Amihud非流动性因子"""
        ret = frame['close'].pct_change()
        volume = frame['volume']
        avg_volume = frame.groupby('symbol')['volume'].transform(lambda x: x.rolling(window).mean())
        illiq = ret.abs() / volume.replace(0, np.nan) * 1e6
        return illiq.rolling(window).mean()
    
    def _turnover_rate(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """换手率因子 (需要市值数据，暂用成交量代理)"""
        vol_now = frame['volume']
        vol_avg = frame.groupby('symbol')['volume'].transform(lambda x: x.rolling(window).mean())
        return vol_now / vol_avg.replace(0, np.nan) - 1
    
    def _volume_price_correlation(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """量价相关性"""
        def corr(x):
            if len(x) < 10:
                return np.nan
            return x['close'].corr(x['volume'])
        return frame.groupby('symbol').apply(corr)
    
    def _relative_volume(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """相对成交量"""
        vol = frame.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(window).mean())
        market_avg = float(vol.mean())
        if market_avg == 0:
            market_avg = 1.0
        return vol / market_avg
    
    def _buy_sell_pressure(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """买卖压力因子"""
        up = (frame['close'] > frame['close'].shift(1)).astype(int)
        down = (frame['close'] < frame['close'].shift(1)).astype(int)
        vol_up = frame['volume'] * up
        vol_down = frame['volume'] * down
        vol_sum = frame['volume'].rolling(window).sum()
        pressure = (vol_up.rolling(window).sum() - vol_down.rolling(window).sum()) / vol_sum.replace(0, np.nan)
        return pressure
    
    def _volume_skewness(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """成交量偏度"""
        return frame.groupby('symbol')['volume'].transform(lambda x: x.rolling(window).skew())
    
    def _price_impact(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """价格冲击因子"""
        ret = frame['close'].pct_change().abs()
        vol_ratio = frame['volume'] / frame.groupby('symbol')['volume'].transform(lambda x: x.rolling(window).mean())
        return ret / vol_ratio.replace(0, np.nan)
    
    # ===== Distribution Factors =====
    def _return_skewness(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """收益率偏度"""
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(window).skew())
    
    def _return_kurtosis(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """收益率峰度"""
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(window).kurt())
    
    def _volume_skew(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """成交量偏度因子"""
        return frame.groupby('symbol')['volume'].transform(lambda x: x.rolling(window).skew())
    
    def _high_low_range(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """高低点范围"""
        high = frame.groupby('symbol')['high'].transform(lambda x: x.rolling(window).max())
        low = frame.groupby('symbol')['low'].transform(lambda x: x.rolling(window).min())
        return (high - low) / frame['close'].replace(0, np.nan)
    
    # ===== Short-term Reversal Factors =====
    def _short_term_reversal(frame: pd.DataFrame, window: int = 5) -> pd.Series:
        """短期反转因子"""
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: -x.rolling(window).mean())
    
    def _mid_term_reversal(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """中期反转因子"""
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: -x.rolling(window).mean())
    
    def _opening_gap(frame: pd.DataFrame) -> pd.Series:
        """跳空缺口因子"""
        prev_close = frame.groupby('symbol')['close'].shift(1)
        return (frame['open'] - prev_close) / prev_close.replace(0, np.nan)
    
    def _intraday_reversal(frame: pd.DataFrame) -> pd.Series:
        """日内反转因子"""
        return (frame['close'] - frame['open']) / frame['open'].replace(0, np.nan)
    
    # Register liquidity factors
    liquidity_factors = [
        ('amihud_illiq_20d', lambda f: _amihud_illiquidity(f, 20), 'Amihud illiquidity 20d', 21),
        ('relative_volume_20d', lambda f: _relative_volume(f, 20), 'Relative volume 20d', 21),
        ('volume_skew_20d', lambda f: _volume_skew(f, 20), 'Volume skewness 20d', 21),
        ('price_impact_20d', lambda f: _price_impact(f, 20), 'Price impact 20d', 21),
        ('high_low_range_20d', lambda f: _high_low_range(f, 20), 'High-low range 20d', 21),
        ('turnover_rate_20d', lambda f: _turnover_rate(f, 20), 'Turnover rate 20d', 21),
    ]
    
    for name, compute_fn, desc, lookback in liquidity_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('open', 'high', 'low', 'close', 'volume'),
            lookback=lookback,
            description=f'Liquidity: {desc}',
            compute=compute_fn,
            category='liquidity',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
    
    # Register distribution factors
    distribution_factors = [
        ('return_skew_20d', lambda f: _return_skewness(f, 20), 'Return skewness 20d', 21),
        ('return_kurt_20d', lambda f: _return_kurtosis(f, 20), 'Return kurtosis 20d', 21),
        ('return_skew_60d', lambda f: _return_skewness(f, 60), 'Return skewness 60d', 61),
        ('high_low_range_60d', lambda f: _high_low_range(f, 60), 'High-low range 60d', 61),
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
    
    # Register reversal factors
    reversal_factors = [
        ('short_term_reversal_5d', lambda f: _short_term_reversal(f, 5), 'Short-term reversal 5d', 5),
        ('mid_term_reversal_20d', lambda f: _mid_term_reversal(f, 20), 'Mid-term reversal 20d', 21),
        ('opening_gap', _opening_gap, 'Opening gap', 1),
        ('intraday_reversal', _intraday_reversal, 'Intraday reversal', 1),
    ]
    
    for name, compute_fn, desc, lookback in reversal_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('open', 'high', 'low', 'close', 'volume', 'pct_chg'),
            lookback=lookback,
            description=f'Reversal: {desc}',
            compute=compute_fn,
            category='reversal',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
    
    # ===== Sector Rotation Factors =====
    def _sector_momentum(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """股票所属行业动量 - 个股与行业平均的比较"""
        stock_ret = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(window))
        market_ret = frame['close'].pct_change(window)
        return stock_ret - market_ret
    
    def _sector_relative_strength(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """行业相对强弱 - 个股相对市场的累积优势"""
        ret = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(window))
        return ret - ret.mean()
    
    def _sector_regime(frame: pd.DataFrame) -> pd.Series:
        """市场状态: 1=上涨趋势, 0=震荡, -1=下跌趋势"""
        market_ret = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(20))
        avg_ret = market_ret.mean()
        return np.sign(avg_ret)
    
    def _sector_momentum_adjusted_pattern(
        pattern_signal: pd.Series,
        sector_mom: pd.Series,
        frame: pd.DataFrame
    ) -> pd.Series:
        """条件形态因子: 根据行业动量调整形态信号
        
        逻辑:
        - 行业强势时, 趋势跟随型形态(如连续上涨)信号增强
        - 行业弱势时, 反转型形态(如新低)信号增强
        """
        regime = _sector_regime(frame)
        adjustment = np.where(
            sector_mom > 0,  # 行业强势
            pattern_signal * 1.2,  # 增强趋势信号
            pattern_signal * 0.8   # 减弱趋势信号
        )
        return adjustment
    
    def _create_sector_adjusted_pattern(pattern_fn, window: int):
        """创建行业调整后的形态因子"""
        def compute(frame: pd.DataFrame):
            pattern_signal = pattern_fn(frame)
            sector_mom = _sector_momentum(frame, window)
            return _sector_momentum_adjusted_pattern(pattern_signal, sector_mom, frame)
        return compute
    
    # Register sector rotation factors
    sector_factors = [
        ('sector_mom_20d', lambda f: _sector_momentum(f, 20), 'Sector momentum 20d', 21),
        ('sector_mom_60d', lambda f: _sector_momentum(f, 60), 'Sector momentum 60d', 61),
        ('sector_rs_20d', lambda f: _sector_relative_strength(f, 20), 'Sector relative strength 20d', 21),
        ('sector_regime', _sector_regime, 'Market regime (bull/neutral/bear)', 21),
    ]
    
    for name, compute_fn, desc, lookback in sector_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('open', 'high', 'low', 'close', 'volume'),
            lookback=lookback,
            description=f'Sector: {desc}',
            compute=compute_fn,
            category='sector',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
    
    # Register pattern factors
    pattern_factors = [
        # Candlestick
        ('candle_body_ratio', _candle_body_ratio, 'Candlestick body ratio', 1),
        ('candle_upper_shadow', _candle_upper_shadow, 'Upper shadow ratio', 1),
        ('candle_lower_shadow', _candle_lower_shadow, 'Lower shadow ratio', 1),
        ('candle_doji', _candle_doji, 'Doji strength', 1),
        
        # Volume patterns
        ('volume_trend_20d', lambda f: _volume_trend(f, 20), '20d volume trend', 21),
        ('volume_momentum_5d', lambda f: _volume_momentum(f, 5), '5d volume momentum', 10),
        ('volume_momentum_20d', lambda f: _volume_momentum(f, 20), '20d volume momentum', 40),
        
        # Price position
        ('close_position_20d', lambda f: _close_position(f, 20), 'Close position 20d', 20),
        ('close_position_60d', lambda f: _close_position(f, 60), 'Close position 60d', 60),
        ('close_position_120d', lambda f: _close_position(f, 120), 'Close position 120d', 120),
        
        # Trend
        ('trend_strength_20d', lambda f: _trend_strength(f, 20), 'Trend strength 20d', 20),
        ('trend_strength_60d', lambda f: _trend_strength(f, 60), 'Trend strength 60d', 60),
        
        # Volatility
        ('vol_ratio_5_20', lambda f: _volatility_ratio(f, 5, 20), 'Volatility ratio 5/20', 20),
        ('vol_ratio_10_60', lambda f: _volatility_ratio(f, 10, 60), 'Volatility ratio 10/60', 60),
        
        # Gap
        ('gap_size', _gap_size, 'Gap size', 1),
        
        # Momentum
        ('momentum_divergence_20d', lambda f: _momentum_divergence(f, 20), 'Momentum divergence 20d', 21),
        
        # New high/low
        ('new_high_20d', lambda f: (f['close'] == f.groupby('symbol')['close'].transform(lambda x: x.rolling(20).max())).astype(float), 'New 20d high', 20),
        ('new_low_20d', lambda f: (f['close'] == f.groupby('symbol')['close'].transform(lambda x: x.rolling(20).min())).astype(float), 'New 20d low', 20),
        
        # Up/down day ratio
        ('up_day_ratio_20d', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: (x > 0).rolling(20).mean()), 'Up day ratio 20d', 20),
        
        # Consecutive up/down
        ('consecutive_up', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: (x > 0).rolling(5).sum()), 'Consecutive up days', 5),
        ('consecutive_down', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: (x < 0).rolling(5).sum()), 'Consecutive down days', 5),
        
        # ===== Sector-Adjusted Pattern Factors =====
        # 形态因子根据行业轮动状态调整
        ('sector_adj_trend_strength', _create_sector_adjusted_pattern(_trend_strength, 20), 'Sector-adjusted trend strength', 21),
        ('sector_adj_close_position', _create_sector_adjusted_pattern(_close_position, 20), 'Sector-adjusted close position', 21),
        ('sector_adj_volume_momentum', _create_sector_adjusted_pattern(_volume_momentum, 20), 'Sector-adjusted volume momentum', 21),
        ('sector_adj_up_day_ratio', lambda f: _sector_momentum_adjusted_pattern(
            f.groupby('symbol')['pct_chg'].transform(lambda x: (x > 0).rolling(20).mean()),
            _sector_momentum(f, 20), f), 'Sector-adjusted up day ratio', 21),
        ('sector_adj_consecutive_up', lambda f: _sector_momentum_adjusted_pattern(
            f.groupby('symbol')['pct_chg'].transform(lambda x: (x > 0).rolling(5).sum()),
            _sector_momentum(f, 20), f), 'Sector-adjusted consecutive up', 21),
    ]
    
    for name, compute_fn, desc, lookback in pattern_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('open', 'high', 'low', 'close', 'volume', 'pct_chg'),
            lookback=lookback,
            description=f'Pattern: {desc}',
            compute=compute_fn,
            category='pattern',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    return registry

