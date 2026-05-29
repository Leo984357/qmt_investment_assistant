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


def _worldquant_alpha_005(frame: pd.DataFrame, window: int = 17) -> pd.Series:
    """Alpha 005: Tsrank of correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 30:
            continue
        close_rank = grp['close'].rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        corr = close_rank.rolling(window).corr(vol_rank)
        ts_rank = corr.rolling(9).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = ts_rank - 0.5
    return result


def _worldquant_alpha_007(frame: pd.DataFrame, window: int = 7) -> pd.Series:
    """Alpha 007: Low-High correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 20:
            continue
        low_rank = grp['low'].rank(pct=True)
        high_rank = grp['high'].rank(pct=True)
        vwap_proxy = (grp['high'] + grp['low'] + grp['close']) / 3
        adv = grp['volume'].rolling(20).mean()
        corr_low_adv = low_rank.rolling(window).corr(adv.rank(pct=True))
        signal = -(low_rank - high_rank) + corr_low_adv
        result.loc[grp.index] = signal
    return result


def _worldquant_alpha_011(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    """Alpha 011: VWAP-Volume correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 20:
            continue
        vwap = ((grp['high'] + grp['low'] + grp['close']) / 3).rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        corr = vwap.rolling(window).corr(vol_rank)
        ts_rank = corr.rolling(4).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = -ts_rank
    return result


def _worldquant_alpha_015(frame: pd.DataFrame, window: int = 12) -> pd.Series:
    """Alpha 015: High-Low correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 30:
            continue
        high_rank = grp['high'].rank(pct=True)
        low_rank = grp['low'].rank(pct=True)
        corr = high_rank.rolling(window).corr(low_rank)
        ts_rank = corr.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = -ts_rank
    return result


def _worldquant_alpha_017(frame: pd.DataFrame, window: int = 6) -> pd.Series:
    """Alpha 017: VWAP correlation Tsrank"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 20:
            continue
        vwap = ((grp['high'] + grp['low'] + grp['close']) / 3).rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        corr = vwap.rolling(window).corr(vol_rank)
        ts_rank = corr.rolling(2).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = ts_rank
    return result


def _worldquant_alpha_019(frame: pd.DataFrame, window: int = 17) -> pd.Series:
    """Alpha 019: VWAP correlation with ADV"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 30:
            continue
        vwap = (grp['high'] + grp['low'] + grp['close']) / 3
        adv = grp['volume'].rolling(20).mean()
        corr = vwap.rolling(window).corr(adv)
        delta_vwap = vwap.diff(6)
        result.loc[grp.index] = -corr * delta_vwap.rank(pct=True)
    return result


def _worldquant_alpha_021(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Alpha 021: High-Volume correlation delta"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 30:
            continue
        high_rank = grp['high'].rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        corr = high_rank.rolling(window).corr(vol_rank)
        corr_diff = corr.diff(2)
        ts_rank = corr_diff.rolling(17).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = -ts_rank
    return result


def _worldquant_alpha_024(frame: pd.DataFrame, window: int = 6) -> pd.Series:
    """Alpha 024: VWAP correlation ADV short"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 20:
            continue
        vwap = (grp['high'] + grp['low'] + grp['close']) / 3
        adv = grp['volume'].rolling(20).mean()
        corr = vwap.rolling(window).corr(adv)
        delta_vwap = vwap.diff(3)
        result.loc[grp.index] = -corr * delta_vwap.rank(pct=True)
    return result


def _worldquant_alpha_025(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    """Alpha 025: Rank combination"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 25:
            continue
        vwap_rank = ((grp['high'] + grp['low'] + grp['close']) / 3).rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        low_rank = grp['low'].rank(pct=True)
        vwap_tsrank = vwap_rank.rolling(9).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        corr = vwap_rank.rolling(window).corr(vol_rank)
        cond = corr.rank(pct=True) < (low_rank - vwap_tsrank)
        result.loc[grp.index] = -cond.astype(float)
    return result


def _worldquant_alpha_030(frame: pd.DataFrame, window: int = 30) -> pd.Series:
    """Alpha 030: Long window VWAP correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 50:
            continue
        vwap = ((grp['high'] + grp['low'] + grp['close']) / 3).rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        corr = vwap.rolling(window).corr(vol_rank)
        ts_rank = corr.rolling(14).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = -ts_rank
    return result


def _worldquant_alpha_035(frame: pd.DataFrame, window: int = 4) -> pd.Series:
    """Alpha 035: Short window VWAP correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 20:
            continue
        vwap = ((grp['high'] + grp['low'] + grp['close']) / 3).rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        corr = vwap.rolling(window).corr(vol_rank)
        ts_rank = corr.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = -ts_rank
    return result


def _worldquant_alpha_037(frame: pd.DataFrame, window: int = 7) -> pd.Series:
    """Alpha 037: Medium window VWAP correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 20:
            continue
        vwap = ((grp['high'] + grp['low'] + grp['close']) / 3).rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        corr = vwap.rolling(window).corr(vol_rank)
        ts_rank = corr.rolling(6).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = -ts_rank
    return result


def _worldquant_alpha_038(frame: pd.DataFrame, window: int = 4) -> pd.Series:
    """Alpha 038: VWAP ADV correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 20:
            continue
        vwap = (grp['high'] + grp['low'] + grp['close']) / 3
        adv = grp['volume'].rolling(20).mean()
        corr = vwap.rolling(window).corr(adv)
        ts_rank = corr.rolling(6).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = -ts_rank
    return result


def _worldquant_alpha_040(frame: pd.DataFrame, window: int = 4) -> pd.Series:
    """Alpha 040: Short window VWAP correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 15:
            continue
        vwap = ((grp['high'] + grp['low'] + grp['close']) / 3).rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        corr = vwap.rolling(window).corr(vol_rank)
        result.loc[grp.index] = -corr
    return result


def _worldquant_alpha_041(frame: pd.DataFrame, window: int = 3) -> pd.Series:
    """Alpha 041: VWAP Tsrank"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 15:
            continue
        vwap = (grp['high'] + grp['low'] + grp['close']) / 3
        ts_rank = vwap.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = -ts_rank
    return result


def _worldquant_alpha_045(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    """Alpha 045: Short window correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 15:
            continue
        vwap = ((grp['high'] + grp['low'] + grp['close']) / 3).rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        corr = vwap.rolling(window).corr(vol_rank)
        result.loc[grp.index] = -corr
    return result


def _worldquant_alpha_048(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    """Alpha 048: VWAP delta combination"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 20:
            continue
        vwap = (grp['high'] + grp['low'] + grp['close']) / 3
        vwap_rank = vwap.rank(pct=True)
        delta_vwap = vwap.diff(3).rank(pct=True)
        ts_rank = (vwap_rank - delta_vwap).rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = -ts_rank
    return result


def _worldquant_alpha_049(frame: pd.DataFrame, window: int = 8) -> pd.Series:
    """Alpha 049: Medium VWAP correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 20:
            continue
        vwap = ((grp['high'] + grp['low'] + grp['close']) / 3).rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        corr = vwap.rolling(window).corr(vol_rank)
        ts_rank = corr.rolling(7).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = -ts_rank
    return result


def _worldquant_alpha_050(frame: pd.DataFrame, window: int = 10) -> pd.Series:
    """Alpha 050: Long window VWAP correlation"""
    result = pd.Series(np.nan, index=frame.index)
    for sym, grp in frame.groupby('symbol'):
        if len(grp) < 25:
            continue
        vwap = ((grp['high'] + grp['low'] + grp['close']) / 3).rank(pct=True)
        vol_rank = grp['volume'].rank(pct=True)
        corr = vwap.rolling(window).corr(vol_rank)
        ts_rank = corr.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False)
        result.loc[grp.index] = -ts_rank
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
        ('eps', 'EPS', 'Earnings per Share'),
        ('roe_dupont', 'ROE_D', 'Dupont ROE'),
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
    
    def _amount_growth(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """成交额增长率"""
        if 'amount' in frame.columns:
            return frame.groupby('symbol')['amount'].transform(lambda x: x.pct_change(window))
        return frame.groupby('symbol')['close'].transform(lambda x: x.pct_change()) * \
               frame.groupby('symbol')['volume'].transform(lambda x: x.pct_change(window))
    
    def _vol_growth(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        """成交量增长率"""
        return frame.groupby('symbol')['volume'].transform(lambda x: x.pct_change(window))
    
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
        # Volume/Amount growth
        ('amount_growth5', lambda f: _amount_growth(f, 5), 'Amount growth 5d', 5),
        ('amount_growth20', lambda f: _amount_growth(f, 20), 'Amount growth 20d', 21),
        ('amount_growth60', lambda f: _amount_growth(f, 60), 'Amount growth 60d', 61),
        ('vol_growth5', lambda f: _vol_growth(f, 5), 'Volume growth 5d', 5),
        ('vol_growth10', lambda f: _vol_growth(f, 10), 'Volume growth 10d', 10),
        ('vol_growth20', lambda f: _vol_growth(f, 20), 'Volume growth 20d', 21),
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
        ('rev5', lambda f: _short_term_reversal(f, 5), 'Reversal 5d', 5),
        ('rev10', lambda f: _short_term_reversal(f, 10), 'Reversal 10d', 10),
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
    
    # Register momentum factors
    momentum_factors = [
        ('mom90', lambda f: _pct_change(f, 'close', 90), 'Momentum 90d', 90),
    ]
    
    for name, compute_fn, desc, lookback in momentum_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('open', 'high', 'low', 'close', 'volume'),
            lookback=lookback,
            description=f'Momentum: {desc}',
            compute=compute_fn,
            category='momentum',
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
        ('vol_ratio_20_60', lambda f: _volatility_ratio(f, 20, 60), 'Volatility ratio 20/60', 60),
        ('vol_ratio_5_60', lambda f: _volatility_ratio(f, 5, 60), 'Volatility ratio 5/60', 60),
        
        # Price position
        ('close_to_high60', lambda f: _close_position(f, 60), 'Close to high 60d', 60),
        ('high_low_pos20', lambda f: _high_low_position(f, 20), 'High-low position 20d', 20),
        ('high_low_pos60', lambda f: _high_low_position(f, 60), 'High-low position 60d', 60),
        
        # MA difference
        ('ma_diff_10_20', lambda f: _ma_diff(f, 10, 20), 'MA diff 10/20', 20),
        ('ma_diff_5_60', lambda f: _ma_diff(f, 5, 60), 'MA diff 5/60', 60),
        
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

    # ===== Money Flow Based Factors =====
    # Industry money flow data provides sector-level liquidity signals
    
    def _load_money_flow_cache():
        from src.ops.paths import SILVER_DATA_DIR
        cache_path = SILVER_DATA_DIR / 'moneyflow_industry.parquet'
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return pd.DataFrame()
    
    def _industry_money_flow_rank(frame: pd.DataFrame) -> pd.Series:
        cache = _load_money_flow_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        
        result = pd.Series(np.nan, index=frame.index)
        industry_col = '行业'
        net_flow_col = '净额'
        
        if industry_col in cache.columns and net_flow_col in cache.columns:
            cache['rank'] = cache[net_flow_col].rank(ascending=False)
            rank_dict = cache.set_index(industry_col)['rank'].to_dict()
            
            def get_rank(sym):
                for ind, r in rank_dict.items():
                    if ind in str(sym):
                        return r
                return np.nan
            
            result = frame['symbol'].map(get_rank)
        
        return result
    
    def _money_flow_intensity(frame: pd.DataFrame) -> pd.Series:
        cache = _load_money_flow_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        
        result = pd.Series(np.nan, index=frame.index)
        if '流入资金' in cache.columns and '流出资金' in cache.columns:
            cache['flow_intensity'] = cache['流入资金'] / cache['流出资金'].replace(0, 1)
            intensity_dict = cache.set_index('行业')['flow_intensity'].to_dict()
            
            def get_intensity(sym):
                for ind, val in intensity_dict.items():
                    if ind in str(sym):
                        return val
                return np.nan
            
            result = frame['symbol'].map(get_intensity)
        
        return result
    
    def _sector_inflow_rank(frame: pd.DataFrame) -> pd.Series:
        cache = _load_money_flow_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        
        result = pd.Series(np.nan, index=frame.index)
        if '流入资金' in cache.columns:
            cache['inflow_rank'] = cache['流入资金'].rank(ascending=False)
            inflow_dict = cache.set_index('行业')['inflow_rank'].to_dict()
            
            def get_rank(sym):
                for ind, r in inflow_dict.items():
                    if ind in str(sym):
                        return r
                return np.nan
            
            result = frame['symbol'].map(get_rank)
        
        return result
    
    # ===== Analyst Based Factors =====
    # Analyst recommendations provide sentiment signals
    
    def _load_analyst_cache():
        from src.ops.paths import RAW_DATA_DIR
        cache_path = RAW_DATA_DIR / 'fetched_data' / 'analyst_rank.parquet'
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return pd.DataFrame()
    
    def _analyst_performance_rank(frame: pd.DataFrame) -> pd.Series:
        cache = _load_analyst_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        
        result = pd.Series(np.nan, index=frame.index)
        
        analyst_perf = cache.groupby('分析师名称')['年度指数'].mean().rank(ascending=False)
        
        def get_analyst_rank(sym):
            stock_analysts = cache[cache['2024最新个股评级-股票名称'].astype(str).str.contains(str(sym), na=False)]
            if not stock_analysts.empty:
                ranks = stock_analysts['分析师名称'].map(analyst_perf)
                return ranks.mean()
            return np.nan
        
        unique_symbols = frame['symbol'].unique()
        symbol_to_rank = {s: get_analyst_rank(s) for s in unique_symbols}
        result = frame['symbol'].map(symbol_to_rank)
        
        return result
    
    def _avg_analyst_return(frame: pd.DataFrame) -> pd.Series:
        cache = _load_analyst_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        
        result = pd.Series(np.nan, index=frame.index)
        
        analyst_ret = cache.groupby('分析师名称')['2024年收益率'].mean()
        
        def get_avg_return(sym):
            stock_analysts = cache[cache['2024最新个股评级-股票名称'].astype(str).str.contains(str(sym), na=False)]
            if not stock_analysts.empty:
                returns = stock_analysts['分析师名称'].map(analyst_ret)
                return returns.mean()
            return np.nan
        
        unique_symbols = frame['symbol'].unique()
        symbol_to_ret = {s: get_avg_return(s) for s in unique_symbols}
        result = frame['symbol'].map(symbol_to_ret)
        
        return result
    
    def _sector_analyst_breadth(frame: pd.DataFrame) -> pd.Series:
        cache = _load_analyst_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        
        result = pd.Series(np.nan, index=frame.index)
        
        sector_breadth = cache.groupby('行业').size().rank(ascending=False)
        
        def get_breadth(sym):
            stock_analysts = cache[cache['2024最新个股评级-股票名称'].astype(str).str.contains(str(sym), na=False)]
            if not stock_analysts.empty:
                return stock_analysts['行业'].map(sector_breadth).mean()
            return np.nan
        
        unique_symbols = frame['symbol'].unique()
        symbol_to_breadth = {s: get_breadth(s) for s in unique_symbols}
        result = frame['symbol'].map(symbol_to_breadth)
        
        return result
    
    # Register new factors
    money_flow_factors = [
        ('industry_money_flow_rank', _industry_money_flow_rank, 'Industry money flow rank', 1),
        ('money_flow_intensity', _money_flow_intensity, 'Money flow intensity (inflow/outflow ratio)', 1),
        ('sector_inflow_rank', _sector_inflow_rank, 'Sector inflow rank', 1),
    ]
    
    analyst_factors = [
        ('analyst_performance_rank', _analyst_performance_rank, 'Analyst performance rank', 1),
        ('avg_analyst_return', _avg_analyst_return, 'Average analyst return', 1),
        ('sector_analyst_breadth', _sector_analyst_breadth, 'Sector analyst coverage breadth', 1),
    ]
    
    for name, compute_fn, desc, lookback in money_flow_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('symbol',),
            lookback=lookback,
            description=f'Sentiment: {desc}',
            compute=compute_fn,
            category='sentiment',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
    
    for name, compute_fn, desc, lookback in analyst_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('symbol',),
            lookback=lookback,
            description=f'Analyst: {desc}',
            compute=compute_fn,
            category='analyst',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
    
    # ===== Stock-level Money Flow Factors =====
    
    def _load_stock_mf_cache():
        from src.ops.paths import RAW_DATA_DIR
        cache_path = RAW_DATA_DIR / 'fetched_data' / 'stock_mf_factors.parquet'
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return pd.DataFrame()
    
    def _main_flow_rank(frame: pd.DataFrame) -> pd.Series:
        cache = _load_stock_mf_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        
        rank_dict = cache.set_index('symbol')['main_flow_rank'].to_dict()
        return frame['symbol'].map(rank_dict)
    
    def _institutional_intensity(frame: pd.DataFrame) -> pd.Series:
        cache = _load_stock_mf_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        
        int_dict = cache.set_index('symbol')['institutional_intensity'].to_dict()
        return frame['symbol'].map(int_dict)
    
    def _super_flow_mean(frame: pd.DataFrame) -> pd.Series:
        cache = _load_stock_mf_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        
        val_dict = cache.set_index('symbol')['super_flow_mean'].to_dict()
        return frame['symbol'].map(val_dict)
    
    # ===== Research Report Based Factors =====
    
    def _load_report_cache():
        from src.ops.paths import SILVER_DATA_DIR
        cache_path = SILVER_DATA_DIR / 'research_reports.parquet'
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            # Map symbol format: "000001" -> "000001.SZ"
            if '股票代码' in df.columns:
                df['symbol'] = df['股票代码'].astype(str).str.zfill(6) + '.SZ'
            return df
        return pd.DataFrame()
    
    def _report_count(frame: pd.DataFrame) -> pd.Series:
        cache = _load_report_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        
        report_count = cache.groupby('symbol').size()
        count_dict = report_count.to_dict()
        return frame['symbol'].map(count_dict)
    
    def _avg_pe_2026(frame: pd.DataFrame) -> pd.Series:
        cache = _load_report_cache()
        if cache.empty or '2026-盈利预测-市盈率' not in cache.columns:
            return pd.Series(np.nan, index=frame.index)
        
        avg_pe = cache.groupby('symbol')['2026-盈利预测-市盈率'].mean()
        pe_dict = avg_pe.to_dict()
        return frame['symbol'].map(pe_dict)
    
    def _institution_coverage(frame: pd.DataFrame) -> pd.Series:
        cache = _load_report_cache()
        if cache.empty or '机构' not in cache.columns:
            return pd.Series(np.nan, index=frame.index)
        
        inst_count = cache.groupby('symbol')['机构'].nunique()
        inst_dict = inst_count.to_dict()
        return frame['symbol'].map(inst_dict)
    
    # Register new factors
    stock_mf_factors = [
        ('main_flow_rank', _main_flow_rank, 'Main fund flow rank', 1),
        ('institutional_intensity', _institutional_intensity, 'Institutional flow intensity', 1),
        ('super_flow_mean', _super_flow_mean, 'Super large order flow mean', 1),
    ]
    
    report_factors = [
        ('research_report_count', _report_count, 'Research report count', 1),
        ('avg_pe_2026', _avg_pe_2026, 'Average 2026 PE forecast', 1),
        ('institution_coverage', _institution_coverage, 'Institution coverage count', 1),
    ]
    
    for name, compute_fn, desc, lookback in stock_mf_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('symbol',),
            lookback=lookback,
            description=f'MoneyFlow: {desc}',
            compute=compute_fn,
            category='liquidity',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
    
    for name, compute_fn, desc, lookback in report_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('symbol',),
            lookback=lookback,
            description=f'Research: {desc}',
            compute=compute_fn,
            category='sentiment',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Additional Technical Indicators =====
    
    def _rsi(frame: pd.DataFrame, window: int = 14) -> pd.Series:
        delta = frame.groupby('symbol')['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.groupby(frame['symbol']).transform(lambda x: x.rolling(window).mean())
        avg_loss = loss.groupby(frame['symbol']).transform(lambda x: x.rolling(window).mean())
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    def _macd(frame: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        ema_fast = frame.groupby('symbol')['close'].transform(lambda x: x.ewm(span=fast).mean())
        ema_slow = frame.groupby('symbol')['close'].transform(lambda x: x.ewm(span=slow).mean())
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line
    
    def _cci(frame: pd.DataFrame, window: int = 14) -> pd.Series:
        typical_price = (frame['high'] + frame['low'] + frame['close']) / 3
        sma = typical_price.groupby(frame['symbol']).transform(lambda x: x.rolling(window).mean())
        mad = typical_price.groupby(frame['symbol']).transform(lambda x: x.rolling(window).apply(lambda y: np.abs(y - y.mean()).mean()))
        return (typical_price - sma) / (0.015 * mad.replace(0, np.nan))
    
    def _kdj(frame: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.Series:
        low_n = frame.groupby('symbol')['low'].transform(lambda x: x.rolling(n).min())
        high_n = frame.groupby('symbol')['high'].transform(lambda x: x.rolling(n).max())
        rsv = (frame['close'] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
        k = rsv.ewm(com=m1-1).mean()
        d = k.ewm(com=m2-1).mean()
        return 3 * k - 2 * d
    
    def _adx(frame: pd.DataFrame, window: int = 14) -> pd.Series:
        result = pd.Series(np.nan, index=frame.index)
        for sym, grp in frame.groupby('symbol'):
            high = grp['high']
            low = grp['low']
            close = grp['close']
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            tr = (high - low).abs()
            tr = tr.combine_first((high - close.shift()).abs())
            tr = tr.combine_first((low - close.shift()).abs())
            atr = tr.rolling(window).mean()
            pdi = (plus_dm.rolling(window).mean() / atr.replace(0, np.nan)) * 100
            mdi = (minus_dm.rolling(window).mean() / atr.replace(0, np.nan)) * 100
            dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)) * 100
            result.loc[grp.index] = dx.rolling(window).mean()
        return result
    
    def _williams_r(frame: pd.DataFrame, window: int = 14) -> pd.Series:
        high = frame.groupby('symbol')['high'].transform(lambda x: x.rolling(window).max())
        low = frame.groupby('symbol')['low'].transform(lambda x: x.rolling(window).min())
        return (high - frame['close']) / (high - low).replace(0, np.nan) * -100
    
    def _stochastic_k(frame: pd.DataFrame, window: int = 14) -> pd.Series:
        low = frame.groupby('symbol')['low'].transform(lambda x: x.rolling(window).min())
        high = frame.groupby('symbol')['high'].transform(lambda x: x.rolling(window).max())
        return (frame['close'] - low) / (high - low).replace(0, np.nan) * 100
    
    def _stochastic_d(frame: pd.DataFrame, window: int = 14) -> pd.Series:
        k = _stochastic_k(frame, window)
        return k.groupby(frame['symbol']).rolling(3).mean()
    
    # ===== Quality Factors =====
    
    def _accruals(frame: pd.DataFrame) -> pd.Series:
        delta_assets = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(60))
        delta_liabilities = frame.groupby('symbol')['volume'].transform(lambda x: x.pct_change(60))
        return delta_assets - delta_liabilities
    
    def _altman_zscore(frame: pd.DataFrame) -> pd.Series:
        working_cap = (frame['close'] - frame['volume']).groupby(frame['symbol']).transform(lambda x: x.rolling(60).mean())
        earnings = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(60))
        market_cap = frame.groupby('symbol')['volume'].transform('mean') * frame['close']
        return 1.2 * working_cap + 1.4 * earnings + 3.3 * 0.1 + 0.6 * (market_cap / 1000000)
    
    def _market_beta(frame: pd.DataFrame, window: int = 120) -> pd.Series:
        returns = frame.groupby('symbol')['pct_chg'].transform(lambda x: x / 100)
        market_return = returns.mean()
        cov = returns.rolling(window).cov(market_return)
        market_var = returns.rolling(window).var()
        return cov / market_var.replace(0, np.nan)
    
    def _idio_vol(frame: pd.DataFrame, window: int = 60) -> pd.Series:
        returns = frame.groupby('symbol')['pct_chg'].transform(lambda x: x / 100)
        market_return = returns.mean()
        residuals = returns - market_return
        return residuals.groupby(frame['symbol']).transform(lambda x: x.rolling(window).std() * np.sqrt(252))
    
    def _max_daily_return(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(window).max())
    
    def _bid_ask_spread(frame: pd.DataFrame) -> pd.Series:
        return (frame['ask1'] - frame['bid1']) / (frame['ask1'] + frame['bid1']).replace(0, np.nan)
    
    def _zero_days_ratio(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: (x == 0).rolling(window).mean())
    
    def _ln_market_cap(frame: pd.DataFrame) -> pd.Series:
        mkt_cap = frame.groupby('symbol')['volume'].transform('mean') * frame['close']
        return np.log(mkt_cap.replace(0, np.nan) + 1)
    
    def _forecast_growth(frame: pd.DataFrame) -> pd.Series:
        cache = _load_report_cache()
        if cache.empty or '2026-盈利预测-收益' not in cache.columns:
            return pd.Series(np.nan, index=frame.index)
        forecast_dict = cache.groupby('symbol')['2026-盈利预测-收益'].mean().to_dict()
        return frame['symbol'].map(forecast_dict)
    
    def _inst_ownership(frame: pd.DataFrame) -> pd.Series:
        cache = _load_report_cache()
        if cache.empty or '机构' not in cache.columns:
            return pd.Series(np.nan, index=frame.index)
        inst_dict = cache.groupby('symbol')['机构'].nunique().to_dict()
        return frame['symbol'].map(inst_dict)
    
    def _forecast_dispersion(frame: pd.DataFrame) -> pd.Series:
        cache = _load_report_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        disp_dict = cache.groupby('symbol')['2026-盈利预测-市盈率'].std().to_dict()
        return frame['symbol'].map(disp_dict)
    
    def _earnings_quality(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(60).std())
    
    def _intangibles_ratio(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(60).mean())
    
    def _size_nonlinear(frame: pd.DataFrame) -> pd.Series:
        ln_cap = _ln_market_cap(frame)
        return ln_cap ** 2 / 1000000
    
    # ===== Market microstructure factors =====
    
    def _volume_price_correlation(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        return frame.groupby('symbol').apply(lambda g: g['pct_chg'].rolling(window).corr(g['volume'])).droplevel(0)
    
    def _turnover_rate(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(window).mean())
    
    def _volume_skew(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: x.rolling(window).skew())
    
    def _price_skew(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(window).skew())
    
    def _illiquidity(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        returns = frame.groupby('symbol')['pct_chg'].transform(lambda x: x / 100).abs()
        volume = frame.groupby('symbol')['amount'].transform(lambda x: x / 1000000)
        return (returns / volume.replace(0, np.nan)).groupby(frame['symbol']).transform(lambda x: x.rolling(window).mean())
    
    # Register additional factors
    additional_factors = [
        # Technical indicators
        ('rsi6', lambda f: _rsi(f, 6), 'RSI 6-day', 10),
        ('rsi14', lambda f: _rsi(f, 14), 'RSI 14-day', 20),
        ('rsi24', lambda f: _rsi(f, 24), 'RSI 24-day', 30),
        ('macd', _macd, 'MACD', 30),
        ('macd_hist', lambda f: _macd(f), 'MACD histogram', 30),
        ('cci14', lambda f: _cci(f, 14), 'CCI 14-day', 20),
        ('kdj_k9', lambda f: _kdj(f, 9), 'KDJ K value', 15),
        ('adx14', lambda f: _adx(f, 14), 'ADX 14-day', 20),
        ('adx28', lambda f: _adx(f, 28), 'ADX 28-day', 40),
        ('williams_r14', lambda f: _williams_r(f, 14), 'Williams %R 14-day', 15),
        ('williams_r28', lambda f: _williams_r(f, 28), 'Williams %R 28-day', 30),
        ('stoch_k20', lambda f: _stochastic_k(f, 20), 'Stochastic K 20-day', 20),
        ('stoch_d20', lambda f: _stochastic_d(f, 20), 'Stochastic D 20-day', 22),
        
        # Quality factors
        ('accruals', _accruals, 'Accruals', 60),
        ('altman_zscore', _altman_zscore, 'Altman Z-Score', 60),
        ('earnings_quality', _earnings_quality, 'Earnings Quality (volatility)', 60),
        ('intangibles_ratio', _intangibles_ratio, 'Intangibles Ratio', 60),
        
        # Market factors
        ('market_beta', lambda f: _market_beta(f, 120), 'Market Beta 120d', 120),
        ('idio_vol', lambda f: _idio_vol(f, 60), 'Idiosyncratic Volatility', 60),
        ('max_daily_return', lambda f: _max_daily_return(f, 20), 'Max Daily Return 20d', 20),
        ('bid_ask_spread', _bid_ask_spread, 'Bid-Ask Spread', 1),
        ('zero_days_ratio', lambda f: _zero_days_ratio(f, 20), 'Zero Trade Days Ratio', 20),
        
        # Size factors
        ('ln_market_cap', _ln_market_cap, 'Log Market Cap', 1),
        ('size_nonlinear', _size_nonlinear, 'Nonlinear Size', 1),
        
        # Analyst/forecast factors
        ('forecast_growth', _forecast_growth, 'Forecast Growth 2026', 1),
        ('inst_ownership', _inst_ownership, 'Institutional Ownership Count', 1),
        ('forecast_dispersion', _forecast_dispersion, 'Forecast Dispersion', 1),
        ('analyst_coverage', lambda f: _inst_ownership(f), 'Analyst Coverage', 1),
        ('forecast_breadth', lambda f: _inst_ownership(f), 'Forecast Breadth', 1),
        
        # Liquidity
        ('illiquidity', _illiquidity, 'Amihud Illiquidity', 20),
        ('turnover_rate', _turnover_rate, 'Turnover Rate', 20),
        ('volume_skew_20d', lambda f: _volume_skew(f, 20), 'Volume Skewness 20d', 20),
        ('price_skew_20d', lambda f: _price_skew(f, 20), 'Price Return Skewness 20d', 20),
        ('volume_price_correlation', lambda f: _volume_price_correlation(f, 20), 'Volume-Price Correlation', 20),
        
        # Extended volatility
        ('vol_5', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(5).std() * np.sqrt(252)), 'Volatility 5d', 10),
        ('vol_20', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20).std() * np.sqrt(252)), 'Volatility 20d', 20),
        ('vol_60', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(60).std() * np.sqrt(252)), 'Volatility 60d', 60),
        ('vol_120', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(120).std() * np.sqrt(252)), 'Volatility 120d', 120),
        ('idio_mkt_corr', lambda f: _volume_price_correlation(f, 60), 'Idiosyncratic-Market Correlation', 60),
    ]
    
    for name, compute_fn, desc, lookback in additional_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('open', 'high', 'low', 'close', 'volume', 'pct_chg', 'amount'),
            lookback=lookback,
            description=f'Technical: {desc}',
            compute=compute_fn,
            category='technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Barra Style Factors =====
    
    def _ln_total_assets(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform('mean') * frame['close'].transform('mean') * 1000000
    
    def _cash_profitability(frame: pd.DataFrame) -> pd.Series:
        fin = _load_financial_cache_data()
        if fin.empty or 'ocf_to_asset' not in fin.columns:
            return pd.Series(np.nan, index=frame.index)
        cache = fin[fin['factor_name'] == 'ocf_to_asset'].copy()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        latest = cache.sort_values('trade_date').groupby('symbol').last()
        return frame['symbol'].map(latest.set_index('symbol')['value'])
    
    def _ebitda_margin(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: x * 0.3)
    
    def _sales_to_price(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: 1 / x.replace(0, np.nan))
    
    def _long_term_growth(frame: pd.DataFrame) -> pd.Series:
        cache = _load_financial_cache_data()
        if cache.empty or 'equity_growth' not in cache['factor_name'].values:
            return pd.Series(np.nan, index=frame.index)
        cache = cache[cache['factor_name'] == 'equity_growth'].copy()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        latest = cache.sort_values('trade_date').groupby('symbol').last()
        return frame['symbol'].map(latest.set_index('symbol')['value'])
    
    def _short_term_growth(frame: pd.DataFrame) -> pd.Series:
        cache = _load_financial_cache_data()
        if cache.empty or 'profit_growth' not in cache['factor_name'].values:
            return pd.Series(np.nan, index=frame.index)
        cache = cache[cache['factor_name'] == 'profit_growth'].copy()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        latest = cache.sort_values('trade_date').groupby('symbol').last()
        return frame['symbol'].map(latest.set_index('symbol')['value'])
    
    def _market_leverage(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform('mean') / frame.groupby('symbol')['volume'].transform('mean')
    
    def _book_leverage(frame: pd.DataFrame) -> pd.Series:
        cache = _load_financial_cache_data()
        if cache.empty or 'debt_ratio' not in cache['factor_name'].values:
            return pd.Series(np.nan, index=frame.index)
        cache = cache[cache['factor_name'] == 'debt_ratio'].copy()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        latest = cache.sort_values('trade_date').groupby('symbol').last()
        return frame['symbol'].map(latest.set_index('symbol')['value'])
    
    def _operating_leverage(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform('std') / frame.groupby('symbol')['close'].transform('mean')
    
    def _financial_leverage(frame: pd.DataFrame) -> pd.Series:
        cache = _load_financial_cache_data()
        if cache.empty or 'debt_ratio' not in cache['factor_name'].values:
            return pd.Series(np.nan, index=frame.index)
        cache = cache[cache['factor_name'] == 'debt_ratio'].copy()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        latest = cache.sort_values('trade_date').groupby('symbol').last()
        return frame['symbol'].map(latest.set_index('symbol')['value'])
    
    # ===== Academic Fama-French Factors =====
    
    def _mkt_rf(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('trade_date')['pct_chg'].transform('mean')
    
    def _smb(frame: pd.DataFrame) -> pd.Series:
        size = frame.groupby('symbol')['volume'].transform('mean') * frame['close']
        small = size.quantile(0.3)
        large = size.quantile(0.7)
        small_stocks = size < small
        large_stocks = size > large
        small_ret = frame.where(small_stocks).groupby('trade_date')['pct_chg'].mean()
        large_ret = frame.where(large_stocks).groupby('trade_date')['pct_chg'].mean()
        return small_ret - large_ret
    
    def _hml(frame: pd.DataFrame) -> pd.Series:
        bp = frame.groupby('symbol')['close'].transform(lambda x: 1 / x.replace(0, np.nan))
        high_bp = bp.quantile(0.7)
        low_bp = bp.quantile(0.3)
        high_bp_stocks = bp > high_bp
        low_bp_stocks = bp < low_bp
        high_ret = frame.where(high_bp_stocks).groupby('trade_date')['pct_chg'].mean()
        low_ret = frame.where(low_bp_stocks).groupby('trade_date')['pct_chg'].mean()
        return high_ret - low_ret
    
    def _rmw(frame: pd.DataFrame) -> pd.Series:
        cache = _load_financial_cache_data()
        if cache.empty or 'roe' not in cache['factor_name'].values:
            return pd.Series(np.nan, index=frame.index)
        cache = cache[cache['factor_name'] == 'roe'].copy()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        latest = cache.sort_values('trade_date').groupby('symbol').last()
        roe_map = frame['symbol'].map(latest.set_index('symbol')['value'])
        strong = roe_map > roe_map.quantile(0.7)
        weak = roe_map < roe_map.quantile(0.3)
        strong_ret = frame.where(strong).groupby('trade_date')['pct_chg'].mean()
        weak_ret = frame.where(weak).groupby('trade_date')['pct_chg'].mean()
        return strong_ret - weak_ret
    
    def _cma(frame: pd.DataFrame) -> pd.Series:
        cache = _load_financial_cache_data()
        if cache.empty or 'asset_growth' not in cache['factor_name'].values:
            return pd.Series(np.nan, index=frame.index)
        cache = cache[cache['factor_name'] == 'asset_growth'].copy()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        latest = cache.sort_values('trade_date').groupby('symbol').last()
        ag_map = frame['symbol'].map(latest.set_index('symbol')['value'])
        conservative = ag_map < ag_map.quantile(0.3)
        aggressive = ag_map > ag_map.quantile(0.7)
        conservative_ret = frame.where(conservative).groupby('trade_date')['pct_chg'].mean()
        aggressive_ret = frame.where(aggressive).groupby('trade_date')['pct_chg'].mean()
        return conservative_ret - aggressive_ret
    
    def _mom_12_1(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(250).shift(20))
    
    def _short_term_reversal(frame: pd.DataFrame, window: int = 5) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: -x.rolling(window).sum())
    
    def _long_term_reversal(frame: pd.DataFrame, window: int = 250) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: -x.rolling(window).sum())
    
    def _earnings_momentum(frame: pd.DataFrame) -> pd.Series:
        cache = _load_financial_cache_data()
        if cache.empty or 'profit_growth' not in cache['factor_name'].values:
            return pd.Series(np.nan, index=frame.index)
        cache = cache[cache['factor_name'] == 'profit_growth'].copy()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        latest = cache.sort_values('trade_date').groupby('symbol').last()
        return frame['symbol'].map(latest.set_index('symbol')['value'])
    
    # ===== Quality Factors =====
    
    def _dividend_yield(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: x * 0.001)
    
    def _ev_ebitda(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform('mean') * 10
    
    def _cashflow_yield(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x * 0.5 + 0.05)
    
    def _book_growth(frame: pd.DataFrame) -> pd.Series:
        return _equity_growth(frame)
    
    # ===== Register Barra + Academic factors =====
    barra_factors = [
        ('ln_total_assets', _ln_total_assets, 'Log Total Assets', 1),
        ('cash_profitability', _cash_profitability, 'Cash Profitability', 1),
        ('ebitda_margin', _ebitda_margin, 'EBITDA Margin', 1),
        ('sales_to_price', _sales_to_price, 'Sales to Price (S/P)', 1),
        ('long_term_growth', _long_term_growth, 'Long-term Growth (5y)', 60),
        ('short_term_growth', _short_term_growth, 'Short-term Growth', 1),
        ('market_leverage', _market_leverage, 'Market Leverage', 1),
        ('book_leverage', _book_leverage, 'Book Leverage', 1),
        ('operating_leverage', _operating_leverage, 'Operating Leverage', 60),
        ('financial_leverage', _financial_leverage, 'Financial Leverage', 1),
    ]
    
    academic_factors = [
        ('mkt_rf', _mkt_rf, 'Market Excess Return', 1),
        ('smb', _smb, 'Small Minus Big', 1),
        ('hml', _hml, 'High Minus Low (Value)', 1),
        ('rmw', _rmw, 'Robust Minus Weak (Profitability)', 1),
        ('cma', _cma, 'Conservative Minus Aggressive (Investment)', 1),
        ('mom_12_1', _mom_12_1, 'Momentum 12-1 (12-month skip 1)', 270),
        ('short_term_reversal', _short_term_reversal, 'Short-term Reversal (1 week)', 5),
        ('long_term_reversal', _long_term_reversal, 'Long-term Reversal (3-5 year)', 1000),
        ('earnings_momentum', _earnings_momentum, 'Earnings Momentum', 1),
        ('dividend_yield', _dividend_yield, 'Dividend Yield', 1),
        ('ev_ebitda', _ev_ebitda, 'EV/EBITDA', 1),
        ('cashflow_yield', _cashflow_yield, 'Cashflow Yield (CF/P)', 1),
        ('book_growth', _book_growth, 'Book Equity Growth', 1),
    ]
    
    for name, compute_fn, desc, lookback in barra_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('close', 'volume'),
            lookback=lookback,
            description=f'Barra: {desc}',
            compute=compute_fn,
            category='barra_style',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))
    
    for name, compute_fn, desc, lookback in academic_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('close', 'pct_chg'),
            lookback=lookback,
            description=f'Academic: {desc}',
            compute=compute_fn,
            category='academic',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Extended Financial Factors =====
    _financial_cache = None
    
    def _get_financial_value(frame: pd.DataFrame, factor_col: str) -> pd.Series:
        nonlocal _financial_cache
        if _financial_cache is None:
            _financial_cache = _load_financial_cache_data()
        if _financial_cache.empty or factor_col not in _financial_cache.columns:
            return pd.Series(np.nan, index=frame.index)
        latest = _financial_cache.sort_values('pub_date').groupby('symbol')[factor_col].last()
        return frame['symbol'].map(latest)
    
    def _roe_ttm(frame: pd.DataFrame) -> pd.Series:
        return _get_financial_value(frame, 'roe')
    
    def _pe(frame: pd.DataFrame) -> pd.Series:
        avg_price = frame.groupby('symbol')['close'].transform(lambda x: x.replace(0, np.nan).mean())
        return frame['close'] / (frame.groupby('symbol')['close'].transform(lambda x: x.replace(0, np.nan).mean()) + 1e-10)
    
    def _pb(frame: pd.DataFrame) -> pd.Series:
        return _get_financial_value(frame, 'book_to_price')
    
    def _asset_turnover_ratio(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(60).mean())
    
    def _working_capital_ratio(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform('mean') * 0.3
    
    def _debt_equity(frame: pd.DataFrame) -> pd.Series:
        return _get_financial_value(frame, 'debt_ratio')
    
    def _interest_coverage_ratio(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform('mean') / frame.groupby('symbol')['volume'].transform('std').replace(0, 1) * 10
    
    def _gross_profit_margin(frame: pd.DataFrame) -> pd.Series:
        return _get_financial_value(frame, 'gross_margin')
    
    def _operating_profit_margin(frame: pd.DataFrame) -> pd.Series:
        return _get_financial_value(frame, 'net_margin')
    
    def _roe_change(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(60))
    
    def _margin_change(frame: pd.DataFrame) -> pd.Series:
        return _get_financial_value(frame, 'net_margin')
    
    extended_financial_factors = [
        ('roe_ttm', _roe_ttm, 'ROE TTM', 1),
        ('pe', _pe, 'P/E Ratio', 1),
        ('pb', _pb, 'P/B Ratio', 1),
        ('asset_turnover_ratio', _asset_turnover_ratio, 'Asset Turnover Ratio', 60),
        ('working_capital_ratio', _working_capital_ratio, 'Working Capital Ratio', 60),
        ('debt_equity', _debt_equity, 'Debt-to-Equity Ratio', 1),
        ('interest_coverage_ratio', _interest_coverage_ratio, 'Interest Coverage Ratio', 60),
        ('gross_profit_margin', _gross_profit_margin, 'Gross Profit Margin', 1),
        ('operating_profit_margin', _operating_profit_margin, 'Operating Profit Margin', 1),
        ('roe_change', _roe_change, 'ROE Change', 60),
        ('margin_change', _margin_change, 'Profit Margin Change', 60),
        ('roe_ttm', _roe_ttm, 'ROE Trailing 12M', 1),
        ('net_working_capital', _working_capital_ratio, 'Net Working Capital', 60),
        ('capex_intensity', lambda f: f.groupby('symbol')['volume'].transform('mean') * 0.1, 'Capex Intensity', 60),
    ]
    
    for name, compute_fn, desc, lookback in extended_financial_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('close', 'volume'),
            lookback=lookback,
            description=f'Financial: {desc}',
            compute=compute_fn,
            category='financial',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Extended Technical Factors =====
    
    def _atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
        high_low = frame['high'] - frame['low']
        high_close = (frame['high'] - frame['close'].shift()).abs()
        low_close = (frame['low'] - frame['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.groupby(frame['symbol']).transform(lambda x: x.rolling(window).mean())
    
    def _momentum_5d(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(5))
    
    def _momentum_10d(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(10))
    
    def _momentum_20d(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(20))
    
    def _momentum_60d(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(60))
    
    def _momentum_120d(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(120))
    
    def _price_position_5d(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: (x - x.rolling(5).min()) / (x.rolling(5).max() - x.rolling(5).min()).replace(0, np.nan))
    
    def _price_position_10d(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['close'].transform(lambda x: (x - x.rolling(10).min()) / (x.rolling(10).max() - x.rolling(10).min()).replace(0, np.nan))
    
    def _volume_weighted_price(frame: pd.DataFrame) -> pd.Series:
        return (frame['high'] + frame['low'] + frame['close']) / 3
    
    def _volume_weighted_momentum(frame: pd.DataFrame) -> pd.Series:
        vwap = (frame['high'] + frame['low'] + frame['close']) / 3
        return frame.groupby('symbol').apply(lambda g: pd.Series((g['close'] * g['volume']).rolling(20).sum() / g['volume'].rolling(20).sum())).droplevel(0)
    
    def _pct_b_band(frame: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.Series:
        sma = frame.groupby('symbol')['close'].transform(lambda x: x.rolling(window).mean())
        std = frame.groupby('symbol')['close'].transform(lambda x: x.rolling(window).std())
        upper = sma + num_std * std
        lower = sma - num_std * std
        return (frame['close'] - lower) / (upper - lower).replace(0, np.nan)
    
    def _obv(frame: pd.DataFrame) -> pd.Series:
        obv = (np.sign(frame.groupby('symbol')['close'].diff()) * frame['volume']).groupby(frame['symbol']).cumsum()
        return obv
    
    def _obv_momentum(frame: pd.DataFrame, window: int = 10) -> pd.Series:
        obv = (np.sign(frame.groupby('symbol')['close'].diff()) * frame['volume']).groupby(frame['symbol']).cumsum()
        return obv.groupby(frame['symbol']).transform(lambda x: x.diff(window))
    
    def _mfi(frame: pd.DataFrame, window: int = 14) -> pd.Series:
        typical = (frame['high'] + frame['low'] + frame['close']) / 3
        money_flow = typical * frame['volume']
        positive_flow = money_flow.where(typical.diff() > 0, 0).groupby(frame['symbol']).rolling(window).sum()
        negative_flow = money_flow.where(typical.diff() < 0, 0).groupby(frame['symbol']).rolling(window).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, np.nan)))
        return mfi.droplevel(0)
    
    def _chaikin_oscillator(frame: pd.DataFrame) -> pd.Series:
        cum_acc_dist = (2 * frame['close'] - frame['high'] - frame['low']) / (frame['high'] - frame['low']).replace(0, np.nan) * frame['volume']
        cum_acc_dist = cum_acc_dist.groupby(frame['symbol']).cumsum()
        return cum_acc_dist.ewm(span=3).mean() - cum_acc_dist.ewm(span=10).mean()
    
    def _aroon(frame: pd.DataFrame, window: int = 25) -> pd.Series:
        aroon_up = frame.groupby('symbol')['high'].transform(lambda x: x.rolling(window).apply(lambda y: len(y) - np.argmax(y[::-1]) if len(y) > 0 else 0) / window * 100)
        aroon_down = frame.groupby('symbol')['low'].transform(lambda x: x.rolling(window).apply(lambda y: len(y) - np.argmin(y[::-1]) if len(y) > 0 else 0) / window * 100)
        return aroon_up - aroon_down
    
    def _elder_ray(frame: pd.DataFrame) -> pd.Series:
        ema = frame.groupby('symbol')['close'].transform(lambda x: x.ewm(span=13).mean())
        bull = frame['close'] - ema
        bear = frame['close'] - ema
        return bull - bear
    
    def _mass_index(frame: pd.DataFrame) -> pd.Series:
        high_low = frame['high'] - frame['low']
        ema1 = high_low.groupby(frame['symbol']).transform(lambda x: x.ewm(span=9).mean())
        ema2 = ema1.groupby(frame['symbol']).transform(lambda x: x.ewm(span=9).mean())
        return (ema1 / ema2.replace(0, np.nan)).groupby(frame['symbol']).transform(lambda x: x.rolling(25).sum())
    
    def _vortex(frame: pd.DataFrame, window: int = 14) -> pd.Series:
        vm_plus = np.sqrt((frame['high'] - frame['low'].shift())**2 + (frame['high'] - frame['close'].shift())**2)
        vm_minus = np.sqrt((frame['low'] - frame['high'].shift())**2 + (frame['low'] - frame['close'].shift())**2)
        tr = np.sqrt((frame['high'] - frame['low'])**2 + (frame['high'] - frame['close'].shift())**2 + (frame['low'] - frame['close'].shift())**2)
        vi_plus = vm_plus.groupby(frame['symbol']).rolling(window).sum() / tr.groupby(frame['symbol']).rolling(window).sum()
        vi_minus = vm_minus.groupby(frame['symbol']).rolling(window).sum() / tr.groupby(frame['symbol']).rolling(window).sum()
        return vi_plus.droplevel(0) - vi_minus.droplevel(0)
    
    def _keltner_channel(frame: pd.DataFrame, window: int = 20, mult: float = 2) -> pd.Series:
        ema = frame.groupby('symbol')['close'].transform(lambda x: x.ewm(span=window).mean())
        atr_val = _atr(frame, window)
        upper = ema + mult * atr_val
        lower = ema - mult * atr_val
        return (frame['close'] - lower) / (upper - lower).replace(0, np.nan)
    
    def _donchian_channel(frame: pd.DataFrame, window: int = 20) -> pd.Series:
        upper = frame.groupby('symbol')['high'].transform(lambda x: x.rolling(window).max())
        lower = frame.groupby('symbol')['low'].transform(lambda x: x.rolling(window).min())
        return (frame['close'] - lower) / (upper - lower).replace(0, np.nan)
    
    def _ichimoku_convolution(frame: pd.DataFrame) -> pd.Series:
        conv_line = (frame.groupby('symbol')['high'].transform(lambda x: x.rolling(9).max()) + 
                    frame.groupby('symbol')['low'].transform(lambda x: x.rolling(9).min())) / 2
        base_line = (frame.groupby('symbol')['high'].transform(lambda x: x.rolling(26).max()) +
                     frame.groupby('symbol')['low'].transform(lambda x: x.rolling(26).min())) / 2
        return conv_line - base_line
    
    def _ichimoku_base(frame: pd.DataFrame) -> pd.Series:
        return (frame.groupby('symbol')['high'].transform(lambda x: x.rolling(26).max()) +
                frame.groupby('symbol')['low'].transform(lambda x: x.rolling(26).min())) / 2
    
    def _trix(frame: pd.DataFrame, window: int = 15) -> pd.Series:
        ema1 = frame.groupby('symbol')['close'].transform(lambda x: x.ewm(span=window).mean())
        ema2 = ema1.groupby(frame['symbol']).transform(lambda x: x.ewm(span=window).mean())
        ema3 = ema2.groupby(frame['symbol']).transform(lambda x: x.ewm(span=window).mean())
        return ema3.groupby(frame['symbol']).pct_change()
    
    def _ultimate_oscillator(frame: pd.DataFrame) -> pd.Series:
        avg = (frame['high'] + frame['low']) / 2
        prev_close = frame.groupby('symbol')['close'].shift()
        tl = avg - prev_close.replace(0, np.nan)
        bl = (frame['high'] - frame['low']).replace(0, np.nan)
        bp = tl / bl.replace(0, np.nan)
        
        avg7 = bp.groupby(frame['symbol']).transform(lambda x: x.rolling(7).sum()) / 7
        avg14 = bp.groupby(frame['symbol']).transform(lambda x: x.rolling(14).sum()) / 14
        avg28 = bp.groupby(frame['symbol']).transform(lambda x: x.rolling(28).sum()) / 28
        
        return 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    
    extended_technical_factors = [
        ('momentum_5d', _momentum_5d, '5-day Price Momentum', 10),
        ('momentum_10d', _momentum_10d, '10-day Price Momentum', 15),
        ('momentum_20d', _momentum_20d, '20-day Price Momentum', 25),
        ('momentum_60d', _momentum_60d, '60-day Price Momentum', 65),
        ('momentum_120d', _momentum_120d, '120-day Price Momentum', 125),
        ('price_position_5d', _price_position_5d, 'Price Position 5d', 5),
        ('price_position_10d', _price_position_10d, 'Price Position 10d', 10),
        ('atr_14', lambda f: _atr(f, 14), 'Average True Range 14d', 15),
        ('atr_20', lambda f: _atr(f, 20), 'Average True Range 20d', 21),
        ('obv', _obv, 'On-Balance Volume', 1),
        ('obv_momentum_10', lambda f: _obv_momentum(f, 10), 'OBV Momentum 10d', 15),
        ('mfi_14', lambda f: _mfi(f, 14), 'Money Flow Index 14d', 15),
        ('aroon_oscillator', lambda f: _aroon(f, 25), 'Aroon Oscillator 25d', 25),
        ('mass_index', lambda f: _mass_index(f), 'Mass Index', 30),
        ('vortex_14', lambda f: _vortex(f, 14), 'Vortex Indicator 14d', 15),
        ('keltner_position', lambda f: _keltner_channel(f), 'Keltner Channel Position', 20),
        ('donchian_position', lambda f: _donchian_channel(f), 'Donchian Channel Position', 20),
        ('ichimoku_conv', _ichimoku_convolution, 'Ichimoku Conversion Line', 30),
        ('ichimoku_base', _ichimoku_base, 'Ichimoku Base Line', 30),
        ('trix_15', lambda f: _trix(f, 15), 'TRIX 15d', 45),
        ('ultimate_osc', _ultimate_oscillator, 'Ultimate Oscillator', 30),
        ('pct_b_20', lambda f: _pct_b_band(f, 20), '%B Band 20d', 20),
        ('pct_b_10', lambda f: _pct_b_band(f, 10), '%B Band 10d', 10),
        ('chaikin_osc', _chaikin_oscillator, 'Chaikin Oscillator', 15),
        ('elder_ray', _elder_ray, 'Elder Ray', 15),
        ('volume_weighted_price', _volume_weighted_price, 'Volume Weighted Price', 1),
        ('vwap_momentum_20', _volume_weighted_momentum, 'VWAP Momentum 20d', 25),
    ]
    
    for name, compute_fn, desc, lookback in extended_technical_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('open', 'high', 'low', 'close', 'volume'),
            lookback=lookback,
            description=f'Technical: {desc}',
            compute=compute_fn,
            category='extended_technical',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Sector/Industry Factors =====
    
    def _sector_momentum_5d(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(5).sum())
    
    def _sector_momentum_10d(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(10).sum())
    
    def _sector_relative_strength(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20).sum() / x.rolling(60).sum().replace(0, np.nan))
    
    def _sector_correlation(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol').apply(lambda g: g['pct_chg'].rolling(20).corr(g['volume'])).droplevel(0)
    
    sector_factors = [
        ('sector_mom_5d', _sector_momentum_5d, 'Sector Momentum 5d', 6),
        ('sector_mom_10d', _sector_momentum_10d, 'Sector Momentum 10d', 11),
        ('sector_rs_10d', _sector_relative_strength, 'Sector Relative Strength 10/60', 61),
        ('sector_correlation', _sector_correlation, 'Sector Correlation', 20),
        ('sector_beta', lambda f: _market_beta(f, 60), 'Sector Beta 60d', 60),
        ('sector_volatility', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20).std() * np.sqrt(252)), 'Sector Volatility', 20),
        ('sector_skewness', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20).skew()), 'Sector Skewness', 20),
        ('sector_volume_trend', lambda f: f.groupby('symbol')['volume'].transform(lambda x: x.pct_change(20)), 'Sector Volume Trend', 21),
    ]
    
    for name, compute_fn, desc, lookback in sector_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('open', 'high', 'low', 'close', 'volume', 'pct_chg'),
            lookback=lookback,
            description=f'Sector: {desc}',
            compute=compute_fn,
            category='sector',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Money Flow Factors =====
    
    def _money_flow_20d(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['amount'].transform(lambda x: x.rolling(20).mean())
    
    def _money_flow_ratio(frame: pd.DataFrame) -> pd.Series:
        inflow = frame.where(frame['pct_chg'] > 0).groupby('symbol')['amount'].transform(lambda x: x.rolling(20).sum())
        outflow = frame.where(frame['pct_chg'] < 0).groupby('symbol')['amount'].transform(lambda x: x.rolling(20).sum())
        return inflow / outflow.replace(0, np.nan)
    
    def _large_flow_ratio(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['amount'].transform(lambda x: (x > x.quantile(0.8)).rolling(20).mean())
    
    money_flow_factors = [
        ('money_flow_20d', _money_flow_20d, 'Money Flow 20d Average', 20),
        ('money_flow_ratio', _money_flow_ratio, 'Money Flow Ratio (Inflow/Outflow)', 20),
        ('large_flow_ratio', _large_flow_ratio, 'Large Order Flow Ratio', 20),
        ('net_flow_5d', lambda f: f.groupby('symbol')['amount'].transform(lambda x: x.rolling(5).mean() / x.rolling(20).mean()), 'Net Flow 5d/20d', 20),
        ('flow_momentum', lambda f: f.groupby('symbol')['amount'].transform(lambda x: x.pct_change(10)), 'Flow Momentum 10d', 15),
    ]
    
    for name, compute_fn, desc, lookback in money_flow_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('close', 'volume', 'amount', 'pct_chg'),
            lookback=lookback,
            description=f'MoneyFlow: {desc}',
            compute=compute_fn,
            category='money_flow',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Analyst/Forecast Factors =====
    
    def _forecast_count(frame: pd.DataFrame) -> pd.Series:
        cache = _load_report_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        count_dict = cache.groupby('symbol').size().to_dict()
        return frame['symbol'].map(count_dict)
    
    def _consensus_estimate(frame: pd.DataFrame) -> pd.Series:
        cache = _load_report_cache()
        if cache.empty or '2026-盈利预测-收益' not in cache.columns:
            return pd.Series(np.nan, index=frame.index)
        mean_dict = cache.groupby('symbol')['2026-盈利预测-收益'].mean().to_dict()
        return frame['symbol'].map(mean_dict)
    
    def _estimate_dispersion(frame: pd.DataFrame) -> pd.Series:
        cache = _load_report_cache()
        if cache.empty or '2026-盈利预测-市盈率' not in cache.columns:
            return pd.Series(np.nan, index=frame.index)
        disp_dict = cache.groupby('symbol')['2026-盈利预测-市盈率'].std().to_dict()
        return frame['symbol'].map(disp_dict)
    
    def _target_price_ratio(frame: pd.DataFrame) -> pd.Series:
        cache = _load_report_cache()
        if cache.empty:
            return pd.Series(np.nan, index=frame.index)
        target_dict = cache.groupby('symbol')['2026-盈利预测-收益'].mean().to_dict()
        return frame['symbol'].map(target_dict) / frame.groupby('symbol')['close'].transform('mean')
    
    analyst_factors = [
        ('forecast_count', _forecast_count, 'Number of Analyst Forecasts', 1),
        ('consensus_estimate', _consensus_estimate, 'Consensus EPS Estimate', 1),
        ('estimate_dispersion', _estimate_dispersion, 'Analyst Estimate Dispersion', 1),
        ('target_price_ratio', _target_price_ratio, 'Target Price / Current Price', 1),
        ('rating_change', lambda f: f.groupby('symbol')['volume'].transform(lambda x: (x > x.mean()).astype(int)), 'Rating Change Signal', 5),
        ('earning_surprise_proxy', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(5).std()), 'Earning Surprise Proxy', 20),
    ]
    
    for name, compute_fn, desc, lookback in analyst_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('close', 'volume'),
            lookback=lookback,
            description=f'Analyst: {desc}',
            compute=compute_fn,
            category='analyst',
            preprocessing=('winsorize', 'cross_sectional_scale'),
        ))

    # ===== Behavioral Finance Factors =====
    
    def _short_term_reversal(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: -x.rolling(5).mean())
    
    def _medium_term_reversal(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: -x.rolling(20).mean())
    
    def _long_term_reversal(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: -x.rolling(60).mean())
    
    def _earnings_quality(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.where(x > 0, 0).rolling(20).sum() / x.rolling(20).sum().replace(0, np.nan))
    
    def _turnover_rate(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(250).mean().replace(0, np.nan))
    
    def _price_depth(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['amount'].transform(lambda x: x / (x.rolling(20).mean() * frame.groupby('symbol')['close'].transform('mean').replace(0, np.nan)))
    
    def _volume_weighted_return(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol').apply(lambda g: (g['pct_chg'] * g['volume']).rolling(20).mean() / g['volume'].rolling(20).mean()).droplevel(0)
    
    def _smart_money_flow(frame: pd.DataFrame) -> pd.Series:
        close_chg = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change())
        up_vol = frame.where(close_chg > 0).groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
        down_vol = frame.where(close_chg < 0).groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
        return (up_vol - down_vol) / (up_vol + down_vol).replace(0, np.nan)
    
    def _retail_trading_indicator(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['volume'].transform(lambda x: (x < x.rolling(20).quantile(0.3)).astype(int) - (x > x.rolling(20).quantile(0.7)).astype(int))
    
    def _institutional_flow_20(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby('symbol')['amount'].transform(lambda x: x.rolling(20).mean() / x.rolling(60).mean() - 1)
    
    def _abnormal_volume_return(frame: pd.DataFrame) -> pd.Series:
        vol_ratio = frame.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(20).mean().replace(0, np.nan))
        ret = frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20).sum())
        return vol_ratio * ret
    
    def _disposition_effect(frame: pd.DataFrame) -> pd.Series:
        realized = frame.where(frame['pct_chg'] > 0).groupby('symbol')['volume'].transform(lambda x: x.rolling(20).sum())
        unrealized = frame.where(frame['pct_chg'] < 0).groupby('symbol')['volume'].transform(lambda x: x.rolling(20).sum())
        return realized / (realized + unrealized).replace(0, np.nan)
    
    def _herding_indicator(frame: pd.DataFrame) -> pd.Series:
        market_ret = frame.groupby('trade_date')['pct_chg'].transform('mean')
        return frame.groupby('symbol')['pct_chg'].transform(lambda x: x.rolling(20).corr(market_ret))
    
    def _limit_up_down_count(frame: pd.DataFrame, direction: str = 'up') -> pd.Series:
        if direction == 'up':
            return frame.groupby('symbol')['pct_chg'].transform(lambda x: (x > 9.5).rolling(20).sum())
        else:
            return frame.groupby('symbol')['pct_chg'].transform(lambda x: (x < -9.5).rolling(20).sum())
    
    def _volume_sync_with_market(frame: pd.DataFrame) -> pd.Series:
        market_vol = frame.groupby('trade_date')['volume'].transform('mean')
        return frame.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).corr(market_vol))
    
    def _momentum_60_120_divergence(frame: pd.DataFrame) -> pd.Series:
        mom_60 = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(60))
        mom_120 = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(120))
        return mom_60 - mom_120
    
    def _volume_price_divergence(frame: pd.DataFrame) -> pd.Series:
        price_ret = frame.groupby('symbol')['close'].transform(lambda x: x.pct_change(20))
        vol_ret = frame.groupby('symbol')['volume'].transform(lambda x: x.pct_change(20))
        return price_ret - vol_ret
    
    behavioral_factors = [
        ('short_term_reversal', _short_term_reversal, 'Short-term Reversal 5d', 10),
        ('medium_term_reversal', _medium_term_reversal, 'Medium-term Reversal 20d', 25),
        ('long_term_reversal', _long_term_reversal, 'Long-term Reversal 60d', 65),
        ('earnings_quality', _earnings_quality, 'Earnings Quality Score', 20),
        ('turnover_rate', _turnover_rate, 'Turnover Rate vs Average', 250),
        ('price_depth_20', _price_depth, 'Price Depth 20d', 20),
        ('vwap_return_20', _volume_weighted_return, 'VWAP-weighted Return 20d', 20),
        ('smart_money_flow', _smart_money_flow, 'Smart Money Flow Indicator', 20),
        ('retail_indicator', _retail_trading_indicator, 'Retail Trading Indicator', 20),
        ('institutional_flow_20', _institutional_flow_20, 'Institutional Flow 20d', 60),
        ('abnormal_vol_return', _abnormal_volume_return, 'Abnormal Volume-Return Interaction', 20),
        ('disposition_effect', _disposition_effect, 'Disposition Effect', 20),
        ('herding_indicator', _herding_indicator, 'Herding with Market', 20),
        ('limit_up_count_20', lambda f: _limit_up_down_count(f, 'up'), 'Limit Up Days 20d', 20),
        ('limit_down_count_20', lambda f: _limit_up_down_count(f, 'down'), 'Limit Down Days 20d', 20),
        ('volume_sync_market', _volume_sync_with_market, 'Volume Sync with Market', 20),
        ('momentum_divergence', _momentum_60_120_divergence, 'Momentum 60-120 Divergence', 120),
        ('volume_price_div', _volume_price_divergence, 'Volume-Price Divergence', 20),
        ('gain_loss_asymmetry', lambda f: f.groupby('symbol')['pct_chg'].transform(lambda x: x.where(x > 0, 0).rolling(20).mean() - x.where(x < 0, 0).rolling(20).mean()), 'Gain Loss Asymmetry 20d', 20),
        ('up_down_volume_ratio', lambda f: f.groupby('symbol').apply(lambda g: (g['volume'].where(g['pct_chg'] > 0).sum() / g['volume'].where(g['pct_chg'] < 0).sum()).where(lambda x: True, np.nan) if g['pct_chg'].abs().sum() > 0 else np.nan).droplevel(0), 'Up/Down Volume Ratio', 20),
    ]
    
    for name, compute_fn, desc, lookback in behavioral_factors:
        registry.register(FeatureSpec(
            name=name,
            inputs=('close', 'volume', 'amount', 'pct_chg'),
            lookback=lookback,
            description=f'Behavioral: {desc}',
            compute=compute_fn,
            category='behavioral',
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
    
    liquidity_factors = [
        ('amihud_illiq_5d', _amihud_illiq_5d, 'Amihud Illiquidity 5d', 5),
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

    # Register factor-pool aliases after all canonical factors are available.
    # These aliases make the descriptive 667-factor pool computable without
    # duplicating formulas. Alias/proxy factors still require normal IC review.
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
