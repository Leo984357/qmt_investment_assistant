"""WorldQuant-style alpha factor functions used in the simplified factor registry."""
from __future__ import annotations

import numpy as np
import pandas as pd


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
