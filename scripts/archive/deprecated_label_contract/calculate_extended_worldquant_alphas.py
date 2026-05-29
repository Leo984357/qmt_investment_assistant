"""
快速计算剩余WorldQuant Alpha因子 IC - 批量优化版
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from scipy.stats import spearmanr

DATA_DIR = Path('data/raw')
CACHE_DIR = Path('data/factor_cache')


def load_data():
    """加载并预处理数据"""
    print("Loading data...")
    bars = pd.read_parquet(DATA_DIR / 'daily_bar.parquet')
    bars['trade_date'] = pd.to_datetime(bars['trade_date'])
    bars = bars.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    bars['returns'] = bars.groupby('symbol')['adj_close'].pct_change()
    bars['vwap'] = bars['amount'] / bars['volume'].replace(0, np.nan)
    bars['adv20'] = bars.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    bars['fwd_return_20d'] = bars.groupby('symbol')['adj_close'].pct_change(20).shift(-20)
    print(f"Data loaded: {len(bars):,} rows")
    return bars


def calculate_ic_fast(factor_values, label_values):
    """快速计算IC"""
    mask = ~(np.isnan(factor_values) | np.isnan(label_values))
    if mask.sum() < 30:
        return np.nan
    ic, _ = spearmanr(factor_values[mask], label_values[mask])
    return ic


def main():
    print("=" * 60)
    print("WorldQuant Alpha快速IC测试")
    print("=" * 60)
    
    bars = load_data()
    
    n = len(bars)
    returns = bars['returns'].values
    volume = bars['volume'].values
    adv20 = bars['adv20'].values
    close = bars['adj_close'].values
    open_price = bars['open'].values
    high = bars['high'].values
    low = bars['low'].values
    vwap = bars['vwap'].values
    fwd_ret = bars['fwd_return_20d'].values
    symbol_idx = bars.groupby('symbol').cumcount().values
    max_idx = bars.groupby('symbol').cumcount().values
    
    results = []
    
    # Define alpha calculations (simplified versions)
    alphas = [
        ('alpha_031', lambda i: _alpha_031(volume, high, low, returns, symbol_idx, max_idx, i)),
        ('alpha_032', lambda i: _alpha_032(returns, symbol_idx, max_idx, i)),
        ('alpha_033', lambda i: _alpha_033(high, low, symbol_idx, max_idx, i)),
        ('alpha_034', lambda i: _alpha_034(high, volume, symbol_idx, max_idx, i)),
        ('alpha_035', lambda i: _alpha_035(close, open_price, low, symbol_idx, max_idx, i)),
        ('alpha_036', lambda i: _alpha_036(close, open_price, symbol_idx, max_idx, i)),
        ('alpha_037', lambda i: _alpha_037(volume, adv20, close, open_price, symbol_idx, max_idx, i)),
        ('alpha_038', lambda i: _alpha_038(close, volume, adv20, high, symbol_idx, max_idx, i)),
        ('alpha_039', lambda i: _alpha_039(close, volume, symbol_idx, max_idx, i)),
        ('alpha_040', lambda i: _alpha_040(high, close, adv20, symbol_idx, max_idx, i)),
        ('alpha_041', lambda i: _alpha_041(close, vwap, symbol_idx, max_idx, i)),
        ('alpha_042', lambda i: _alpha_042(returns, close, open_price, symbol_idx, max_idx, i)),
        ('alpha_043', lambda i: _alpha_043(close, symbol_idx, max_idx, i)),
        ('alpha_044', lambda i: _alpha_044(close, volume, symbol_idx, max_idx, i)),
        ('alpha_045', lambda i: _alpha_045(volume, adv20, close, low, high, symbol_idx, max_idx, i)),
        ('alpha_046', lambda i: _alpha_046(volume, adv20, close, symbol_idx, max_idx, i)),
        ('alpha_047', lambda i: _alpha_047(volume, adv20, close, symbol_idx, max_idx, i)),
        ('alpha_048', lambda i: _alpha_048(volume, adv20, close, symbol_idx, max_idx, i)),
        ('alpha_049', lambda i: _alpha_049(volume, close, symbol_idx, max_idx, i)),
        ('alpha_050', lambda i: _alpha_050(high, close, volume, symbol_idx, max_idx, i)),
        ('alpha_055', lambda i: _alpha_055(open_price, close, volume, symbol_idx, max_idx, i)),
        ('alpha_056', lambda i: _alpha_056(vwap, close, symbol_idx, max_idx, i)),
        ('alpha_057', lambda i: _alpha_057(vwap, adv20, symbol_idx, max_idx, i)),
        ('alpha_058', lambda i: _alpha_058(high, vwap, symbol_idx, max_idx, i)),
        ('alpha_059', lambda i: _alpha_059(high, close, volume, symbol_idx, max_idx, i)),
        ('alpha_060', lambda i: _alpha_060(vwap, adv20, close, symbol_idx, max_idx, i)),
        ('alpha_061', lambda i: _alpha_061(high, close, adv20, symbol_idx, max_idx, i)),
        ('alpha_062', lambda i: _alpha_062(open_price, returns, volume, symbol_idx, max_idx, i)),
        ('alpha_063', lambda i: _alpha_063(open_price, volume, adv20, close, symbol_idx, max_idx, i)),
        ('alpha_064', lambda i: _alpha_064(open_price, high, low, volume, symbol_idx, max_idx, i)),
        ('alpha_065', lambda i: _alpha_065(open_price, high, adv20, symbol_idx, max_idx, i)),
        ('alpha_066', lambda i: _alpha_066(open_price, volume, symbol_idx, max_idx, i)),
        ('alpha_067', lambda i: _alpha_067(high, low, returns, symbol_idx, max_idx, i)),
        ('alpha_068', lambda i: _alpha_068(close, adv20, symbol_idx, max_idx, i)),
        ('alpha_069', lambda i: _alpha_069(vwap, volume, close, symbol_idx, max_idx, i)),
        ('alpha_070', lambda i: _alpha_070(vwap, volume, symbol_idx, max_idx, i)),
        ('alpha_071', lambda i: _alpha_071(open_price, high, low, close, volume, symbol_idx, max_idx, i)),
        ('alpha_072', lambda i: _alpha_072(volume, adv20, symbol_idx, max_idx, i)),
        ('alpha_073', lambda i: _alpha_073(close, volume, symbol_idx, max_idx, i)),
        ('alpha_074', lambda i: _alpha_074(high, adv20, close, symbol_idx, max_idx, i)),
        ('alpha_075', lambda i: _alpha_075(adv20, low, vwap, close, symbol_idx, max_idx, i)),
        ('alpha_076', lambda i: _alpha_076(high, close, volume, symbol_idx, max_idx, i)),
        ('alpha_077', lambda i: _alpha_077(high, volume, symbol_idx, max_idx, i)),
        ('alpha_078', lambda i: _alpha_078(close, open_price, symbol_idx, max_idx, i)),
        ('alpha_079', lambda i: _alpha_079(open_price, volume, adv20, vwap, symbol_idx, max_idx, i)),
        ('alpha_080', lambda i: _alpha_080(adv20, volume, close, symbol_idx, max_idx, i)),
        ('alpha_081', lambda i: _alpha_081(vwap, close, volume, symbol_idx, max_idx, i)),
        ('alpha_082', lambda i: _alpha_082(vwap, symbol_idx, max_idx, i)),
        ('alpha_083', lambda i: _alpha_083(close, volume, vwap, symbol_idx, max_idx, i)),
        ('alpha_084', lambda i: _alpha_084(adv20, open_price, vwap, symbol_idx, max_idx, i)),
        ('alpha_085', lambda i: _alpha_085(close, open_price, volume, adv20, symbol_idx, max_idx, i)),
        ('alpha_086', lambda i: _alpha_086(high, close, adv20, vwap, symbol_idx, max_idx, i)),
        ('alpha_087', lambda i: _alpha_087(open_price, close, symbol_idx, max_idx, i)),
        ('alpha_088', lambda i: _alpha_088(high, close, adv20, symbol_idx, max_idx, i)),
        ('alpha_089', lambda i: _alpha_089(volume, adv20, close, symbol_idx, max_idx, i)),
        ('alpha_090', lambda i: _alpha_090(vwap, volume, symbol_idx, max_idx, i)),
        ('alpha_091', lambda i: _alpha_091(high, close, volume, symbol_idx, max_idx, i)),
        ('alpha_092', lambda i: _alpha_092(high, low, vwap, close, symbol_idx, max_idx, i)),
        ('alpha_093', lambda i: _alpha_093(close, adv20, symbol_idx, max_idx, i)),
        ('alpha_094', lambda i: _alpha_094(high, vwap, close, volume, symbol_idx, max_idx, i)),
        ('alpha_095', lambda i: _alpha_095(vwap, adv20, volume, symbol_idx, max_idx, i)),
        ('alpha_096', lambda i: _alpha_096(vwap, close, adv20, symbol_idx, max_idx, i)),
        ('alpha_097', lambda i: _alpha_097(vwap, close, symbol_idx, max_idx, i)),
        ('alpha_098', lambda i: _alpha_098(vwap, volume, symbol_idx, max_idx, i)),
        ('alpha_099', lambda i: _alpha_099(close, volume, adv20, symbol_idx, max_idx, i)),
        ('alpha_100', lambda i: _alpha_100(vwap, close, volume, symbol_idx, max_idx, i)),
        ('alpha_101', lambda i: _alpha_101(vwap, volume, symbol_idx, max_idx, i)),
    ]
    
    # Create symbol groups for rolling operations
    symbols = bars['symbol'].values
    unique_symbols, inv_idx = np.unique(symbols, return_inverse=True)
    n_symbols = len(unique_symbols)
    
    for name, calc_func in alphas:
        print(f"Calculating {name}...", end=" ", flush=True)
        start = time.time()
        try:
            factor_values = calc_func(inv_idx)
            ic = calculate_ic_fast(factor_values, fwd_ret)
            elapsed = time.time() - start
            print(f"IC = {ic:.4f} ({elapsed:.1f}s)" if not np.isnan(ic) else "nan")
            results.append({'factor': name, 'source': 'worldquant', 'rank_ic': ic})
        except Exception as e:
            print(f"ERROR - {e}")
            results.append({'factor': name, 'source': 'worldquant', 'rank_ic': np.nan})
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rank_ic', ascending=False)
    output_file = CACHE_DIR / 'worldquant_alpha_extended_ic.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    print("\n" + "=" * 60)
    print("Top 10 Alphas:")
    print(results_df.head(10).to_string(index=False))
    print("\nBottom 10 Alphas:")
    print(results_df.tail(10).to_string(index=False))
    
    return results_df


# ============ NumPy-based alpha calculations ============

def rolling_rank(values, window, idx, max_idx):
    """Efficient rolling rank using numpy"""
    result = np.full_like(values, np.nan, dtype=float)
    for i in range(len(values)):
        if max_idx[i] >= window - 1:
            start = i - max_idx[i]
            end = i + 1
            window_values = values[start:end]
            valid = ~np.isnan(window_values)
            if valid.sum() > 0:
                result[i] = (window_values[valid] < values[i]).sum() / valid.sum()
    return result


def rolling_sum(values, window, max_idx):
    """Efficient rolling sum"""
    result = np.full_like(values, np.nan, dtype=float)
    for i in range(len(values)):
        if max_idx[i] >= window - 1:
            start = i - max_idx[i]
            end = i + 1
            result[i] = np.nansum(values[start:end])
    return result


def rolling_mean(values, window, max_idx):
    """Efficient rolling mean"""
    result = np.full_like(values, np.nan, dtype=float)
    for i in range(len(values)):
        if max_idx[i] >= window - 1:
            start = i - max_idx[i]
            end = i + 1
            result[i] = np.nanmean(values[start:end])
    return result


def rolling_max(values, window, max_idx):
    """Efficient rolling max"""
    result = np.full_like(values, np.nan, dtype=float)
    for i in range(len(values)):
        if max_idx[i] >= window - 1:
            start = i - max_idx[i]
            end = i + 1
            result[i] = np.nanmax(values[start:end])
    return result


def rolling_min(values, window, max_idx):
    """Efficient rolling min"""
    result = np.full_like(values, np.nan, dtype=float)
    for i in range(len(values)):
        if max_idx[i] >= window - 1:
            start = i - max_idx[i]
            end = i + 1
            result[i] = np.nanmin(values[start:end])
    return result


def delay(values, period, max_idx):
    """Delay/shift"""
    result = np.full_like(values, np.nan, dtype=float)
    for i in range(len(values)):
        if max_idx[i] >= period:
            result[i] = values[i - period]
    return result


def delta(values, period, max_idx):
    """Delta/diff"""
    result = np.full_like(values, np.nan, dtype=float)
    for i in range(len(values)):
        if max_idx[i] >= period:
            result[i] = values[i] - values[i - period]
    return result


# Simplified alpha implementations
def _alpha_031(volume, high, low, returns, inv_idx, max_idx, n):
    vol_rank = rolling_rank(volume, 32, inv_idx, max_idx)
    price_range = high + low
    price_rank = rolling_rank(price_range, 16, inv_idx, max_idx)
    ret_rank = rolling_rank(returns, 32, inv_idx, max_idx)
    return vol_rank * (1 - price_rank) * (1 - ret_rank)


def _alpha_032(returns, inv_idx, max_idx, n):
    sum_ret_10 = rolling_sum(returns, 10, max_idx)
    sum_ret_2 = rolling_sum(returns, 2, max_idx)
    sum_sum_ret = rolling_sum(sum_ret_2, 3, max_idx)
    ratio = sum_ret_10 / (sum_sum_ret + 1e-10)
    return ratio


def _alpha_033(high, low, inv_idx, max_idx, n):
    high_std = rolling_mean(high, 10, max_idx)
    return -high_std * low


def _alpha_034(high, volume, inv_idx, max_idx, n):
    high_rank = high / (high + 1e-10)
    vol_rank = volume / (volume + 1e-10)
    combined = high_rank * vol_rank
    return -rolling_rank(combined, 3, inv_idx, max_idx)


def _alpha_035(close, open_price, low, inv_idx, max_idx, n):
    delay_close = delay(close, 20, max_idx)
    delay_open = delay(open_price, 20, max_idx)
    price_diff = delay_close - delay_open
    low_rank = low / (low + 1e-10)
    return -price_diff * low_rank


def _alpha_036(close, open_price, inv_idx, max_idx, n):
    up_days = (close > open_price).astype(float)
    sum_up = rolling_sum(up_days, 10, max_idx) / 10
    return sum_up


def _alpha_037(volume, adv20, close, open_price, inv_idx, max_idx, n):
    vol_ratio = volume / (adv20 + 1e-10)
    close_open = (close - open_price) / (open_price + 1e-10)
    return -vol_ratio * close_open


def _alpha_038(close, volume, adv20, high, inv_idx, max_idx, n):
    inv_close = 1 / (close + 1e-10)
    vol_ratio = (inv_close * volume) / (adv20 + 1e-10)
    sum_high_5 = rolling_mean(high, 5, max_idx)
    high_diff = high - close
    return vol_ratio * (high_diff / (sum_high_5 + 1e-10))


def _alpha_039(close, volume, inv_idx, max_idx, n):
    delay_5 = delay(close, 5, max_idx)
    sum_delay = rolling_mean(delay_5, 20, max_idx)
    close_rank = close / (close + sum_delay + 1e-10)
    return -close_rank


def _alpha_040(high, close, adv20, inv_idx, max_idx, n):
    weighted = high * 0.9 + close * 0.1
    vol_ratio = weighted / (adv20 + 1e-10)
    return -vol_ratio


def _alpha_041(close, vwap, inv_idx, max_idx, n):
    ma_close = rolling_mean(close, 7, max_idx)
    diff = close - ma_close
    vwap_diff = close - vwap
    return diff + vwap_diff


def _alpha_042(returns, close, open_price, inv_idx, max_idx, n):
    sum_ret_10 = rolling_sum(returns, 10, max_idx)
    product = open_price * close
    return -sum_ret_10 * product


def _alpha_043(close, inv_idx, max_idx, n):
    delta_1 = delta(close, 1, max_idx)
    ts_min_delta = rolling_min(delta_1, 3, max_idx)
    ts_max_delta = rolling_max(delta_1, 3, max_idx)
    return np.where(ts_min_delta > 0, delta_1, np.where(ts_max_delta < 0, delta_1, -delta_1))


def _alpha_044(close, volume, inv_idx, max_idx, n):
    delta_1 = delta(close, 1, max_idx)
    ts_min = rolling_min(delta_1, 4, max_idx)
    ts_max = rolling_max(delta_1, 4, max_idx)
    vol_rank = rolling_rank(volume, 32, inv_idx, max_idx)
    return (np.sign(ts_min) - np.sign(ts_max)) * vol_rank


def _alpha_045(volume, adv20, close, low, high, inv_idx, max_idx, n):
    vol_ratio = volume / (adv20 + 1e-10)
    price_pos = (close - low) / (close - high + 1e-10)
    return vol_ratio * price_pos


def _alpha_046(volume, adv20, close, inv_idx, max_idx, n):
    vol_ratio = volume / (adv20 + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 20, inv_idx, max_idx)
    delta_close = delta(close, 3, max_idx)
    delta_rank = rolling_rank(-delta_close, 16, inv_idx, max_idx)
    return vol_rank * delta_rank


def _alpha_047(volume, adv20, close, inv_idx, max_idx, n):
    vol_rank = rolling_rank(volume, 32, inv_idx, max_idx)
    delta_close = delta(close, 3, max_idx)
    delta_ratio = -delta_close / (adv20 + 1e-10)
    delta_rank = rolling_rank(delta_ratio, 5, inv_idx, max_idx)
    return vol_rank * delta_rank * -1


def _alpha_048(volume, adv20, close, inv_idx, max_idx, n):
    vol_ratio = volume / (adv20 + 1e-10)
    delta_vol = delta(vol_ratio, 4, max_idx)
    sign_vol = np.sign(delta_vol)
    delta_close = delta(close, 4, max_idx)
    delta_rank = rolling_rank(delta_close, 12, inv_idx, max_idx)
    return sign_vol * -delta_rank


def _alpha_049(volume, close, inv_idx, max_idx, n):
    vol_rank = rolling_rank(volume, 12, inv_idx, max_idx)
    delta_close = delta(close, 7, max_idx)
    delta_rank = rolling_rank(-delta_close, 12, inv_idx, max_idx)
    return np.maximum(vol_rank, delta_rank) * -1


def _alpha_050(high, close, volume, inv_idx, max_idx, n):
    vol_rank = rolling_rank(volume, 5, inv_idx, max_idx)
    high_close = high - close
    avg_high = rolling_mean(high, 5, max_idx)
    ratio = high_close / (avg_high + 1e-10)
    ratio_rank = rolling_rank(ratio, 3, inv_idx, max_idx)
    return vol_rank * ratio_rank


def _alpha_055(open_price, close, volume, inv_idx, max_idx, n):
    weighted = open_price * 0.9 + close * 0.1
    weighted_rank = rolling_rank(weighted, 10, inv_idx, max_idx)
    delta_close = delta(close, 2, max_idx)
    delta_rank = rolling_rank(delta_close, 12, inv_idx, max_idx)
    return -weighted_rank * delta_rank


def _alpha_056(vwap, close, inv_idx, max_idx, n):
    vwap_diff = vwap - close
    vwap_sum = vwap + close
    diff_rank = rolling_rank(vwap_diff, 20, inv_idx, max_idx)
    sum_rank = rolling_rank(vwap_sum, 4, inv_idx, max_idx)
    return diff_rank / (sum_rank + 1e-10)


def _alpha_057(vwap, adv20, inv_idx, max_idx, n):
    ts_min_vwap = rolling_min(vwap, 16, max_idx)
    diff = vwap - ts_min_vwap
    diff_rank = rolling_rank(diff, 17, inv_idx, max_idx)
    vol_ratio = vwap / (adv20 + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 16, inv_idx, max_idx)
    return -diff_rank * vol_rank


def _alpha_058(high, vwap, inv_idx, max_idx, n):
    high_rank = high / (high + 1e-10)
    vwap_rank = vwap / (vwap + 1e-10)
    combined = high_rank * vwap_rank
    combined_rank = rolling_rank(combined, 15, inv_idx, max_idx)
    return -combined_rank


def _alpha_059(high, close, volume, inv_idx, max_idx, n):
    weighted = high * 0.9 + close * 0.1
    delta_weighted = delta(weighted, 3, max_idx)
    delta_rank = rolling_rank(delta_weighted, 14, inv_idx, max_idx)
    vol_rank = rolling_rank(volume, 19, inv_idx, max_idx)
    return delta_rank * vol_rank


def _alpha_060(vwap, adv20, close, inv_idx, max_idx, n):
    vol_ratio = vwap / (adv20 + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 4, inv_idx, max_idx)
    vwap_close = vwap / (close + 1e-10)
    delta_ratio = delta(vwap_close, 3, max_idx)
    delta_rank = rolling_rank(delta_ratio, 14, inv_idx, max_idx)
    return np.maximum(vol_rank, delta_rank) * -1


def _alpha_061(high, close, adv20, inv_idx, max_idx, n):
    weighted = high * 0.8 + close * 0.2
    vol_ratio = weighted / (adv20 + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 4, inv_idx, max_idx)
    delta_close = delta(close, 1, max_idx)
    delta_delta = delta(delta_close, 1, max_idx)
    delta_rank = rolling_rank(delta_delta, 4, inv_idx, max_idx)
    return vol_rank * delta_rank


def _alpha_062(open_price, returns, volume, inv_idx, max_idx, n):
    sum_open = rolling_mean(open_price, 5, max_idx)
    sum_ret = rolling_mean(returns, 5, max_idx)
    product = sum_open * sum_ret
    delay_product = delay(product, 10, max_idx)
    diff = product - delay_product
    diff_rank = rolling_rank(diff, 3, inv_idx, max_idx)
    vol_rank = rolling_rank(volume, 10, inv_idx, max_idx)
    return -vol_rank * diff_rank


def _alpha_063(open_price, volume, adv20, close, inv_idx, max_idx, n):
    delta_open = delta(open_price, 4, max_idx) * 0.9
    vol_rank = rolling_rank(volume, 9, inv_idx, max_idx)
    term1 = rolling_rank(delta_open * vol_rank, 13, inv_idx, max_idx)
    delta_close = delta(close, 2, max_idx)
    ts_min_delta = rolling_min(delta_close, 3, max_idx)
    term2 = rolling_rank(ts_min_delta, 1, inv_idx, max_idx)
    return np.minimum(term1, term2) * -1


def _alpha_064(open_price, high, low, volume, inv_idx, max_idx, n):
    sum_open = rolling_mean(open_price * 0.9, 4, max_idx)
    mid = (high + low) / 2
    mid_rank = rolling_rank(mid, 14, inv_idx, max_idx)
    vol_rank = rolling_rank(volume, 5, inv_idx, max_idx)
    return rolling_rank(sum_open + mid_rank - vol_rank, 4, inv_idx, max_idx)


def _alpha_065(open_price, high, adv20, inv_idx, max_idx, n):
    open_adv = open_price / (adv20 + 1e-10)
    high_adv = high / (adv20 + 1e-10)
    open_rank = rolling_rank(open_adv, 10, inv_idx, max_idx)
    high_rank = rolling_rank(high_adv, 7, inv_idx, max_idx)
    high_rank2 = rolling_rank(high_rank, 9, inv_idx, max_idx)
    combined = rolling_rank(open_rank + high_rank2, 17, inv_idx, max_idx)
    return -combined


def _alpha_066(open_price, volume, inv_idx, max_idx, n):
    vol_ratio = volume / (open_price + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 14, inv_idx, max_idx)
    return -vol_rank


def _alpha_067(high, low, returns, inv_idx, max_idx, n):
    mid = (high + low) / 2
    delta_mid = delta(mid, 4, max_idx)
    delta_rank = rolling_rank(delta_mid, 12, inv_idx, max_idx)
    delta_ret = delta(returns, 6, max_idx)
    ret_rank = rolling_rank(-delta_ret, 9, inv_idx, max_idx)
    return delta_rank * ret_rank


def _alpha_068(close, adv20, inv_idx, max_idx, n):
    sum_adv = rolling_mean(adv20, 25, max_idx)
    ratio = close / (sum_adv + 1e-10)
    ratio_rank = rolling_rank(ratio, 7, inv_idx, max_idx)
    delta_close = delta(close, 3, max_idx)
    delta_rank = rolling_rank(delta_close, 5, inv_idx, max_idx)
    return ratio_rank * delta_rank * -1


def _alpha_069(vwap, volume, close, inv_idx, max_idx, n):
    delta_vwap = delta(vwap, 4, max_idx)
    vol_rank = rolling_rank(volume, 11, inv_idx, max_idx)
    term1 = rolling_rank(delta_vwap * vol_rank, 6, inv_idx, max_idx)
    delay_close = delay(close, 3, max_idx)
    close_change = (close - delay_close) / (delay_close + 1e-10)
    term2 = rolling_rank(-close_change, 2, inv_idx, max_idx)
    return np.maximum(term1, term2) * -1


def _alpha_070(vwap, volume, inv_idx, max_idx, n):
    vol_rank = rolling_rank(volume, 5, inv_idx, max_idx)
    delta_vwap = delta(vwap, 3, max_idx)
    delta_rank = rolling_rank(delta_vwap, 5, inv_idx, max_idx)
    return delta_rank - vol_rank


def _alpha_071(open_price, high, low, close, volume, inv_idx, max_idx, n):
    open_rank = open_price / (open_price + rolling_mean(open_price, 10, max_idx) + 1e-10)
    high_rank = high / (high + rolling_mean(high, 10, max_idx) + 1e-10)
    low_rank = low / (low + rolling_mean(low, 10, max_idx) + 1e-10)
    close_rank = close / (close + rolling_mean(close, 10, max_idx) + 1e-10)
    combined = (open_rank + low_rank) - (high_rank + close_rank)
    vol_rank = rolling_rank(volume, 9, inv_idx, max_idx)
    return combined * vol_rank


def _alpha_072(volume, adv20, inv_idx, max_idx, n):
    vol_ratio = volume / (adv20 + 1e-10)
    return rolling_rank(vol_ratio, 12, inv_idx, max_idx)


def _alpha_073(close, volume, inv_idx, max_idx, n):
    vol_rank = rolling_rank(volume, 8, inv_idx, max_idx)
    delta_close = delta(close, 3, max_idx)
    delta_rank = rolling_rank(delta_close, 3, inv_idx, max_idx)
    return delta_rank * vol_rank


def _alpha_074(high, adv20, close, inv_idx, max_idx, n):
    high_rank = high / (high + 1e-10)
    adv_rank = adv20 / (adv20 + rolling_mean(adv20, 8, max_idx) + 1e-10)
    combined = high_rank * adv_rank
    combined_rank = rolling_rank(combined, 19, inv_idx, max_idx)
    delta_close = delta(close, 1, max_idx)
    delta_rank = rolling_rank(delta_close, 1, inv_idx, max_idx)
    return combined_rank * delta_rank


def _alpha_075(adv20, low, vwap, close, inv_idx, max_idx, n):
    vol_ratio = adv20 / (low + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 4, inv_idx, max_idx)
    close_vwap = close - vwap
    return vol_rank * close_vwap


def _alpha_076(high, close, volume, inv_idx, max_idx, n):
    weighted = high * 0.9 + close * 0.1
    delta_weighted = delta(weighted, 2, max_idx)
    ratio = delta_weighted / (close + 1e-10)
    ratio_rank = rolling_rank(ratio, 12, inv_idx, max_idx)
    vol_rank = rolling_rank(volume, 10, inv_idx, max_idx)
    return ratio_rank * rolling_rank(-vol_rank, 2, inv_idx, max_idx)


def _alpha_077(high, volume, inv_idx, max_idx, n):
    high_rank = high / (high + 1e-10)
    vol_rank = volume / (volume + rolling_mean(volume, 5, max_idx) + 1e-10)
    combined = high_rank * vol_rank
    combined_rank = rolling_rank(combined, 4, inv_idx, max_idx)
    return -combined_rank


def _alpha_078(close, open_price, inv_idx, max_idx, n):
    delta_close = delta(close, 3, max_idx)
    delta_rank = rolling_rank(-delta_close, 13, inv_idx, max_idx)
    ratio = open_price / (close + 1e-10)
    ratio_rank = ratio / (ratio + rolling_mean(ratio, 10, max_idx) + 1e-10)
    return delta_rank * ratio_rank


def _alpha_079(open_price, volume, adv20, vwap, inv_idx, max_idx, n):
    delta_open = delta(open_price, 2, max_idx) * 0.9
    vol_rank = rolling_rank(volume, 5, inv_idx, max_idx)
    term1 = rolling_rank(delta_open * vol_rank, 3, inv_idx, max_idx)
    sum_vwap = rolling_mean(vwap, 25, max_idx)
    open_diff = open_price - sum_vwap
    term2 = rolling_rank(-open_diff, 7, inv_idx, max_idx)
    return np.minimum(term1, term2) * -1


def _alpha_080(adv20, volume, close, inv_idx, max_idx, n):
    vol_ratio = adv20 / (volume + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 15, inv_idx, max_idx)
    delta_close = delta(close, 2, max_idx)
    delta_rank = rolling_rank(delta_close, 16, inv_idx, max_idx)
    return vol_rank * delta_rank * -1


def _alpha_081(vwap, close, volume, inv_idx, max_idx, n):
    diff = vwap - close
    diff_rank = rolling_rank(diff, 20, inv_idx, max_idx)
    vol_rank = rolling_rank(volume, 5, inv_idx, max_idx)
    return diff_rank * vol_rank * -1


def _alpha_082(vwap, inv_idx, max_idx, n):
    ts_min_vwap = rolling_min(vwap, 13, max_idx)
    diff = vwap - ts_min_vwap
    diff_rank = rolling_rank(diff, 13, inv_idx, max_idx)
    delta_vwap = delta(vwap, 3, max_idx)
    delta_rank = rolling_rank(-delta_vwap, 10, inv_idx, max_idx)
    return diff_rank * delta_rank


def _alpha_083(close, volume, vwap, inv_idx, max_idx, n):
    vol_ratio = volume / (close + 1e-10)
    delta_vol = delta(vol_ratio, 2, max_idx)
    delta_rank = rolling_rank(delta_vol, 3, inv_idx, max_idx)
    vwap_rank = rolling_rank(vwap, 5, inv_idx, max_idx)
    return delta_rank * vwap_rank


def _alpha_084(adv20, open_price, vwap, inv_idx, max_idx, n):
    vol_ratio = adv20 / (open_price + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 4, inv_idx, max_idx)
    delta_vwap = delta(vwap, 4, max_idx)
    delta_rank = rolling_rank(-delta_vwap, 4, inv_idx, max_idx)
    return np.maximum(vol_rank, delta_rank) * -1


def _alpha_085(close, open_price, volume, adv20, inv_idx, max_idx, n):
    up_days = (close > open_price).astype(float)
    sum_up = rolling_mean(up_days, 4, max_idx)
    rank1 = rolling_rank(sum_up, 12, inv_idx, max_idx)
    vol_ratio = volume / (adv20 + 1e-10)
    sum_vol = rolling_mean(vol_ratio, 4, max_idx)
    rank2 = rolling_rank(sum_vol, 12, inv_idx, max_idx)
    delta_close = delta(close, 3, max_idx)
    rank3 = rolling_rank(delta_close, 4, inv_idx, max_idx)
    return rank1 * rank2 * rank3 * -1


def _alpha_086(high, close, adv20, vwap, inv_idx, max_idx, n):
    weighted = high * 0.9 + close * 0.1
    vol_ratio = weighted / (adv20 + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 7, inv_idx, max_idx)
    delta_vwap = delta(vwap, 3, max_idx)
    delta_rank = rolling_rank(delta_vwap, 7, inv_idx, max_idx)
    return vol_rank - delta_rank


def _alpha_087(open_price, close, inv_idx, max_idx, n):
    ts_min_open = rolling_min(open_price, 12, max_idx)
    diff = open_price - ts_min_open
    diff_rank = rolling_rank(diff, 12, inv_idx, max_idx)
    delta_close = delta(close, 12, max_idx)
    delta_rank = rolling_rank(-delta_close, 10, inv_idx, max_idx)
    return diff_rank * delta_rank


def _alpha_088(high, close, adv20, inv_idx, max_idx, n):
    weighted = high * 0.8 + close * 0.2
    delta_weighted = delta(weighted, 8, max_idx)
    vol_ratio = delta_weighted / (adv20 + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 5, inv_idx, max_idx)
    delta_close = delta(close, 3, max_idx)
    ts_min = rolling_min(delta_close, 9, max_idx)
    term2 = rolling_rank(ts_min, 4, inv_idx, max_idx)
    return np.maximum(vol_rank, term2)


def _alpha_089(volume, adv20, close, inv_idx, max_idx, n):
    vol_ratio = volume / (adv20 + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 5, inv_idx, max_idx)
    delta_close = delta(close, 3, max_idx)
    delta_rank = rolling_rank(-delta_close, 10, inv_idx, max_idx)
    return vol_rank * delta_rank


def _alpha_090(vwap, volume, inv_idx, max_idx, n):
    sum_vol = rolling_sum(volume, 8, max_idx)
    vol_ratio = vwap / (sum_vol / 8 + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 4, inv_idx, max_idx)
    delta_vwap = delta(vwap, 2, max_idx)
    delta_rank = rolling_rank(delta_vwap, 4, inv_idx, max_idx)
    return vol_rank - delta_rank


def _alpha_091(high, close, volume, inv_idx, max_idx, n):
    vol_ratio = high / (volume + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 5, inv_idx, max_idx)
    high_close = high / (close + 1e-10)
    ratio_rank = high_close / (high_close + rolling_mean(high_close, 5, max_idx) + 1e-10)
    return vol_rank * ratio_rank


def _alpha_092(high, low, vwap, close, inv_idx, max_idx, n):
    mid = (high + low) / 2
    combined = mid - vwap
    combined_rank = rolling_rank(combined, 11, inv_idx, max_idx)
    delta_close = delta(close, 3, max_idx)
    delta_rank = rolling_rank(-delta_close, 10, inv_idx, max_idx)
    return combined_rank * delta_rank


def _alpha_093(close, adv20, inv_idx, max_idx, n):
    ratio = close / (adv20 + 1e-10)
    ratio_rank = rolling_rank(ratio, 10, inv_idx, max_idx)
    delta_neg = -delta(close, 2, max_idx)
    delta_neg_rank = rolling_rank(delta_neg, 5, inv_idx, max_idx)
    return ratio_rank * delta_neg_rank


def _alpha_094(high, vwap, close, volume, inv_idx, max_idx, n):
    weighted = high * 0.3 + vwap * 0.7
    vol_ratio = weighted / (volume + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 4, inv_idx, max_idx)
    close_vwap = close - vwap
    delta_diff = delta(close_vwap, 3, max_idx)
    delta_rank = rolling_rank(delta_diff, 5, inv_idx, max_idx)
    return vol_rank * delta_rank


def _alpha_095(vwap, adv20, volume, inv_idx, max_idx, n):
    delta_vwap = delta(vwap, 5, max_idx)
    delta_rank = rolling_rank(delta_vwap, 10, inv_idx, max_idx)
    vol_ratio = adv20 / (volume + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 5, inv_idx, max_idx)
    return delta_rank * vol_rank


def _alpha_096(vwap, close, adv20, inv_idx, max_idx, n):
    ratio = vwap / (close + 1e-10)
    ratio_rank = rolling_rank(ratio, 19, inv_idx, max_idx)
    diff = vwap - adv20
    diff_rank = diff / (diff + rolling_mean(diff, 10, max_idx) + 1e-10)
    return ratio_rank * diff_rank


def _alpha_097(vwap, close, inv_idx, max_idx, n):
    ts_min_vwap = rolling_min(vwap, 5, max_idx)
    diff = vwap - ts_min_vwap
    diff_rank = rolling_rank(diff, 11, inv_idx, max_idx)
    ratio = vwap / (close + 1e-10)
    ratio_rank = ratio / (ratio + rolling_mean(ratio, 10, max_idx) + 1e-10)
    return diff_rank * ratio_rank


def _alpha_098(vwap, volume, inv_idx, max_idx, n):
    sum_vol = rolling_sum(volume, 9, max_idx)
    ratio = vwap / (sum_vol / 9 + 1e-10)
    ratio_rank = rolling_rank(ratio, 5, inv_idx, max_idx)
    delta_vwap = delta(vwap, 3, max_idx)
    delta_rank = rolling_rank(delta_vwap, 5, inv_idx, max_idx)
    return ratio_rank * delta_rank


def _alpha_099(close, volume, adv20, inv_idx, max_idx, n):
    delay_close = delay(close, 3, max_idx)
    diff = close - delay_close
    diff_rank = rolling_rank(diff, 10, inv_idx, max_idx)
    vol_ratio = volume / (adv20 + 1e-10)
    vol_rank = rolling_rank(vol_ratio, 10, inv_idx, max_idx)
    return diff_rank * vol_rank


def _alpha_100(vwap, close, volume, inv_idx, max_idx, n):
    delta_vwap = delta(vwap, 4, max_idx)
    ratio = vwap / (close + 1e-10)
    ratio_rank = rolling_rank(ratio, 9, inv_idx, max_idx)
    combined = delta_vwap * ratio_rank
    combined_rank = rolling_rank(combined, 7, inv_idx, max_idx)
    return -combined_rank


def _alpha_101(vwap, volume, inv_idx, max_idx, n):
    delta_vwap = delta(vwap, 3, max_idx)
    vol_rank = rolling_rank(volume, 9, inv_idx, max_idx)
    combined = delta_vwap * vol_rank
    combined_rank = rolling_rank(combined, 7, inv_idx, max_idx)
    return -combined_rank


if __name__ == '__main__':
    results = main()
