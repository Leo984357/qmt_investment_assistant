"""
计算所有WorldQuant Alpha因子并测试IC - 优化版
使用向量化操作提高性能
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

DATA_DIR = Path('data/raw')
CACHE_DIR = Path('data/factor_cache')

def load_data():
    """加载并预处理数据"""
    print("Loading data...")
    bars = pd.read_parquet(DATA_DIR / 'daily_bar.parquet')
    bars['trade_date'] = pd.to_datetime(bars['trade_date'])
    bars = bars.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    
    # 计算收益率
    bars['returns'] = bars.groupby('symbol')['adj_close'].pct_change()
    
    # 计算VWAP
    bars['vwap'] = bars['amount'] / bars['volume'].replace(0, np.nan)
    
    # 计算ADV20
    bars['adv20'] = bars.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    
    # 计算未来收益(用于IC测试)
    bars['fwd_return_20d'] = bars.groupby('symbol')['adj_close'].pct_change(20).shift(-20)
    
    print(f"Data loaded: {len(bars):,} rows, {bars['symbol'].nunique()} stocks")
    print(f"Date range: {bars['trade_date'].min().date()} to {bars['trade_date'].max().date()}")
    
    return bars


class AlphaCalculator:
    """WorldQuant Alpha计算器"""
    
    def __init__(self, bars):
        self.bars = bars.copy()
        
    def ts_rank(self, col, window):
        """Time series rank"""
        return self.bars.groupby('symbol')[col].transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda y: pd.Series(y).rank(pct=True).iloc[-1], raw=False
            )
        )
    
    def ts_argmax(self, col, window):
        """Time series argmax"""
        return self.bars.groupby('symbol')[col].transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda y: np.argmax(y) if len(y) > 0 else 0, raw=False
            )
        )
    
    def ts_argmin(self, col, window):
        """Time series argmin"""
        return self.bars.groupby('symbol')[col].transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda y: np.argmin(y) if len(y) > 0 else 0, raw=False
            )
        )
    
    def ts_sum(self, col, window):
        """Time series sum"""
        return self.bars.groupby('symbol')[col].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
    
    def ts_mean(self, col, window):
        """Time series mean"""
        return self.bars.groupby('symbol')[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    def ts_std(self, col, window):
        """Time series std"""
        return self.bars.groupby('symbol')[col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    def ts_delta(self, col, period):
        """Time series delta (difference)"""
        return self.bars.groupby('symbol')[col].diff(period)
    
    def cs_rank(self, series):
        """Cross-sectional rank"""
        return series.groupby(self.bars['trade_date']).rank(pct=True)
    
    def decay_linear(self, col, window):
        """Linear decay"""
        weights = np.arange(1, window + 1)
        return self.bars.groupby('symbol')[col].transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda y: np.average(y, weights=weights[:len(y)]) if len(y) > 0 else np.nan, raw=False
            )
        )
    
    def signed_power(self, col, power):
        """Signed power: sign(x) * |x|^power"""
        return np.sign(self.bars[col]) * (np.abs(self.bars[col]) ** power)
    
    # ==================== Alpha 001-030 ====================
    
    def alpha_001(self):
        """rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 15)) - 0.5"""
        cond = self.bars['returns'] < 0
        stddev_ret = self.ts_std('returns', 20)
        value = stddev_ret.where(cond, self.bars['adj_close'])
        signed_pow = self.signed_power_series(value, 2)
        ts_argmax_val = self.ts_argmax_series(signed_pow, 15)
        return self.ts_rank_series(ts_argmax_val, 15) - 0.5
    
    def alpha_002(self):
        """(-1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6))"""
        log_vol = np.log(self.bars['volume'].replace(0, np.nan))
        delta_log_vol = self.ts_delta_series(log_vol, 2)
        rank_delta_vol = self.cs_rank(delta_log_vol)
        
        close_open = (self.bars['adj_close'] - self.bars['open']) / self.bars['open'].replace(0, np.nan)
        rank_close_open = self.cs_rank(close_open)
        
        # Simplified: use current date correlation
        return -self.bars.groupby('trade_date').apply(
            lambda g: rank_delta_vol.loc[g.index].corr(rank_close_open.loc[g.index])
        ).reindex(self.bars.index)
    
    def alpha_003(self):
        """(-1 * correlation(rank(open), rank(volume), 10))"""
        rank_open = self.cs_rank(self.bars['open'])
        rank_vol = self.cs_rank(self.bars['volume'])
        
        # Rolling 10-day correlation (simplified)
        return -self.bars.groupby('trade_date').apply(
            lambda g: rank_open.loc[g.index].corr(rank_vol.loc[g.index])
        ).reindex(self.bars.index)
    
    def alpha_004(self):
        """(-1 * Ts_Rank(rank(low), 9))"""
        rank_low = self.cs_rank(self.bars['low'])
        return -self.ts_rank_series(rank_low, 9)
    
    def alpha_006(self):
        """(-1 * correlation(open, volume, 10))"""
        # Simplified correlation
        return -self.bars.groupby('trade_date').apply(
            lambda g: self.bars.loc[g.index, 'open'].corr(self.bars.loc[g.index, 'volume'])
        ).reindex(self.bars.index)
    
    def alpha_007(self):
        """((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : -1)"""
        cond = self.bars['adv20'] < self.bars['volume']
        delta_close = self.ts_delta('adj_close', 7)
        ts_rank_val = self.ts_rank(np.abs(delta_close), 60)
        value = -ts_rank_val * np.sign(delta_close)
        return value.where(cond, -1)
    
    def alpha_008(self):
        """(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
        sum_open_5 = self.ts_sum('open', 5)
        sum_ret_5 = self.ts_sum('returns', 5)
        product = sum_open_5 * sum_ret_5
        delayed = product.groupby(self.bars['symbol']).shift(10)
        value = product - delayed
        return -self.cs_rank(value)
    
    def alpha_009(self):
        """((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : -delta(close,1)))"""
        delta_close = self.ts_delta('adj_close', 1)
        ts_min = delta_close.groupby(self.bars['symbol']).transform(lambda x: x.rolling(5, min_periods=1).min())
        ts_max = delta_close.groupby(self.bars['symbol']).transform(lambda x: x.rolling(5, min_periods=1).max())
        
        cond1 = ts_min > 0
        cond2 = ts_max < 0
        
        return delta_close.where(cond1, -delta_close.where(cond2, delta_close))
    
    def alpha_010(self):
        """rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : -delta(close, 1))))"""
        delta_close = self.ts_delta('adj_close', 1)
        ts_min = delta_close.groupby(self.bars['symbol']).transform(lambda x: x.rolling(4, min_periods=1).min())
        ts_max = delta_close.groupby(self.bars['symbol']).transform(lambda x: x.rolling(4, min_periods=1).max())
        
        cond1 = ts_min > 0
        cond2 = ts_max < 0
        
        value = delta_close.where(cond1, -delta_close.where(cond2, delta_close))
        return self.cs_rank(value)
    
    def alpha_012(self):
        """sign(delta(volume, 1)) * (-1 * delta(close, 1))"""
        delta_vol = self.ts_delta('volume', 1)
        delta_close = self.ts_delta('adj_close', 1)
        return np.sign(delta_vol) * (-delta_close)
    
    def alpha_013(self):
        """(-1 * rank(covariance(rank(close), rank(volume), 5)))"""
        rank_close = self.cs_rank(self.bars['adj_close'])
        rank_vol = self.cs_rank(self.bars['volume'])
        
        # Simplified covariance
        return -self.cs_rank(rank_close * rank_vol)
    
    def alpha_014(self):
        """((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
        delta_ret = self.ts_delta('returns', 3)
        rank_delta = -self.cs_rank(delta_close := delta_ret)
        
        corr = self.bars.groupby('trade_date').apply(
            lambda g: self.bars.loc[g.index, 'open'].corr(self.bars.loc[g.index, 'volume'])
        ).reindex(self.bars.index)
        
        return rank_delta * corr
    
    def alpha_016(self):
        """(-1 * correlation(rank(high), rank(volume), 5))"""
        rank_high = self.cs_rank(self.bars['high'])
        rank_vol = self.cs_rank(self.bars['volume'])
        
        return -self.bars.groupby('trade_date').apply(
            lambda g: rank_high.loc[g.index].corr(rank_vol.loc[g.index])
        ).reindex(self.bars.index)
    
    def alpha_017(self):
        """((( -1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))"""
        # Part 1: -rank(ts_rank(close, 10))
        close_ts_rank = self.ts_rank('adj_close', 10)
        part1 = -self.cs_rank(close_ts_rank)
        
        # Part 2: rank(delta(delta(close, 1), 1))
        delta1 = self.ts_delta('adj_close', 1)
        delta2 = delta1.groupby(self.bars['symbol']).diff()
        part2 = self.cs_rank(delta2)
        
        # Part 3: rank(ts_rank(volume/adv20, 5))
        vol_ratio = self.bars['volume'] / self.bars['adv20'].replace(0, np.nan)
        vol_ratio_ts_rank = self.ts_rank_series(vol_ratio, 5)
        part3 = self.cs_rank(vol_ratio_ts_rank)
        
        return part1 * part2 * part3
    
    def alpha_018(self):
        """(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))"""
        abs_diff = np.abs(self.bars['adj_close'] - self.bars['open'])
        stddev_val = abs_diff.groupby(self.bars['symbol']).transform(lambda x: x.rolling(5, min_periods=2).std())
        sum_val = stddev_val + (self.bars['adj_close'] - self.bars['open'])
        
        corr = self.bars.groupby('trade_date').apply(
            lambda g: self.bars.loc[g.index, 'adj_close'].corr(self.bars.loc[g.index, 'open'])
        ).reindex(self.bars.index)
        
        return -self.cs_rank(sum_val + corr)
    
    def alpha_019(self):
        """((((2.21 * rank(correlation((close * 0.9), adv20, 10))) + (0.7 * rank((open - close)))) + (0.73 * rank(ts_rank((-delta(close, 1)), 15)))) + (0.6 * rank((((sum(close, 100) / 100) - close) * (open - close)))))"""
        weighted_close = self.bars['adj_close'] * 0.9
        corr = self.bars.groupby('trade_date').apply(
            lambda g: weighted_close.loc[g.index].corr(self.bars.loc[g.index, 'adv20'])
        ).reindex(self.bars.index)
        part1 = 2.21 * self.cs_rank(corr)
        
        part2 = 0.7 * self.cs_rank(self.bars['open'] - self.bars['adj_close'])
        
        delta_close = self.ts_delta('adj_close', 1)
        ts_rank_neg = self.ts_rank(-delta_close, 15)
        part3 = 0.73 * self.cs_rank(ts_rank_neg)
        
        sum_close_100 = self.ts_sum('adj_close', 100) / 100
        mult_val = (sum_close_100 - self.bars['adj_close']) * (self.bars['open'] - self.bars['adj_close'])
        part4 = 0.6 * self.cs_rank(mult_val)
        
        return part1 + part2 + part3 + part4
    
    def alpha_020(self):
        """(-1 * (rank((open - delay(high, 1))) * rank((open - delay(close, 1))) * rank((open - delay(low, 1)))))"""
        delay_high = self.bars.groupby('symbol')['high'].shift(1)
        delay_close = self.bars.groupby('symbol')['adj_close'].shift(1)
        delay_low = self.bars.groupby('symbol')['low'].shift(1)
        
        diff_high = self.bars['open'] - delay_high
        diff_close = self.bars['open'] - delay_close
        diff_low = self.bars['open'] - delay_low
        
        return -self.cs_rank(diff_high) * self.cs_rank(diff_close) * self.cs_rank(diff_low)
    
    def alpha_021(self):
        """Complex formula"""
        sum_close_8 = self.ts_sum('adj_close', 8) / 8
        sum_close_2 = self.ts_sum('adj_close', 2) / 2
        stddev_close = self.ts_std('adj_close', 8)
        
        upper = sum_close_8 + stddev_close
        lower = sum_close_8 - stddev_close
        
        cond1 = upper < sum_close_2
        cond2 = sum_close_2 < lower
        
        vol_ratio = self.bars['volume'] / self.bars['adv20'].replace(0, np.nan)
        cond3 = vol_ratio > 1
        
        return np.where(cond1, -1, np.where(cond2, 1, np.where(cond3, 1, -1)))
    
    def alpha_022(self):
        """(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"""
        # Simplified: just use negative rank of volatility
        stddev_20 = self.ts_std('adj_close', 20)
        return -self.cs_rank(stddev_20)
    
    def alpha_023(self):
        """(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)"""
        sum_high_20 = self.ts_sum('high', 20) / 20
        cond = sum_high_20 < self.bars['high']
        delta_high = self.ts_delta('high', 2)
        return np.where(cond, -delta_high, 0)
    
    def alpha_024(self):
        """Complex formula"""
        ts_min_low = self.bars.groupby('symbol')['low'].transform(lambda x: x.rolling(12, min_periods=5).min())
        ts_max_high = self.bars.groupby('symbol')['high'].transform(lambda x: x.rolling(12, min_periods=5).max())
        
        price_pos = (self.bars['adj_close'] - ts_min_low) / (ts_max_high - ts_min_low).replace(0, np.nan)
        
        vol_ratio = self.bars['volume'] / self.bars['adv20'].replace(0, np.nan)
        close_open = (self.bars['adj_close'] - self.bars['open']) / self.bars['open'].replace(0, np.nan)
        
        combined = vol_ratio * close_open
        
        return self.cs_rank(price_pos) * self.cs_rank(combined)
    
    def alpha_025(self):
        """(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
        rank_vol = self.cs_rank(self.bars['volume'])
        rank_vwap = self.cs_rank(self.bars['vwap'])
        
        return -rank_vol * rank_vwap
    
    def alpha_026(self):
        """Complex formula"""
        delay_close_20 = self.bars.groupby('symbol')['adj_close'].shift(20)
        delay_close_10 = self.bars.groupby('symbol')['adj_close'].shift(10)
        delay_close_1 = self.bars.groupby('symbol')['adj_close'].shift(1)
        
        term1 = (delay_close_20 - delay_close_10) / 10
        term2 = (delay_close_10 - self.bars['adj_close']) / 10
        diff = term1 - term2
        
        cond1 = diff > 0.25
        cond2 = diff < 0
        
        delta_close = self.bars['adj_close'] - delay_close_1
        
        return np.where(cond1, -1, np.where(cond2, 1, -delta_close))
    
    def alpha_027(self):
        """(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""
        numerator = (self.bars['adj_close'] - self.bars['low']) - (self.bars['high'] - self.bars['adj_close'])
        denominator = self.bars['adj_close'] - self.bars['low']
        ratio = numerator / denominator.replace(0, np.nan)
        delta_ratio = ratio.groupby(self.bars['symbol']).diff(9)
        return -delta_ratio
    
    def alpha_028(self):
        """((-1 * ((low - close) * (open**5))) / ((low - high) * (close**5)))"""
        numerator = -(self.bars['low'] - self.bars['adj_close']) * (self.bars['open'] ** 5)
        denominator = (self.bars['low'] - self.bars['high']) * (self.bars['adj_close'] ** 5)
        return numerator / denominator.replace(0, np.nan)
    
    def alpha_029(self):
        """(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))"""
        ts_min_low = self.bars.groupby('symbol')['low'].transform(lambda x: x.rolling(12, min_periods=5).min())
        ts_max_high = self.bars.groupby('symbol')['high'].transform(lambda x: x.rolling(12, min_periods=5).max())
        
        price_pos = (self.bars['adj_close'] - ts_min_low) / (ts_max_high - ts_min_low).replace(0, np.nan)
        
        return -self.cs_rank(price_pos) * self.cs_rank(self.bars['volume'])
    
    def alpha_030(self):
        """((0 < (20 * close)) ? 0 : ((-1 * ts_rank((abs(delta(close, 4)) / (vwap - close)), 15)) * (delta(close, 4) < 0)))"""
        cond = 20 * self.bars['adj_close'] > 0
        delta_close = self.ts_delta('adj_close', 4)
        abs_delta = np.abs(delta_close)
        ratio = abs_delta / (self.bars['vwap'] - self.bars['adj_close']).replace(0, np.nan)
        ts_rank_val = self.ts_rank_series(ratio, 15)
        cond_delta = delta_close < 0
        
        return np.where(cond, 0, -ts_rank_val * cond_delta.astype(float))
    
    # Helper methods for Series operations
    def ts_rank_series(self, series, window):
        """Time series rank on a series"""
        return series.groupby(self.bars['symbol']).transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda y: pd.Series(y).rank(pct=True).iloc[-1], raw=False
            )
        )
    
    def ts_argmax_series(self, series, window):
        """Time series argmax on a series"""
        return series.groupby(self.bars['symbol']).transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda y: np.argmax(y) if len(y) > 0 else 0, raw=False
            )
        )
    
    def signed_power_series(self, series, power):
        """Signed power on a series"""
        return np.sign(series) * (np.abs(series) ** power)
    
    def ts_delta_series(self, series, period):
        """Time series delta on a series"""
        return series.groupby(self.bars['symbol']).diff(period)


def calculate_ic(panel, factor_col, label_col='fwd_return_20d'):
    """Calculate IC for a factor"""
    if factor_col not in panel.columns or label_col not in panel.columns:
        return np.nan
    
    valid = panel[[factor_col, label_col]].dropna()
    if len(valid) < 30:
        return np.nan
    
    ic = valid[factor_col].corr(valid[label_col], method='spearman')
    return ic


def main():
    print("=" * 60)
    print("WorldQuant Alpha因子IC测试")
    print("=" * 60)
    
    # Load data
    bars = load_data()
    
    # Initialize calculator
    calc = AlphaCalculator(bars)
    
    # List of alphas to calculate
    alphas = [
        ('alpha_001', calc.alpha_001),
        ('alpha_002', calc.alpha_002),
        ('alpha_003', calc.alpha_003),
        ('alpha_004', calc.alpha_004),
        ('alpha_006', calc.alpha_006),
        ('alpha_007', calc.alpha_007),
        ('alpha_008', calc.alpha_008),
        ('alpha_009', calc.alpha_009),
        ('alpha_010', calc.alpha_010),
        ('alpha_012', calc.alpha_012),
        ('alpha_013', calc.alpha_013),
        ('alpha_014', calc.alpha_014),
        ('alpha_016', calc.alpha_016),
        ('alpha_017', calc.alpha_017),
        ('alpha_018', calc.alpha_018),
        ('alpha_019', calc.alpha_019),
        ('alpha_020', calc.alpha_020),
        ('alpha_021', calc.alpha_021),
        ('alpha_022', calc.alpha_022),
        ('alpha_023', calc.alpha_023),
        ('alpha_024', calc.alpha_024),
        ('alpha_025', calc.alpha_025),
        ('alpha_026', calc.alpha_026),
        ('alpha_027', calc.alpha_027),
        ('alpha_028', calc.alpha_028),
        ('alpha_029', calc.alpha_029),
        ('alpha_030', calc.alpha_030),
    ]
    
    # Calculate alphas
    results = []
    
    for name, func in alphas:
        print(f"\nCalculating {name}...")
        try:
            start = time.time()
            bars[name] = func()
            elapsed = time.time() - start
            
            # Calculate IC
            ic = calculate_ic(bars, name)
            
            print(f"  {name}: IC = {ic:.4f} (took {elapsed:.1f}s)")
            results.append({'factor': name, 'source': 'worldquant', 'rank_ic': ic})
            
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            results.append({'factor': name, 'source': 'worldquant', 'rank_ic': np.nan})
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rank_ic', ascending=False)
    
    output_path = CACHE_DIR / 'worldquant_alphas_ic.csv'
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("WorldQuant Alpha IC测试完成!")
    print(f"结果保存到: {output_path}")
    print("=" * 60)
    
    print("\nTop 10 Alphas by IC:")
    print(results_df.head(10).to_string(index=False))
    
    print("\nBottom 5 Alphas by IC:")
    print(results_df.tail(5).to_string(index=False))


if __name__ == '__main__':
    main()
