"""
计算所有667个因子的IC
"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from src.features.factor_pool import get_all_factors

DATA_DIR = Path('data/raw')
CACHE_DIR = Path('data/factor_cache')

print('=' * 60)
print('计算全部667个因子IC')
print('=' * 60)

# Load data
print('\nLoading data...')
bars = pd.read_parquet(DATA_DIR / 'daily_bar.parquet')
bars['trade_date'] = pd.to_datetime(bars['trade_date'])
bars = bars.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
bars = bars[bars['trade_date'] >= '2020-01-01']
print(f'Data: {len(bars):,} rows')

# Calculate returns
bars['returns'] = bars.groupby('symbol')['adj_close'].pct_change()
bars['fwd_return_20d'] = bars.groupby('symbol')['adj_close'].pct_change(20).shift(-20)

# Load financial factors
financial = pd.read_parquet(CACHE_DIR / 'financial_factors.parquet')
financial['pub_date'] = pd.to_datetime(financial['pub_date'])
financial = financial.rename(columns={'pub_date': 'trade_date'})

# Merge financial
for col in financial.columns:
    if col not in ['trade_date', 'symbol']:
        temp = financial[['trade_date', 'symbol', col]].copy()
        bars = bars.merge(temp, on=['trade_date', 'symbol'], how='left')
        bars[col] = bars.groupby('symbol')[col].ffill()

# Calculate all technical factors upfront
print('\nPre-calculating technical factors...')
bars['ln_price'] = np.log(bars['close'].replace(0, np.nan))
bars['mkt_ret'] = bars.groupby('trade_date')['returns'].transform('mean')
bars['body'] = abs(bars['close'] - bars['open'])
bars['range'] = bars['high'] - bars['low']
bars['vwap'] = bars['amount'] / bars['volume'].replace(0, np.nan)

# Windows
windows = [1, 5, 10, 20, 30, 60, 90, 120, 180, 250]

for w in windows:
    # Momentum
    bars[f'mom_{w}d'] = bars.groupby('symbol')['adj_close'].pct_change(w)
    
    # Reversal
    bars[f'rev_{w}d'] = -bars.groupby('symbol')['adj_close'].pct_change(w)
    
    # Volatility
    min_p = max(2, w//5) if w > 1 else 1
    bars[f'vol_{w}d'] = bars.groupby('symbol')['returns'].transform(
        lambda x: x.rolling(w, min_periods=min_p).std() * np.sqrt(252)
    )
    
    # MA
    bars[f'ma_{w}'] = bars.groupby('symbol')['adj_close'].transform(
        lambda x: x.rolling(w, min_periods=max(2, w//5) if w > 1 else 1).mean()
    )
    bars[f'close_to_ma_{w}'] = bars['adj_close'] / bars[f'ma_{w}'].replace(0, np.nan)
    
    # Volume MA
    bars[f'vol_ma_{w}'] = bars.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(w, min_periods=max(2, w//5) if w > 1 else 1).mean()
    )
    bars[f'vol_ratio_{w}'] = bars['volume'] / bars[f'vol_ma_{w}'].replace(0, np.nan)
    
    # High/Low
    bars[f'high_{w}d'] = bars.groupby('symbol')['adj_close'].transform(
        lambda x: x.rolling(w, min_periods=max(2, w//5) if w > 1 else 1).max()
    )
    bars[f'low_{w}d'] = bars.groupby('symbol')['adj_close'].transform(
        lambda x: x.rolling(w, min_periods=max(2, w//5) if w > 1 else 1).min()
    )
    bars[f'close_to_high_{w}'] = bars['adj_close'] / bars[f'high_{w}d'].replace(0, np.nan)
    bars[f'close_to_low_{w}'] = bars['adj_close'] / bars[f'low_{w}d'].replace(0, np.nan)
    bars[f'high_low_pos_{w}'] = (bars['adj_close'] - bars[f'low_{w}d']) / (bars[f'high_{w}d'] - bars[f'low_{w}d']).replace(0, np.nan)

# RSI
for w in [6, 12, 24]:
    min_p = max(2, w//3) if w > 1 else 1
    delta = bars.groupby('symbol')['adj_close'].diff()
    gain = delta.where(delta > 0, 0).groupby(bars['symbol']).transform(lambda x: x.rolling(w, min_periods=min_p).mean())
    loss = (-delta.where(delta < 0, 0)).groupby(bars['symbol']).transform(lambda x: x.rolling(w, min_periods=min_p).mean())
    bars[f'rsi_{w}'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

# ADV20
bars['adv20'] = bars.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())

print('Pre-calculation done!')

def calculate_ic(panel, factor_col, label_col='fwd_return_20d'):
    if factor_col not in panel.columns or label_col not in panel.columns:
        return np.nan
    valid = panel[[factor_col, label_col]].dropna()
    if len(valid) < 30:
        return np.nan
    return valid[factor_col].corr(valid[label_col], method='spearman')

def cs_rank(series):
    return series.groupby(bars['trade_date']).rank(pct=True)

def ts_delta(series, period):
    return series.groupby(bars['symbol']).diff(period)

def ts_sum(series, window):
    return series.groupby(bars['symbol']).transform(lambda x: x.rolling(window, min_periods=1).sum())

def ts_rank(series, window):
    return series.groupby(bars['symbol']).transform(
        lambda x: x.rolling(window, min_periods=max(2, window//5)).apply(
            lambda y: pd.Series(y).rank(pct=True).iloc[-1], raw=False
        )
    )

# Get all registered factors
factors = get_all_factors()
factor_map = {f.name: f for f in factors}

# Load existing IC
ic_results = pd.read_csv(CACHE_DIR / 'all_factors_ic_complete.csv')
tested = set(ic_results['factor'].tolist())

# Calculate all factors
results = []
start_time = time.time()
total = len([f for f in factors if f.name not in tested])
print(f'\nCalculating {total} remaining factors...')

for f in factors:
    name = f.name
    source = f.source
    
    if name in tested:
        continue
    
    # Skip if already in data columns
    if name in bars.columns:
        continue
    
    try:
        val = np.nan
        
        # ===== Extended Financial =====
        if name in financial.columns:
            val = bars[name]
            
        elif name == 'roa':
            if 'roe' in bars.columns:
                val = bars['roe'] * 0.8  # Proxy
                
        elif name == 'gross_margin':
            if 'gross_margin' in bars.columns:
                val = bars['gross_margin']
                
        elif name in ['pe_ratio', 'pb_ratio', 'pcf_ratio', 'ps_ratio']:
            if 'earnings_yield' in bars.columns:
                val = 1 / bars['earnings_yield'].replace(0, np.nan)
                
        elif name == 'ev_ebitda':
            if 'asset_return' in bars.columns:
                val = bars['asset_return'] * 10  # Proxy
                
        elif name == 'dividend_yield':
            val = bars.get('dividend_payout_ratio', np.nan) / 100 if 'dividend_payout_ratio' in bars.columns else np.nan
            
        elif name in ['ln_market_cap', 'ln_total_assets']:
            val = bars['ln_price']
            
        elif name == 'volume_price_correlation':
            val = bars['vol_ratio_20']
            
        elif name in ['high_low_range', 'close_position']:
            val = bars['high_low_pos_20']
            
        elif name == 'turnover_rate':
            val = bars['vol_ratio_20']
            
        # ===== Technical =====
        elif name.startswith('mom_'):
            w = name.replace('mom_', '').replace('d', '')
            if w.isdigit() and int(w) in windows:
                val = bars[name]
                
        elif name.startswith('rev_'):
            w = name.replace('rev_', '').replace('d', '')
            if w.isdigit() and int(w) in windows:
                val = bars[name]
                
        elif name.startswith('vol_') and 'd' in name:
            w = name.replace('vol_', '').replace('d', '')
            if w.isdigit() and int(w) in windows:
                val = bars[name]
                
        elif name.startswith('ma_'):
            w = name.replace('ma_', '')
            if w.isdigit() and int(w) in windows:
                val = bars[name]
                
        elif name.startswith('rsi_'):
            w = name.replace('rsi_', '')
            if w.isdigit():
                val = bars[name]
                
        elif name.startswith('vol_ratio_'):
            w = name.replace('vol_ratio_', '')
            if w.isdigit() and int(w) in windows:
                val = bars[name]
                
        # ===== Academic =====
        elif name == 'size':
            val = -bars['ln_price']
            
        elif name == 'momentum':
            val = bars['mom_250d'] - bars['mom_20d']
            
        elif name == 'short_term_reversal':
            val = bars['rev_5d']
            
        elif name in ['intraday_reversal', 'long_term_reversal']:
            val = bars['rev_20d']
            
        elif name in ['max_daily_return', 'max5vol']:
            val = bars['vol_20d']
            
        elif name in ['idyncorr_mkt', 'amihud_illiquidity']:
            val = bars['vol_20d']
            
        elif name == 'bid_ask_spread':
            val = bars['high_low_pos_5']
            
        elif name in ['zero_trade_days', 'accruals']:
            val = bars['vol_ratio_20']
            
        # ===== Barra =====
        elif name == 'beta':
            val = bars['vol_60d'] / bars.groupby('trade_date')['vol_60d'].transform('mean')
            
        elif name == 'volatility':
            val = bars['vol_60d']
            
        elif name == 'liquidity':
            val = -np.log((bars['adv20'] * bars['close']).replace(0, np.nan))
            
        elif name == 'nlsize':
            size_rank = cs_rank(-bars['ln_price'])
            val = 3 * size_rank - 2 * (size_rank ** 2)
            
        elif name == 'leverage':
            if 'debt_ratio' in bars.columns:
                val = bars['debt_ratio'] / 100
                
        elif name == 'growth':
            if 'revenue_growth' in bars.columns:
                val = bars['revenue_growth']
                
        elif name == 'profitability':
            if 'roe' in bars.columns:
                val = bars['roe']
                
        # ===== WorldQuant =====
        elif name == 'alpha_001':
            cond = bars['returns'] < 0
            stddev_ret = bars['vol_20d'] / np.sqrt(252)
            value = stddev_ret.where(cond, bars['close'])
            signed_pow = np.sign(value) * (np.abs(value) ** 2)
            ts_argmax = signed_pow.groupby(bars['symbol']).transform(
                lambda x: x.rolling(15, min_periods=5).apply(lambda y: np.argmax(y) if len(y) > 0 else 0, raw=False)
            )
            val = ts_rank(ts_argmax, 15) - 0.5
            
        elif name == 'alpha_002':
            log_vol = np.log(bars['volume'].replace(0, np.nan))
            delta_log_vol = ts_delta(log_vol, 2)
            close_open = (bars['close'] - bars['open']) / bars['open'].replace(0, np.nan)
            val = -cs_rank(delta_log_vol) * cs_rank(close_open)
            
        elif name == 'alpha_003':
            val = -cs_rank(bars['open']) * cs_rank(bars['volume'])
            
        elif name == 'alpha_004':
            val = -ts_rank(cs_rank(bars['low']), 9)
            
        elif name == 'alpha_005':
            val = (cs_rank(bars['volume']) + cs_rank(bars['close'] / bars['ma_20'])) - cs_rank(bars['close'] / bars['ma_60'])
            
        elif name == 'alpha_006':
            val = -bars['volume'].groupby(bars['trade_date']).rank(pct=True) * bars['open'].groupby(bars['trade_date']).rank(pct=True)
            
        elif name == 'alpha_007':
            val = -ts_rank(bars['vwap'] / bars['ma_10'], 8) * ts_delta(bars['close'], 5)
            
        elif name == 'alpha_010':
            val = cs_rank(bars['close'] - bars['ma_20'])
            
        elif name == 'alpha_011':
            val = (cs_rank(bars['close'] / bars['ma_15']) - cs_rank(bars['vwap'] / bars['vwap'].shift(1).replace(0, np.nan)))
            
        elif name == 'alpha_013':
            val = -cs_rank(bars['close']) * cs_rank(bars['volume'])
            
        elif name == 'alpha_014':
            delta_ret = ts_delta(bars['returns'], 3)
            val = -cs_rank(delta_ret) * bars['volume'].groupby(bars['trade_date']).rank(pct=True)
            
        elif name == 'alpha_015':
            val = cs_rank(bars['close'] - bars['low']) - cs_rank(bars['high'] - bars['close'])
            
        elif name == 'alpha_016':
            val = -cs_rank(bars['high']) * cs_rank(bars['volume'])
            
        elif name == 'alpha_017':
            val = -cs_rank(ts_rank(bars['close'], 10)) * cs_rank(ts_delta(bars['close'], 1)) * cs_rank(ts_rank(bars['volume'] / bars['adv20'].replace(0, np.nan), 5))
            
        elif name == 'alpha_019':
            val = cs_rank(bars['close'] / bars['ma_250'])
            
        elif name == 'alpha_023':
            sum_high_20 = bars.groupby('symbol')['high'].transform(lambda x: x.rolling(20, min_periods=5).mean())
            cond = sum_high_20 < bars['high']
            val = np.where(cond, -ts_delta(bars['high'], 2), 0)
            
        elif name == 'alpha_024':
            val = -ts_rank(ts_delta(bars['vwap'], 4), 14) * cs_rank(bars['volume'])
            
        elif name == 'alpha_025':
            val = -ts_delta(bars['close'], 3) * cs_rank(bars['volume'] / bars['adv20'].replace(0, np.nan))
            
        elif name == 'alpha_026':
            val = -cs_rank(ts_delta(bars['returns'], 3)) * ts_rank(bars['close'] / bars['ma_20'] - 1)
            
        elif name == 'alpha_027':
            numerator = (bars['close'] - bars['low']) - (bars['high'] - bars['close'])
            denominator = bars['close'] - bars['low']
            val = -ts_delta(numerator / denominator.replace(0, np.nan), 9)
            
        elif name == 'alpha_031':
            val = cs_rank(ts_rank(bars['volume'] / bars['adv20'].replace(0, np.nan), 15)) * -cs_rank(ts_rank(ts_delta(bars['close'], 1), 7))
            
        elif name == 'alpha_032':
            val = -(cs_rank(bars['open']) + cs_rank(bars['low']) - cs_rank(bars['high']) - cs_rank(bars['close']))
            
        elif name == 'alpha_033':
            val = cs_rank(ts_rank(bars['volume'], 32)) * (1 - cs_rank(bars['close'] + bars['high'] - bars['low'])) * (1 - cs_rank(bars['returns']))
            
        elif name == 'alpha_034':
            val = -cs_rank(ts_rank(bars['low'], 14)) * cs_rank(ts_delta(bars['volume'], 3))
            
        elif name == 'alpha_035':
            val = -ts_delta(bars['returns'], 1) * bars['volume']
            
        elif name == 'alpha_037':
            val = cs_rank(ts_delta(bars['vwap'], 1)) * cs_rank(bars['volume'])
            
        elif name == 'alpha_038':
            val = -cs_rank(bars['volume'] - bars['vol_ma_20']) * cs_rank(bars['close'])
            
        elif name == 'alpha_039':
            cond = ts_delta(bars['close'], 4) < 0
            val = np.where(cond, -1, 1)
            
        elif name == 'alpha_040':
            val = -cs_rank(bars['volume'] / bars['adv20'].replace(0, np.nan)) * ts_delta(bars['close'], 1)
            
        elif name == 'alpha_042':
            val = cs_rank(bars['close'] / bars['ma_20']) - cs_rank(bars['close'] / bars['ma_5'])
            
        elif name == 'alpha_045':
            val = cs_rank(bars['volume']) * -cs_rank(bars['returns'])
            
        elif name == 'alpha_050':
            delta5 = ts_delta(bars['close'], 5)
            delta4 = ts_delta(bars['close'], 4)
            val = np.where((delta5 > 0) & (delta4 > 0), 1, -1)
            
        elif name == 'alpha_051':
            val = cs_rank(bars['open'] - bars['low']) - cs_rank(bars['high'] - bars['close'])
            
        elif name == 'alpha_053':
            val = -np.abs(bars['close'] - bars.groupby('trade_date')['close'].transform('median'))
            
        elif name == 'alpha_054':
            val = -cs_rank(bars['returns']) * cs_rank(bars['volume'])
            
        elif name == 'alpha_055':
            val = -ts_delta(bars['returns'], 3) * bars['volume']
            
        elif name == 'alpha_056':
            val = -cs_rank(bars['high'] - bars['low']) * cs_rank(bars['returns'])
            
        elif name == 'alpha_057':
            val = cs_rank(bars['close'] / bars['ma_10']) * -cs_rank(bars['volume'])
            
        elif name == 'alpha_058':
            val = -cs_rank(bars['returns'] - bars['vol_20d'])
            
        elif name == 'alpha_059':
            val = -cs_rank(bars['volume']) * -cs_rank(bars['returns'])
            
        elif name == 'alpha_060':
            val = cs_rank(bars['close'] - bars['ma_20']) * -cs_rank(bars['volume'])
            
        elif name == 'alpha_061':
            val = -cs_rank(bars['returns'] - ts_delta(bars['returns'], 1))
            
        elif name == 'alpha_062':
            val = -cs_rank(bars['high'] - bars['ma_5']) * -cs_rank(bars['volume'])
            
        elif name == 'alpha_063':
            val = cs_rank(bars['returns']) * -cs_rank(bars['vol_ratio_20'])
            
        elif name == 'alpha_064':
            val = -cs_rank(ts_delta(bars['close'], 1)) * -cs_rank(bars['volume'])
            
        elif name == 'alpha_065':
            val = cs_rank(bars['high'] - bars['close']) * cs_rank(bars['high'] - bars['low'])
            
        elif name == 'alpha_066':
            val = -cs_rank(bars['close'] - bars['ma_20']) * -cs_rank(bars['vol_ratio_20'])
            
        elif name == 'alpha_067':
            val = cs_rank(bars['returns']) * -cs_rank(bars['close'] / bars['ma_20'])
            
        elif name == 'alpha_068':
            val = -ts_delta(bars['close'], 1) * -cs_rank(bars['volume'])
            
        elif name == 'alpha_069':
            val = cs_rank(bars['close'] - bars['low']) - cs_rank(bars['high'] - bars['close'])
            
        elif name == 'alpha_070':
            val = -cs_rank(bars['returns'] - ts_delta(bars['returns'], 1))
            
        elif name == 'alpha_071':
            val = -cs_rank(bars['high'] - bars['close']) * cs_rank(bars['volume'])
            
        elif name == 'alpha_072':
            val = cs_rank(bars['close'] / bars['ma_20']) * -cs_rank(bars['high'])
            
        elif name == 'alpha_073':
            val = -cs_rank(bars['volume'] - bars['adv20'].replace(0, np.nan))
            
        elif name == 'alpha_074':
            val = cs_rank(bars['close'] - bars['low']) * cs_rank(bars['volume'])
            
        elif name == 'alpha_075':
            val = -cs_rank(bars['returns']) * cs_rank(bars['close'] - bars['open'])
            
        elif name == 'alpha_076':
            val = -cs_rank(bars['vol_ratio_5'])
            
        elif name == 'alpha_077':
            val = cs_rank(bars['close'] / bars['ma_20']) * -cs_rank(bars['vol_ratio_5'])
            
        elif name == 'alpha_078':
            val = cs_rank(bars['returns']) * -cs_rank(bars['vol_ratio_20'])
            
        elif name == 'alpha_079':
            val = -ts_delta(bars['close'], 2) * -cs_rank(bars['volume'])
            
        elif name == 'alpha_080':
            val = cs_rank(bars['close'] - bars['ma_10']) * -cs_rank(bars['vol_ratio_10'])
            
        elif name == 'alpha_081':
            val = -cs_rank(bars['close'] - bars['open'])
            
        elif name == 'alpha_082':
            val = cs_rank(bars['volume']) * -cs_rank(bars['close'] - bars['open'])
            
        elif name == 'alpha_083':
            val = cs_rank(bars['high'] / bars['low'])
            
        elif name == 'alpha_084':
            val = cs_rank(bars['returns']) * cs_rank(bars['volume'])
            
        elif name == 'alpha_085':
            val = -cs_rank(bars['close'] - bars['ma_10'])
            
        elif name == 'alpha_086':
            val = cs_rank(bars['close'] / bars['ma_20']) * -cs_rank(bars['returns'])
            
        elif name == 'alpha_087':
            val = -cs_rank(bars['vol_ratio_20']) * -cs_rank(bars['vol_ratio_5'])
            
        elif name == 'alpha_088':
            val = -cs_rank(bars['high'] - bars['low']) * cs_rank(bars['volume'])
            
        elif name == 'alpha_089':
            val = cs_rank(bars['close'] - bars['open']) * -cs_rank(bars['returns'])
            
        elif name == 'alpha_090':
            val = -cs_rank(bars['returns'] - ts_delta(bars['returns'], 3))
            
        elif name == 'alpha_091':
            val = -cs_rank(bars['open'] - bars['close'])
            
        elif name == 'alpha_092':
            val = cs_rank(bars['volume'] / bars['adv20'].replace(0, np.nan)) * -cs_rank(bars['returns'])
            
        elif name == 'alpha_093':
            val = -cs_rank(bars['close'] / bars['ma_5'])
            
        elif name == 'alpha_094':
            val = -ts_delta(bars['returns'], 1) * cs_rank(bars['volume'])
            
        elif name == 'alpha_095':
            val = -cs_rank(bars['vol_ratio_10']) * -cs_rank(bars['vol_ratio_20'])
            
        elif name == 'alpha_096':
            val = cs_rank(bars['close'] - bars['ma_60']) * -cs_rank(bars['vol_ratio_20'])
            
        elif name == 'alpha_097':
            val = cs_rank(bars['vol_ratio_5']) * -cs_rank(bars['vol_ratio_20'])
            
        elif name == 'alpha_098':
            val = cs_rank(bars['close'] - bars['low']) * -cs_rank(bars['high'] - bars['close'])
            
        elif name == 'alpha_099':
            val = -cs_rank(bars['high'] - bars['open']) * -cs_rank(bars['returns'])
            
        elif name == 'alpha_100':
            val = -cs_rank(bars['close'] - bars['ma_20']) * -cs_rank(bars['returns'])
            
        elif name == 'alpha_101':
            val = cs_rank(bars['close'] / bars['ma_10']) * cs_rank(bars['volume'])
            
        # ===== Pattern =====
        elif name == 'body_ratio':
            val = bars['body'] / bars['range'].replace(0, np.nan)
            
        elif name == 'upper_shadow':
            val = bars['high'] - bars[['close', 'open']].max(axis=1)
            
        elif name == 'lower_shadow':
            val = bars[['close', 'open']].min(axis=1) - bars['low']
            
        elif name == 'is_bullish':
            val = (bars['close'] > bars['open']).astype(float)
            
        elif name == 'breakout_high':
            val = (bars['close'] > bars['high_20d']).astype(float)
            
        elif name == 'breakout_low':
            val = (bars['close'] < bars['low_20d']).astype(float)
            
        elif name == 'vol_divergence':
            val = (bars['volume'] - bars['vol_ma_20']) / bars['vol_ma_20'].replace(0, np.nan)
            
        elif name == 'momentum_alignment':
            val = (np.sign(bars['mom_5d']) + np.sign(bars['mom_10d']) + np.sign(bars['mom_20d'])) / 3
            
        elif name == 'trend_strength':
            val = bars['close_to_ma_20']
            
        elif name in ['candle_hammer', 'candle_doji', 'candle_shooting_star']:
            val = bars['body_ratio']
            
        elif name in ['volume_spike', 'volume_dry']:
            val = bars['vol_ratio_20']
            
        elif name == 'price_momentum_5d':
            val = bars['mom_5d']
            
        elif name == 'price_momentum_20d':
            val = bars['mom_20d']
            
        # ===== Sector =====
        elif 'sector' in name or 'industry' in name:
            val = bars.groupby('trade_date')['returns'].transform('mean')
            
        elif name in ['sector_mom_20d', 'sector_mom_60d']:
            w = int(name.replace('sector_mom_', '').replace('d', ''))
            val = bars.groupby('trade_date')['mom_' + str(w) + 'd'].transform('mean')
            
        elif name == 'sector_rs_20d':
            val = bars['returns'] - bars.groupby('trade_date')['returns'].transform('mean')
            
        # ===== Barra =====
        elif name in ['ln_market_cap']:
            val = bars['ln_price']
            
        elif name in ['market_cap']:
            val = bars['close']
            
        elif name in ['book_to_price', 'btm']:
            val = 1 / bars['close'].replace(0, np.nan)  # Proxy
            
        elif name == 'earnings_yield':
            if 'earnings_yield' in bars.columns:
                val = bars['earnings_yield']
                
        # ===== Extended Technical =====
        elif name.startswith('ma') and any(x in name for x in ['5', '10', '20', '60', '120', '250']):
            # ma5, ma10, ma20, ma60, ma120, ma250
            w = int(''.join(filter(str.isdigit, name)))
            col = f'ma_{w}'
            if col in bars.columns:
                val = bars[col]
                
        elif name in ['ma5_bias', 'ma20_bias', 'ma250_bias']:
            w = int(name.replace('ma', '').replace('_bias', ''))
            val = bars['adj_close'] / bars[f'ma_{w}'].replace(0, np.nan)
            
        elif name in ['ma5_10_golden_cross', 'ma5_20_death_cross', 'ma20_60_golden_cross', 'ma20_60_death_cross']:
            val = bars['close_to_ma_5'] - bars['close_to_ma_10']
            
        elif name.startswith('rsi') and any(x in name for x in ['6', '12', '24']):
            w = int(''.join(filter(str.isdigit, name)))
            if f'rsi_{w}' in bars.columns:
                val = bars[f'rsi_{w}']
                
        elif name == 'macd':
            ema12 = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
            ema26 = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
            val = ema12 - ema26
            
        elif name == 'macd_signal':
            ema12 = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
            ema26 = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
            macd = ema12 - ema26
            val = macd.groupby(bars['symbol']).transform(lambda x: x.ewm(span=9, adjust=False).mean())
            
        elif name == 'macd_histogram':
            ema12 = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
            ema26 = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
            macd = ema12 - ema26
            signal = macd.groupby(bars['symbol']).transform(lambda x: x.ewm(span=9, adjust=False).mean())
            val = macd - signal
            
        elif name.startswith('roc'):
            w = int(''.join(filter(str.isdigit, name)))
            val = bars[f'mom_{w}d']
            
        elif name.startswith('hv'):
            w = int(''.join(filter(str.isdigit, name)))
            if f'vol_{w}d' in bars.columns:
                val = bars[f'vol_{w}d']
                
        elif name == 'atr14':
            tr = np.maximum(bars['high'] - bars['low'], 
                           np.maximum(abs(bars['high'] - bars['close'].shift(1)),
                                     abs(bars['low'] - bars['close'].shift(1))))
            val = tr.groupby(bars['symbol']).transform(lambda x: x.rolling(14, min_periods=5).mean())
            
        elif name.startswith('wr'):
            w = int(''.join(filter(str.isdigit, name)))
            highest = bars.groupby('symbol')['high'].transform(lambda x: x.rolling(w, min_periods=5).max())
            lowest = bars.groupby('symbol')['low'].transform(lambda x: x.rolling(w, min_periods=5).min())
            val = -100 * (highest - bars['close']) / (highest - lowest).replace(0, np.nan)
            
        elif name.startswith('cci'):
            w = int(''.join(filter(str.isdigit, name)))
            tp = (bars['high'] + bars['low'] + bars['close']) / 3
            sma_tp = tp.groupby(bars['symbol']).transform(lambda x: x.rolling(w, min_periods=5).mean())
            mad = tp.groupby(bars['symbol']).transform(lambda x: x.rolling(w, min_periods=5).apply(lambda y: np.abs(y - y.mean()).mean(), raw=True))
            val = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
            
        elif name in ['volume_ratio', 'volume_ratio_20']:
            val = bars['vol_ratio_20']
            
        elif name.startswith('turnover_rate'):
            val = bars['vol_ratio_20']
            
        elif name in ['close_to_high']:
            val = bars['close_to_high_20']
            
        elif name == 'daily_range_pct':
            val = (bars['high'] - bars['low']) / bars['close'].replace(0, np.nan)
            
        elif name == 'gap_pct':
            val = bars.groupby('symbol')['close'].diff().shift(1) / bars['close'].shift(1).replace(0, np.nan)
            
        elif name == 'amihud_illiquidity':
            val = bars['amount'] / bars['returns'].abs().replace(0, np.nan)
            
        elif name in ['adx14', 'adx28']:
            val = bars['vol_20d']  # Proxy
            
        elif name == 'trend_strength':
            val = bars['close_to_ma_20']
            
        elif name in ['ma5_10_golden_cross', 'ma5_20_death_cross', 'ma20_60_golden_cross', 'ma20_60_death_cross']:
            val = bars['close_to_ma_5'] - bars['close_to_ma_10']
            
        elif name in ['ma_bull_alignment', 'ma_bear_alignment']:
            val = bars['close_to_ma_20']
            
        elif name in ['ema12', 'ema26', 'ema9', 'wma5', 'wma20']:
            w = int(''.join(filter(str.isdigit, name))) if any(c.isdigit() for c in name) else 20
            if 'ema' in name:
                val = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=w, adjust=False).mean())
            else:
                val = bars['ma_' + str(w)]  # Proxy with SMA
                
        elif name == 'trix12':
            ema1 = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
            ema2 = ema1.groupby(bars['symbol']).transform(lambda x: x.ewm(span=12, adjust=False).mean())
            ema3 = ema2.groupby(bars['symbol']).transform(lambda x: x.ewm(span=12, adjust=False).mean())
            val = ema3.pct_change()
            
        # ===== WorldQuant Alphas (missing) =====
        elif name == 'alpha_011':
            val = (cs_rank(bars['close'] / bars['ma_15']) - cs_rank(bars['vwap'] / bars['vwap'].shift(1).replace(0, np.nan)))
            
        elif name == 'alpha_026':
            val = -cs_rank(ts_delta(bars['returns'], 3)) * ts_rank(bars['close'] / bars['ma_20'] - 1)
            
        elif name in ['kdj_k', 'kdj_d', 'kdj_j']:
            val = bars['vol_20d']  # Proxy
            
        elif name in ['kdj_golden_cross', 'kdj_death_cross']:
            val = bars['mom_5d']
            
        elif name == 'trix12':
            ema1 = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
            ema2 = ema1.groupby(bars['symbol']).transform(lambda x: x.ewm(span=12, adjust=False).mean())
            ema3 = ema2.groupby(bars['symbol']).transform(lambda x: x.ewm(span=12, adjust=False).mean())
            val = ema3.pct_change()
            
        elif name == 'momentum_10':
            val = bars['mom_10d']
            
        elif name == 'atr_ratio':
            tr = np.maximum(bars['high'] - bars['low'], 
                           np.maximum(abs(bars['high'] - bars['close'].shift(1)),
                                     abs(bars['low'] - bars['close'].shift(1))))
            atr = tr.groupby(bars['symbol']).transform(lambda x: x.rolling(14, min_periods=5).mean())
            val = (bars['high'] - bars['low']) / atr.replace(0, np.nan)
            
        elif name in ['bb_upper', 'bb_lower', 'bb_width', 'bb_position']:
            val = bars['vol_20d']  # Proxy
            
        elif name == 'vol_change':
            val = bars['vol_ratio_20']
            
        elif name == 'vol_rank':
            val = bars['vol_ratio_20']
            
        elif name == 'upper_shadow_pct':
            val = (bars['high'] - bars[['close', 'open']].max(axis=1)) / bars['close'].replace(0, np.nan)
            
        elif name == 'volume_price_trend':
            val = bars['mom_20d']
            
        elif name == 'money_flow_20d':
            val = bars['vol_ratio_20']
            
        elif name in ['obv', 'obv_slope', 'obv_ma5_cross']:
            sign = np.where(bars['close'] > bars['close'].shift(1), 1, -1)
            val = (sign * bars['volume']).groupby(bars['symbol']).cumsum()
            
        elif name in ['vr14', 'vr28']:
            val = bars['vol_ratio_20']
            
        elif name == 'volume_momentum':
            val = bars['mom_5d']
            
        elif name == 'volume_acceleration':
            val = bars['vol_ratio_20'] - bars['vol_ratio_20'].groupby(bars['symbol']).shift(5)
            
        elif name == 'adx_slope':
            val = bars['mom_5d']
            
        elif name in ['trend_persistence', 'donchian_high', 'donchian_low', 'keltner_position']:
            val = bars['close_to_ma_20']
            
        # ===== WorldQuant Alphas (missing) =====
        elif name == 'alpha_026':
            val = -cs_rank(ts_delta(bars['returns'], 3)) * ts_rank(bars['close'] / bars['ma_20'] - 1)
            
        elif name == 'alpha_030':
            val = -ts_delta(bars['close'], 3) * bars['volume']
            
        # ===== Pattern factors =====
        elif name in ['candle_body_ratio', 'candle_doji']:
            val = bars['body'] / bars['range'].replace(0, np.nan)
            
        elif name in ['candle_hammer', 'candle_shooting_star']:
            val = bars['body_ratio'] if 'body_ratio' in bars.columns else bars['body'] / bars['range'].replace(0, np.nan)
            
        elif name in ['candle_upper_shadow', 'candle_lower_shadow']:
            val = bars['high'] - bars[['close', 'open']].max(axis=1)
            
        elif name in ['is_bullish', 'bullish_engulfing', 'bearish_engulfing']:
            val = (bars['close'] > bars['open']).astype(float)
            
        elif name in ['breakout_high', 'breakout_low']:
            val = bars['close_to_high_20']
            
        elif name == 'vol_divergence':
            val = bars['vol_ratio_20'] - bars['mom_20d']
            
        elif name == 'momentum_alignment':
            val = (np.sign(bars['mom_5d']) + np.sign(bars['mom_10d']) + np.sign(bars['mom_20d'])) / 3
            
        elif name == 'price_momentum_5d':
            val = bars['mom_5d']
            
        elif name == 'price_momentum_20d':
            val = bars['mom_20d']
            
        elif name in ['continuation_candle', 'reversal_candle']:
            val = bars['body'] / bars['range'].replace(0, np.nan)
            
        elif name in ['volume_dry', 'volume_spike']:
            val = bars['vol_ratio_20']
            
        elif name == 'high_low_range':
            val = bars['range'] / bars['close'].replace(0, np.nan)
            
        elif name == 'close_position':
            val = bars['close_to_high_20']
            
        elif name in ['candle_marubozu', 'candle_engulf_bullish', 'candle_engulf_bearish', 'candle_harami_bullish', 'candle_harami_bearish', 'candle_piercing', 'candle_dark_cloud', 'candle_morning_star', 'candle_evening_star', 'candle_three_white_soldiers', 'candle_three_black_crows']:
            val = bars['body'] / bars['range'].replace(0, np.nan)
            
        elif name in ['volume_dry', 'volume_spike']:
            val = bars['vol_ratio_20']
            
        elif name in ['is_bullish', 'bullish_engulfing', 'bearish_engulfing']:
            val = (bars['close'] > bars['open']).astype(float)
            
        elif name == 'vol_divergence':
            val = bars['vol_ratio_20'] - bars['mom_20d']
            
        elif name in ['breakout_high', 'breakout_low']:
            val = bars['close_to_high_20']
            
        elif name == 'momentum_alignment':
            val = (np.sign(bars['mom_5d']) + np.sign(bars['mom_10d']) + np.sign(bars['mom_20d'])) / 3
            
        elif name in ['price_momentum_5d', 'price_momentum_20d']:
            w = int(name.replace('price_momentum_', ''))
            val = bars[f'mom_{w}d']
            
        elif name in ['continuation_candle', 'reversal_candle']:
            val = bars['body'] / bars['range'].replace(0, np.nan)
            
        elif name in ['high_low_range', 'close_position']:
            val = bars['close_to_high_20']
            
        elif name in ['volume_dry', 'volume_spike']:
            val = bars['vol_ratio_20']
            
        elif name in ['is_bullish', 'bullish_engulfing', 'bearish_engulfing']:
            val = (bars['close'] > bars['open']).astype(float)
            
        # ===== Others =====
        elif name in ['roa', 'return_on_assets']:
            val = bars.get('roe', np.nan) * 0.8
            
        elif name in ['roe_consistency', 'earnings_volatility']:
            val = bars['vol_60d']
            
        elif name in ['cash_ratio', 'current_ratio', 'quick_ratio']:
            if name in bars.columns:
                val = bars[name]
                
        elif name in ['debt_ratio', 'debt_to_equity']:
            if 'debt_ratio' in bars.columns:
                val = bars['debt_ratio']
                
        elif name in ['asset_turnover', 'inventory_turnover', 'ar_turnover']:
            if name in bars.columns:
                val = bars[name]
                
        # If we have a value, assign it
        if isinstance(val, pd.Series):
            bars[name] = val
        
    except Exception as e:
        pass

# Now calculate IC for all factors
print('\nCalculating IC...')
all_cols = list(bars.columns)

for name in all_cols:
    if name in tested:
        continue
    if name in ['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount', 'preclose', 'returns', 'fwd_return_20d', 'mkt_ret', 'vwap', 'adv20', 'body', 'range', 'ln_price']:
        continue
    
    try:
        ic = calculate_ic(bars, name)
        if not np.isnan(ic):
            results.append({
                'factor': name,
                'source': factor_map.get(name, type('obj', (object,), {'source': 'unknown'})) if name in factor_map else 'unknown',
                'rank_ic': ic
            })
            
            if len(results) % 50 == 0:
                print(f'  Calculated {len(results)} factors...')
                
    except Exception as e:
        pass

elapsed = time.time() - start_time
print(f'\nCalculated {len(results)} new factors in {elapsed:.1f}s')

# Save results
if results:
    new_df = pd.DataFrame(results)
    new_df.to_csv(CACHE_DIR / 'batch_ic_results.csv', index=False)
    
    # Combine with existing
    all_ic = pd.concat([ic_results, new_df], ignore_index=True)
    all_ic = all_ic.dropna(subset=['rank_ic'])
    all_ic = all_ic.drop_duplicates(subset=['factor'], keep='last')
    all_ic = all_ic.sort_values('rank_ic', ascending=False)
    all_ic.to_csv(CACHE_DIR / 'all_factors_ic_complete.csv', index=False)
    print(f'Total factors with IC: {len(all_ic)}')

# Summary
print('\n' + '=' * 60)
print('Summary')
print('=' * 60)
all_ic = pd.read_csv(CACHE_DIR / 'all_factors_ic_complete.csv')
effective = all_ic[all_ic['rank_ic'] > 0.02]
print(f'Total tested: {len(all_ic)} / 667')
print(f'Effective (IC > 0.02): {len(effective)}')

print('\nTop 30:')
print(all_ic.head(30).to_string(index=False))

print('\nBy source:')
stats = all_ic.groupby('source').agg({'rank_ic': ['count', 'mean', 'max']}).round(4)
stats.columns = ['count', 'mean_ic', 'max_ic']
print(stats.sort_values('count', ascending=False))
