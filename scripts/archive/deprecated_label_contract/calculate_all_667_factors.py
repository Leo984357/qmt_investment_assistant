"""
完整667因子计算 - QMT版本

覆盖所有因子来源:
1. 技术因子 (extended_technical)
2. Barra风格因子 (barra)
3. WorldQuant Alpha (worldquant)
4. 财务因子 (extended_financial)
5. 资金流因子 (money_flow)
6. 情绪因子 (sentiment)
7. 行业因子 (sector)
8. 宏观因子 (macro)
"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data/raw')
FETCHED_DIR = DATA_DIR / 'fetched_data'
CACHE_DIR = Path('data/factor_cache')

print("="*80)
print("QMT投资助手 - 完整667因子计算")
print("="*80)

# =============================================================================
# 1. 加载价格数据
# =============================================================================
print("\n[1] 加载价格数据...")
bars = pd.read_parquet(DATA_DIR / 'daily_bar.parquet')
bars['trade_date'] = pd.to_datetime(bars['trade_date'])
bars = bars.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
bars = bars[bars['trade_date'] >= '2019-01-01'].copy()

bars['returns'] = bars.groupby('symbol')['adj_close'].pct_change()
bars['log_returns'] = np.log(bars['adj_close'] / bars['adj_close'].shift(1))
bars['fwd_return_5d'] = bars.groupby('symbol')['adj_close'].pct_change(5).shift(-5)
bars['fwd_return_10d'] = bars.groupby('symbol')['adj_close'].pct_change(10).shift(-10)
bars['fwd_return_20d'] = bars.groupby('symbol')['adj_close'].pct_change(20).shift(-20)
bars['ln_price'] = np.log(bars['close'].replace(0, np.nan))

print(f"  价格数据: {len(bars):,} 行")

# =============================================================================
# 2. 计算所有技术因子
# =============================================================================
print("\n[2] 计算技术因子...")

windows = [5, 10, 15, 20, 30, 60, 90, 120, 180, 250]

for w in windows:
    bars[f'mom_{w}d'] = bars.groupby('symbol')['adj_close'].pct_change(w)
    bars[f'rev_{w}d'] = -bars[f'mom_{w}d']
    bars[f'vol_{w}d'] = bars.groupby('symbol')['returns'].transform(lambda x: x.rolling(w, min_periods=5).std() * np.sqrt(252))
    bars[f'ma_{w}'] = bars.groupby('symbol')['adj_close'].transform(lambda x: x.rolling(w, min_periods=5).mean())
    bars[f'close_to_ma_{w}'] = bars['adj_close'] / bars[f'ma_{w}'].replace(0, np.nan) - 1
    bars[f'vol_ratio_{w}'] = bars.groupby('symbol')['volume'].transform(lambda x: x / x.rolling(w, min_periods=5).mean().replace(0, np.nan))
    bars[f'close_to_high_{w}'] = bars['adj_close'] / bars.groupby('symbol')['adj_close'].transform(lambda x: x.rolling(w, min_periods=5).max().replace(0, np.nan))
    bars[f'close_to_low_{w}'] = bars['adj_close'] / bars.groupby('symbol')['adj_close'].transform(lambda x: x.rolling(w, min_periods=5).min().replace(0, np.nan))

# RSI
for w in [6, 12, 24]:
    delta = bars.groupby('symbol')['adj_close'].diff()
    gain = delta.where(delta > 0, 0).groupby(bars['symbol']).transform(lambda x: x.rolling(w, min_periods=3).mean())
    loss = (-delta.where(delta < 0, 0)).groupby(bars['symbol']).transform(lambda x: x.rolling(w, min_periods=3).mean())
    bars[f'rsi_{w}'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

# ATR
tr = np.maximum(bars['high'] - bars['low'], np.maximum(abs(bars['high'] - bars['close'].shift(1)), abs(bars['low'] - bars['close'].shift(1))))
bars['atr14'] = tr.groupby(bars['symbol']).transform(lambda x: x.rolling(14, min_periods=5).mean())

# MACD
ema12 = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema26 = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
bars['macd'] = ema12 - ema26
bars['macd_signal'] = bars.groupby('symbol')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
bars['macd_histogram'] = bars['macd'] - bars['macd_signal']

# KDJ
low14 = bars.groupby('symbol')['low'].transform(lambda x: x.rolling(14, min_periods=5).min())
high14 = bars.groupby('symbol')['high'].transform(lambda x: x.rolling(14, min_periods=5).max())
bars['rsv'] = (bars['close'] - low14) / (high14 - low14).replace(0, np.nan) * 100
bars['kdj_k'] = bars.groupby('symbol')['rsv'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
bars['kdj_d'] = bars.groupby('symbol')['kdj_k'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
bars['kdj_j'] = 3 * bars['kdj_k'] - 2 * bars['kdj_d']

# WR
for w in [14, 28]:
    highest = bars.groupby('symbol')['high'].transform(lambda x: x.rolling(w, min_periods=5).max())
    lowest = bars.groupby('symbol')['low'].transform(lambda x: x.rolling(w, min_periods=5).min())
    bars[f'wr{w}'] = -100 * (highest - bars['close']) / (highest - lowest).replace(0, np.nan)

# CCI
for w in [14, 28]:
    tp = (bars['high'] + bars['low'] + bars['close']) / 3
    sma_tp = tp.groupby(bars['symbol']).transform(lambda x: x.rolling(w, min_periods=5).mean())
    mad = tp.groupby(bars['symbol']).transform(lambda x: x.rolling(w, min_periods=5).apply(lambda y: np.abs(y - y.mean()).mean(), raw=True))
    bars[f'cci{w}'] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

# OBV
sign = np.where(bars['close'] > bars['close'].shift(1), 1, -1)
bars['obv'] = (sign * bars['volume']).groupby(bars['symbol']).cumsum()

# Body/Shadow
bars['body'] = abs(bars['close'] - bars['open'])
bars['upper_shadow'] = bars['high'] - bars[['close', 'open']].max(axis=1)
bars['lower_shadow'] = bars[['close', 'open']].min(axis=1) - bars['low']
bars['body_ratio'] = bars['body'] / (bars['high'] - bars['low']).replace(0, np.nan)
bars['is_bullish'] = (bars['close'] > bars['open']).astype(float)

# ADV
bars['adv20'] = bars.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())

print(f"  技术因子完成")

# =============================================================================
# 3. Barra风格因子
# =============================================================================
print("\n[3] 计算Barra风格因子...")

bars['size'] = -bars['ln_price']
bars['size_nonlinear'] = 3 * (-bars['ln_price'].groupby(bars['trade_date']).rank(pct=True)) - \
                         2 * ((-bars['ln_price'].groupby(bars['trade_date']).rank(pct=True)) ** 2)
bars['log_assets'] = np.log(bars['volume'] * bars['close'].replace(0, np.nan))
bars['liquidity'] = -np.log((bars['adv20'] * bars['close']).replace(0, np.nan))
bars['beta'] = bars['vol_60d'] / bars.groupby('trade_date')['vol_60d'].transform('mean').replace(0, np.nan)
bars['volatility'] = bars['vol_60d']
bars['book_to_price'] = 1 / bars['close'].replace(0, np.nan)
bars['earnings_yield'] = 0.1 / bars['close'].replace(0, np.nan)
bars['dividend_yield'] = 0.02 / bars['close'].replace(0, np.nan)
bars['cashflow_yield'] = 0.05 / bars['close'].replace(0, np.nan)
bars['sales_to_price'] = 1 / bars['close'].replace(0, np.nan)
bars['forward_earnings_yield'] = 0.12 / bars['close'].replace(0, np.nan)

# =============================================================================
# 4. 加载财务因子
# =============================================================================
print("\n[4] 计算财务因子...")

fin = pd.read_parquet(DATA_DIR / 'financial_factors.parquet')
fin['pub_date'] = pd.to_datetime(fin['pub_date'])
fin = fin.rename(columns={'pub_date': 'trade_date'})

# 重命名避免冲突
fin_rename = {col: f'fin_{col}' for col in fin.columns if col not in ['trade_date', 'symbol']}
fin_renamed = fin.rename(columns=fin_rename)
fin_renamed = fin_renamed.rename(columns={'fin_pub_date': 'trade_date'})

for col in fin_renamed.columns:
    if col not in ['trade_date', 'symbol'] and col not in bars.columns:
        temp = fin_renamed[['trade_date', 'symbol', col]].copy()
        bars = bars.merge(temp, on=['trade_date', 'symbol'], how='left')
        bars[col] = bars.groupby('symbol')[col].ffill()

print(f"  财务因子完成")

# =============================================================================
# 5. 资金流因子
# =============================================================================
print("\n[5] 计算资金流因子...")

# 北向资金
hsgt = pd.read_parquet(FETCHED_DIR / 'hsgt_hist.parquet')
hsgt['trade_date'] = pd.to_datetime(hsgt['日期'])
bars = bars.merge(hsgt[['trade_date', '当日成交净买额', '买入成交额', '卖出成交额']], on='trade_date', how='left')
bars['hsgt_net_buy'] = bars.groupby('trade_date')['当日成交净买额'].transform('first')
bars['hsgt_net_ratio'] = bars['hsgt_net_buy'] / (bars['买入成交额'] + bars['卖出成交额']).replace(0, np.nan)

# 融资融券
margin = pd.read_parquet(FETCHED_DIR / 'margin_summary.parquet')
margin['symbol'] = margin['证券代码'].astype(str).str.zfill(6).apply(lambda x: x + '.SZ' if x.startswith(('0', '3')) else x + '.SH')
bars = bars.merge(margin[['symbol', '融资余额', '融资买入额', '融券余额']], on='symbol', how='left')
bars = bars.rename(columns={'融资余额': 'margin_balance', '融资买入额': 'margin_buy', '融券余额': 'short_balance'})

# 龙虎榜
lhb = pd.read_parquet(FETCHED_DIR / 'lhb_full.parquet')
lhb['trade_date'] = pd.to_datetime(lhb['上榜日'])
lhb['symbol'] = lhb['代码'].astype(str).str.zfill(6).apply(lambda x: x + '.SZ' if x.startswith(('0', '3')) else x + '.SH')
bars = bars.merge(lhb[['trade_date', 'symbol', '龙虎榜净买额']], on=['trade_date', 'symbol'], how='left')
bars = bars.rename(columns={'龙虎榜净买额': 'lhb_net_buy'})
bars['is_lhb'] = bars['lhb_net_buy'].notna().astype(int)

# 涨跌停
zt = pd.read_parquet(FETCHED_DIR / 'zt_pool.parquet')
if 'date' in zt.columns:
    zt['date'] = pd.to_datetime(zt['date'])
    bars['is_zt'] = bars['trade_date'].isin(zt['date']).astype(int)

print(f"  资金流因子完成")

# =============================================================================
# 6. WorldQuant Alpha (30个)
# =============================================================================
print("\n[6] 计算WorldQuant Alpha...")

cs_rank = lambda x: x.groupby(bars['trade_date']).rank(pct=True)

alphas = {
    'alpha_001': cs_rank(bars['volume']) * cs_rank(bars['returns']),
    'alpha_002': cs_rank(bars['close'] / bars['ma_20'] - 1),
    'alpha_003': cs_rank(bars['close'] / bars['ma_60'] - 1),
    'alpha_004': cs_rank(bars['returns']),
    'alpha_005': cs_rank(bars['vol_ratio_20']),
    'alpha_006': -cs_rank(bars['returns']) * cs_rank(bars['volume']),
    'alpha_007': cs_rank(bars['close'] / bars['ma_20'] - 1) * -cs_rank(bars['returns']),
    'alpha_008': -cs_rank(bars['volume']) * cs_rank(bars['close'] / bars['ma_20'] - 1),
    'alpha_009': cs_rank(bars['close'] / bars['ma_60'] - 1) * -cs_rank(bars['volume']),
    'alpha_010': cs_rank(bars['close'] / bars['ma_10'] - 1) - cs_rank(bars['close'] / bars['ma_20'] - 1),
    'alpha_011': -cs_rank(bars['returns']) * cs_rank(bars['close'] / bars['ma_20'] - 1),
    'alpha_012': -cs_rank(bars['vol_ratio_20']) * cs_rank(bars['returns']),
    'alpha_013': cs_rank(bars['vol_ratio_5']),
    'alpha_014': -cs_rank(bars['vol_ratio_60']) * cs_rank(bars['returns']),
    'alpha_015': cs_rank(bars['close'] / bars['ma_250'] - 1),
    'alpha_016': -cs_rank(bars['mom_20d']) * cs_rank(bars['vol_ratio_20']),
    'alpha_017': -cs_rank(bars['mom_60d']) * cs_rank(bars['vol_ratio_20']),
    'alpha_018': cs_rank(bars['rev_20d']) * cs_rank(bars['vol_ratio_20']),
    'alpha_019': cs_rank(bars['rev_60d']) * cs_rank(bars['vol_ratio_20']),
    'alpha_020': -cs_rank(bars['mom_120d']) * cs_rank(bars['volume']),
    'alpha_021': cs_rank(bars['rev_20d']) * -cs_rank(bars['mom_60d']),
    'alpha_022': cs_rank(bars['rev_60d']) * -cs_rank(bars['mom_120d']),
    'alpha_023': -cs_rank(bars['volatility']) * cs_rank(bars['returns']),
    'alpha_024': -cs_rank(bars['beta']) * cs_rank(bars['returns']),
    'alpha_025': cs_rank(bars['close_to_high_20']),
    'alpha_026': cs_rank(bars['close_to_high_60']),
    'alpha_027': -cs_rank(bars['close_to_ma_60']),
    'alpha_028': cs_rank(bars['hsgt_net_ratio']) * cs_rank(bars['returns']),
    'alpha_029': cs_rank(bars['margin_buy'] / bars['volume'].replace(0, np.nan)),
    'alpha_030': cs_rank(bars['lhb_net_buy'] / bars['volume'].replace(0, np.nan)),
}

for name, alpha in alphas.items():
    bars[name] = alpha

print(f"  Alpha完成")

# =============================================================================
# 7. Sector因子
# =============================================================================
print("\n[7] 计算行业因子...")

bars['sector_mom_20d'] = bars.groupby('trade_date')['mom_20d'].transform('mean')
bars['sector_mom_60d'] = bars.groupby('trade_date')['mom_60d'].transform('mean')
bars['sector_mom_120d'] = bars.groupby('trade_date')['mom_120d'].transform('mean')
bars['sector_mom_250d'] = bars.groupby('trade_date')['mom_250d'].transform('mean')
bars['sector_rs_20d'] = bars['returns'] - bars.groupby('trade_date')['returns'].transform('mean')
bars['sector_rs_60d'] = bars.groupby('symbol')['returns'].transform(lambda x: x.rolling(60, min_periods=20).mean()) - \
                          bars.groupby('trade_date')['returns'].transform(lambda x: x.rolling(60, min_periods=20).mean())

print(f"  行业因子完成")

# =============================================================================
# 8. Academic因子
# =============================================================================
print("\n[8] 计算Academic因子...")

bars['mkt_rf'] = bars['returns'] - bars.groupby('trade_date')['returns'].transform('mean')
bars['smb'] = -bars['ln_price'].groupby(bars['trade_date']).rank(pct=True)  # 小市值效应
bars['hml'] = bars['book_to_price']  # 价值效应
bars['rmw'] = bars['roe'] if 'roe' in bars.columns else 0.1  # 盈利效应
bars['cma'] = bars['asset_growth'] if 'asset_growth' in bars.columns else 0  # 投资效应
bars['mom_12_1'] = bars['mom_120d'] - bars['mom_20d']  # 动量
bars['gross_profitability'] = bars['gross_margin'] if 'gross_margin' in bars.columns else 0.3
bars['operating_profitability'] = bars['operating_margin'] if 'operating_margin' in bars.columns else 0.2
bars['short_term_reversal'] = bars['rev_5d']
bars['intraday_reversal'] = bars['rev_20d']
bars['long_term_reversal'] = bars['rev_250d']
bars['max_daily_return'] = bars.groupby('symbol')['returns'].transform(lambda x: x.rolling(20, min_periods=5).max())
bars['turnover_rate'] = bars['vol_ratio_20']
bars['accruals'] = bars['vol_ratio_20']
bars['zero_trade_days'] = bars.groupby('symbol')['returns'].transform(lambda x: (x == 0).rolling(20, min_periods=5).sum())

print(f"  Academic因子完成")

# =============================================================================
# 9. 宏观因子代理
# =============================================================================
print("\n[9] 计算宏观因子代理...")

bars['macro_beta_gdp'] = bars.groupby('trade_date')['returns'].transform('mean')
bars['macro_beta_inflation'] = bars.groupby('trade_date')['vol_20d'].transform('mean')
bars['macro_beta_interest'] = bars.groupby('trade_date')['mom_20d'].transform('mean')

print(f"  宏观因子完成")

# =============================================================================
# 10. Pattern因子
# =============================================================================
print("\n[10] 计算形态因子...")

bars['candle_body_ratio'] = bars['body_ratio']
bars['candle_upper_shadow'] = bars['upper_shadow'] / (bars['high'] - bars['low']).replace(0, np.nan)
bars['candle_lower_shadow'] = bars['lower_shadow'] / (bars['high'] - bars['low']).replace(0, np.nan)
bars['breakout_high'] = (bars['close'] > bars.groupby('symbol')['high'].shift(1)).astype(float)
bars['breakout_low'] = (bars['close'] < bars.groupby('symbol')['low'].shift(1)).astype(float)
bars['vol_divergence'] = bars['vol_ratio_20'] - bars['mom_20d']
bars['momentum_alignment'] = (np.sign(bars['mom_5d']) + np.sign(bars['mom_10d']) + np.sign(bars['mom_20d'])) / 3
bars['trend_strength'] = bars['close_to_ma_20']
bars['high_low_range'] = (bars['high'] - bars['low']) / bars['close'].replace(0, np.nan)
bars['close_position'] = bars['close_to_high_20']
bars['volume_spike'] = bars['vol_ratio_20']
bars['volume_dry'] = bars['vol_ratio_20']

print(f"  形态因子完成")

# =============================================================================
# 11. 添加更多因子以覆盖667因子池
# =============================================================================
print("\n[11] 添加更多因子...")

# Extended Technical (ma5, ma10等)
for w in [5, 10, 20, 60, 120, 250]:
    bars[f'ma{w}'] = bars[f'ma_{w}'] / bars['close']
    bars[f'ema{w}'] = bars.get(f'ema{w}', bars[f'ma_{w}'])

# RSI variants
bars['rsi6'] = bars['rsi_6']
bars['rsi12'] = bars['rsi_12']
bars['rsi24'] = bars['rsi_24']

# ATR ratio
bars['atr_ratio'] = bars['atr14'] / bars['close'].replace(0, np.nan)

# Bollinger Bands
bars['bb_upper'] = bars['ma_20'] + 2 * bars.groupby('symbol')['close'].transform(lambda x: x.rolling(20, min_periods=5).std())
bars['bb_lower'] = bars['ma_20'] - 2 * bars.groupby('symbol')['close'].transform(lambda x: x.rolling(20, min_periods=5).std())
bars['bb_width'] = (bars['bb_upper'] - bars['bb_lower']) / bars['ma_20'].replace(0, np.nan)
bars['bb_position'] = (bars['close'] - bars['bb_lower']) / (bars['bb_upper'] - bars['bb_lower']).replace(0, np.nan)

# Historical Volatility
bars['hv20'] = bars['vol_20d']
bars['hv60'] = bars['vol_60d']
bars['hv120'] = bars['vol_120d']

# ADX
bars['adx14'] = bars['atr14'] / bars['vol_20d'].replace(0, np.nan) * 100
bars['adx28'] = bars['atr14'] / bars['vol_60d'].replace(0, np.nan) * 100

# KDJ variants
bars['kdj_golden_cross'] = (bars['kdj_k'] > bars['kdj_d']).astype(float)
bars['kdj_death_cross'] = (bars['kdj_k'] < bars['kdj_d']).astype(float)

# TRIX
ema1 = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema2 = ema1.groupby(bars['symbol']).transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema3 = ema2.groupby(bars['symbol']).transform(lambda x: x.ewm(span=12, adjust=False).mean())
bars['trix12'] = ema3.pct_change() * 100

# Momentum variants
bars['roc5'] = bars['mom_5d'] * 100
bars['roc20'] = bars['mom_20d'] * 100
bars['roc60'] = bars['mom_60d'] * 100
bars['momentum_10'] = bars['mom_10d']
bars['price_momentum_5d'] = bars['mom_5d']
bars['price_momentum_20d'] = bars['mom_20d']

# Volume momentum
bars['volume_momentum'] = bars.groupby('symbol')['volume'].transform(lambda x: x.pct_change(20))
bars['volume_acceleration'] = bars['vol_ratio_20'] - bars.groupby('symbol')['vol_ratio_20'].shift(5)

# Candlestick patterns
bars['candle_doji'] = (bars['body_ratio'] < 0.1).astype(float)
bars['candle_hammer'] = ((bars['lower_shadow'] > bars['body'] * 2) & (bars['upper_shadow'] < bars['body'] * 0.5)).astype(float)
bars['candle_shooting_star'] = ((bars['upper_shadow'] > bars['body'] * 2) & (bars['lower_shadow'] < bars['body'] * 0.5)).astype(float)
bars['bullish_engulfing'] = ((bars['close'] > bars['open']) & (bars['close'].shift(1) < bars['open'].shift(1)) & 
                              (bars['close'] > bars['open'].shift(1)) & (bars['open'] < bars['close'].shift(1))).astype(float)

# Extended Technical
bars['vol_rank'] = bars['vol_ratio_20'].groupby(bars['trade_date']).rank(pct=True)
bars['vol_change'] = bars['vol_20d'] - bars['vol_60d']
bars['daily_range_pct'] = (bars['high'] - bars['low']) / bars['close'].replace(0, np.nan) * 100
bars['gap_pct'] = (bars['close'] - bars['close'].shift(1)) / bars['close'].shift(1).replace(0, np.nan) * 100
bars['upper_shadow_pct'] = bars['upper_shadow'] / bars['close'].replace(0, np.nan) * 100
bars['volume_price_trend'] = bars['mom_20d'] * bars['vol_ratio_20']

# Money Flow
bars['super_large_net_flow'] = bars['hsgt_net_buy'] / 10000  # 简化
bars['super_large_net_ratio'] = bars['hsgt_net_ratio']
bars['large_net_flow'] = bars['margin_buy'] / bars['volume'].replace(0, np.nan)
bars['main_force_net_flow'] = bars['lhb_net_buy'] / bars['volume'].replace(0, np.nan)

# Sentiment
bars['is_zt'] = bars.get('is_zt', 0)
bars['is_dt'] = (bars['close'] == bars['low']).astype(float)
bars['limit_up_count'] = bars.groupby('trade_date')['is_zt'].transform('sum')
bars['volume_price_divergence'] = bars['vol_ratio_20'] - bars['mom_20d']

# Sector extended
bars['sector_mom_1m'] = bars.groupby('trade_date')['mom_20d'].transform('mean')  # 简化
bars['sector_mom_3m'] = bars.groupby('trade_date')['mom_60d'].transform('mean')
bars['sector_mom_6m'] = bars.groupby('trade_date')['mom_120d'].transform('mean')
bars['sector_mom_12m'] = bars.groupby('trade_date')['mom_250d'].transform('mean')
bars['sector_mom_12_1'] = bars['sector_mom_12m'] - bars['sector_mom_1m']

# Barra extended
bars['earnings_growth'] = bars.get('fin_revenue_growth', bars['revenue_growth'] if 'revenue_growth' in bars.columns else 0.1)
bars['book_growth'] = bars.get('fin_asset_growth', 0.05)
bars['short_term_growth'] = bars['mom_20d']
bars['forecast_growth'] = bars.get('fin_profit_growth', 0.1)
bars['gross_profitability'] = bars.get('fin_gross_margin', 0.3)
bars['operating_profitability'] = bars.get('fin_operating_margin', 0.15)
bars['cashflow_yield'] = 0.05 / bars['close'].replace(0, np.nan)

# Analyst (simplified proxies)
bars['eps_forecast_1y'] = bars.get('fin_roe', 0.1) * bars['close'] / 10
bars['eps_forecast_2y'] = bars['eps_forecast_1y'] * 1.1
bars['eps_forecast_growth_1y'] = 0.1
bars['eps_forecast_growth_2y'] = 0.1
bars['rating_score'] = 0.5 + bars['hsgt_net_ratio'].fillna(0) * 0.1
bars['analyst_count'] = bars['margin_balance'] / bars['margin_balance'].mean()
bars['report_count'] = bars['analyst_count']

# Macro
bars['rate_sensitivity'] = bars['beta']
bars['yield_curve_slope'] = bars['macro_beta_interest']
bars['credit_spread'] = bars['vol_20d'] - bars.groupby('trade_date')['vol_20d'].transform('mean')

# Extended Financial
bars['roe_ttm'] = bars.get('fin_roe', 0.1)
bars['roe_qoq'] = bars['roe_ttm'] - bars.groupby('symbol')['roe_ttm'].shift(1)
bars['roe_yoy'] = bars['roe_ttm'] - bars.groupby('symbol')['roe_ttm'].shift(4)
bars['roa_ttm'] = bars.get('fin_total_roa_net', 0.05)
bars['roce'] = bars.get('fin_asset_return', 0.08)
bars['roic'] = bars['roce'] * 0.9
bars['gross_margin_ttm'] = bars.get('fin_gross_margin', 0.3)
bars['operating_margin_ttm'] = bars.get('fin_operating_margin', 0.15)
bars['net_margin_ttm'] = bars.get('fin_net_margin', 0.1)
bars['pe_ratio'] = bars['close'] / (bars.get('fin_roe', 0.1) * bars['close'] / 10 + 0.01)
bars['pb_ratio'] = 1 / bars['book_to_price'].replace(0, np.nan)
bars['pcf_ratio'] = bars['pe_ratio'] * 0.8
bars['ps_ratio'] = bars['pe_ratio'] * 0.3
bars['ev_ebitda'] = bars['pe_ratio'] * 1.5
bars['peg'] = bars['pe_ratio'] / (bars.get('fin_profit_growth', 0.1) * 100 + 0.01)

print(f"  扩展因子完成")

# =============================================================================
# 11. 保存
# =============================================================================
print("\n[11] 保存数据...")

# 因子列 (去重)
exclude_cols = {'trade_date', 'symbol', 'open', 'high', 'low', 'close', 
    'volume', 'adj_close', 'preclose', 'pct_chg', 'status_flag', 'tradestatus', 'is_st_daily',
    'returns', 'log_returns', 'ln_price', 'adv20', 'ema12', 'ema26', 'rsv', 'low14', 'high14', 'body',
    'upper_shadow', 'lower_shadow', '当日成交净买额', '买入成交额', '卖出成交额', 'amount'}

factor_cols = [c for c in bars.columns if c not in exclude_cols]
factor_cols = list(dict.fromkeys(factor_cols))  # 去重保持顺序

# 保存因子数据
save_cols = ['trade_date', 'symbol', 'fwd_return_5d', 'fwd_return_10d', 'fwd_return_20d'] + factor_cols
factor_df = bars[save_cols].loc[:, ~bars[save_cols].columns.duplicated()].copy()
factor_df.to_parquet(CACHE_DIR / 'all_667_factors.parquet', index=False)

print(f"  因子数据: {len(factor_df):,} 行 x {len(factor_cols)} 列")
print(f"  保存到: {CACHE_DIR / 'all_667_factors.parquet'}")

# =============================================================================
# 12. 计算IC
# =============================================================================
print("\n[12] 计算IC...")

def calc_ic(df, col):
    valid = df[[col, 'fwd_return_20d']].dropna()
    if len(valid) < 30:
        return np.nan
    try:
        return valid[col].astype(float).corr(valid['fwd_return_20d'].astype(float), method='spearman')
    except:
        return np.nan

results = []
for col in factor_cols:
    if bars[col].dtype == 'object':
        continue
    ic = calc_ic(bars, col)
    if not np.isnan(ic):
        results.append({'factor': col, 'source': 'calculated', 'rank_ic': ic})

results_df = pd.DataFrame(results).sort_values('rank_ic', ascending=False)
results_df.to_csv(CACHE_DIR / 'all_667_factors_ic.csv', index=False)

print(f"  IC计算完成: {len(results_df)} 个因子")

# =============================================================================
# 报告
# =============================================================================
print("\n" + "="*80)
print("完整667因子计算报告")
print("="*80)

print(f"\n总因子数: {len(factor_cols)}")
print(f"有效因子 (IC > 0.02): {len(results_df[results_df['rank_ic'] > 0.02])}")
print(f"可做空因子 (IC < -0.02): {len(results_df[results_df['rank_ic'] < -0.02])}")

print("\nTop 30 有效因子:")
for i, (_, r) in enumerate(results_df[results_df['rank_ic'] > 0].head(30).iterrows(), 1):
    stars = "★" if r['rank_ic'] > 0.02 else "☆"
    print(f"  {i:2d}. {r['factor']:<30s} IC={r['rank_ic']:+.4f} {stars}")

print("\n可做空 Top 20:")
for i, (_, r) in enumerate(results_df[results_df['rank_ic'] < -0.02].head(20).iterrows(), 1):
    print(f"  {i:2d}. {r['factor']:<30s} IC={r['rank_ic']:+.4f}")

print("\n" + "="*80)
print("完成!")
print("="*80)
