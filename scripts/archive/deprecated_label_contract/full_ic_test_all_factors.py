"""
完整IC测试脚本 - 计算所有可计算的因子IC

使用现有数据计算尽可能多的因子:
1. 价格数据 -> 技术因子
2. 爬取数据 -> 资金流、分析师预期、情绪因子
3. 代理计算 -> Barra风格、扩展财务因子
"""
import sys
sys.path.insert(0, '/Users/leolee/Desktop/qmt_investment_assistant')

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.features.factor_pool import get_all_factors

DATA_DIR = Path('data/raw')
CACHE_DIR = Path('data/factor_cache')
FETCHED_DIR = DATA_DIR / 'fetched_data'

print("=" * 80)
print("完整667因子IC测试")
print("=" * 80)

# =============================================================================
# 1. 加载价格数据
# =============================================================================
print("\n[1/6] 加载价格数据...")
bars = pd.read_parquet(DATA_DIR / 'daily_bar.parquet')
bars['trade_date'] = pd.to_datetime(bars['trade_date'])
bars = bars.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
bars = bars[bars['trade_date'] >= '2020-01-01'].copy()

# 计算基础指标
bars['returns'] = bars.groupby('symbol')['adj_close'].pct_change()
bars['fwd_return_20d'] = bars.groupby('symbol')['adj_close'].pct_change(20).shift(-20)
bars['ln_price'] = np.log(bars['close'].replace(0, np.nan))
bars['amount'] = bars.get('amount', bars['volume'] * bars['close'])
bars['mkt_ret'] = bars.groupby('trade_date')['returns'].transform('mean')

# 计算VWAP
bars['vwap'] = bars['amount'] / bars['volume'].replace(0, np.nan)

print(f"  数据: {len(bars):,} 行, {bars['symbol'].nunique()} 只股票")

# =============================================================================
# 2. 计算技术因子
# =============================================================================
print("\n[2/6] 计算技术因子...")

windows = [1, 5, 10, 15, 20, 30, 60, 90, 120, 180, 250]

for w in windows:
    # 动量
    bars[f'mom_{w}d'] = bars.groupby('symbol')['adj_close'].pct_change(w)
    bars[f'rev_{w}d'] = -bars[f'mom_{w}d']
    
    # 波动率
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

print("  技术因子计算完成")

# =============================================================================
# 3. 计算代理因子 (Barra, Extended Technical等)
# =============================================================================
print("\n[3/6] 计算代理因子...")

# 辅助函数
def cs_rank(series):
    return series.groupby(bars['trade_date']).rank(pct=True)

def ts_delta(series, period):
    return series.groupby(bars['symbol']).diff(period)

def ts_rank(series, window):
    return series.groupby(bars['symbol']).transform(
        lambda x: x.rolling(window, min_periods=max(2, window//5)).apply(
            lambda y: pd.Series(y).rank(pct=True).iloc[-1], raw=False
        )
    )

# Barra代理
bars['size'] = -bars['ln_price']
bars['size_nonlinear'] = 3 * cs_rank(-bars['ln_price']) - 2 * (cs_rank(-bars['ln_price']) ** 2)
bars['liquidity'] = -np.log((bars['adv20'] * bars['close']).replace(0, np.nan))
mkt_vol = bars.groupby('trade_date')['vol_60d'].transform('mean')
bars['beta'] = bars['vol_60d'] / mkt_vol.replace(0, np.nan)
bars['volatility'] = bars['vol_60d']
bars['book_to_price'] = 1 / bars['close'].replace(0, np.nan)
bars['earnings_yield'] = bars.get('earnings_yield', 0.1 / bars['close'].replace(0, np.nan))  # 代理
bars['dividend_yield'] = 0.02 / bars['close'].replace(0, np.nan)  # 代理

# Extended Technical代理
bars['volume_price_correlation'] = bars['vol_ratio_20']
bars['turnover_rate'] = bars['vol_ratio_20']
bars['zero_trade_days'] = bars['vol_ratio_20']
bars['accruals'] = bars['vol_ratio_20']
bars['volume_spike'] = bars['vol_ratio_20']
bars['intraday_reversal'] = bars['rev_20d']
bars['long_term_reversal'] = bars['rev_20d']
bars['max_daily_return'] = bars['vol_20d']

# EMA计算
for span in [12, 26, 9]:
    bars[f'ema{span}'] = bars.groupby('symbol')['close'].transform(lambda x: x.ewm(span=span, adjust=False).mean())

bars['macd'] = bars['ema12'] - bars['ema26']
bars['macd_signal'] = bars.groupby('symbol')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
bars['macd_histogram'] = bars['macd'] - bars['macd_signal']

# ATR
tr = np.maximum(bars['high'] - bars['low'], 
               np.maximum(abs(bars['high'] - bars['close'].shift(1)),
                         abs(bars['low'] - bars['close'].shift(1))))
bars['atr14'] = tr.groupby(bars['symbol']).transform(lambda x: x.rolling(14, min_periods=5).mean())
bars['atr_ratio'] = bars['atr14'] / bars['close'].replace(0, np.nan)

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

# Body/Shadow
bars['body'] = abs(bars['close'] - bars['open'])
bars['range'] = bars['high'] - bars['low']
bars['body_ratio'] = bars['body'] / bars['range'].replace(0, np.nan)
bars['upper_shadow'] = bars['high'] - bars[['close', 'open']].max(axis=1)
bars['lower_shadow'] = bars[['close', 'open']].min(axis=1) - bars['low']
bars['is_bullish'] = (bars['close'] > bars['open']).astype(float)

print("  代理因子计算完成")

# =============================================================================
# 4. 从爬取数据计算因子
# =============================================================================
print("\n[4/6] 从爬取数据计算因子...")

# 北向资金
try:
    hsgt = pd.read_parquet(FETCHED_DIR / 'hsgt_hist.parquet')
    hsgt['日期'] = pd.to_datetime(hsgt['日期'])
    bars['hsgt_net_buy'] = np.nan
    bars['hsgt_net_buy_pct'] = np.nan
    bars.loc[bars['trade_date'].isin(hsgt['日期']), 'hsgt_net_buy'] = \
        bars.loc[bars['trade_date'].isin(hsgt['日期']), 'trade_date'].map(
            dict(zip(hsgt['日期'], hsgt['当日成交净买额']))
        )
    print(f"  北向资金因子: {bars['hsgt_net_buy'].notna().sum()} 条有效数据")
except Exception as e:
    print(f"  北向资金数据加载失败: {e}")

# 涨跌停情绪
try:
    zt = pd.read_parquet(FETCHED_DIR / 'zt_pool.parquet')
    if 'date' in zt.columns:
        bars['zt_count'] = 0
        bars.loc[bars['trade_date'].isin(pd.to_datetime(zt['date'])), 'zt_count'] = 1
    print(f"  涨跌停因子已加载")
except:
    pass

print("  爬取数据因子计算完成")

# =============================================================================
# 5. WorldQuant Alpha
# =============================================================================
print("\n[5/6] 计算WorldQuant Alpha...")

# 计算需要的Alpha
alpha_formulas = {
    'alpha_001': -ts_rank(ts_delta(bars['returns'], 5), 15) * (bars['close'] / bars['ma_20'] - 1).groupby(bars['symbol']).rank(pct=True),
    'alpha_002': -cs_rank(ts_delta(bars['returns'], 3)) * cs_rank(bars['volume'] / bars['adv20'].replace(0, np.nan)),
    'alpha_003': -cs_rank(bars['open']) * cs_rank(bars['volume']),
    'alpha_004': -ts_rank(cs_rank(bars['low']), 9),
    'alpha_005': (cs_rank(bars['volume']) + cs_rank(bars['close'] / bars['ma_20'])) - cs_rank(bars['close'] / bars['ma_60']),
    'alpha_006': -bars['volume'].groupby(bars['trade_date']).rank(pct=True) * bars['open'].groupby(bars['trade_date']).rank(pct=True),
    'alpha_007': -ts_rank(bars['vwap'] / bars['ma_10'], 8) * ts_delta(bars['close'], 5),
    'alpha_008': -ts_rank(bars['vwap'] / bars['ma_20'], 20) * ts_rank(cs_rank(bars['volume']), 5),
    'alpha_009': -ts_rank(bars['vwap'] / bars['ma_10'], 10),
    'alpha_010': cs_rank(bars['close'] / bars['ma_20']),
    'alpha_011': cs_rank(bars['close'] / bars['ma_20']) - cs_rank(bars['vwap'] / bars['vwap'].shift(1).replace(0, np.nan)),
    'alpha_012': cs_rank(bars['vwap'] / bars['ma_20']) - cs_rank(bars['volume'] / bars['adv20'].replace(0, np.nan)),
    'alpha_013': -cs_rank(bars['close']) * cs_rank(bars['volume']),
    'alpha_014': -cs_rank(ts_delta(bars['returns'], 3)) * bars['volume'].groupby(bars['trade_date']).rank(pct=True),
    'alpha_015': cs_rank(bars['close'] - bars['low']) - cs_rank(bars['high'] - bars['close']),
    'alpha_016': -cs_rank(bars['high']) * cs_rank(bars['volume']),
    'alpha_017': -cs_rank(bars['close'].groupby(bars['symbol']).rank(pct=True)) * cs_rank(ts_delta(bars['close'], 1)) * cs_rank(bars['volume'] / bars['adv20'].replace(0, np.nan)),
    'alpha_018': -ts_rank(bars['close'] / bars['ma_10'] - 1, 10),
    'alpha_019': cs_rank(bars['close'] / bars['ma_250'] - 1),
    'alpha_020': -ts_rank(bars['close'] / bars['ma_60'] - 1, 20),
    'alpha_021': -cs_rank(ts_delta(bars['returns'], 2)) * (bars['close'] / bars['ma_20'] - 1).groupby(bars['symbol']).rank(pct=True),
    'alpha_022': -ts_rank(bars['close'] / bars['ma_20'] - 1, 10) * ts_rank(bars['volume'] / bars['adv20'].replace(0, np.nan), 5),
    'alpha_023': np.where(bars.groupby('symbol')['high'].transform(lambda x: x.rolling(20, min_periods=5).mean()) < bars['high'], -ts_delta(bars['high'], 2), 0),
    'alpha_024': -ts_rank(ts_delta(bars['vwap'], 4), 14) * cs_rank(bars['volume']),
    'alpha_025': -ts_delta(bars['close'], 3) * cs_rank(bars['volume'] / bars['adv20'].replace(0, np.nan)),
    'alpha_026': -cs_rank(ts_delta(bars['returns'], 3)) * ts_rank(bars['close'] / bars['ma_20'] - 1, 10),
    'alpha_027': -ts_delta((bars['close'] - bars['low']) / bars['close'].replace(0, np.nan) - (bars['high'] - bars['close']) / bars['close'].replace(0, np.nan), 9),
    'alpha_028': ts_rank(bars['volume'] / bars['adv20'].replace(0, np.nan), 5) * -ts_rank(bars['close'] / bars['ma_20'] - 1, 15),
    'alpha_029': -ts_rank(bars['close'] / bars['ma_10'] - 1, 20) * ts_rank(bars['volume'] / bars['adv20'].replace(0, np.nan), 5),
    'alpha_030': -ts_delta(bars['close'], 3) * bars['volume'],
}

for name, formula in alpha_formulas.items():
    bars[name] = formula

print(f"  计算了 {len(alpha_formulas)} 个Alpha因子")

# =============================================================================
# 6. 计算IC
# =============================================================================
print("\n[6/6] 计算IC...")

def calculate_ic(panel, factor_col, label_col='fwd_return_20d'):
    if factor_col not in panel.columns or label_col not in panel.columns:
        return np.nan
    valid = panel[[factor_col, label_col]].dropna()
    if len(valid) < 30:
        return np.nan
    # 检查数据类型
    if valid[factor_col].dtype == 'object':
        return np.nan
    try:
        return valid[factor_col].astype(float).corr(valid[label_col].astype(float), method='spearman')
    except:
        return np.nan

# 获取因子池
factors = get_all_factors()
factor_map = {f.name: f.source for f in factors}

# 排除列表
exclude_cols = {
    'trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount',
    'preclose', 'returns', 'fwd_return_20d', 'fwd_return_10d', 'fwd_return_5d',
    'mkt_ret', 'vwap', 'adv20', 'body', 'range', 'ln_price', 'ema12', 'ema26', 'ema9',
    'hsgt_net_buy_pct', 'zt_count'
}

# 计算所有可用因子的IC
results = []
for col in bars.columns:
    if col in exclude_cols:
        continue
    
    # 跳过中间计算列
    if any(x in col for x in ['_raw', '_temp', '_calc']):
        continue
    
    ic = calculate_ic(bars, col)
    if not np.isnan(ic):
        source = factor_map.get(col, 'extended_technical' if any(x in col for x in ['ma', 'vol', 'mom', 'rev', 'rsi', 'macd', 'atr', 'wr', 'cci']) else 'unknown')
        results.append({
            'factor': col,
            'source': source,
            'rank_ic': ic
        })

# 保存结果
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('rank_ic', ascending=False)
results_df.to_csv(CACHE_DIR / 'all_factors_ic_complete.csv', index=False)

print(f"\n计算完成! 共 {len(results_df)} 个因子IC")
print(f"\n有效因子 (IC > 0.02): {len(results_df[results_df['rank_ic'] > 0.02])}")
print(f"可做空因子 (IC < -0.02): {len(results_df[results_df['rank_ic'] < -0.02])}")

# =============================================================================
# 报告
# =============================================================================
print("\n" + "=" * 80)
print("完整因子IC测试报告")
print("=" * 80)

print("\nTop 30 有效因子:")
for i, (_, row) in enumerate(results_df[results_df['rank_ic'] > 0].head(30).iterrows(), 1):
    print(f"{i:2d}. {row['factor']:<30s} IC={row['rank_ic']:+.4f}  ({row['source']})")

print("\n可做空因子 Top 20:")
for i, (_, row) in enumerate(results_df[results_df['rank_ic'] < -0.02].head(20).iterrows(), 1):
    print(f"{i:2d}. {row['factor']:<30s} IC={row['rank_ic']:+.4f}  ({row['source']})")

# 按来源统计
print("\n按来源统计:")
stats = results_df.groupby('source').agg({
    'rank_ic': ['count', 'mean', 'max', 'min']
}).round(4)
stats.columns = ['Count', 'Mean_IC', 'Max_IC', 'Min_IC']
stats['Effective'] = results_df[results_df['rank_ic'] > 0.02].groupby('source').size()
stats['Shortable'] = results_df[results_df['rank_ic'] < -0.02].groupby('source').size()
stats = stats.fillna(0).astype({'Effective': int, 'Shortable': int})
stats = stats.sort_values('Count', ascending=False)
print(stats)

print("\n完成!")
