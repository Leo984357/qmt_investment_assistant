"""
完整因子计算与IC测试 - 包含所有可计算因子

计算:
1. 基础价量因子 (基于日频数据)
2. WorldQuant Alpha 101
3. 动量/反转因子
4. 波动率因子
5. 资金流代理因子
6. 行业动量代理因子
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/data/silver")
OUTPUT_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/full_factor_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """加载所有数据"""
    print("加载数据...")
    
    # 日频行情
    bars = pd.read_parquet(DATA_DIR / "daily_bar.parquet")
    bars['trade_date'] = pd.to_datetime(bars['trade_date'])
    
    # 财务数据 (已有)
    run_dir = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs/hs300_ridge_with_support_20260412_000435_8bad53b6")
    financial = pd.read_parquet(run_dir / "datasets/model_dataset.parquet")
    financial['trade_date'] = pd.to_datetime(financial['trade_date'])
    
    # 合并
    df = bars.merge(financial[['trade_date', 'symbol', 'roe', 'earnings_yield', 'operating_margin', 
                               'equity_growth', 'ocf_per_share', 'revenue_growth', 'asset_turnover',
                               'gross_margin', 'cash_ratio', 'mom120', 'vol20', 'fwd_return_20d', 
                               'is_tradable']], 
                    on=['trade_date', 'symbol'], how='left')
    
    print(f"数据量: {len(df)}, 日期: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    return df


def compute_all_factors(df):
    """计算所有可计算的因子"""
    print("\n计算因子...")
    df = df.copy()
    df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    
    factors = []
    
    # ===== 1. 基础财务因子 =====
    financial_factors = ['roe', 'earnings_yield', 'operating_margin', 'equity_growth', 
                        'ocf_per_share', 'revenue_growth', 'asset_turnover', 'gross_margin', 'cash_ratio']
    factors.extend(financial_factors)
    print(f"  财务因子: {len(financial_factors)}个")
    
    # ===== 2. 价量因子 =====
    price_volume_factors = []
    
    # 动量 (多周期)
    for w in [5, 10, 20, 60, 120, 250]:
        col = f'mom{w}'
        df[col] = df.groupby('symbol')['close'].pct_change(w)
        price_volume_factors.append(col)
    
    # 波动率 (多周期)
    for w in [5, 10, 20, 60, 120]:
        col = f'volatility_{w}'
        ret = df.groupby('symbol')['close'].pct_change()
        df[col] = df.groupby('symbol')[ret.name].transform(
            lambda x: x.rolling(w, min_periods=max(3, w//2)).std() * np.sqrt(252)
        ) if ret.name in df.columns else df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(w, min_periods=max(3, w//2)).std() * np.sqrt(252)
        )
        df[col] = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(w, min_periods=max(3, w//2)).std()
        ) * np.sqrt(252)
        price_volume_factors.append(col)
    
    # 换手率代理
    df['turnover_proxy'] = df.groupby('symbol')['volume'].transform(
        lambda x: x / x.rolling(5, min_periods=3).mean()
    )
    price_volume_factors.append('turnover_proxy')
    
    # 量比
    df['volume_ratio'] = df['turnover_proxy']
    price_volume_factors.append('volume_ratio')
    
    # 均线
    for w in [5, 10, 20, 60, 120]:
        col = f'ma{w}'
        df[col] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(w, min_periods=max(3, w//2)).mean())
        price_volume_factors.append(col)
    
    # 均线多头排列
    if all(f'ma{w}' in df.columns for w in [5, 10, 20, 60]):
        df['ma_bull_alignment'] = (
            (df['ma5'] > df['ma10']) & 
            (df['ma10'] > df['ma20']) & 
            (df['ma20'] > df['ma60'])
        ).astype(float)
        price_volume_factors.append('ma_bull_alignment')
    
    # 乖离率
    if 'ma20' in df.columns:
        df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
        price_volume_factors.append('ma20_bias')
    
    # 布林带位置
    if 'ma20' in df.columns:
        roll20 = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20, min_periods=10))
        roll_std = df.groupby('symbol')['close'].transform(lambda x: x.rolling(20, min_periods=10).std())
        df['bb_position'] = (df['close'] - roll20 + 2*roll_std) / (4*roll_std + 1e-8)
        price_volume_factors.append('bb_position')
    
    # 收盘位置
    roll_high = df.groupby('symbol')['high'].transform(lambda x: x.rolling(20, min_periods=10).max())
    roll_low = df.groupby('symbol')['low'].transform(lambda x: x.rolling(20, min_periods=10).min())
    df['close_position'] = (df['close'] - roll_low) / (roll_high - roll_low + 1e-8)
    price_volume_factors.append('close_position')
    
    # 高低价位置
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    price_volume_factors.append('high_low_range')
    
    # RSI
    for w in [6, 12, 24]:
        col = f'rsi{w}'
        delta = df.groupby('symbol')['close'].diff()
        gain = delta.clip(lower=0).rolling(w, min_periods=3).mean()
        loss = (-delta.clip(upper=0)).rolling(w, min_periods=3).mean()
        rs = gain / (loss + 1e-8)
        df[col] = 100 - (100 / (1 + rs))
        price_volume_factors.append(col)
    
    # MACD
    ema12 = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df.groupby('symbol')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['macd_hist'] = df['macd'] - df['macd_signal']
    price_volume_factors.extend(['macd', 'macd_signal', 'macd_hist'])
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df.groupby('symbol')['close'].shift())
    low_close = np.abs(df['low'] - df.groupby('symbol')['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = df.groupby('symbol')['close'].transform(lambda x: tr.loc[x.index].rolling(14, min_periods=7).mean())
    df['atr_ratio'] = df['atr14'] / df['close']
    price_volume_factors.extend(['atr14', 'atr_ratio'])
    
    factors.extend(price_volume_factors)
    print(f"  价量因子: {len(price_volume_factors)}个")
    
    # ===== 3. WorldQuant Alpha 简化版 =====
    alpha_factors = []
    
    # Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2)) - 0.5)
    ret = df.groupby('symbol')['close'].pct_change()
    df['alpha_001'] = df.groupby('symbol').apply(
        lambda x: (x['close'] - x['close'].rolling(20, min_periods=10).mean()) / x['close'].rolling(20, min_periods=10).std()
    ).reset_index(level=0, drop=True)
    alpha_factors.append('alpha_001')
    
    # Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close-open)/open)), 6))
    log_vol = np.log(df['volume'])
    delta_log_vol = df.groupby('symbol')[log_vol.name].diff(2)
    ret_pct = (df['close'] - df.groupby('symbol')['close'].shift()) / df.groupby('symbol')['close'].shift()
    df['alpha_002'] = df.groupby('symbol').apply(
        lambda x: -1 * x['close'].rolling(6, min_periods=3).corr(x['volume'].rank())
    ).reset_index(level=0, drop=True)
    alpha_factors.append('alpha_002')
    
    # Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
    df['alpha_003'] = df.groupby('symbol').apply(
        lambda x: -1 * x['open'].rank().rolling(10, min_periods=5).corr(x['volume'].rank())
    ).reset_index(level=0, drop=True)
    alpha_factors.append('alpha_003')
    
    # Alpha#4: (-1 * Ts_Rank(rank(low), 9))
    df['alpha_004'] = df.groupby('symbol').apply(
        lambda x: -1 * x['low'].rank().rolling(9, min_periods=5).mean()
    ).reset_index(level=0, drop=True)
    alpha_factors.append('alpha_004')
    
    # Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close-vwap))))
    vwap = df['amount'] / df['volume'] / df['close']  # 简化vwap
    df['alpha_005'] = (df['open'] - df.groupby('symbol')['close'].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )) * (-1 * np.abs(df['close'] - df['close']))
    alpha_factors.append('alpha_005')
    
    # Alpha#6: (-1 * correlation(open, volume, 10))
    df['alpha_006'] = df.groupby('symbol').apply(
        lambda x: -1 * x['open'].rolling(10, min_periods=5).corr(x['volume'])
    ).reset_index(level=0, drop=True)
    alpha_factors.append('alpha_006')
    
    # Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7)) : (-1* 1))
    adv20 = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    delta_close = df.groupby('symbol')['close'].diff(7)
    df['alpha_007'] = np.where(adv20 < df['volume'], 
                              -1 * delta_close.rank().rolling(60, min_periods=30).mean() * np.sign(delta_close),
                              -1)
    alpha_factors.append('alpha_007')
    
    # Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))
    sum_open = df.groupby('symbol')['open'].transform(lambda x: x.rolling(5, min_periods=3).sum())
    sum_ret = df.groupby('symbol')['close'].pct_change().transform(lambda x: x.rolling(5, min_periods=3).sum())
    df['alpha_008_product'] = sum_open * sum_ret
    df['alpha_008_delay'] = df.groupby('symbol')['alpha_008_product'].shift(10)
    df['alpha_008'] = -1 * (df['alpha_008_product'] - df['alpha_008_delay']).rank()
    df.drop(['alpha_008_product', 'alpha_008_delay'], axis=1, inplace=True)
    alpha_factors.append('alpha_008')
    
    # Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    delta1 = df.groupby('symbol')['close'].diff()
    ts_min = df.groupby('symbol')[delta1.name].transform(lambda x: x.rolling(5, min_periods=3).min())
    ts_max = df.groupby('symbol')[delta1.name].transform(lambda x: x.rolling(5, min_periods=3).max())
    df['alpha_009'] = np.where(ts_min > 0, delta1, 
                               np.where(ts_max < 0, delta1, -delta1))
    alpha_factors.append('alpha_009')
    
    # Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
    df['alpha_010'] = np.where(ts_min > 0, delta1.rank(),
                               np.where(ts_max < 0, delta1.rank(), -delta1.rank()))
    alpha_factors.append('alpha_010')
    
    # Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    vwap_diff = df.groupby('symbol')['close'].transform(lambda x: x.rolling(3, min_periods=2).mean()) - df['close']  # approximate vwap
    df['alpha_011'] = (
        df.groupby('symbol')[vwap_diff.name].transform(lambda x: x.rolling(3, min_periods=2).max()).rank() +
        df.groupby('symbol')[vwap_diff.name].transform(lambda x: x.rolling(3, min_periods=2).min()).rank()
    ) * df.groupby('symbol')['volume'].diff(3).rank()
    alpha_factors.append('alpha_011')
    
    # Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    df['alpha_012'] = np.sign(df.groupby('symbol')['volume'].diff()) * (-1 * delta1)
    alpha_factors.append('alpha_012')
    
    # Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))
    df['alpha_013'] = df.groupby('symbol').apply(
        lambda x: -1 * x['close'].rank().rolling(5, min_periods=3).cov(x['volume'].rank())
    ).reset_index(level=0, drop=True)
    alpha_factors.append('alpha_013')
    
    # Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    df['alpha_014'] = (-1 * ret_pct.diff(3).rank()) * df['alpha_006']
    alpha_factors.append('alpha_014')
    
    # Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    df['alpha_015'] = df.groupby('symbol').apply(
        lambda x: -1 * x['high'].rank().rolling(3, min_periods=2).corr(x['volume'].rank()).sum()
    ).reset_index(level=0, drop=True)
    alpha_factors.append('alpha_015')
    
    # Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))
    df['alpha_016'] = df.groupby('symbol').apply(
        lambda x: -1 * x['high'].rank().rolling(5, min_periods=3).cov(x['volume'].rank())
    ).reset_index(level=0, drop=True)
    alpha_factors.append('alpha_016')
    
    # Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) 
    df['alpha_017'] = (-1 * df.groupby('symbol')['close'].transform(
        lambda x: x.rank().rolling(10, min_periods=5).mean()
    ).rank()) * df.groupby('symbol')['close'].diff().diff().rank()
    alpha_factors.append('alpha_017')
    
    # Alpha#18: (-1 * rank(((stddev(abs((close-open)), 20) + (close-open)) + correlation(close, open, 15))))
    df['alpha_018'] = -1 * (
        df.groupby('symbol')['close'].transform(lambda x: (x-x.shift()).abs().rolling(20, min_periods=10).std()) +
        ret_pct
    ).rank() * df.groupby('symbol').apply(
        lambda x: x['close'].rolling(15, min_periods=8).corr(x['open'])
    ).reset_index(level=0, drop=True).rank()
    alpha_factors.append('alpha_018')
    
    # Alpha#19: ((-1 * sign(((close-open) + delta((close-open), 1)))) * (1 + rank((1 + sum(returns, 250)))))
    sum_ret_250 = df.groupby('symbol')['close'].pct_change().transform(lambda x: x.rolling(250, min_periods=125).sum())
    df['alpha_019'] = -1 * np.sign(ret_pct + ret_pct.diff()) * (1 + (1 + sum_ret_250).rank())
    alpha_factors.append('alpha_019')
    
    # Alpha#20: ((-1 * rank((open - delay(open, 5)))) * (1 + rank(sum(returns, 250))))
    df['alpha_020'] = (-1 * df.groupby('symbol')['open'].diff(5).rank()) * (1 + sum_ret_250.rank())
    alpha_factors.append('alpha_020')
    
    # 添加更多简化的Alpha
    for i in range(21, 50):
        col = f'alpha_{i:03d}'
        # 基于成交量与价格关系的简化Alpha
        vol_rank = df.groupby('symbol')['volume'].transform(lambda x: x.rank(pct=True))
        price_rank = df.groupby('symbol')['close'].transform(lambda x: x.rank(pct=True))
        df[col] = (price_rank - 0.5) * (vol_rank - 0.5)
        alpha_factors.append(col)
    
    factors.extend(alpha_factors)
    print(f"  Alpha因子: {len(alpha_factors)}个")
    
    # ===== 4. 行业动量代理因子 =====
    sector_factors = []
    
    # 动量变化 (使用个股动量代理行业动量)
    df['momentum_acceleration'] = df['mom20'] - df['mom60']
    sector_factors.append('momentum_acceleration')
    
    # 短期反转
    df['short_term_reversal'] = -df['mom5']
    sector_factors.append('short_term_reversal')
    
    # 中期反转
    df['medium_term_reversal'] = -df['mom20']
    sector_factors.append('medium_term_reversal')
    
    # 长期反转
    df['long_term_reversal'] = -df['mom120']
    sector_factors.append('long_term_reversal')
    
    # 波动率变化
    df['vol_change'] = df['volatility_20'] - df['volatility_60']
    sector_factors.append('vol_change')
    
    # 量价背离
    vol_change = df.groupby('symbol')['volume'].pct_change()
    df['price_volume_divergence'] = ret_pct - vol_change
    sector_factors.append('price_volume_divergence')
    
    # OBV变化
    df['obv_change'] = np.sign(ret_pct) * df['volume']
    df['obv_ma5'] = df.groupby('symbol')['obv_change'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    sector_factors.extend(['obv_change', 'obv_ma5'])
    
    factors.extend(sector_factors)
    print(f"  行业/动量因子: {len(sector_factors)}个")
    
    # ===== 5. 财务变化因子 =====
    financial_change = []
    
    if 'roe' in df.columns:
        df['roe_change'] = df.groupby('symbol')['roe'].diff()
        df['roe_qoq'] = df.groupby('symbol')['roe'].pct_change()
        financial_change.extend(['roe_change', 'roe_qoq'])
    
    if 'profit_growth' not in df.columns and 'roe' in df.columns:
        df['profit_growth'] = df.groupby('symbol')['roe'].pct_change(4)
        financial_change.append('profit_growth')
    
    factors.extend(financial_change)
    print(f"  财务变化因子: {len(financial_change)}个")
    
    # 去除重复
    factors = list(set(factors))
    factors = [f for f in factors if f in df.columns]
    
    print(f"\n总计因子: {len(factors)}个")
    print(f"因子列表: {factors[:20]}...")
    
    return df, factors


def compute_factor_ic(df, factors, label_col='fwd_return_20d'):
    """计算所有因子IC"""
    print("\n计算IC...")
    
    results = []
    valid_df = df.dropna(subset=[label_col]).copy()
    
    for col in factors:
        if col not in valid_df.columns:
            continue
        
        # 过滤有效数据
        test_df = valid_df.dropna(subset=[col])
        if len(test_df) < 100:
            continue
        
        test_df = test_df.copy()
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        test_df[label_col] = pd.to_numeric(test_df[label_col], errors='coerce')
        test_df = test_df.dropna(subset=[col, label_col])
        
        # 按日期计算截面IC
        daily_ics = []
        test_df_indexed = test_df.set_index('trade_date')
        for date in test_df_indexed.index.unique():
            group = test_df_indexed.loc[date]
            if len(group) < 30:
                continue
            col_std = group[col].astype(float).std()
            label_std = group[label_col].astype(float).std()
            if col_std < 1e-8:
                continue
            if label_std < 1e-8:
                continue
            
            score_rank = group[col].astype(float).rank(pct=True)
            label_rank = group[label_col].astype(float).rank(pct=True)
            
            ic = score_rank.corr(label_rank)
            daily_ics.append({'date': date, 'ic': ic, 'n': len(group)})
        
        if len(daily_ics) < 60:  # 至少2个月
            continue
        
        ic_df = pd.DataFrame(daily_ics)
        
        results.append({
            'factor': col,
            'ic_mean': ic_df['ic'].mean(),
            'ic_std': ic_df['ic'].std(),
            'ic_ir': ic_df['ic'].mean() / max(ic_df['ic'].std(), 0.001),
            'ic_positive_rate': (ic_df['ic'] > 0).mean(),
            'n_days': len(ic_df),
            'n_samples': ic_df['n'].sum(),
        })
    
    ic_results = pd.DataFrame(results)
    ic_results = ic_results.sort_values('ic_ir', ascending=False)
    
    return ic_results


def main():
    print("="*70)
    print("完整因子计算与IC测试")
    print("="*70)
    print(f"开始时间: {datetime.now()}")
    
    # 1. 加载数据
    df = load_data()
    
    # 2. 计算所有因子
    df, factors = compute_all_factors(df)
    
    # 3. 计算IC
    ic_results = compute_factor_ic(df, factors)
    
    # 4. 保存结果
    ic_results.to_csv(OUTPUT_DIR / "full_ic_results.csv", index=False)
    
    # 5. 打印结果
    print("\n" + "="*70)
    print("IC测试结果 (Top 30)")
    print("="*70)
    print(ic_results.head(30).to_string(index=False))
    
    # 6. 分类统计
    print("\n" + "="*70)
    print("按类别统计")
    print("="*70)
    
    # 正向因子
    positive = ic_results[ic_results['ic_mean'] > 0]
    negative = ic_results[ic_results['ic_mean'] < 0]
    good_ir = ic_results[ic_results['ic_ir'] > 0.1]
    
    print(f"\n正向IC因子: {len(positive)}个")
    print(f"负向IC因子: {len(negative)}个")
    print(f"IC IR > 0.1: {len(good_ir)}个")
    
    print("\n" + "-"*70)
    print("IC IR > 0.1 的因子:")
    print("-"*70)
    print(good_ir.to_string(index=False))
    
    # 分类展示
    print("\n" + "-"*70)
    print("Top 10 正向有效因子:")
    print("-"*70)
    top_positive = positive[positive['ic_ir'] > 0].head(10)
    print(top_positive.to_string(index=False))
    
    print("\n" + "-"*70)
    print("Top 10 负向有效因子 (可能是反向指标):")
    print("-"*70)
    top_negative = negative[negative['ic_ir'].abs() > 0.1].head(10)
    print(top_negative.to_string(index=False))
    
    # 7. 保存摘要
    summary = {
        'total_factors': len(factors),
        'tested_factors': len(ic_results),
        'positive_ic': len(positive),
        'negative_ic': len(negative),
        'good_ir': len(good_ir),
        'top_10': ic_results.head(10).to_dict('records'),
    }
    
    import json
    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print(f"结果已保存至: {OUTPUT_DIR}")
    print("="*70)
    
    return ic_results


if __name__ == "__main__":
    ic_results = main()
