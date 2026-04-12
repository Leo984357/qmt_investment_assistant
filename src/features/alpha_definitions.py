"""
Alpha因子定义 - 标准化版

包含：
1. WorldQuant Alpha 101 的标准定义
2. 简化版 vs 原版的对比
3. IC验证结果
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable


# ==================== Alpha定义 ====================

def alpha_004_original(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#4 原版: (-1 * Ts_Rank(rank(low), 9))
    
    伪代码:
    for each day:
        rank_low = rank(low)  # 截面排名
        ts_rank = mean(rank_low[-9:])  # 过去9天的排名均值（Ts_Rank近似）
        alpha = -ts_rank
    
    逻辑: 低价在过去9天排名越高，alpha越负（看空）
    """
    def calc(s):
        rank_low = s['low'].rank(pct=True)  # 截面排名
        ts_rank = rank_low.rolling(9, min_periods=5).mean()  # Ts_Rank近似
        return -ts_rank
    
    return df.groupby('symbol', group_keys=False).apply(calc)


def alpha_004_simplified(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#4 简化版
    
    问题: 用rolling mean代替Ts_Rank概念不同
    """
    def calc(s):
        return -s['low'].rank().rolling(9, min_periods=5).mean()
    
    return df.groupby('symbol', group_keys=False).apply(calc)


def alpha_004_corrected(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#4 修正版
    
    Ts_Rank正确实现:
    - 计算过去9天的排名
    - 返回最近一天的排名（ts_rank = value at lag 0）
    """
    def calc(s):
        rank_low = s['low'].rank()  # 绝对排名
        # 过去9天的排名，不是均值，而是ts_rank = rank of current value in history
        ts_rank = rank_low.rolling(9, min_periods=5).apply(
            lambda x: pd.Series(x).rank().iloc[-1] if len(x) > 0 else np.nan, raw=False
        )
        return -ts_rank
    
    return df.groupby('symbol', group_keys=False).apply(calc)


def alpha_017_original(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#17 原版: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)))
    
    复合动量因子:
    1. rank(ts_rank(close, 10)): 收盘价的ts_rank
    2. delta(delta(close, 1), 1): 价格加速度
    3. 两者相乘
    """
    def calc(s):
        ts_rank = s['close'].rank().rolling(10, min_periods=5).mean()
        rank_ts_rank = (-ts_rank).rank()
        
        delta1 = s['close'].diff()
        delta2 = delta1.diff()
        rank_delta = delta2.rank()
        
        return rank_ts_rank * rank_delta
    
    return df.groupby('symbol', group_keys=False).apply(calc)


def alpha_006_original(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#6 原版: (-1 * correlation(open, volume, 10))
    
    开盘价与成交量的相关性（过去10天）
    """
    def calc(s):
        return -s['open'].rolling(10, min_periods=5).corr(s['volume'])
    
    return df.groupby('symbol', group_keys=False).apply(calc)


def alpha_002_original(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#2 原版: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close-open)/open)), 6))
    """
    def calc(s):
        log_vol = np.log(s['volume'])
        delta_log_vol = log_vol.diff(2)
        ret_pct = (s['close'] - s['open']) / s['open']
        
        return -s['open'].rolling(6, min_periods=3).corr(s['volume'].rank())
    
    return df.groupby('symbol', group_keys=False).apply(calc)


# ==================== Alpha注册表 ====================

ALPHA_REGISTRY: Dict[str, Dict] = {
    'alpha_004': {
        'original_formula': '(-1 * Ts_Rank(rank(low), 9))',
        'description': '低价动量',
        'logic': '低价排名越高，alpha越负',
        'implementation': {
            'original': alpha_004_original,
            'simplified': alpha_004_simplified,
            'corrected': alpha_004_corrected,
        },
        'expected_ic': 0.05,  # 预期IC
        'notes': '简化版与原版概念不同，需修正',
    },
    'alpha_017': {
        'original_formula': '((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)))',
        'description': '复合动量',
        'logic': '价格加速度 × ts_rank',
        'implementation': alpha_017_original,
        'expected_ic': 0.03,
        'notes': '',
    },
    'alpha_006': {
        'original_formula': '(-1 * correlation(open, volume, 10))',
        'description': '量价相关',
        'logic': '开盘价与成交量负相关意味着...',
        'implementation': alpha_006_original,
        'expected_ic': 0.02,
        'notes': '',
    },
    'alpha_002': {
        'original_formula': '(-1 * correlation(rank(delta(log(volume), 2)), rank(((close-open)/open)), 6))',
        'description': '成交量变化与收益',
        'logic': '成交量变化与收益率的相关性',
        'implementation': alpha_002_original,
        'expected_ic': 0.02,
        'notes': '',
    },
}


def get_alpha_implementation(name: str, version: str = 'corrected') -> Callable:
    """获取Alpha因子的实现"""
    if name not in ALPHA_REGISTRY:
        raise ValueError(f"Unknown alpha: {name}")
    
    impl = ALPHA_REGISTRY[name]['implementation']
    if isinstance(impl, dict):
        if version not in impl:
            raise ValueError(f"Unknown version: {version}")
        return impl[version]
    return impl


def compare_alpha_versions(df: pd.DataFrame, alpha_name: str) -> pd.DataFrame:
    """对比不同版本的Alpha IC"""
    if alpha_name not in ['alpha_004']:
        return pd.DataFrame()
    
    # 计算各版本
    versions = {}
    for version in ['original', 'simplified', 'corrected']:
        try:
            impl = get_alpha_implementation(alpha_name, version)
            versions[version] = impl(df)
        except Exception as e:
            print(f"  {version}: {e}")
    
    # 计算IC
    results = []
    for version, alpha_values in versions.items():
        df_temp = df.copy()
        df_temp[alpha_name] = alpha_values
        
        valid_df = df_temp.dropna(subset=[alpha_name, 'fwd_return_20d'])
        
        if len(valid_df) > 0:
            daily_ics = []
            for date, group in valid_df.groupby('trade_date'):
                if len(group) >= 30:
                    ic = group[alpha_name].rank().corr(group['fwd_return_20d'].rank())
                    daily_ics.append(ic)
            
            if len(daily_ics) > 60:
                results.append({
                    'version': version,
                    'ic_mean': np.mean(daily_ics),
                    'ic_std': np.std(daily_ics),
                    'ic_ir': np.mean(daily_ics) / np.std(daily_ics),
                    'positive_rate': np.mean([x > 0 for x in daily_ics]),
                    'n_days': len(daily_ics),
                })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("Alpha因子定义表")
    print("="*60)
    for name, info in ALPHA_REGISTRY.items():
        print(f"\n{name}: {info['description']}")
        print(f"  原版公式: {info['original_formula']}")
        print(f"  逻辑: {info['logic']}")
        if info['notes']:
            print(f"  注意: {info['notes']}")
