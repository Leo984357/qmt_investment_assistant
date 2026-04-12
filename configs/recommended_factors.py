# 推荐的11个有效因子
# 基于71因子信号研究结果

recommended_factors = [
    # 强烈推荐 (5个)
    "close_to_high250",   # 创新高, IC=0.023, 最稳健
    "mom250",            # 长周期动量, IC=0.035
    "high_low_pos120",   # 120日高低位, IC=0.005
    "rsi6",              # 短期RSI, IC=0.003
    "close_to_high120",  # 120日新高, IC=0.007
    
    # 可考虑 (6个)
    "mom120",            # 中周期动量, IC=0.010
    "rev20",             # 20日反转, IC=0.019
    "high_low_pos20",    # 20日高低位, IC=0.001
    "close_to_high20",   # 20日新高, IC=0.006
    "close_to_high60",   # 60日新高, IC=0.000
    "vol_growth5",       # 5日量增, IC=0.001
]

# 应该剔除的因子 (60个)
rejected_factors = [
    # 短期动量 - 全部失效
    "mom3", "mom5", "mom10", "mom20", "mom30", "mom60", "mom90",
    
    # 波动率因子 - IC全负
    "vol5", "vol10", "vol20", "vol30", "vol60", "vol120",
    "vol_ratio_5_20", "vol_ratio_10_60", "vol_ratio_20_60", "vol_ratio_5_60",
    "vol_std_ratio5_20", "vol_std_ratio10_60", "vol_std_ratio20_60",
    "atr14", "atr20",
    
    # 技术指标 - 大部分失效
    "kdj_k9", "kdj_k14", "kdj_d9", "kdj_d14",
    "williams_r14", "williams_r28",
    "macd_diff", "macd_dea", "macd_hist",
    "rsi12", "rsi24",
    "cci14", "cci20",
    
    # 其他
    "rev1", "rev3", "rev5", "rev10",
    "price_to_ma20", "price_to_ma60", "price_to_ma120",
    "ma_diff_5_20", "ma_diff_10_20", "ma_diff_5_60", "ma_diff_20_60",
    "close_to_low20", "close_to_low60", "close_to_low120", "close_to_low250",
    "return_skew20", "return_skew60",
    "return_kurt20", "return_kurt60",
    "amount_growth5", "amount_growth20", "amount_growth60",
    "vol_growth10", "vol_growth20",
]
