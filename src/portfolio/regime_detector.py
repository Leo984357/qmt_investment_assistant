"""
市场状态检测器

用于检测当前市场环境：牛市、熊市、震荡市、趋势市
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """市场状态"""
    BULL = "bull"           # 牛市
    BEAR = "bear"           # 熊市
    TRENDING = "trending"    # 趋势市
    RANGING = "ranging"     # 震荡市


@dataclass
class RegimeConfig:
    """市场状态检测配置"""
    short_window: int = 20      # 短期均线窗口
    long_window: int = 120      # 长期均线窗口
    volatility_window: int = 20  # 波动率窗口
    trend_threshold: float = 0.05  # 趋势阈值 (5%)


def detect_market_regime(
    market_index: pd.Series,
    signal_date: pd.Timestamp,
    config: RegimeConfig | None = None,
) -> tuple[MarketRegime, dict]:
    """
    简化市场状态检测 - 基于价格与均线的关系
    
    逻辑：
    - 牛市: 价格在长期均线上方，且均线向上
    - 熊市: 价格在长期均线下方，或均线向下
    - 震荡: 其他情况
    
    Args:
        market_index: 市场指数序列
        signal_date: 信号日期
        config: 配置
    
    Returns:
        (市场状态, 诊断信息)
    """
    config = config or RegimeConfig()
    
    history = market_index.loc[market_index.index <= signal_date].dropna()
    if len(history) < config.long_window + 10:
        return MarketRegime.RANGING, {"reason": "insufficient_data"}
    
    # 计算均线
    short_ma = history.tail(config.short_window).mean()
    long_ma = history.tail(config.long_window).mean()
    current = history.iloc[-1]
    
    # 趋势位置: 价格相对于长期均线
    trend_position = (current / long_ma - 1) if long_ma > 0 else 0
    
    # 均线方向: 使用短期均线与长期均线的比值
    short_vs_long = short_ma / long_ma - 1 if long_ma > 0 else 0
    
    diagnostics = {
        "trend_position": trend_position,
        "short_vs_long": short_vs_long,
        "short_ma": short_ma,
        "long_ma": long_ma,
        "current": current,
    }
    
    # 阈值
    threshold = 0.03  # 3%作为判断阈值
    
    # 均线方向
    ma_up = short_vs_long > 0.005
    ma_down = short_vs_long < -0.005
    
    # 价格与均线关系
    above_ma = trend_position > threshold
    below_ma = trend_position < -threshold
    
    # 牛市: 在均线上方 + 均线向上
    if above_ma and ma_up:
        return MarketRegime.BULL, diagnostics
    
    # 熊市: 在均线下方 + 均线向下，或者价格在均线下方
    if below_ma or (trend_position < 0 and ma_down):
        return MarketRegime.BEAR, diagnostics
    
    # 价格在均线附近
    if abs(trend_position) < threshold:
        return MarketRegime.RANGING, diagnostics
    
    # 其他情况
    return MarketRegime.TRENDING, diagnostics


def get_regime_exposure_multiplier(regime: MarketRegime) -> float:
    """
    根据市场状态返回仓位乘数
    
    经济逻辑：
    - 牛市: 满仓享受趋势收益
    - 熊市: 空仓规避系统性风险
    - 趋势市: 满仓跟随趋势
    - 震荡市: 降低仓位减少磨损
    
    Args:
        regime: 市场状态
    
    Returns:
        仓位乘数
    """
    multipliers = {
        MarketRegime.BULL: 1.0,      # 牛市满仓
        MarketRegime.BEAR: 0.0,      # 熊市空仓（完全规避）
        MarketRegime.TRENDING: 1.0,  # 趋势市满仓
        MarketRegime.RANGING: 0.3,   # 震荡市轻仓
    }
    return multipliers.get(regime, 0.5)


def detect_and_apply_regime(
    market_index: pd.Series,
    signal_date: pd.Timestamp,
    base_exposure: float,
    config: RegimeConfig | None = None,
) -> tuple[float, MarketRegime, dict]:
    """
    检测市场状态并调整仓位
    
    Args:
        market_index: 市场指数序列
        signal_date: 信号日期
        base_exposure: 基础仓位
        config: 配置
    
    Returns:
        (调整后仓位, 市场状态, 诊断信息)
    """
    regime, diagnostics = detect_market_regime(market_index, signal_date, config)
    multiplier = get_regime_exposure_multiplier(regime)
    adjusted_exposure = base_exposure * multiplier
    
    return adjusted_exposure, regime, diagnostics
