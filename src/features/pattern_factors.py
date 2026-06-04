"""
形态学因子库 - K线形态、技术图形、价格模式

来源:
1. K线形态 (Candlestick Patterns)
2. 技术图形 (Chart Patterns)
3. 价格形态 (Price Patterns)
4. 成交量形态 (Volume Patterns)
5. 波动率形态 (Volatility Patterns)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatternSpec:
    name: str
    category: str
    description: str
    pattern_type: str
    economic_interpretation: str
    lookback: int
    data_requirement: list[str]


PATTERN_FACTORS = [
    # ===== K线形态 - 单根蜡烛 =====
    PatternSpec(
        name="candle_body_ratio",
        category="candlestick_single",
        description="蜡烛实体占比",
        pattern_type="single",
        economic_interpretation="实体大表示趋势强",
        lookback=1,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_upper_shadow",
        category="candlestick_single",
        description="上影线占比",
        pattern_type="single",
        economic_interpretation="上影长表示上方压力大",
        lookback=1,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_lower_shadow",
        category="candlestick_single",
        description="下影线占比",
        pattern_type="single",
        economic_interpretation="下影长表示下方支撑强",
        lookback=1,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_doji",
        category="candlestick_single",
        description="十字星强度",
        pattern_type="single",
        economic_interpretation="多空力量均衡",
        lookback=1,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_hammer",
        category="candlestick_single",
        description="锤子线信号",
        pattern_type="single",
        economic_interpretation="底部反转信号",
        lookback=1,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_shooting_star",
        category="candlestick_single",
        description="流星线信号",
        pattern_type="single",
        economic_interpretation="顶部反转信号",
        lookback=1,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_marubozu",
        category="candlestick_single",
        description="光头光脚线",
        pattern_type="single",
        economic_interpretation="强烈趋势信号",
        lookback=1,
        data_requirement=["open", "close", "high", "low"],
    ),

    # ===== K线形态 - 双根蜡烛 =====
    PatternSpec(
        name="candle_engulf_bullish",
        category="candlestick_double",
        description="看涨吞没形态",
        pattern_type="double",
        economic_interpretation="空转多反转",
        lookback=2,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_engulf_bearish",
        category="candlestick_double",
        description="看跌吞没形态",
        pattern_type="double",
        economic_interpretation="多转空反转",
        lookback=2,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_harami_bullish",
        category="candlestick_double",
        description="看涨孕线形态",
        pattern_type="double",
        economic_interpretation="底部整固",
        lookback=2,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_harami_bearish",
        category="candlestick_double",
        description="看跌孕线形态",
        pattern_type="double",
        economic_interpretation="顶部整固",
        lookback=2,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_piercing",
        category="candlestick_double",
        description="刺透形态",
        pattern_type="double",
        economic_interpretation="底部反转",
        lookback=2,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_dark_cloud",
        category="candlestick_double",
        description="乌云盖顶形态",
        pattern_type="double",
        economic_interpretation="顶部反转",
        lookback=2,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_morning_star",
        category="candlestick_double",
        description="晨星形态",
        pattern_type="double",
        economic_interpretation="三日后底部反转",
        lookback=3,
        data_requirement=["open", "close", "high", "low"],
    ),
    PatternSpec(
        name="candle_evening_star",
        category="candlestick_double",
        description="黄昏星形态",
        pattern_type="double",
        economic_interpretation="三日后顶部反转",
        lookback=3,
        data_requirement=["open", "close", "high", "low"],
    ),

    # ===== 技术图形 - 突破 =====
    PatternSpec(
        name="pattern_breakout_strength",
        category="breakout",
        description="突破强度",
        pattern_type="pattern",
        economic_interpretation="突破重要价位",
        lookback=20,
        data_requirement=["high", "low", "close"],
    ),
    PatternSpec(
        name="pattern_support_resistance_strength",
        category="breakout",
        description="支撑阻力位强度",
        pattern_type="pattern",
        economic_interpretation="测试次数与成交量",
        lookback=60,
        data_requirement=["high", "low", "volume"],
    ),
    PatternSpec(
        name="pattern_volume_breakout",
        category="breakout",
        description="量价配合突破",
        pattern_type="pattern",
        economic_interpretation="放量确认突破",
        lookback=5,
        data_requirement=["volume", "price_change"],
    ),
    PatternSpec(
        name="pattern_close_breakout",
        category="breakout",
        description="收盘价突破",
        pattern_type="pattern",
        economic_interpretation="收盘站稳突破位",
        lookback=1,
        data_requirement=["close", "resistance_level"],
    ),

    # ===== 技术图形 - 趋势 =====
    PatternSpec(
        name="pattern_trendline_slope",
        category="trendline",
        description="趋势线斜率",
        pattern_type="pattern",
        economic_interpretation="趋势强度",
        lookback=60,
        data_requirement=["high", "low"],
    ),
    PatternSpec(
        name="pattern_trendline_angle",
        category="trendline",
        description="趋势线角度",
        pattern_type="pattern",
        economic_interpretation="趋势陡峭程度",
        lookback=60,
        data_requirement=["high", "low"],
    ),
    PatternSpec(
        name="pattern_channel_width",
        category="trendline",
        description="通道宽度",
        pattern_type="pattern",
        economic_interpretation="波动范围",
        lookback=60,
        data_requirement=["high", "low"],
    ),
    PatternSpec(
        name="pattern_price_channel_position",
        category="trendline",
        description="价格在通道中位置",
        pattern_type="pattern",
        economic_interpretation="相对强弱位置",
        lookback=60,
        data_requirement=["close", "high", "low"],
    ),

    # ===== 技术图形 - 整理形态 =====
    PatternSpec(
        name="pattern_triangle_convergence",
        category="consolidation",
        description="三角收敛程度",
        pattern_type="pattern",
        economic_interpretation="即将选择方向",
        lookback=60,
        data_requirement=["high", "low"],
    ),
    PatternSpec(
        name="pattern_consolidation_range",
        category="consolidation",
        description="整理区间大小",
        pattern_type="pattern",
        economic_interpretation="横盘震荡幅度",
        lookback=20,
        data_requirement=["high", "low", "close"],
    ),
    PatternSpec(
        name="pattern_rectangle_stability",
        category="consolidation",
        description="矩形整理稳定性",
        pattern_type="pattern",
        economic_interpretation="盘整持续时间",
        lookback=60,
        data_requirement=["high", "low"],
    ),
    PatternSpec(
        name="pattern_flag_pole_height",
        category="consolidation",
        description="旗形竿高度",
        pattern_type="pattern",
        economic_interpretation="旗形突破目标位",
        lookback=60,
        data_requirement=["high", "low", "close"],
    ),

    # ===== 技术图形 - 反转形态 =====
    PatternSpec(
        name="pattern_head_shoulders_score",
        category="reversal",
        description="头肩顶/底形态得分",
        pattern_type="pattern",
        economic_interpretation="趋势反转信号",
        lookback=120,
        data_requirement=["high", "low"],
    ),
    PatternSpec(
        name="pattern_double_top_bottom_score",
        category="reversal",
        description="双顶/底形态得分",
        pattern_type="pattern",
        economic_interpretation="关键价位反转",
        lookback=60,
        data_requirement=["high", "low"],
    ),
    PatternSpec(
        name="pattern_triple_top_bottom_score",
        category="reversal",
        description="三顶/底形态得分",
        pattern_type="pattern",
        economic_interpretation="强反转信号",
        lookback=90,
        data_requirement=["high", "low"],
    ),
    PatternSpec(
        name="pattern_v_reversal_strength",
        category="reversal",
        description="V型反转强度",
        pattern_type="pattern",
        economic_interpretation="快速反转",
        lookback=20,
        data_requirement=["close"],
    ),

    # ===== 价格形态 - 均线系统 =====
    PatternSpec(
        name="ma_alignment_short",
        category="moving_average",
        description="短期均线多头排列",
        pattern_type="pattern",
        economic_interpretation="上升趋势",
        lookback=20,
        data_requirement=["close"],
    ),
    PatternSpec(
        name="ma_alignment_long",
        category="moving_average",
        description="长期均线多头排列",
        pattern_type="pattern",
        economic_interpretation="强势上升趋势",
        lookback=60,
        data_requirement=["close"],
    ),
    PatternSpec(
        name="ma_golden_cross",
        category="moving_average",
        description="均线金叉信号",
        pattern_type="signal",
        economic_interpretation="短期均线从下穿越长期",
        lookback=5,
        data_requirement=["close"],
    ),
    PatternSpec(
        name="ma_death_cross",
        category="moving_average",
        description="均线死叉信号",
        pattern_type="signal",
        economic_interpretation="短期均线从上穿越长期",
        lookback=5,
        data_requirement=["close"],
    ),
    PatternSpec(
        name="ma_bands_width",
        category="moving_average",
        description="布林带宽度",
        pattern_type="pattern",
        economic_interpretation="波动率收缩/扩张",
        lookback=20,
        data_requirement=["close", "stddev"],
    ),
    PatternSpec(
        name="ma_bands_position",
        category="moving_average",
        description="价格在布林带中位置",
        pattern_type="pattern",
        economic_interpretation="超买超卖",
        lookback=20,
        data_requirement=["close", "upper_band", "lower_band"],
    ),

    # ===== 价格形态 - 波动率 =====
    PatternSpec(
        name="volatility_breakout",
        category="volatility",
        description="波动率突破",
        pattern_type="pattern",
        economic_interpretation="波动率从低突然放大",
        lookback=60,
        data_requirement=["daily_range"],
    ),
    PatternSpec(
        name="volatility_squeeze",
        category="volatility",
        description="波动率收缩",
        pattern_type="pattern",
        economic_interpretation="暴风雨前的宁静",
        lookback=60,
        data_requirement=["historical_volatility"],
    ),
    PatternSpec(
        name="volatility_asymmetry",
        category="volatility",
        description="波动率非对称性",
        pattern_type="pattern",
        economic_interpretation="上行vs下行波动",
        lookback=20,
        data_requirement=["up_days", "down_days"],
    ),
    PatternSpec(
        name="gap_size",
        category="volatility",
        description="跳空缺口大小",
        pattern_type="pattern",
        economic_interpretation="跳空幅度",
        lookback=1,
        data_requirement=["open", "prev_close"],
    ),
    PatternSpec(
        name="gap_fill_ratio",
        category="volatility",
        description="缺口回补比例",
        pattern_type="pattern",
        economic_interpretation="缺口是否回补",
        lookback=5,
        data_requirement=["high", "low", "gap_level"],
    ),

    # ===== 价格形态 - 动能 =====
    PatternSpec(
        name="momentum_divergence",
        category="momentum",
        description="价格与动能背离",
        pattern_type="pattern",
        economic_interpretation="趋势可能反转",
        lookback=60,
        data_requirement=["price", "oscillator"],
    ),
    PatternSpec(
        name="momentum_acceleration",
        category="momentum",
        description="动能加速",
        pattern_type="pattern",
        economic_interpretation="趋势正在加速",
        lookback=20,
        data_requirement=["price_change"],
    ),
    PatternSpec(
        name="momentum_exhaustion",
        category="momentum",
        description="动能衰竭",
        pattern_type="pattern",
        economic_interpretation="趋势可能结束",
        lookback=20,
        data_requirement=["price", "volume"],
    ),
    PatternSpec(
        name="momentum_sequencing",
        category="momentum",
        description="动能序列",
        pattern_type="pattern",
        economic_interpretation="连续动能柱",
        lookback=10,
        data_requirement=["price_change"],
    ),

    # ===== 成交量形态 =====
    PatternSpec(
        name="volume_price_divergence",
        category="volume_pattern",
        description="量价背离",
        pattern_type="pattern",
        economic_interpretation="量不支持价格",
        lookback=20,
        data_requirement=["volume", "price"],
    ),
    PatternSpec(
        name="volume_climax",
        category="volume_pattern",
        description="成交量高潮",
        pattern_type="pattern",
        economic_interpretation="可能见顶/底",
        lookback=5,
        data_requirement=["volume"],
    ),
    PatternSpec(
        name="volume_dry_up",
        category="volume_pattern",
        description="成交量枯竭",
        pattern_type="pattern",
        economic_interpretation="趋势将变",
        lookback=20,
        data_requirement=["volume"],
    ),
    PatternSpec(
        name="volume_accumulation",
        category="volume_pattern",
        description="成交量累积",
        pattern_type="pattern",
        economic_interpretation="主力建仓",
        lookback=20,
        data_requirement=["volume", "price"],
    ),
    PatternSpec(
        name="volume_distribution",
        category="volume_pattern",
        description="成交量分布",
        pattern_type="pattern",
        economic_interpretation="高低价成交量",
        lookback=20,
        data_requirement=["volume", "price"],
    ),
    PatternSpec(
        name="volume_on_up_days",
        category="volume_pattern",
        description="上涨日成交量占比",
        pattern_type="pattern",
        economic_interpretation="资金流入流出",
        lookback=20,
        data_requirement=["volume", "price_change"],
    ),

    # ===== 市场广度 =====
    PatternSpec(
        name="breadth_momentum",
        category="market_breadth",
        description="广度动量",
        pattern_type="pattern",
        economic_interpretation="市场参与度",
        lookback=20,
        data_requirement=["advancing", "declining"],
    ),
    PatternSpec(
        name="breadth_divergence",
        category="market_breadth",
        description="广度背离",
        pattern_type="pattern",
        economic_interpretation="价格与广度背离",
        lookback=60,
        data_requirement=["price", "breadth"],
    ),
    PatternSpec(
        name="breadth_thrust",
        category="market_breadth",
        description="广度突破",
        pattern_type="pattern",
        economic_interpretation="广度突然放大",
        lookback=5,
        data_requirement=["advancing", "declining"],
    ),
    PatternSpec(
        name="new_high_low_ratio",
        category="market_breadth",
        description="新高/新低比",
        pattern_type="pattern",
        economic_interpretation="市场极端位置",
        lookback=20,
        data_requirement=["new_high", "new_low"],
    ),

    # ===== A股特有形态 =====
    PatternSpec(
        name="limit_up_consecutive",
        category="china_pattern",
        description="连续涨停",
        pattern_type="pattern",
        economic_interpretation="强势股信号",
        lookback=10,
        data_requirement=["limit_up_days"],
    ),
    PatternSpec(
        name="turnover_rate_spike",
        category="china_pattern",
        description="换手率突增",
        pattern_type="pattern",
        economic_interpretation="资金进出",
        lookback=5,
        data_requirement=["turnover_rate"],
    ),
    PatternSpec(
        name="investor_tracking",
        category="china_pattern",
        description="主力追踪",
        pattern_type="pattern",
        economic_interpretation="大单净流入",
        lookback=5,
        data_requirement=["big_order_flow"],
    ),
    PatternSpec(
        name="st_speculation",
        category="china_pattern",
        description="ST股投机度",
        pattern_type="pattern",
        economic_interpretation="市场风险偏好",
        lookback=20,
        data_requirement=["st_stock_return"],
    ),
    PatternSpec(
        name="small_cap_speculation",
        category="china_pattern",
        description="小盘股投机度",
        pattern_type="pattern",
        economic_interpretation="壳价值/炒作情绪",
        lookback=20,
        data_requirement=["small_cap_return"],
    ),
    PatternSpec(
        name="board_leading",
        category="china_pattern",
        description="板块轮动领先",
        pattern_type="pattern",
        economic_interpretation="领先板块切换",
        lookback=5,
        data_requirement=["sector_return"],
    ),
]


def get_pattern_factor_names() -> list[str]:
    """获取所有形态因子名称"""
    return [f.name for f in PATTERN_FACTORS]


def get_pattern_factors_by_category(category: str) -> list[PatternSpec]:
    """按类别获取形态因子"""
    return [f for f in PATTERN_FACTORS if f.category == category]


def print_pattern_factor_summary():
    """打印形态因子库汇总"""
    print("=" * 100)
    print("形态学因子库汇总")
    print("=" * 100)

    categories = {}
    for f in PATTERN_FACTORS:
        if f.category not in categories:
            categories[f.category] = []
        categories[f.category].append(f)

    for cat, factors in sorted(categories.items()):
        print(f"\n【{cat.upper()}】{len(factors)}个因子")
        for f in factors:
            print(f"  {f.name:<35} {f.description}")

    print(f"\n总计: {len(PATTERN_FACTORS)}个形态学因子")


if __name__ == "__main__":
    print_pattern_factor_summary()
