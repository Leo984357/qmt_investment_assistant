"""
扩展技术因子库 - 从32个扩展到90个

覆盖: 均线系统、动量指标、波动率、成交量、趋势指标等五大类
"""

from dataclasses import dataclass


@dataclass
class ExtendedTechnicalFactor:
    """扩展技术因子定义"""
    name: str
    category: str
    sub_category: str
    description: str
    economic_interpretation: str
    lookback: int
    data_requirement: list[str]
    formula: str
    ic_direction: str


EXTENDED_TECHNICAL_FACTORS: list[ExtendedTechnicalFactor] = [

    # ========== 一、均线系统 (20个) ==========

    # 简单均线
    ExtendedTechnicalFactor(
        name="ma5",
        category="moving_average",
        sub_category="sma",
        description="5日简单均线",
        economic_interpretation="短期价格趋势",
        lookback=5,
        data_requirement=["close"],
        formula="mean(close, 5)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="ma10",
        category="moving_average",
        sub_category="sma",
        description="10日简单均线",
        economic_interpretation="短期价格趋势",
        lookback=10,
        data_requirement=["close"],
        formula="mean(close, 10)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="ma20",
        category="moving_average",
        sub_category="sma",
        description="20日简单均线",
        economic_interpretation="中期价格趋势",
        lookback=20,
        data_requirement=["close"],
        formula="mean(close, 20)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="ma60",
        category="moving_average",
        sub_category="sma",
        description="60日简单均线",
        economic_interpretation="中长期价格趋势",
        lookback=60,
        data_requirement=["close"],
        formula="mean(close, 60)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="ma120",
        category="moving_average",
        sub_category="sma",
        description="120日简单均线",
        economic_interpretation="长期价格趋势",
        lookback=120,
        data_requirement=["close"],
        formula="mean(close, 120)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="ma250",
        category="moving_average",
        sub_category="sma",
        description="250日简单均线",
        economic_interpretation="年线，长期趋势",
        lookback=250,
        data_requirement=["close"],
        formula="mean(close, 250)",
        ic_direction="conditional",
    ),

    # 均线交叉
    ExtendedTechnicalFactor(
        name="ma5_10_golden_cross",
        category="moving_average",
        sub_category="crossover",
        description="5日线上穿10日线",
        economic_interpretation="短期看涨信号",
        lookback=10,
        data_requirement=["close"],
        formula="ma5 > ma10 AND ma5_1 < ma10_1",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="ma5_20_death_cross",
        category="moving_average",
        sub_category="crossover",
        description="5日线下穿20日线",
        economic_interpretation="短期看跌信号",
        lookback=20,
        data_requirement=["close"],
        formula="ma5 < ma20 AND ma5_1 > ma20_1",
        ic_direction="negative",
    ),
    ExtendedTechnicalFactor(
        name="ma20_60_golden_cross",
        category="moving_average",
        sub_category="crossover",
        description="20日线上穿60日线",
        economic_interpretation="中期看涨信号",
        lookback=60,
        data_requirement=["close"],
        formula="ma20 > ma60 AND ma20_1 < ma60_1",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="ma20_60_death_cross",
        category="moving_average",
        sub_category="crossover",
        description="20日线下穿60日线",
        economic_interpretation="中期看跌信号",
        lookback=60,
        data_requirement=["close"],
        formula="ma20 < ma60 AND ma20_1 > ma60_1",
        ic_direction="negative",
    ),

    # 均线多头/空头排列
    ExtendedTechnicalFactor(
        name="ma_bull_alignment",
        category="moving_average",
        sub_category="alignment",
        description="均线多头排列强度",
        economic_interpretation="多头排列=强势",
        lookback=120,
        data_requirement=["close"],
        formula="(ma5 > ma10 > ma20 > ma60) * (ma5/ma60 - 1)",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="ma_bear_alignment",
        category="moving_average",
        sub_category="alignment",
        description="均线空头排列强度",
        economic_interpretation="空头排列=弱势",
        lookback=120,
        data_requirement=["close"],
        formula="(ma5 < ma10 < ma20 < ma60) * (1 - ma5/ma60)",
        ic_direction="negative",
    ),

    # 指数移动均线
    ExtendedTechnicalFactor(
        name="ema12",
        category="moving_average",
        sub_category="ema",
        description="12日指数均线",
        economic_interpretation="短期EMA",
        lookback=12,
        data_requirement=["close"],
        formula="ema(close, 12)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="ema26",
        category="moving_average",
        sub_category="ema",
        description="26日指数均线",
        economic_interpretation="中长期EMA",
        lookback=26,
        data_requirement=["close"],
        formula="ema(close, 26)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="ema9",
        category="moving_average",
        sub_category="ema",
        description="9日指数均线 (MACD用)",
        economic_interpretation="MACD信号线",
        lookback=9,
        data_requirement=["close"],
        formula="ema(close, 9)",
        ic_direction="conditional",
    ),

    # 加权均线
    ExtendedTechnicalFactor(
        name="wma5",
        category="moving_average",
        sub_category="wma",
        description="5日加权均线",
        economic_interpretation="近期权重更高",
        lookback=5,
        data_requirement=["close"],
        formula="wma(close, 5)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="wma20",
        category="moving_average",
        sub_category="wma",
        description="20日加权均线",
        economic_interpretation="近期权重更高",
        lookback=20,
        data_requirement=["close"],
        formula="wma(close, 20)",
        ic_direction="conditional",
    ),

    # 均线乖离
    ExtendedTechnicalFactor(
        name="ma5_bias",
        category="moving_average",
        sub_category="bias",
        description="5日均线乖离率",
        economic_interpretation="偏离均线的程度",
        lookback=5,
        data_requirement=["close"],
        formula="(close - ma5) / ma5",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="ma20_bias",
        category="moving_average",
        sub_category="bias",
        description="20日均线乖离率",
        economic_interpretation="偏离均线的程度",
        lookback=20,
        data_requirement=["close"],
        formula="(close - ma20) / ma20",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="ma250_bias",
        category="moving_average",
        sub_category="bias",
        description="250日均线乖离率",
        economic_interpretation="年线乖离，极端值可能反转",
        lookback=250,
        data_requirement=["close"],
        formula="(close - ma250) / ma250",
        ic_direction="conditional",
    ),

    # ========== 二、动量指标 (20个) ==========

    # RSI及其变体
    ExtendedTechnicalFactor(
        name="rsi6",
        category="momentum",
        sub_category="rsi",
        description="6日RSI",
        economic_interpretation="短期超买超卖",
        lookback=6,
        data_requirement=["close"],
        formula="100 - 100/(1 + rs)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="rsi12",
        category="momentum",
        sub_category="rsi",
        description="12日RSI",
        economic_interpretation="中期超买超卖",
        lookback=12,
        data_requirement=["close"],
        formula="100 - 100/(1 + rs)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="rsi24",
        category="momentum",
        sub_category="rsi",
        description="24日RSI",
        economic_interpretation="长期超买超卖",
        lookback=24,
        data_requirement=["close"],
        formula="100 - 100/(1 + rs)",
        ic_direction="conditional",
    ),

    # MACD及其变体
    ExtendedTechnicalFactor(
        name="macd",
        category="momentum",
        sub_category="macd",
        description="MACD柱 (DIF-DEA)",
        economic_interpretation="趋势动量",
        lookback=26,
        data_requirement=["close"],
        formula="ema12 - ema26",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="macd_signal",
        category="momentum",
        sub_category="macd",
        description="MACD信号线穿越",
        economic_interpretation="MACD穿越信号线",
        lookback=9,
        data_requirement=["close"],
        formula="macd - ema(macd, 9)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="macd_histogram",
        category="momentum",
        sub_category="macd",
        description="MACD柱状图变化",
        economic_interpretation="动量加速/减速",
        lookback=26,
        data_requirement=["close"],
        formula="macd_histogram - macd_histogram_1",
        ic_direction="conditional",
    ),

    # KDJ指标
    ExtendedTechnicalFactor(
        name="kdj_k",
        category="momentum",
        sub_category="kdj",
        description="KDJ K值",
        economic_interpretation="随机指标",
        lookback=9,
        data_requirement=["high", "low", "close"],
        formula="rsv的3日移动平均",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="kdj_d",
        category="momentum",
        sub_category="kdj",
        description="KDJ D值",
        economic_interpretation="随机指标",
        lookback=9,
        data_requirement=["high", "low", "close"],
        formula="K的3日移动平均",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="kdj_j",
        category="momentum",
        sub_category="kdj",
        description="KDJ J值",
        economic_interpretation="敏感指标",
        lookback=9,
        data_requirement=["high", "low", "close"],
        formula="3*K - 2*D",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="kdj_golden_cross",
        category="momentum",
        sub_category="kdj",
        description="KDJ金叉",
        economic_interpretation="K上穿D",
        lookback=9,
        data_requirement=["high", "low", "close"],
        formula="kdj_k > kdj_d AND kdj_k_1 < kdj_d_1",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="kdj_death_cross",
        category="momentum",
        sub_category="kdj",
        description="KDJ死叉",
        economic_interpretation="K下穿D",
        lookback=9,
        data_requirement=["high", "low", "close"],
        formula="kdj_k < kdj_d AND kdj_k_1 > kdj_d_1",
        ic_direction="negative",
    ),

    # WR威廉指标
    ExtendedTechnicalFactor(
        name="wr14",
        category="momentum",
        sub_category="wr",
        description="14日威廉指标",
        economic_interpretation="超买超卖",
        lookback=14,
        data_requirement=["high", "low", "close"],
        formula="(highest - close) / (highest - lowest)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="wr28",
        category="momentum",
        sub_category="wr",
        description="28日威廉指标",
        economic_interpretation="中期超买超卖",
        lookback=28,
        data_requirement=["high", "low", "close"],
        formula="(highest - close) / (highest - lowest)",
        ic_direction="conditional",
    ),

    # 动量变化率
    ExtendedTechnicalFactor(
        name="roc5",
        category="momentum",
        sub_category="roc",
        description="5日变化率",
        economic_interpretation="短期动量",
        lookback=5,
        data_requirement=["close"],
        formula="(close - close_5d_ago) / close_5d_ago",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="roc20",
        category="momentum",
        sub_category="roc",
        description="20日变化率",
        economic_interpretation="中期动量",
        lookback=20,
        data_requirement=["close"],
        formula="(close - close_20d_ago) / close_20d_ago",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="roc60",
        category="momentum",
        sub_category="roc",
        description="60日变化率",
        economic_interpretation="长期动量",
        lookback=60,
        data_requirement=["close"],
        formula="(close - close_60d_ago) / close_60d_ago",
        ic_direction="positive",
    ),

    # 其他动量
    ExtendedTechnicalFactor(
        name="cci14",
        category="momentum",
        sub_category="cci",
        description="14日顺势指标",
        economic_interpretation="超买超卖",
        lookback=14,
        data_requirement=["high", "low", "close"],
        formula="(typical_price - sma) / (0.015 * mad)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="cci28",
        category="momentum",
        sub_category="cci",
        description="28日顺势指标",
        economic_interpretation="中期超买超卖",
        lookback=28,
        data_requirement=["high", "low", "close"],
        formula="(typical_price - sma) / (0.015 * mad)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="trix12",
        category="momentum",
        sub_category="trix",
        description="12日TRIX",
        economic_interpretation="三重指数平均",
        lookback=12,
        data_requirement=["close"],
        formula="triple_ema的变动率",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="momentum_10",
        category="momentum",
        sub_category="momentum",
        description="10日动量",
        economic_interpretation="简单动量",
        lookback=10,
        data_requirement=["close"],
        formula="close - close_10d_ago",
        ic_direction="positive",
    ),

    # ========== 三、波动率指标 (15个) ==========

    # ATR及其变体
    ExtendedTechnicalFactor(
        name="atr14",
        category="volatility",
        sub_category="atr",
        description="14日平均真实波幅",
        economic_interpretation="日内波动幅度",
        lookback=14,
        data_requirement=["high", "low", "close"],
        formula="average_true_range",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="atr28",
        category="volatility",
        sub_category="atr",
        description="28日平均真实波幅",
        economic_interpretation="中期波动幅度",
        lookback=28,
        data_requirement=["high", "low", "close"],
        formula="average_true_range",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="atr_ratio",
        category="volatility",
        sub_category="atr",
        description="ATR/价格比率",
        economic_interpretation="相对波动率",
        lookback=14,
        data_requirement=["high", "low", "close"],
        formula="atr / close",
        ic_direction="conditional",
    ),

    # 布林带
    ExtendedTechnicalFactor(
        name="bb_upper",
        category="volatility",
        sub_category="bollinger",
        description="布林带上轨偏离",
        economic_interpretation="价格上轨关系",
        lookback=20,
        data_requirement=["close"],
        formula="(close - upper_band) / upper_band",
        ic_direction="negative",
    ),
    ExtendedTechnicalFactor(
        name="bb_lower",
        category="volatility",
        sub_category="bollinger",
        description="布林带下轨偏离",
        economic_interpretation="价格下轨关系",
        lookback=20,
        data_requirement=["close"],
        formula="(close - lower_band) / lower_band",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="bb_width",
        category="volatility",
        sub_category="bollinger",
        description="布林带宽度",
        economic_interpretation="波动率水平",
        lookback=20,
        data_requirement=["close"],
        formula="(upper - lower) / middle",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="bb_position",
        category="volatility",
        sub_category="bollinger",
        description="布林带位置",
        economic_interpretation="价格在布林带中的位置",
        lookback=20,
        data_requirement=["close"],
        formula="(close - lower) / (upper - lower)",
        ic_direction="conditional",
    ),

    # 历史波动率
    ExtendedTechnicalFactor(
        name="hv20",
        category="volatility",
        sub_category="historical_vol",
        description="20日历史波动率",
        economic_interpretation="实现波动率",
        lookback=20,
        data_requirement=["close"],
        formula="std(log(close/close_1)) * sqrt(252)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="hv60",
        category="volatility",
        sub_category="historical_vol",
        description="60日历史波动率",
        economic_interpretation="中期实现波动率",
        lookback=60,
        data_requirement=["close"],
        formula="std(log(close/close_1)) * sqrt(252)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="hv120",
        category="volatility",
        sub_category="historical_vol",
        description="120日历史波动率",
        economic_interpretation="长期实现波动率",
        lookback=120,
        data_requirement=["close"],
        formula="std(log(close/close_1)) * sqrt(252)",
        ic_direction="conditional",
    ),

    # 波动率变化
    ExtendedTechnicalFactor(
        name="vol_change",
        category="volatility",
        sub_category="vol_change",
        description="波动率变化 (短期/长期)",
        economic_interpretation="波动率是否扩大",
        lookback=60,
        data_requirement=["close"],
        formula="hv20 / hv60 - 1",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="vol_rank",
        category="volatility",
        sub_category="vol_rank",
        description="波动率历史分位",
        economic_interpretation="相对历史波动",
        lookback=250,
        data_requirement=["close"],
        formula="percentile_rank(hv20, 1y)",
        ic_direction="conditional",
    ),

    # 其他波动
    ExtendedTechnicalFactor(
        name="daily_range_pct",
        category="volatility",
        sub_category="range",
        description="日内振幅百分比",
        economic_interpretation="日内波动幅度",
        lookback=20,
        data_requirement=["high", "low", "close"],
        formula="(high - low) / close",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="close_to_high",
        category="volatility",
        sub_category="range",
        description="收盘价相对高低点位置",
        economic_interpretation="收盘在日内区间位置",
        lookback=20,
        data_requirement=["high", "low", "close"],
        formula="(close - low) / (high - low)",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="gap_pct",
        category="volatility",
        sub_category="gap",
        description="跳空幅度",
        economic_interpretation="跳空缺口",
        lookback=1,
        data_requirement=["open", "preclose"],
        formula="(open - preclose) / preclose",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="upper_shadow_pct",
        category="volatility",
        sub_category="candle",
        description="上影线比例",
        economic_interpretation="上影线长度",
        lookback=20,
        data_requirement=["high", "low", "close", "open"],
        formula="(high - max(open, close)) / (high - low)",
        ic_direction="negative",
    ),

    # ========== 四、成交量指标 (15个) ==========

    # 量价相关
    ExtendedTechnicalFactor(
        name="volume_price_trend",
        category="volume",
        sub_category="vp",
        description="量价趋势",
        economic_interpretation="价增量涨/跌",
        lookback=20,
        data_requirement=["close", "volume"],
        formula="sum((close - close_1) * volume)",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="money_flow_20d",
        category="volume",
        sub_category="money_flow",
        description="20日资金流向",
        economic_interpretation="累计资金净流入",
        lookback=20,
        data_requirement=["close", "volume"],
        formula="sum((close - close_1) * volume)",
        ic_direction="positive",
    ),

    # OBV及其变体
    ExtendedTechnicalFactor(
        name="obv",
        category="volume",
        sub_category="obv",
        description="能量潮指标",
        economic_interpretation="累计成交量",
        lookback=20,
        data_requirement=["close", "volume"],
        formula="cumsum(sign(close-change) * volume)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="obv_slope",
        category="volume",
        sub_category="obv",
        description="OBV斜率",
        economic_interpretation="成交量趋势",
        lookback=20,
        data_requirement=["close", "volume"],
        formula="slope(obv, 20)",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="obv_ma5_cross",
        category="volume",
        sub_category="obv",
        description="OBV上穿MA5",
        economic_interpretation="成交量放大信号",
        lookback=5,
        data_requirement=["close", "volume"],
        formula="obv > ma(obv, 5) AND obv_1 < ma(obv_1, 5)",
        ic_direction="positive",
    ),

    # VR成交量比
    ExtendedTechnicalFactor(
        name="vr14",
        category="volume",
        sub_category="vr",
        description="14日成交量比",
        economic_interpretation="量能水平",
        lookback=14,
        data_requirement=["volume"],
        formula="上升量/下降量 * 100",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="vr28",
        category="volume",
        sub_category="vr",
        description="28日成交量比",
        economic_interpretation="中期量能水平",
        lookback=28,
        data_requirement=["volume"],
        formula="上升量/下降量 * 100",
        ic_direction="conditional",
    ),

    # 量比
    ExtendedTechnicalFactor(
        name="volume_ratio",
        category="volume",
        sub_category="ratio",
        description="量比 (当日/5日平均)",
        economic_interpretation="相对成交量",
        lookback=5,
        data_requirement=["volume"],
        formula="volume / ma(volume, 5)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="volume_ratio_20",
        category="volume",
        sub_category="ratio",
        description="量比 (当日/20日平均)",
        economic_interpretation="中期相对成交量",
        lookback=20,
        data_requirement=["volume"],
        formula="volume / ma(volume, 20)",
        ic_direction="conditional",
    ),

    # 换手率
    ExtendedTechnicalFactor(
        name="turnover_rate",
        category="volume",
        sub_category="turnover",
        description="换手率",
        economic_interpretation="日换手比例",
        lookback=1,
        data_requirement=["volume", "float_shares"],
        formula="volume / float_shares",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="turnover_rate_5d",
        category="volume",
        sub_category="turnover",
        description="5日平均换手率",
        economic_interpretation="短期换手水平",
        lookback=5,
        data_requirement=["volume", "float_shares"],
        formula="ma(volume / float_shares, 5)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="turnover_rate_20d",
        category="volume",
        sub_category="turnover",
        description="20日平均换手率",
        economic_interpretation="中期换手水平",
        lookback=20,
        data_requirement=["volume", "float_shares"],
        formula="ma(volume / float_shares, 20)",
        ic_direction="conditional",
    ),

    # 量价背离
    ExtendedTechnicalFactor(
        name="price_volume_divergence",
        category="volume",
        sub_category="divergence",
        description="量价背离",
        economic_interpretation="价涨量缩/价跌量增",
        lookback=20,
        data_requirement=["close", "volume"],
        formula="corr(returns, volume_change, 20)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="volume_momentum",
        category="volume",
        sub_category="momentum",
        description="成交量动量",
        economic_interpretation="量能变化趋势",
        lookback=20,
        data_requirement=["volume"],
        formula="volume_ma20 / volume_ma60 - 1",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="volume_acceleration",
        category="volume",
        sub_category="acceleration",
        description="成交量加速度",
        economic_interpretation="量能变化加速",
        lookback=20,
        data_requirement=["volume"],
        formula="(vol_now - vol_5d_ago) - (vol_5d_ago - vol_10d_ago)",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="amihud_illiquidity",
        category="volume",
        sub_category="illiquidity",
        description="Amihud非流动性",
        economic_interpretation="流动性冲击成本",
        lookback=20,
        data_requirement=["volume", "close"],
        formula="mean(abs(return) / volume)",
        ic_direction="negative",
    ),

    # ========== 五、趋势指标 (10个) ==========

    # ADX
    ExtendedTechnicalFactor(
        name="adx14",
        category="trend",
        sub_category="adx",
        description="14日ADX",
        economic_interpretation="趋势强度",
        lookback=14,
        data_requirement=["high", "low", "close"],
        formula="average_directional_index",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="adx28",
        category="trend",
        sub_category="adx",
        description="28日ADX",
        economic_interpretation="中期趋势强度",
        lookback=28,
        data_requirement=["high", "low", "close"],
        formula="average_directional_index",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="plus_di",
        category="trend",
        sub_category="dmi",
        description="DMI+",
        economic_interpretation="上升趋势强度",
        lookback=14,
        data_requirement=["high", "low", "close"],
        formula="directional_indicator_plus",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="minus_di",
        category="trend",
        sub_category="dmi",
        description="DMI-",
        economic_interpretation="下降趋势强度",
        lookback=14,
        data_requirement=["high", "low", "close"],
        formula="directional_indicator_minus",
        ic_direction="conditional",
    ),
    ExtendedTechnicalFactor(
        name="adx_slope",
        category="trend",
        sub_category="adx",
        description="ADX斜率",
        economic_interpretation="趋势是否增强",
        lookback=14,
        data_requirement=["high", "low", "close"],
        formula="slope(adx, 10)",
        ic_direction="positive",
    ),

    # 趋势强度
    ExtendedTechnicalFactor(
        name="trend_strength",
        category="trend",
        sub_category="strength",
        description="趋势强度",
        economic_interpretation="R平方",
        lookback=20,
        data_requirement=["close"],
        formula="r_squared(close, time)",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="trend_persistence",
        category="trend",
        sub_category="persistence",
        description="趋势持续性",
        economic_interpretation="方向一致性",
        lookback=20,
        data_requirement=["close"],
        formula="sum(sign(return)) / n",
        ic_direction="positive",
    ),

    # 通道
    ExtendedTechnicalFactor(
        name="donchian_high",
        category="trend",
        sub_category="channel",
        description="唐奇安通道高点偏离",
        economic_interpretation="突破历史高点",
        lookback=20,
        data_requirement=["high"],
        formula="(close - highest(high, 20)) / highest(high, 20)",
        ic_direction="positive",
    ),
    ExtendedTechnicalFactor(
        name="donchian_low",
        category="trend",
        sub_category="channel",
        description="唐奇安通道低点偏离",
        economic_interpretation="跌破历史低点",
        lookback=20,
        data_requirement=["low"],
        formula="(close - lowest(low, 20)) / lowest(low, 20)",
        ic_direction="negative",
    ),
    ExtendedTechnicalFactor(
        name="keltner_position",
        category="trend",
        sub_category="channel",
        description="肯特纳通道位置",
        economic_interpretation="价格在通道中位置",
        lookback=20,
        data_requirement=["high", "low", "close"],
        formula="(close - middle) / (upper - lower)",
        ic_direction="conditional",
    ),
]


def get_extended_technical_factors() -> list[ExtendedTechnicalFactor]:
    """获取扩展技术因子列表"""
    return EXTENDED_TECHNICAL_FACTORS


def get_technical_factors_by_category(category: str) -> list[ExtendedTechnicalFactor]:
    """按类别获取技术因子"""
    return [f for f in EXTENDED_TECHNICAL_FACTORS if f.category == category]


def get_technical_factor_names() -> list[str]:
    """获取所有技术因子名称"""
    return [f.name for f in EXTENDED_TECHNICAL_FACTORS]


def print_technical_factor_summary():
    """打印技术因子汇总"""
    print("=" * 80)
    print("扩展技术因子库汇总")
    print("=" * 80)
    print(f"总计: {len(EXTENDED_TECHNICAL_FACTORS)}个因子")
    print()

    categories = {}
    for f in EXTENDED_TECHNICAL_FACTORS:
        if f.category not in categories:
            categories[f.category] = []
        categories[f.category].append(f)

    for cat, factors in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"【{cat}】{len(factors)}个")
        for f in factors[:5]:
            print(f"  - {f.name}: {f.description}")
        if len(factors) > 5:
            print(f"  ... 还有{len(factors)-5}个")
        print()


if __name__ == "__main__":
    print_technical_factor_summary()
