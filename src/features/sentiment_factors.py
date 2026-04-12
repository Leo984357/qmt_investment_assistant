"""
情绪/另类因子库 - 35个因子

覆盖: 舆情因子、关注度因子、波动率情绪、异常交易等四大类
数据来源: akshare财经新闻、东方财富资金流等
"""

from dataclasses import dataclass
from typing import List


@dataclass
class SentimentFactor:
    """情绪因子定义"""
    name: str
    category: str
    sub_category: str
    description: str
    economic_interpretation: str
    lookback: int
    data_requirement: List[str]
    formula: str
    ic_direction: str
    update_frequency: str = "daily"


SENTIMENT_FACTORS: List[SentimentFactor] = [

    # ========== 一、舆情因子 (10个) ==========
    
    SentimentFactor(
        name="news_sentiment",
        category="sentiment",
        sub_category="news",
        description="新闻舆情分数",
        economic_interpretation="正面vs负面新闻比例",
        lookback=1,
        data_requirement=["news_content"],
        formula="NLP_sentiment_score(news)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="news_sentiment_3d",
        category="sentiment",
        sub_category="news",
        description="3日平均新闻舆情",
        economic_interpretation="中期舆情",
        lookback=3,
        data_requirement=["news_content"],
        formula="mean(news_sentiment, 3d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="news_sentiment_7d",
        category="sentiment",
        sub_category="news",
        description="7日平均新闻舆情",
        economic_interpretation="周度舆情",
        lookback=7,
        data_requirement=["news_content"],
        formula="mean(news_sentiment, 7d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="news_count_7d",
        category="sentiment",
        sub_category="news",
        description="7日新闻数量",
        economic_interpretation="中期关注度",
        lookback=7,
        data_requirement=["news_count"],
        formula="sum(news_count, 7d)",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="news_acceleration",
        category="sentiment",
        sub_category="news",
        description="新闻加速",
        economic_interpretation="新闻数量变化趋势",
        lookback=7,
        data_requirement=["news_count"],
        formula="news_7d / news_14d - 1",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="negative_news_ratio",
        category="sentiment",
        sub_category="news",
        description="负面新闻占比",
        economic_interpretation="负面关注度",
        lookback=7,
        data_requirement=["news_sentiment"],
        formula="count(sentiment < 0) / count(total)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="news_sentiment_momentum",
        category="sentiment",
        sub_category="news",
        description="舆情动量",
        economic_interpretation="舆情变化趋势",
        lookback=14,
        data_requirement=["news_sentiment"],
        formula="sentiment_7d - sentiment_14d",
        ic_direction="positive",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="analyst_report_sentiment",
        category="sentiment",
        sub_category="news",
        description="研报舆情",
        economic_interpretation="研报情绪",
        lookback=1,
        data_requirement=["report_content"],
        formula="NLP_sentiment_score(reports)",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    SentimentFactor(
        name="social_media_buzz",
        category="sentiment",
        sub_category="social",
        description="社交媒体热度",
        economic_interpretation="社交媒体讨论量",
        lookback=1,
        data_requirement=["social_mentions"],
        formula="social_mentions / avg_social_mentions",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="search_index_trend",
        category="sentiment",
        sub_category="social",
        description="搜索指数趋势",
        economic_interpretation="搜索热度变化",
        lookback=7,
        data_requirement=["search_index"],
        formula="search_index / search_index_7d_ago - 1",
        ic_direction="conditional",
        update_frequency="daily",
    ),

    # ========== 二、关注度因子 (8个) ==========
    
    SentimentFactor(
        name="attention_rank",
        category="attention",
        sub_category="search",
        description="关注度排名",
        economic_interpretation="在同行业中的关注度",
        lookback=1,
        data_requirement=["attention_score"],
        formula="percentile_rank(attention, industry)",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="hot_stock_rank",
        category="attention",
        sub_category="search",
        description="人气排名",
        economic_interpretation="东方财富人气排名",
        lookback=1,
        data_requirement=["hot_rank"],
        formula="1 / hot_rank",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="hot_stock_change",
        category="attention",
        sub_category="search",
        description="人气变化",
        economic_interpretation="人气排名变化",
        lookback=5,
        data_requirement=["hot_rank"],
        formula="hot_rank_5d_ago - hot_rank_now",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="stock_mention_frequency",
        category="attention",
        sub_category="media",
        description="被提及频率",
        economic_interpretation="媒体/社交平台提及次数",
        lookback=7,
        data_requirement=["mention_count"],
        formula="sum(mention_count, 7d)",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="institutional_follow_count",
        category="attention",
        sub_category="media",
        description="机构关注度",
        economic_interpretation="机构研报数量",
        lookback=30,
        data_requirement=["report_count"],
        formula="count(reports, 30d)",
        ic_direction="positive",
        update_frequency="weekly",
    ),
    SentimentFactor(
        name="public_follower_count",
        category="attention",
        sub_category="media",
        description="公开关注度",
        economic_interpretation="雪球/东财粉丝数",
        lookback=0,
        data_requirement=["follower_count"],
        formula="log(follower_count + 1)",
        ic_direction="conditional",
        update_frequency="weekly",
    ),
    SentimentFactor(
        name="discussion_intensity",
        category="attention",
        sub_category="media",
        description="讨论强度",
        economic_interpretation="帖子回复数",
        lookback=1,
        data_requirement=["post_replies"],
        formula="sum(post_replies, 1d)",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="media_coverage_change",
        category="attention",
        sub_category="media",
        description="媒体覆盖变化",
        economic_interpretation="媒体报道数量变化",
        lookback=7,
        data_requirement=["media_count"],
        formula="media_count_7d - media_count_14d",
        ic_direction="conditional",
        update_frequency="daily",
    ),

    # ========== 三、波动率情绪 (10个) ==========
    
    SentimentFactor(
        name="volatility_sentiment",
        category="volatility_sentiment",
        sub_category="regime",
        description="波动率情绪",
        economic_interpretation="市场恐慌/贪婪",
        lookback=20,
        data_requirement=["hv20", "hv60"],
        formula="hv20 / hv60 - 1",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="vix_correlation",
        category="volatility_sentiment",
        sub_category="regime",
        description="VIX相关性",
        economic_interpretation="市场波动敏感度",
        lookback=20,
        data_requirement=["stock_return", "vix"],
        formula="corr(stock_return, vix_change)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="tail_risk_sentiment",
        category="volatility_sentiment",
        sub_category="regime",
        description="尾部风险情绪",
        economic_interpretation="极端收益频率",
        lookback=20,
        data_requirement=["daily_returns"],
        formula="count(|return| > 2*std, last_20d) / 20",
        ic_direction="negative",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="skew_sentiment",
        category="volatility_sentiment",
        sub_category="regime",
        description="偏度情绪",
        economic_interpretation="收益分布偏度",
        lookback=20,
        data_requirement=["daily_returns"],
        formula="skewness(daily_returns)",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="kurtosis_sentiment",
        category="volatility_sentiment",
        sub_category="regime",
        description="峰度情绪",
        economic_interpretation="收益分布峰度",
        lookback=20,
        data_requirement=["daily_returns"],
        formula="kurtosis(daily_returns)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="drawdown_sentiment",
        category="volatility_sentiment",
        sub_category="regime",
        description="回撤情绪",
        economic_interpretation="当前回撤程度",
        lookback=60,
        data_requirement=["nav"],
        formula="current_drawdown",
        ic_direction="negative",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="recovery_sentiment",
        category="volatility_sentiment",
        sub_category="regime",
        description="恢复情绪",
        economic_interpretation="从低点反弹幅度",
        lookback=60,
        data_requirement=["nav"],
        formula="(current_price - 60d_low) / 60d_low",
        ic_direction="positive",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="volatility_clustering",
        category="volatility_sentiment",
        sub_category="regime",
        description="波动聚集",
        economic_interpretation="波动是否持续",
        lookback=20,
        data_requirement=["returns"],
        formula="corr(|return_t|, |return_t-1|)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="regime_switch_prob",
        category="volatility_sentiment",
        sub_category="regime",
        description="状态切换概率",
        economic_interpretation="高波动状态概率",
        lookback=60,
        data_requirement=["hv20", "hv60"],
        formula="P(high_vol_regime)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="fear_greed_index",
        category="volatility_sentiment",
        sub_category="regime",
        description="恐慌贪婪指数",
        economic_interpretation="综合情绪指标",
        lookback=20,
        data_requirement=["multiple_signals"],
        formula="composite_fear_greed",
        ic_direction="positive",
        update_frequency="daily",
    ),

    # ========== 四、异常交易 (7个) ==========
    
    SentimentFactor(
        name="volume_spike",
        category="abnormal_trading",
        sub_category="volume",
        description="成交量突刺",
        economic_interpretation="成交量异常放大",
        lookback=20,
        data_requirement=["volume"],
        formula="volume / avg_volume_20d > 3",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="price_spike",
        category="abnormal_trading",
        sub_category="price",
        description="价格突刺",
        economic_interpretation="价格异常波动",
        lookback=20,
        data_requirement=["returns"],
        formula="|return| > 3 * std_20d",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="limit_up_count",
        category="abnormal_trading",
        sub_category="price",
        description="涨停次数",
        economic_interpretation="近期涨停",
        lookback=20,
        data_requirement=["limit_up"],
        formula="count(limit_up, last_20d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="limit_down_count",
        category="abnormal_trading",
        sub_category="price",
        description="跌停次数",
        economic_interpretation="近期跌停",
        lookback=20,
        data_requirement=["limit_down"],
        formula="count(limit_down, last_20d)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="abnormal_volume_ratio",
        category="abnormal_trading",
        sub_category="volume",
        description="异常成交量比率",
        economic_interpretation="相对历史平均",
        lookback=60,
        data_requirement=["volume"],
        formula="volume / avg_volume_60d - 1",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="bid_ask_spread",
        category="abnormal_trading",
        sub_category="liquidity",
        description="买卖价差",
        economic_interpretation="流动性指标",
        lookback=1,
        data_requirement=["bid", "ask"],
        formula="(ask - bid) / mid_price",
        ic_direction="negative",
        update_frequency="daily",
    ),
    SentimentFactor(
        name="order_imbalance",
        category="abnormal_trading",
        sub_category="liquidity",
        description="订单不平衡",
        economic_interpretation="买卖盘差异",
        lookback=1,
        data_requirement=["bid_volume", "ask_volume"],
        formula="(bid_volume - ask_volume) / (bid_volume + ask_volume)",
        ic_direction="conditional",
        update_frequency="daily",
    ),
]


def get_sentiment_factors() -> List[SentimentFactor]:
    """获取情绪因子列表"""
    return SENTIMENT_FACTORS


def get_sentiment_factor_names() -> List[str]:
    """获取所有情绪因子名称"""
    return [f.name for f in SENTIMENT_FACTORS]


def print_sentiment_factor_summary():
    """打印情绪因子汇总"""
    print("=" * 80)
    print("情绪/另类因子库汇总")
    print("=" * 80)
    print(f"总计: {len(SENTIMENT_FACTORS)}个因子")
    print()
    
    categories = {}
    for f in SENTIMENT_FACTORS:
        if f.category not in categories:
            categories[f.category] = []
        categories[f.category].append(f)
    
    for cat, factors in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"【{cat}】{len(factors)}个")
        for f in factors[:3]:
            print(f"  - {f.name}: {f.description}")
        print()


if __name__ == "__main__":
    print_sentiment_factor_summary()
