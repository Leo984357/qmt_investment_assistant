"""
资金流因子库 - 40个因子

覆盖: 大单资金流、订单分类、资金流向、资金博弈等四大类
数据来源: akshare资金流数据 (东方财富)
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MoneyFlowFactor:
    """资金流因子定义"""
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


MONEY_FLOW_FACTORS: List[MoneyFlowFactor] = [

    # ========== 一、大单资金流 (12个) ==========
    
    # 超大单
    MoneyFlowFactor(
        name="super_large_net_flow",
        category="large_order_flow",
        sub_category="super_large",
        description="超大单净流入",
        economic_interpretation="机构大资金动向",
        lookback=1,
        data_requirement=["super_large_buy", "super_large_sell"],
        formula="super_large_buy - super_large_sell",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="super_large_net_ratio",
        category="large_order_flow",
        sub_category="super_large",
        description="超大单净流入率",
        economic_interpretation="超大单占比",
        lookback=1,
        data_requirement=["super_large_buy", "super_large_sell", "total_volume"],
        formula="(super_large_buy - super_large_sell) / total_volume",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="super_large_buy_pct",
        category="large_order_flow",
        sub_category="super_large",
        description="超大单买入占比",
        economic_interpretation="超大单买入比例",
        lookback=1,
        data_requirement=["super_large_buy", "total_volume"],
        formula="super_large_buy / total_volume",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="super_large_5d_net_flow",
        category="large_order_flow",
        sub_category="super_large",
        description="超大单5日净流入",
        economic_interpretation="超大单中期动向",
        lookback=5,
        data_requirement=["super_large_buy", "super_large_sell"],
        formula="sum(super_large_net_flow, 5d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="super_large_20d_net_flow",
        category="large_order_flow",
        sub_category="super_large",
        description="超大单20日净流入",
        economic_interpretation="超大单长期动向",
        lookback=20,
        data_requirement=["super_large_buy", "super_large_sell"],
        formula="sum(super_large_net_flow, 20d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    
    # 大单
    MoneyFlowFactor(
        name="large_net_flow",
        category="large_order_flow",
        sub_category="large",
        description="大单净流入",
        economic_interpretation="大资金动向",
        lookback=1,
        data_requirement=["large_buy", "large_sell"],
        formula="large_buy - large_sell",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="large_net_ratio",
        category="large_order_flow",
        sub_category="large",
        description="大单净流入率",
        economic_interpretation="大单占比",
        lookback=1,
        data_requirement=["large_buy", "large_sell", "total_volume"],
        formula="(large_buy - large_sell) / total_volume",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="large_5d_net_flow",
        category="large_order_flow",
        sub_category="large",
        description="大单5日净流入",
        economic_interpretation="大单中期动向",
        lookback=5,
        data_requirement=["large_buy", "large_sell"],
        formula="sum(large_net_flow, 5d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="large_20d_net_flow",
        category="large_order_flow",
        sub_category="large",
        description="大单20日净流入",
        economic_interpretation="大单长期动向",
        lookback=20,
        data_requirement=["large_buy", "large_sell"],
        formula="sum(large_net_flow, 20d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    
    # 合计大单
    MoneyFlowFactor(
        name="big_order_net_flow",
        category="large_order_flow",
        sub_category="combined",
        description="大单合计净流入 (超大+大)",
        economic_interpretation="机构资金总流向",
        lookback=1,
        data_requirement=["super_large", "large"],
        formula="super_large_net + large_net",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="big_order_5d_net_flow",
        category="large_order_flow",
        sub_category="combined",
        description="大单5日净流入合计",
        economic_interpretation="机构中期动向",
        lookback=5,
        data_requirement=["super_large", "large"],
        formula="sum(big_order_net_flow, 5d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="big_order_20d_net_flow",
        category="large_order_flow",
        sub_category="combined",
        description="大单20日净流入合计",
        economic_interpretation="机构长期动向",
        lookback=20,
        data_requirement=["super_large", "large"],
        formula="sum(big_order_net_flow, 20d)",
        ic_direction="positive",
        update_frequency="daily",
    ),

    # ========== 二、订单分类流 (10个) ==========
    
    # 中单
    MoneyFlowFactor(
        name="medium_net_flow",
        category="order_classification",
        sub_category="medium",
        description="中单净流入",
        economic_interpretation="中等资金动向",
        lookback=1,
        data_requirement=["medium_buy", "medium_sell"],
        formula="medium_buy - medium_sell",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="medium_5d_net_flow",
        category="order_classification",
        sub_category="medium",
        description="中单5日净流入",
        economic_interpretation="中等资金中期动向",
        lookback=5,
        data_requirement=["medium_buy", "medium_sell"],
        formula="sum(medium_net_flow, 5d)",
        ic_direction="conditional",
        update_frequency="daily",
    ),
    
    # 小单
    MoneyFlowFactor(
        name="small_net_flow",
        category="order_classification",
        sub_category="small",
        description="小单净流入",
        economic_interpretation="散户资金动向",
        lookback=1,
        data_requirement=["small_buy", "small_sell"],
        formula="small_buy - small_sell",
        ic_direction="negative",  # 散户净流入通常负面
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="small_5d_net_flow",
        category="order_classification",
        sub_category="small",
        description="小单5日净流入",
        economic_interpretation="散户中期动向",
        lookback=5,
        data_requirement=["small_buy", "small_sell"],
        formula="sum(small_net_flow, 5d)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="small_20d_net_flow",
        category="order_classification",
        sub_category="small",
        description="小单20日净流入",
        economic_interpretation="散户长期动向",
        lookback=20,
        data_requirement=["small_buy", "small_sell"],
        formula="sum(small_net_flow, 20d)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    
    # 净流入占比
    MoneyFlowFactor(
        name="inflow_ratio",
        category="order_classification",
        sub_category="ratio",
        description="主动买入占比",
        economic_interpretation="买入量/总成交量",
        lookback=1,
        data_requirement=["buy_volume", "sell_volume"],
        formula="buy_volume / (buy_volume + sell_volume)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="inflow_5d_ratio",
        category="order_classification",
        sub_category="ratio",
        description="5日主动买入占比均值",
        economic_interpretation="中期主动买入",
        lookback=5,
        data_requirement=["buy_volume", "sell_volume"],
        formula="mean(inflow_ratio, 5d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="inflow_20d_ratio",
        category="order_classification",
        sub_category="ratio",
        description="20日主动买入占比均值",
        economic_interpretation="长期主动买入",
        lookback=20,
        data_requirement=["buy_volume", "sell_volume"],
        formula="mean(inflow_ratio, 20d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="institutional_buy_ratio",
        category="order_classification",
        sub_category="ratio",
        description="机构买入占比",
        economic_interpretation="机构资金占总成交",
        lookback=1,
        data_requirement=["big_order_flow", "total_volume"],
        formula="big_order_flow / total_volume",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="retail_sell_ratio",
        category="order_classification",
        sub_category="ratio",
        description="散户卖出占比",
        economic_interpretation="散户资金占总成交",
        lookback=1,
        data_requirement=["small_net_flow", "total_volume"],
        formula="-small_net_flow / total_volume",
        ic_direction="negative",
        update_frequency="daily",
    ),

    # ========== 三、资金流向指标 (10个) ==========
    
    # 主力资金
    MoneyFlowFactor(
        name="main_force_net_flow",
        category="fund_flow_indicators",
        sub_category="main_force",
        description="主力资金净流入",
        economic_interpretation="主力资金总动向",
        lookback=1,
        data_requirement=["main_force_buy", "main_force_sell"],
        formula="main_force_buy - main_force_sell",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="main_force_5d_net_flow",
        category="fund_flow_indicators",
        sub_category="main_force",
        description="主力资金5日净流入",
        economic_interpretation="主力中期动向",
        lookback=5,
        data_requirement=["main_force_net_flow"],
        formula="sum(main_force_net_flow, 5d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="main_force_10d_net_flow",
        category="fund_flow_indicators",
        sub_category="main_force",
        description="主力资金10日净流入",
        economic_interpretation="主力中期动向",
        lookback=10,
        data_requirement=["main_force_net_flow"],
        formula="sum(main_force_net_flow, 10d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="main_force_20d_net_flow",
        category="fund_flow_indicators",
        sub_category="main_force",
        description="主力资金20日净流入",
        economic_interpretation="主力长期动向",
        lookback=20,
        data_requirement=["main_force_net_flow"],
        formula="sum(main_force_net_flow, 20d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="main_force_pct",
        category="fund_flow_indicators",
        sub_category="main_force",
        description="主力资金占比",
        economic_interpretation="主力资金/流通市值",
        lookback=1,
        data_requirement=["main_force_net_flow", "float_market_cap"],
        formula="main_force_net_flow / float_market_cap",
        ic_direction="positive",
        update_frequency="daily",
    ),
    
    # 资金流强度
    MoneyFlowFactor(
        name="flow_strength",
        category="fund_flow_indicators",
        sub_category="strength",
        description="资金流强度",
        economic_interpretation="资金流向相对量",
        lookback=5,
        data_requirement=["net_flow", "volume"],
        formula="net_flow / avg_volume",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="flow_acceleration",
        category="fund_flow_indicators",
        sub_category="strength",
        description="资金流加速度",
        economic_interpretation="资金流变化趋势",
        lookback=10,
        data_requirement=["net_flow"],
        formula="(flow_5d - flow_10d_ago) / abs(flow_10d_ago)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="flow_momentum",
        category="fund_flow_indicators",
        sub_category="strength",
        description="资金流动量",
        economic_interpretation="资金流趋势强度",
        lookback=20,
        data_requirement=["net_flow"],
        formula="flow_5d / flow_20d - 1",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="flow_consistency",
        category="fund_flow_indicators",
        sub_category="strength",
        description="资金流入一致性",
        economic_interpretation="连续净流入天数",
        lookback=20,
        data_requirement=["net_flow"],
        formula="count(net_flow > 0, last_20d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="net_flow_rank",
        category="fund_flow_indicators",
        sub_category="strength",
        description="资金流排名分位",
        economic_interpretation="在市场中的相对位置",
        lookback=1,
        data_requirement=["net_flow"],
        formula="percentile_rank(net_flow, all_stocks)",
        ic_direction="positive",
        update_frequency="daily",
    ),

    # ========== 四、资金博弈 (8个) ==========
    
    # 多空博弈
    MoneyFlowFactor(
        name="buy_sell_imbalance",
        category="fund_game",
        sub_category="multi_short",
        description="买卖不平衡度",
        economic_interpretation="买入vs卖出的不平衡",
        lookback=1,
        data_requirement=["buy_volume", "sell_volume"],
        formula="(buy - sell) / (buy + sell)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="large_vs_small_flow",
        category="fund_game",
        sub_category="multi_short",
        description="大单vs小单比率",
        economic_interpretation="机构vs散户资金对比",
        lookback=1,
        data_requirement=["big_order_flow", "small_order_flow"],
        formula="big_order_flow / (-small_order_flow + 1)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="institutional_pressure",
        category="fund_game",
        sub_category="multi_short",
        description="机构压力指数",
        economic_interpretation="机构卖出压力",
        lookback=5,
        data_requirement=["big_order_sell"],
        formula="sum(big_order_sell, 5d) / sum(volume, 5d)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="retail_activity",
        category="fund_game",
        sub_category="multi_short",
        description="散户活跃度",
        economic_interpretation="小单交易占比",
        lookback=1,
        data_requirement=["small_volume", "total_volume"],
        formula="small_volume / total_volume",
        ic_direction="negative",
        update_frequency="daily",
    ),
    
    # 连续净流入/流出
    MoneyFlowFactor(
        name="consecutive_inflow_days",
        category="fund_game",
        sub_category="consecutive",
        description="连续净流入天数",
        economic_interpretation="资金持续流入",
        lookback=20,
        data_requirement=["net_flow"],
        formula="max_consecutive(net_flow > 0)",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="consecutive_outflow_days",
        category="fund_game",
        sub_category="consecutive",
        description="连续净流出天数",
        economic_interpretation="资金持续流出",
        lookback=20,
        data_requirement=["net_flow"],
        formula="max_consecutive(net_flow < 0)",
        ic_direction="negative",
        update_frequency="daily",
    ),
    
    # 资金流反转
    MoneyFlowFactor(
        name="flow_reversal",
        category="fund_game",
        sub_category="reversal",
        description="资金流反转信号",
        economic_interpretation="连续流出后首日流入",
        lookback=10,
        data_requirement=["net_flow"],
        formula="net_flow > 0 AND sum(net_flow_last_5d) < 0",
        ic_direction="positive",
        update_frequency="daily",
    ),
    MoneyFlowFactor(
        name="flow_divergence",
        category="fund_game",
        sub_category="reversal",
        description="资金流与价格背离",
        economic_interpretation="价跌量涨/价涨量跌",
        lookback=10,
        data_requirement=["net_flow", "price"],
        formula="corr(net_flow, returns, 10d)",
        ic_direction="positive",
        update_frequency="daily",
    ),
]


def get_money_flow_factors() -> List[MoneyFlowFactor]:
    """获取资金流因子列表"""
    return MONEY_FLOW_FACTORS


def get_money_flow_factors_by_category(category: str) -> List[MoneyFlowFactor]:
    """按类别获取资金流因子"""
    return [f for f in MONEY_FLOW_FACTORS if f.category == category]


def get_money_flow_factor_names() -> List[str]:
    """获取所有资金流因子名称"""
    return [f.name for f in MONEY_FLOW_FACTORS]


def print_money_flow_factor_summary():
    """打印资金流因子汇总"""
    print("=" * 80)
    print("资金流因子库汇总")
    print("=" * 80)
    print(f"总计: {len(MONEY_FLOW_FACTORS)}个因子")
    print()
    
    categories = {}
    for f in MONEY_FLOW_FACTORS:
        if f.category not in categories:
            categories[f.category] = []
        categories[f.category].append(f)
    
    for cat, factors in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"【{cat}】{len(factors)}个")
        for f in factors[:3]:
            print(f"  - {f.name}: {f.description}")
        if len(factors) > 3:
            print(f"  ... 还有{len(factors)-3}个")
        print()


if __name__ == "__main__":
    print_money_flow_factor_summary()
