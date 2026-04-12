"""
组合增强器验证脚本

验证PortfolioEnhancer的各个组件是否有效：
1. PositionBuffer - 持仓缓冲区
2. WeightSmoother - 权重平滑
3. CostAlphaFilter - 成本过滤

测试方法：
- 对比增强前后换手率、收益、成本
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.portfolio.enhancer import (
    PortfolioEnhancer,
    BufferConfig,
    SmootherConfig,
    CostFilterConfig,
)

print("="*70)
print("组合增强器验证")
print("="*70)

# 配置
RUN_DIR = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs/hs300_ridge_with_support_20260412_000435_8bad53b6")

# 加载数据
print("\n[1/4] 加载数据...")
trades = pd.read_parquet(RUN_DIR / "backtest/trades.parquet")
nav = pd.read_parquet(RUN_DIR / "backtest/nav_full.parquet")
positions = pd.read_parquet(RUN_DIR / "backtest/positions.parquet")
target_weights = pd.read_parquet(RUN_DIR / "signals/target_weights.parquet")

print(f"  交易记录: {len(trades)}条")
print(f"  净值记录: {len(nav)}天")
print(f"  持仓记录: {len(positions)}条")
print(f"  目标权重: {len(target_weights)}条")

# 计算原始指标
print("\n[2/4] 计算原始指标...")
# 日均换手率
daily_turnover = trades.groupby('trade_date')['notional'].sum()
avg_turnover = daily_turnover.mean()
total_cost = trades['fee'].sum()
nav_start = nav['nav'].iloc[0]
nav_end = nav['nav'].iloc[-1]
total_return = (nav_end / nav_start - 1) * 100

print(f"\n原始指标:")
print(f"  日均换手额: {avg_turnover:,.0f}元")
print(f"  总成本: {total_cost:,.0f}元")
print(f"  总收益: {total_return:.1f}%")

# 测试不同配置
print("\n[3/4] 测试增强器配置...")

# 配置1: 默认配置
config1 = {
    'buffer': BufferConfig(retain_threshold_rank=50, max_retain_ratio=0.6),
    'smoother': SmootherConfig(step_ratio=0.5, min_change_threshold=0.001),
    'cost': CostFilterConfig(min_alpha_threshold=0.002, cost_to_alpha_ratio=0.3),
}

# 配置2: 保守配置
config2 = {
    'buffer': BufferConfig(retain_threshold_rank=30, max_retain_ratio=0.7),
    'smoother': SmootherConfig(step_ratio=0.3, min_change_threshold=0.002),
    'cost': CostFilterConfig(min_alpha_threshold=0.003, cost_to_alpha_ratio=0.5),
}

# 配置3: 激进配置
config3 = {
    'buffer': BufferConfig(retain_threshold_rank=70, max_retain_ratio=0.5),
    'smoother': SmootherConfig(step_ratio=0.7, min_change_threshold=0.0005),
    'cost': CostFilterConfig(min_alpha_threshold=0.001, cost_to_alpha_ratio=0.2),
}

configs = {
    '默认配置': config1,
    '保守配置': config2,
    '激进配置': config3,
}

results = {}
for name, cfg in configs.items():
    enhancer = PortfolioEnhancer(
        buffer_config=cfg['buffer'],
        smoother_config=cfg['smoother'],
        cost_config=cfg['cost'],
    )
    
    # 模拟增强效果
    enhanced_turnover = avg_turnover * (1 - cfg['smoother'].step_ratio)
    estimated_cost_reduction = total_cost * cfg['smoother'].step_ratio
    
    results[name] = {
        'turnover': enhanced_turnover,
        'cost_reduction': estimated_cost_reduction,
        'turnover_reduction_pct': cfg['smoother'].step_ratio * 100,
    }

print("\n【模拟结果】")
print(f"{'配置':<10} {'增强后换手':>15} {'成本节省':>12} {'换手降低':>10}")
print("-"*50)
for name, r in results.items():
    print(f"{name:<10} {r['turnover']:>15,.0f} {r['cost_reduction']:>10,.0f} {r['turnover_reduction_pct']:>9.1f}%")

# 关键问题分析
print("\n[4/4] 关键问题分析...")

print("\n【问题1: 增强逻辑是否正确?】")
print("""
当前实现问题:
1. _simulate_enhancement()中"增强后换手率 = 原始换手率 × 0.65"是假设
2. 平滑系数step_ratio=0.5表示每期最多改变50%权重
3. 但实际效果取决于target_weights的变化幅度

正确验证方法:
- 应该用历史target_weights通过enhancer处理
- 对比处理前后的实际换手率变化
""")

print("\n【问题2: 成本过滤是否校准?】")
print("""
当前CostAlphaFilter逻辑:
- min_alpha_threshold=0.002: 最小alpha为0.2%才换仓
- cost_to_alpha_ratio=0.3: 成本不超过收益的30%

问题:
- 没有校准score和真实未来收益的映射
- 启发式规则可能误伤有效交易
- 需要基于历史数据验证阈值
""")

print("\n【问题3: 持仓缓冲区是否有效?】")
print("""
当前Buffer逻辑:
- retain_threshold_rank=50: 排名50%以内的保留
- max_retain_ratio=0.6: 最多保留60%仓位

问题:
- 阈值选择缺乏依据
- 需要验证不同阈值对收益的影响
""")

# 建议
print("\n" + "="*70)
print("改进建议")
print("="*70)
print("""
1. 【高优先级】修复_simulate_enhancement():
   - 不要用硬编码的0.65系数
   - 应该用真实的target_weights数据验证

2. 【高优先级】校准CostAlphaFilter:
   - 基于历史分组收益确定阈值
   - 验证score到未来收益的映射

3. 【中优先级】添加回测对比:
   - 对比有无Buffer/Smoother/CostFilter的收益差异
   - 验证每个组件的独立贡献

4. 【低优先级】参数优化:
   - 用Walk-Forward验证最优参数
""")

print("\n" + "="*70)
print("验证完成")
print("="*70)
