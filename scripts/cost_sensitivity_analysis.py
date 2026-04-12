"""
成本敏感性分析脚本 - Ridge+Support

在5/10/20/30bp多情景下测试策略是否仍能盈利
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime


def load_experiment_data():
    """加载实验数据"""
    run_dir = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs/hs300_ridge_with_support_20260412_000435_8bad53b6")
    
    nav = pd.read_parquet(run_dir / "backtest/nav_full.parquet")
    trades = pd.read_parquet(run_dir / "backtest/trades_full.parquet")
    positions = pd.read_parquet(run_dir / "backtest/positions_full.parquet")
    benchmark_nav = pd.read_parquet(run_dir / "backtest/benchmark_nav_full.parquet")
    
    return nav, trades, positions, benchmark_nav


def cost_sensitivity_analysis(nav, trades, benchmark_nav):
    """
    成本敏感性分析
    
    分析不同成本情景下的策略表现
    """
    # 基础指标
    total_return = nav['nav'].iloc[-1] / nav['nav'].iloc[0] - 1
    n_days = len(nav)
    annualization_factor = 252 / n_days
    
    # 计算日收益率
    nav['daily_return'] = nav['nav'].pct_change()
    
    # 基准收益 (使用正确的列名)
    benchmark_col = 'benchmark_nav' if 'benchmark_nav' in benchmark_nav.columns else 'nav'
    benchmark_return = benchmark_nav[benchmark_col].iloc[-1] / benchmark_nav[benchmark_col].iloc[0] - 1
    benchmark_annual = (1 + benchmark_return) ** annualization_factor - 1
    
    # 原始成本
    base_cost = trades['fee'].sum() if 'fee' in trades.columns else 0
    
    # 计算换手率
    if 'turnover' in trades.columns:
        total_turnover = trades['turnover'].sum()
    else:
        total_turnover = 0
    
    print("="*70)
    print("成本敏感性分析 - Ridge+Support")
    print("="*70)
    print()
    
    # 定义成本情景
    scenarios = [
        {'name': '乐观 (真实成交)', 'commission': 0.75, 'stamp': 10, 'slippage': 3},
        {'name': '基准 (模拟)', 'commission': 0.75, 'stamp': 10, 'slippage': 5},
        {'name': '10bp成本', 'commission': 0.75, 'stamp': 10, 'slippage': 10},
        {'name': '20bp成本', 'commission': 1.0, 'stamp': 10, 'slippage': 20},
        {'name': '30bp成本', 'commission': 1.5, 'stamp': 10, 'slippage': 30},
        {'name': '50bp成本', 'commission': 2.0, 'stamp': 10, 'slippage': 50},
    ]
    
    results = []
    
    for scenario in scenarios:
        comm = scenario['commission']
        stamp = scenario['stamp']
        slippage = scenario['slippage']
        total_bps = comm + stamp + slippage
        
        # 重新计算成本
        # 假设原始成本按比例缩放
        scale = total_bps / 16.75  # 基准 = 0.75 + 10 + 5 = 15.75
        
        if base_cost > 0:
            adjusted_cost = base_cost * scale
        else:
            # 估算成本 (基于换手)
            adjusted_cost = 1000000 * total_turnover * total_bps / 10000
        
        # 调整后收益
        adjusted_return = total_return - adjusted_cost / 1000000
        
        # 年化收益
        annual_return = (1 + total_return) ** annualization_factor - 1
        adjusted_annual = (1 + adjusted_return) ** annualization_factor - 1
        
        # 超额收益
        excess_return = total_return - benchmark_return
        adjusted_excess = adjusted_return - benchmark_return
        
        # 夏普比率
        daily_mean = nav['daily_return'].mean()
        daily_std = nav['daily_return'].std()
        sharpe = daily_mean / daily_std * np.sqrt(252) if daily_std > 0 else 0
        
        # 调整后夏普
        adjusted_daily_return = adjusted_return / n_days
        adjusted_sharpe = adjusted_daily_return / daily_std * np.sqrt(252) if daily_std > 0 else 0
        
        results.append({
            'scenario': scenario['name'],
            'commission_bps': comm,
            'stamp_bps': stamp,
            'slippage_bps': slippage,
            'total_bps': total_bps,
            'base_cost': base_cost,
            'adjusted_cost': adjusted_cost,
            'total_return': total_return,
            'adjusted_return': adjusted_return,
            'annual_return': annual_return,
            'adjusted_annual': adjusted_annual,
            'excess_return': excess_return,
            'adjusted_excess': adjusted_excess,
            'sharpe': sharpe,
            'adjusted_sharpe': adjusted_sharpe,
            'still_profitable': adjusted_return > 0,
            'beats_benchmark': adjusted_excess > 0,
        })
        
        print(f"{scenario['name']}:")
        print(f"  成本: {total_bps}bp (佣金{comm}+印花{stamp}+滑点{slippage})")
        print(f"  估算成本: {adjusted_cost:,.0f}元")
        print(f"  原始收益: {total_return:.2%} → 调整后: {adjusted_return:.2%}")
        print(f"  年化收益: {annual_return:.2%} → {adjusted_annual:.2%}")
        print(f"  超额收益: {excess_return:.2%} → {adjusted_excess:.2%}")
        print(f"  夏普比率: {sharpe:.3f} → {adjusted_sharpe:.3f}")
        print(f"  仍盈利: {'✅' if adjusted_return > 0 else '❌'} 跑赢基准: {'✅' if adjusted_excess > 0 else '❌'}")
        print()
    
    return pd.DataFrame(results)


def turnover_sensitivity_analysis(trades, nav):
    """
    换手率敏感性分析
    
    分析不同换手率水平下的成本负担
    """
    print("="*70)
    print("换手率敏感性分析")
    print("="*70)
    print()
    
    # 计算实际换手率
    if 'turnover' in trades.columns:
        # 按月聚合
        trades['month'] = pd.to_datetime(trades['trade_date']).dt.to_period('M')
        monthly_turnover = trades.groupby('month')['turnover'].sum()
        
        avg_turnover = monthly_turnover.mean()
        max_turnover = monthly_turnover.max()
        min_turnover = monthly_turnover.min()
    else:
        avg_turnover = 0.03  # 假设3%
        max_turnover = 0.06
        min_turnover = 0.01
    
    print(f"月均换手率: {avg_turnover:.2%}")
    print(f"最大月换手: {max_turnover:.2%}")
    print(f"最小月换手: {min_turnover:.2%}")
    print()
    
    # 换手率情景
    turnover_scenarios = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
    base_return = nav['nav'].iloc[-1] / nav['nav'].iloc[0] - 1
    
    print("不同换手率下的成本估算:")
    print("-"*50)
    
    results = []
    for turnover in turnover_scenarios:
        for cost_bps in [10, 20, 30]:
            cost = 1000000 * turnover * cost_bps / 10000
            adjusted_return = base_return - cost / 1000000
            results.append({
                'turnover': turnover,
                'cost_bps': cost_bps,
                'cost': cost,
                'adjusted_return': adjusted_return,
                'still_profitable': adjusted_return > 0,
            })
            print(f"  换手{turnover:.0%} × 成本{cost_bps}bp = {cost:,.0f}元, 收益{adjusted_return:.2%}")
    
    return pd.DataFrame(results)


def survival_threshold_analysis(nav, benchmark_nav):
    """
    生存阈值分析
    
    找到策略能承受的最大成本
    """
    print("="*70)
    print("生存阈值分析")
    print("="*70)
    print()
    
    total_return = nav['nav'].iloc[-1] / nav['nav'].iloc[0] - 1
    benchmark_col = 'benchmark_nav' if 'benchmark_nav' in benchmark_nav.columns else 'nav'
    benchmark_return = benchmark_nav[benchmark_col].iloc[-1] / benchmark_nav[benchmark_col].iloc[0] - 1
    
    excess_return = total_return - benchmark_return
    
    print(f"总收益: {total_return:.2%}")
    print(f"基准收益: {benchmark_return:.2%}")
    print(f"超额收益: {excess_return:.2%}")
    print()
    
    # 能承受的最大成本
    max_total_cost = total_return * 1000000
    max_cost_bps = max_total_cost / 1000000 * 10000
    
    print(f"能承受的最大总成本: {max_total_cost:,.0f}元 ({max_cost_bps:.0f}bp)")
    print()
    
    # 不同假设下的阈值
    print("不同成本结构下的最大可承受换手率:")
    print("-"*50)
    
    scenarios = [
        {'name': '佣金0.75+印花10', 'fixed': 10.75, 'variable': 1},
        {'name': '佣金1.0+印花10', 'fixed': 11.0, 'variable': 1},
        {'name': '佣金1.5+印花10', 'fixed': 11.5, 'variable': 1},
    ]
    
    results = []
    for scenario in scenarios:
        fixed = scenario['fixed']
        variable = scenario['variable']
        
        # 假设换手T, 成本 = 固定 + T * 可变
        # max_cost = fixed/10000 + T * variable/10000
        # T = (max_cost - fixed/10000) * 10000 / variable
        if variable > 0:
            max_turnover = (max_total_cost - fixed/10000 * 1000000) * 10000 / variable / 1000000
        else:
            max_turnover = float('inf')
        
        print(f"  {scenario['name']}: 最大换手率 = {max_turnover:.2%}")
        
        results.append({
            'name': scenario['name'],
            'fixed_cost': fixed,
            'variable_cost': variable,
            'max_turnover': max_turnover,
        })
    
    return pd.DataFrame(results)


def main():
    print("成本敏感性分析 - Ridge+Support")
    print("="*70)
    print()
    
    # 加载数据
    print("Loading data...")
    nav, trades, positions, benchmark_nav = load_experiment_data()
    print(f"  NAV records: {len(nav)}")
    print(f"  Trade records: {len(trades)}")
    print(f"  Position records: {len(positions)}")
    print(f"  Date range: {nav['trade_date'].min()} to {nav['trade_date'].max()}")
    print()
    
    # 成本敏感性分析
    print("\n" + "="*70)
    cost_results = cost_sensitivity_analysis(nav, trades, benchmark_nav)
    
    # 换手率敏感性分析
    print("\n" + "="*70)
    turnover_results = turnover_sensitivity_analysis(trades, nav)
    
    # 生存阈值分析
    print("\n" + "="*70)
    survival_results = survival_threshold_analysis(nav, benchmark_nav)
    
    # 保存结果
    output_dir = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/cost_sensitivity_ridge_support")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cost_results.to_csv(output_dir / "cost_scenarios.csv", index=False)
    turnover_results.to_csv(output_dir / "turnover_scenarios.csv", index=False)
    survival_results.to_csv(output_dir / "survival_threshold.csv", index=False)
    
    # 保存汇总
    summary = {
        'experiment': 'Ridge+Support',
        'run_date': datetime.now().isoformat(),
        'cost_results': cost_results.to_dict('records'),
        'turnover_results': turnover_results.to_dict('records'),
        'survival_results': survival_results.to_dict('records'),
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print()
    print("="*70)
    print(f"Results saved to {output_dir}")
    print("="*70)
    
    return cost_results, turnover_results, survival_results


if __name__ == "__main__":
    cost_results, turnover_results, survival_results = main()
