"""
多模型成本敏感性对比分析

对比简单平均、Ridge、Ridge+Support、LightGBM在不同成本情景下的表现
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime


def load_experiment(experiment_name):
    """加载实验数据"""
    base_dir = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs")
    
    # 找到对应的实验目录
    for d in base_dir.iterdir():
        if experiment_name in d.name:
            run_dir = d
            break
    else:
        raise ValueError(f"Experiment not found: {experiment_name}")
    
    nav = pd.read_parquet(run_dir / "backtest/nav_full.parquet")
    trades = pd.read_parquet(run_dir / "backtest/trades_full.parquet")
    benchmark_nav = pd.read_parquet(run_dir / "backtest/benchmark_nav_full.parquet")
    
    return nav, trades, benchmark_nav, run_dir.name


def analyze_experiment(name, nav, trades, benchmark_nav):
    """分析单个实验"""
    total_return = nav['nav'].iloc[-1] / nav['nav'].iloc[0] - 1
    benchmark_col = 'benchmark_nav' if 'benchmark_nav' in benchmark_nav.columns else 'nav'
    benchmark_return = benchmark_nav[benchmark_col].iloc[-1] / benchmark_nav[benchmark_col].iloc[0] - 1
    
    n_days = len(nav)
    annualization_factor = 252 / n_days
    
    # 计算日收益
    daily_returns = nav['nav'].pct_change().dropna()
    
    # 年化波动率
    annual_vol = daily_returns.std() * np.sqrt(252)
    
    # 夏普比率
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    
    # 计算成本
    if 'fee' in trades.columns:
        total_cost = trades['fee'].sum()
    elif 'trade_cost' in trades.columns:
        total_cost = trades['trade_cost'].sum()
    else:
        total_cost = 0
    
    # 换手率
    if 'turnover' in trades.columns:
        total_turnover = trades['turnover'].sum()
    elif 'turnover' in nav.columns:
        total_turnover = nav['turnover'].sum()
    else:
        total_turnover = 0
    
    return {
        'name': name,
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'excess_return': total_return - benchmark_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'total_cost': total_cost,
        'total_turnover': total_turnover,
        'n_days': n_days,
    }


def cost_sensitivity_scenarios(base_metrics):
    """计算不同成本情景下的表现"""
    scenarios = [
        {'name': '5bp成本', 'cost_factor': 5/15.75},
        {'name': '10bp成本', 'cost_factor': 10/15.75},
        {'name': '20bp成本', 'cost_factor': 20/15.75},
        {'name': '30bp成本', 'cost_factor': 30/15.75},
        {'name': '50bp成本', 'cost_factor': 50/15.75},
    ]
    
    results = []
    for scenario in scenarios:
        adjusted_cost = base_metrics['total_cost'] * scenario['cost_factor']
        adjusted_return = base_metrics['total_return'] - adjusted_cost / 1000000
        
        results.append({
            'scenario': scenario['name'],
            'cost_factor': scenario['cost_factor'],
            'total_cost': adjusted_cost,
            'total_return': adjusted_return,
            'excess_return': adjusted_return - base_metrics['benchmark_return'],
            'still_profitable': adjusted_return > 0,
            'beats_benchmark': adjusted_return > base_metrics['benchmark_return'],
        })
    
    return results


def main():
    experiments = [
        ('hs300_baseline_diversified_20260411_174815_2aac5224', '简单平均'),
        ('hs300_ridge_baseline_20260411_235534_93a8f463', 'Ridge基线'),
        ('hs300_ridge_with_support_20260412_000435_8bad53b6', 'Ridge+Support'),
    ]
    
    print("="*80)
    print("多模型成本敏感性对比分析")
    print("="*80)
    print()
    
    all_results = []
    
    for exp_name, label in experiments:
        try:
            nav, trades, benchmark_nav, run_name = load_experiment(exp_name)
            metrics = analyze_experiment(label, nav, trades, benchmark_nav)
            
            print(f"\n{label}:")
            print("-"*60)
            print(f"  总收益: {metrics['total_return']:.2%}")
            print(f"  基准收益: {metrics['benchmark_return']:.2%}")
            print(f"  超额收益: {metrics['excess_return']:.2%}")
            print(f"  夏普比率: {metrics['sharpe']:.3f}")
            print(f"  总成本: {metrics['total_cost']:,.0f}元")
            print(f"  总换手: {metrics['total_turnover']:.1%}")
            
            # 成本情景分析
            scenarios = cost_sensitivity_scenarios(metrics)
            
            print(f"\n  成本敏感性:")
            for s in scenarios:
                status = "✅" if s['still_profitable'] else "❌"
                beat = "✅" if s['beats_benchmark'] else "❌"
                print(f"    {s['scenario']}: 收益{s['total_return']:.2%}, 成本{s['total_cost']:,.0f}元 {status}盈利 {beat}跑赢")
            
            all_results.append({
                'label': label,
                'metrics': metrics,
                'scenarios': scenarios,
            })
            
        except Exception as e:
            print(f"  Error loading {label}: {e}")
    
    # 对比表格
    print("\n" + "="*80)
    print("成本敏感性对比总结")
    print("="*80)
    print()
    print(f"{'模型':<20} {'原始收益':<12} {'10bp成本':<12} {'20bp成本':<12} {'30bp成本':<12}")
    print("-"*68)
    
    for r in all_results:
        label = r['label']
        metrics = r['metrics']
        scenarios = {s['scenario']: s for s in r['scenarios']}
        
        orig = f"{metrics['total_return']:.1%}"
        cost10 = f"{scenarios['10bp成本']['total_return']:.1%}"
        cost20 = f"{scenarios['20bp成本']['total_return']:.1%}"
        cost30 = f"{scenarios['30bp成本']['total_return']:.1%}"
        
        print(f"{label:<20} {orig:<12} {cost10:<12} {cost20:<12} {cost30:<12}")
    
    # 生存阈值对比
    print("\n" + "="*80)
    print("成本效率对比")
    print("="*80)
    print()
    print(f"{'模型':<20} {'总成本(元)':<15} {'换手率':<12} {'成本/换手':<15} {'每bp成本(元)':<15}")
    print("-"*77)
    
    for r in all_results:
        label = r['label']
        metrics = r['metrics']
        
        cost_per_turnover = metrics['total_cost'] / metrics['total_turnover'] if metrics['total_turnover'] > 0 else 0
        cost_per_bp = metrics['total_cost'] / metrics['total_turnover'] / 10000 if metrics['total_turnover'] > 0 else 0
        
        print(f"{label:<20} {metrics['total_cost']:<15,.0f} {metrics['total_turnover']:<12.1%} {cost_per_turnover:<15,.0f} {cost_per_bp:<15,.1f}")
    
    # 保存结果
    output_dir = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/cost_sensitivity_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparison_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")
    
    return all_results


if __name__ == "__main__":
    all_results = main()
