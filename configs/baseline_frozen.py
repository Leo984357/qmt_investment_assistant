"""
基线冻结配置

所有实验报告必须使用同一个基线进行比较。

基线定义：简单平均策略
- 模型: 等权平均
- 因子: 使用feature_set_v1的production因子
- 时间: 2022-09-01 ~ 2026-04-09
- 持仓: 15只股票
- 调仓: 每10天
- 成本: 佣金0.075% + 印花税0.1% + 滑点0.05%

基线性能指标（冻结）:
- 总收益: 21.2%
- Sharpe: 0.594
- IC: 0.068
- IC IR: 0.550
- 换手率: 0.83%
- 总成本: 1,092元

注意：
1. 这个基线是"研究基线"，不是"实盘基线"
2. 实盘基线需要单独验证
3. 所有实验必须与这个基线对比
"""

BASELINE_CONFIG = {
    # 基线标识
    'baseline_id': 'frozen_baseline_v1',
    'baseline_name': '简单平均策略（冻结）',
    
    # 模型配置
    'model': {
        'type': 'simple_average',
        'description': '等权平均所有因子Z-score'
    },
    
    # 特征配置
    'features': {
        'source': 'feature_set_v1',
        'factors': [
            'roe',
            'earnings_yield', 
            'operating_margin',
            'equity_growth',
            'ocf_per_share',
            'revenue_growth',
            'asset_turnover',
            'gross_margin',
        ]
    },
    
    # 组合配置
    'portfolio': {
        'top_n': 15,
        'rebalance_days': 10,
        'min_trade_value': 2000,
    },
    
    # 成本配置
    'cost': {
        'commission': 0.00075,
        'stamp_tax': 0.001,
        'slippage': 0.0005,
    },
    
    # 冻结的性能指标（来自历史实验）
    'frozen_metrics': {
        'total_return': 0.212,
        'annual_return': 0.051,
        'sharpe': 0.594,
        'max_drawdown': -0.182,
        'avg_rank_ic': 0.068,
        'ic_ir': 0.550,
        'avg_turnover': 0.0083,
        'total_cost': 1092,
        'excess_return': -0.107,
    },
    
    # 验证时间范围
    'validation_period': {
        'start': '2022-09-01',
        'end': '2026-04-09',
        'n_trading_days': 860,
    },
    
    # 比较规则
    'comparison_rules': {
        'sharpe_improvement_threshold': 0.2,  # Sharpe需提升20%才算显著
        'cost_tolerance': 0.5,  # 成本增加不超过50%
    }
}


def get_baseline_metrics():
    """获取冻结的基线指标"""
    return BASELINE_CONFIG['frozen_metrics'].copy()


def get_baseline_id():
    """获取基线ID"""
    return BASELINE_CONFIG['baseline_id']


def validate_experiment_against_baseline(experiment_metrics: dict) -> dict:
    """
    将实验指标与基线对比
    
    Args:
        experiment_metrics: 实验指标字典
    
    Returns:
        对比结果字典
    """
    baseline = BASELINE_CONFIG['frozen_metrics']
    rules = BASELINE_CONFIG['comparison_rules']
    
    comparison = {
        'baseline_id': BASELINE_CONFIG['baseline_id'],
        'pass': True,
        'reasons': [],
        'metrics': {}
    }
    
    # Sharpe对比
    exp_sharpe = experiment_metrics.get('sharpe', 0)
    base_sharpe = baseline['sharpe']
    sharpe_change = (exp_sharpe - base_sharpe) / base_sharpe if base_sharpe != 0 else 0
    
    comparison['metrics']['sharpe'] = {
        'baseline': base_sharpe,
        'experiment': exp_sharpe,
        'change_pct': sharpe_change * 100,
        'significant': sharpe_change >= rules['sharpe_improvement_threshold']
    }
    
    if sharpe_change < 0:
        comparison['pass'] = False
        comparison['reasons'].append(f'Sharpe下降 {-sharpe_change*100:.1f}%，劣于基线')
    
    # 成本对比
    exp_cost = experiment_metrics.get('total_cost', 0)
    base_cost = baseline['total_cost']
    cost_change = (exp_cost - base_cost) / base_cost if base_cost > 0 else 0
    
    comparison['metrics']['cost'] = {
        'baseline': base_cost,
        'experiment': exp_cost,
        'change_pct': cost_change * 100,
        'acceptable': cost_change <= rules['cost_tolerance']
    }
    
    if cost_change > rules['cost_tolerance']:
        comparison['pass'] = False
        comparison['reasons'].append(f'成本增加 {cost_change*100:.1f}%，超过容忍度')
    
    # 结论
    if comparison['pass']:
        if sharpe_change >= rules['sharpe_improvement_threshold']:
            comparison['conclusion'] = '显著优于基线'
        else:
            comparison['conclusion'] = '与基线相当'
    else:
        comparison['conclusion'] = '劣于基线'
    
    return comparison


if __name__ == '__main__':
    print("="*60)
    print("基线配置 (冻结版)")
    print("="*60)
    print(f"基线ID: {BASELINE_CONFIG['baseline_id']}")
    print(f"基线名称: {BASELINE_CONFIG['baseline_name']}")
    print()
    print("冻结指标:")
    for k, v in BASELINE_CONFIG['frozen_metrics'].items():
        if isinstance(v, float):
            if abs(v) < 1:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    print()
    print("比较规则:")
    print(f"  Sharpe提升阈值: {BASELINE_CONFIG['comparison_rules']['sharpe_improvement_threshold']*100:.0f}%")
    print(f"  成本容忍度: {BASELINE_CONFIG['comparison_rules']['cost_tolerance']*100:.0f}%")
