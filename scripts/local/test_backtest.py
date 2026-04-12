#!/usr/bin/env python
"""
快速回测验证脚本

测试回测系统是否能正常运行

Usage:
    python test_backtest.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.experiment.runner import run_experiment
from src.experiment.spec import load_experiment_spec
from datetime import datetime

def main():
    config_path = "configs/experiments/hs300_lightgbm.yaml"
    
    print("="*60)
    print("HS300 LightGBM 多因子策略 - 优化版回测")
    print("="*60)
    print()
    
    # 加载配置
    spec = load_experiment_spec(config_path)
    print("配置信息:")
    print(f"  回测周期: {spec.data.start_date} ~ {spec.data.end_date}")
    print(f"  调仓周期: {spec.backtest.rebalance_frequency_days}天")
    print(f"  持仓数量: {spec.portfolio.top_n}只")
    print(f"  初始资金: ¥{spec.backtest.initial_cash:,.0f}")
    print(f"  单票上限: {spec.portfolio.max_single_weight:.1%}")
    print(f"  模型: LightGBM")
    print(f"  因子: {spec.features.names}")
    print()
    print("开始回测...")
    print("="*60)
    print()
    
    start = datetime.now()
    
    try:
        result = run_experiment(config_path)
        
        elapsed = (datetime.now() - start).total_seconds()
        
        print()
        print("="*60)
        print("回测完成!")
        print(f"耗时: {elapsed:.1f}秒")
        print("="*60)
        
        # 输出关键指标
        if 'summary' in result:
            s = result['summary']
            print()
            print("【关键绩效指标】")
            print(f"  总收益:      {s.get('total_return', 0)*100:>8.2f}%")
            print(f"  年化收益:    {s.get('annual_return', 0)*100:>8.2f}%")
            print(f"  年化波动:    {s.get('annual_vol', 0)*100:>8.2f}%")
            print(f"  夏普比率:    {s.get('sharpe_like', 0):>8.2f}")
            print(f"  最大回撤:    {s.get('max_drawdown', 0)*100:>8.2f}%")
            if 'excess_total_return' in s:
                print(f"  超额收益:    {s.get('excess_total_return', 0)*100:>8.2f}%")
            if 'avg_rank_ic' in s:
                print(f"  平均IC:     {s.get('avg_rank_ic', 0):>8.4f}")
            
            print()
            print("【交易统计】")
            print(f"  总交易次数:  {s.get('trade_count', 0):>8}")
            print(f"  总手续费:    ¥{s.get('total_cost', 0):>8,.2f}")
            print(f"  平均换手率:  {s.get('avg_turnover', 0)*100:>8.2f}%")
        
        # 报告文件位置
        run_dir = result.get('run_dir')
        if run_dir:
            print()
            print(f"回测结果: {run_dir}")
        
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
