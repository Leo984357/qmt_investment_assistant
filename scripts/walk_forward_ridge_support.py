"""
Walk-Forward OOS验证脚本 - Ridge+Support

每个窗口:
1. 用过去500天数据训练Ridge模型
2. 在未来60天进行真实预测
3. 基于预测建仓并计算真实收益
4. 聚合所有窗口的OOS表现
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from pathlib import Path
import json
from datetime import datetime


def load_data():
    """加载实验数据"""
    run_dir = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs/hs300_ridge_with_support_20260412_000435_8bad53b6")
    
    dataset = pd.read_parquet(run_dir / "datasets/model_dataset.parquet")
    predictions = pd.read_parquet(run_dir / "signals/predictions.parquet")
    nav_full = pd.read_parquet(run_dir / "backtest/nav_full.parquet")
    benchmark_nav = pd.read_parquet(run_dir / "backtest/benchmark_nav_full.parquet")
    
    # 加载价格数据
    bars = pd.read_parquet("/Users/leolee/Desktop/qmt_investment_assistant/data/silver/daily_bar.parquet")
    price_data = bars[['trade_date', 'symbol', 'close']].copy()
    
    # 合并价格到dataset
    dataset = dataset.merge(price_data, on=['trade_date', 'symbol'], how='left')
    
    return dataset, predictions, nav_full, benchmark_nav


def run_walk_forward_validation(dataset: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    运行Walk-Forward验证
    
    Args:
        dataset: 包含因子和标签的完整数据
        feature_cols: 特征列名列表
    
    Returns:
        每个窗口的结果DataFrame
    """
    # 准备数据
    dataset = dataset.copy()
    dataset['trade_date'] = pd.to_datetime(dataset['trade_date'])
    dataset = dataset.sort_values(['trade_date', 'symbol'])
    
    # 只保留有效数据
    dataset = dataset[dataset['is_tradable'] == True]
    dataset = dataset.dropna(subset=feature_cols + ['fwd_return_20d'])
    dataset = dataset[dataset['close'].notna()]
    
    # 获取交易日列表
    dates = sorted(dataset['trade_date'].unique())
    
    # 配置
    train_window_days = 500
    test_window_days = 60
    step_days = 30
    top_n = 15
    
    results = []
    
    print(f"Walk-Forward配置:")
    print(f"  Train窗口: {train_window_days}天")
    print(f"  Test窗口: {test_window_days}天")
    print(f"  步长: {step_days}天")
    print(f"  Top N: {top_n}")
    print()
    
    # 生成窗口
    window_count = 0
    train_end_idx = train_window_days - 1
    
    while train_end_idx < len(dates) - test_window_days:
        train_end = dates[train_end_idx]
        test_start = dates[train_end_idx + 1]
        test_end = dates[train_end_idx + test_window_days]
        
        # 分割数据
        train_data = dataset[dataset['trade_date'] <= train_end].copy()
        test_data = dataset[(dataset['trade_date'] >= test_start) & (dataset['trade_date'] <= test_end)].copy()
        
        if len(train_data) < 100:
            train_end_idx += step_days
            continue
        
        # 训练Ridge模型
        X_train = train_data[feature_cols].values
        y_train = train_data['fwd_return_20d'].values
        
        # 标准化
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - X_mean) / X_std
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_norm, y_train)
        
        train_ic = np.corrcoef(y_train, model.predict(X_train_norm))[0, 1]
        
        # 测试集预测
        test_data['score'] = (test_data[feature_cols].values - X_mean) / X_std @ model.coef_
        
        # 计算每日IC
        daily_ics = []
        for date, group in test_data.groupby('trade_date'):
            if 'fwd_return_20d' in group.columns and group['fwd_return_20d'].notna().sum() > 10:
                score_rank = group['score'].rank(pct=True)
                label_rank = group['fwd_return_20d'].rank(pct=True)
                if score_rank.std() > 0 and label_rank.std() > 0:
                    ic = score_rank.corr(label_rank)
                    daily_ics.append({'date': date, 'ic': ic})
        
        # 获取调仓日的预测
        rebalance_dates = dates[train_end_idx + 1 : train_end_idx + 1 + 10]  # 前10天作为调仓日
        if len(rebalance_dates) == 0:
            train_end_idx += step_days
            continue
        
        rebalance_date = rebalance_dates[0]
        rebalance_preds = test_data[test_data['trade_date'] == rebalance_date].copy()
        
        if len(rebalance_preds) < top_n:
            train_end_idx += step_days
            continue
        
        # 按score排序选股
        rebalance_preds = rebalance_preds.sort_values('score', ascending=False)
        selected_stocks = rebalance_preds.head(top_n)['symbol'].tolist()
        
        # 计算这些股票在test_window内的真实收益
        selected_returns = []
        market_returns = []
        
        for stock in selected_stocks:
            stock_data = test_data[test_data['symbol'] == stock].sort_values('trade_date')
            if len(stock_data) >= 2:
                price_start = stock_data['close'].iloc[0]
                price_end = stock_data['close'].iloc[-1]
                ret = (price_end / price_start) - 1
                selected_returns.append(ret)
        
        # 计算市场收益
        market_by_date = test_data.groupby('trade_date')['close'].last()
        if len(market_by_date) >= 2:
            market_return = (market_by_date.iloc[-1] / market_by_date.iloc[0]) - 1
        else:
            market_return = 0
        
        # 组合收益
        portfolio_return = np.mean(selected_returns) if selected_returns else 0
        
        # 计算市场状态
        is_bull = market_return > 0
        is_bear = market_return < -0.05
        
        # 平均IC
        mean_ic = np.mean([d['ic'] for d in daily_ics]) if daily_ics else 0
        ic_std = np.std([d['ic'] for d in daily_ics]) if daily_ics else 0
        
        results.append({
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'train_ic': train_ic,
            'test_ic': mean_ic,
            'test_ic_std': ic_std,
            'test_ic_ir': mean_ic / max(ic_std, 0.001) if ic_std > 0 else 0,
            'portfolio_return': portfolio_return,
            'market_return': market_return,
            'excess_return': portfolio_return - market_return,
            'is_bull': is_bull,
            'is_bear': is_bear,
            'n_stocks': len(selected_stocks),
        })
        
        window_count += 1
        if window_count % 5 == 0:
            print(f"  Window {window_count}: IC={mean_ic:.4f}, Return={portfolio_return:.2%}, Excess={portfolio_return-market_return:.2%}")
        
        train_end_idx += step_days
    
    return pd.DataFrame(results)


def compute_walk_forward_summary(results_df: pd.DataFrame) -> dict:
    """计算Walk-Forward汇总统计"""
    if len(results_df) == 0:
        return {}
    
    returns = results_df['portfolio_return'].values
    excess_returns = results_df['excess_return'].values
    
    # 年化收益和波动率
    n_windows = len(returns)
    annualization_factor = 12 / n_windows  # 假设每年12个窗口
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    annual_return = mean_return * annualization_factor
    annual_vol = std_return * np.sqrt(annualization_factor)
    sharpe = annual_return / max(annual_vol, 0.001)
    
    # 胜率
    win_rate = (returns > 0).mean()
    excess_win_rate = (excess_returns > 0).mean()
    
    # Regime分析
    bull_mask = results_df['is_bull'].values
    bear_mask = results_df['is_bear'].values
    
    bull_returns = returns[bull_mask] if bull_mask.any() else []
    bear_returns = returns[bear_mask] if bear_mask.any() else []
    
    bull_mean = np.mean(bull_returns) if len(bull_returns) > 0 else 0
    bear_mean = np.mean(bear_returns) if len(bear_returns) > 0 else 0
    
    # 风险指标
    max_drawdown = np.min(returns) if len(returns) > 0 else 0
    
    # IC聚合
    mean_ic = results_df['test_ic'].mean()
    mean_ic_ir = results_df['test_ic_ir'].mean()
    
    return {
        'n_windows': n_windows,
        'OOS Sharpe': sharpe,
        'OOS Annual Return': annual_return,
        'OOS Annual Vol': annual_vol,
        'OOS Mean Return': mean_return,
        'OOS Std Return': std_return,
        'OOS Worst Window': np.min(returns),
        'OOS Best Window': np.max(returns),
        'Win Rate': win_rate,
        'Excess Win Rate': excess_win_rate,
        'Mean IC': mean_ic,
        'Mean IC IR': mean_ic_ir,
        'Bull Return': bull_mean,
        'Bear Return': bear_mean,
        'Bull Count': len(bull_returns),
        'Bear Count': len(bear_returns),
        'Max Drawdown': max_drawdown,
    }


def main():
    print("="*70)
    print("Walk-Forward OOS验证 - Ridge+Support")
    print("="*70)
    print()
    
    # 加载数据
    print("Loading data...")
    dataset, predictions, nav_full, benchmark_nav = load_data()
    print(f"  Dataset: {dataset.shape}")
    print(f"  Date range: {dataset['trade_date'].min()} to {dataset['trade_date'].max()}")
    print()
    
    # 特征列
    feature_cols = ['roe', 'earnings_yield', 'operating_margin', 'equity_growth', 
                   'ocf_per_share', 'revenue_growth', 'asset_turnover', 'gross_margin', 
                   'cash_ratio', 'mom120', 'vol20']
    
    # 运行Walk-Forward验证
    print("Running Walk-Forward validation...")
    results_df = run_walk_forward_validation(dataset, feature_cols)
    
    if len(results_df) == 0:
        print("No valid windows found!")
        return
    
    # 计算汇总
    summary = compute_walk_forward_summary(results_df)
    
    # 打印结果
    print()
    print("="*70)
    print("Walk-Forward OOS Results - Ridge+Support")
    print("="*70)
    print(f"  窗口数量: {summary.get('n_windows', 0)}")
    print(f"  OOS Sharpe: {summary.get('OOS Sharpe', 0):.3f}")
    print(f"  OOS Annual Return: {summary.get('OOS Annual Return', 0):.2%}")
    print(f"  OOS Annual Vol: {summary.get('OOS Annual Vol', 0):.2%}")
    print(f"  OOS Mean Return: {summary.get('OOS Mean Return', 0):.2%}")
    print(f"  OOS Worst Window: {summary.get('OOS Worst Window', 0):.2%}")
    print(f"  OOS Best Window: {summary.get('OOS Best Window', 0):.2%}")
    print(f"  Win Rate: {summary.get('Win Rate', 0):.1%}")
    print(f"  Excess Win Rate: {summary.get('Excess Win Rate', 0):.1%}")
    print(f"  Mean IC: {summary.get('Mean IC', 0):.4f}")
    print(f"  Mean IC IR: {summary.get('Mean IC IR', 0):.3f}")
    print(f"  Max Drawdown: {summary.get('Max Drawdown', 0):.2%}")
    print()
    print("Regime Analysis:")
    print(f"  Bull Markets ({summary.get('Bull Count', 0)} windows): {summary.get('Bull Return', 0):.2%}")
    print(f"  Bear Markets ({summary.get('Bear Count', 0)} windows): {summary.get('Bear Return', 0):.2%}")
    print()
    
    # 保存结果
    output_dir = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/walk_forward_ridge_support")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / "window_results.csv", index=False)
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    
    # 保存详细报告
    print()
    print("Window Details:")
    print("-"*70)
    print(results_df[['train_end', 'test_start', 'test_ic', 'portfolio_return', 'excess_return', 'is_bull', 'is_bear']].to_string(index=False))
    
    return summary, results_df


if __name__ == "__main__":
    summary, results_df = main()
