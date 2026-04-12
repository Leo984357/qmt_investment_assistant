"""
完整因子筛选与策略发现流程

Stage 1: 数据加载与因子计算
Stage 2: 单因子IC测试
Stage 3: 因子去冗余
Stage 4: 多模型对比
Stage 5: Walk-Forward验证
Stage 6: 最终推荐
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 因子池
from src.features.factor_pool import get_all_factors


def load_data():
    """加载数据"""
    print("="*70)
    print("Stage 1: 加载数据")
    print("="*70)
    
    run_dir = Path("/Users/leolee/Desktop/qmt_investment_assistant/artifacts/runs/hs300_ridge_with_support_20260412_000435_8bad53b6")
    
    # 加载模型数据集
    dataset = pd.read_parquet(run_dir / "datasets/model_dataset.parquet")
    
    # 加载价格数据
    bars = pd.read_parquet("/Users/leolee/Desktop/qmt_investment_assistant/data/silver/daily_bar.parquet")
    price_data = bars[['trade_date', 'symbol', 'close', 'volume', 'high', 'low', 'open', 'amount']].copy()
    
    # 合并
    dataset = dataset.merge(price_data, on=['trade_date', 'symbol'], how='left')
    
    print(f"  Dataset shape: {dataset.shape}")
    print(f"  Date range: {dataset['trade_date'].min()} to {dataset['trade_date'].max()}")
    print(f"  Symbols: {dataset['symbol'].nunique()}")
    
    return dataset


def compute_factors(data):
    """计算扩展因子库"""
    print("\n" + "="*70)
    print("Stage 2: 计算因子")
    print("="*70)
    
    # 获取所有注册的因子名称
    factors = get_all_factors()
    print(f"  注册因子总数: {len(factors)}")
    
    # 已有因子
    existing_factors = ['roe', 'earnings_yield', 'operating_margin', 'equity_growth', 
                        'ocf_per_share', 'revenue_growth', 'asset_turnover', 'gross_margin',
                        'cash_ratio', 'mom120', 'vol20']
    
    # 计算额外因子
    data = data.copy()
    data = data.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    
    # 使用transform避免MultiIndex问题
    for window in [5, 10, 20, 60, 120]:
        col = f'ma{window}'
        if col not in data.columns:
            data[col] = data.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window, min_periods=5).mean()
            )
            existing_factors.append(col)
    
    for window in [5, 10, 20, 60]:
        col = f'vol{window}'
        if col not in data.columns:
            ret = data.groupby('symbol')['close'].transform(lambda x: x.pct_change())
            data[col] = data.groupby('symbol')[ret.name].transform(
                lambda x: x.rolling(window, min_periods=5).std() * np.sqrt(252)
            ) if ret.name in data.columns else data.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().rolling(window, min_periods=5).std() * np.sqrt(252)
            )
            # 简化方式
            data[col] = data.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().rolling(window, min_periods=5).std()
            ) * np.sqrt(252)
            existing_factors.append(col)
    
    # 乖离率
    if 'ma20' in data.columns:
        data['ma20_bias'] = (data['close'] - data['ma20']) / data['ma20']
        existing_factors.append('ma20_bias')
    
    # 均线多头
    if all(f'ma{w}' in data.columns for w in [5, 10, 20, 60]):
        data['ma_bull_alignment'] = ((data['ma5'] > data['ma10']) & 
                                      (data['ma10'] > data['ma20']) & 
                                      (data['ma20'] > data['ma60'])).astype(float)
        existing_factors.append('ma_bull_alignment')
    
    # 收盘位置
    data['close_to_high'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    existing_factors.append('close_to_high')
    
    data['high_low_pos'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    existing_factors.append('high_low_pos')
    
    # ROE变化
    data['roe_change'] = data.groupby('symbol')['roe'].transform(lambda x: x.diff())
    existing_factors.append('roe_change')
    
    # 净利润增速
    data['profit_growth'] = data.groupby('symbol')['roe'].transform(lambda x: x.pct_change(4))
    existing_factors.append('profit_growth')
    
    # 量比
    data['volume_ratio'] = data.groupby('symbol')['volume'].transform(
        lambda x: x / x.rolling(5, min_periods=3).mean()
    )
    existing_factors.append('volume_ratio')
    
    print(f"  已计算/可用因子: {len(existing_factors)}")
    print(f"  因子列表: {existing_factors}")
    
    return data, existing_factors


def compute_factor_ic(data, factor_cols, label_col='fwd_return_20d'):
    """计算单因子IC"""
    print("\n" + "="*70)
    print("Stage 3: 单因子IC测试")
    print("="*70)
    
    results = []
    
    for col in factor_cols:
        if col not in data.columns:
            continue
        
        # 过滤有效数据
        valid_data = data.dropna(subset=[col, label_col])
        
        if len(valid_data) < 100:
            continue
        
        # 按日期计算截面IC
        daily_ics = []
        for date, group in valid_data.groupby('trade_date'):
            if len(group) < 20:
                continue
            if group[col].std() < 1e-8:
                continue
            if group[label_col].std() < 1e-8:
                continue
            
            # Rank IC
            score_rank = group[col].rank(pct=True)
            label_rank = group[label_col].rank(pct=True)
            
            ic = score_rank.corr(label_rank)
            daily_ics.append({'date': date, 'ic': ic, 'n': len(group)})
        
        if len(daily_ics) < 30:
            continue
        
        ic_df = pd.DataFrame(daily_ics)
        
        results.append({
            'factor': col,
            'ic_mean': ic_df['ic'].mean(),
            'ic_std': ic_df['ic'].std(),
            'ic_ir': ic_df['ic'].mean() / max(ic_df['ic'].std(), 0.001),
            'ic_positive_rate': (ic_df['ic'] > 0).mean(),
            'n_days': len(ic_df),
            'n_samples': ic_df['n'].sum(),
        })
    
    ic_results = pd.DataFrame(results)
    ic_results = ic_results.sort_values('ic_ir', ascending=False)
    
    print(f"\n  测试因子数: {len(ic_results)}")
    print(f"\n  Top 20 因子 (按IC IR排序):")
    print(ic_results.head(20).to_string(index=False))
    
    return ic_results


def filter_and_deduplicate(ic_results, data, factor_cols, label_col='fwd_return_20d', 
                           ic_threshold=0.02, corr_threshold=0.7):
    """因子筛选与去冗余"""
    print("\n" + "="*70)
    print("Stage 4: 因子筛选与去冗余")
    print("="*70)
    
    # IC过滤
    positive_factors = ic_results[ic_results['ic_mean'] > 0].copy()
    good_factors = positive_factors[positive_factors['ic_ir'] > 0.05].copy()
    
    print(f"  IC > 0 的因子: {len(positive_factors)}")
    print(f"  IC IR > 0.05 的因子: {len(good_factors)}")
    
    if len(good_factors) == 0:
        # 如果没有好因子，放宽阈值
        good_factors = positive_factors[positive_factors['ic_ir'] > 0.01].copy()
        print(f"  放宽到 IC IR > 0.01: {len(good_factors)}")
    
    if len(good_factors) == 0:
        # 使用所有正IC因子
        good_factors = positive_factors
        print(f"  使用所有正IC因子: {len(good_factors)}")
    
    selected_factors = good_factors['factor'].tolist()
    
    # 去冗余
    print(f"\n  去冗余 (相关系数阈值: {corr_threshold})")
    
    # 计算相关性矩阵
    valid_data = data.dropna(subset=selected_factors + [label_col])
    
    # 按日期聚合取平均
    factor_corr = valid_data.groupby('trade_date')[selected_factors].mean().corr()
    
    # 贪婪去冗余
    final_factors = []
    for f in selected_factors:
        if f not in factor_corr.columns:
            continue
        
        # 检查与已选因子的相关性
        correlated = False
        for selected in final_factors:
            if selected in factor_corr.index and f in factor_corr.columns:
                corr = abs(factor_corr.loc[selected, f])
                if corr > corr_threshold:
                    correlated = True
                    break
        
        if not correlated:
            final_factors.append(f)
    
    print(f"  去冗余后因子数: {len(final_factors)}")
    print(f"  最终因子列表: {final_factors}")
    
    return final_factors


def run_experiments(data, final_factors, label_col='fwd_return_20d'):
    """运行多模型实验"""
    print("\n" + "="*70)
    print("Stage 5: 多模型对比实验")
    print("="*70)
    
    from sklearn.linear_model import Ridge, LinearRegression
    
    # 准备数据
    research_start = '2022-09-01'
    research_data = data[
        (data['trade_date'] >= research_start) & 
        data['is_tradable'] == True
    ].dropna(subset=final_factors + [label_col]).copy()
    
    print(f"  研究数据量: {len(research_data)}")
    print(f"  因子数: {len(final_factors)}")
    
    # 计算IC
    daily_ics = []
    for date, group in research_data.groupby('trade_date'):
        if len(group) < 20:
            continue
        
        # 简单平均分数
        scores = group[final_factors].mean(axis=1)
        
        score_rank = scores.rank(pct=True)
        label_rank = group[label_col].rank(pct=True)
        
        if score_rank.std() > 0 and label_rank.std() > 0:
            ic = score_rank.corr(label_rank)
            daily_ics.append({'date': date, 'ic': ic})
    
    ic_df = pd.DataFrame(daily_ics)
    mean_ic = ic_df['ic'].mean()
    ic_ir = ic_df['ic'].mean() / max(ic_df['ic'].std(), 0.001)
    
    print(f"\n  简单平均模型:")
    print(f"    Mean IC: {mean_ic:.4f}")
    print(f"    IC IR: {ic_ir:.4f}")
    
    # Ridge回归
    ridge_ics = []
    for date, group in research_data.groupby('trade_date'):
        if len(group) < 50:
            continue
        
        # 获取过去500天训练
        train_start = pd.Timestamp(date) - pd.Timedelta(days=500)
        train_data = research_data[research_data['trade_date'] < date]
        train_data = train_data[train_data['trade_date'] >= train_start].dropna(subset=final_factors + [label_col])
        
        if len(train_data) < 100:
            # 使用简单平均
            scores = group[final_factors].mean(axis=1)
        else:
            # Ridge回归
            X_train = train_data[final_factors].values
            y_train = train_data[label_col].values
            
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0) + 1e-8
            X_train_norm = (X_train - X_mean) / X_std
            
            model = Ridge(alpha=1.0)
            model.fit(X_train_norm, y_train)
            
            X_test = group[final_factors].values
            X_test_norm = (X_test - X_mean) / X_std
            scores = pd.Series(model.predict(X_test_norm), index=group.index)
        
        score_rank = scores.rank(pct=True)
        label_rank = group[label_col].rank(pct=True)
        
        if score_rank.std() > 0 and label_rank.std() > 0:
            ic = score_rank.corr(label_rank)
            ridge_ics.append({'date': date, 'ic': ic})
    
    ridge_ic_df = pd.DataFrame(ridge_ics)
    ridge_mean_ic = ridge_ic_df['ic'].mean()
    ridge_ic_ir = ridge_ic_df['ic'].mean() / max(ridge_ic_df['ic'].std(), 0.001)
    
    print(f"\n  Ridge回归模型:")
    print(f"    Mean IC: {ridge_mean_ic:.4f}")
    print(f"    IC IR: {ridge_ic_ir:.4f}")
    
    return {
        'factors': final_factors,
        'simple_average': {'ic': mean_ic, 'ic_ir': ic_ir},
        'ridge': {'ic': ridge_mean_ic, 'ic_ir': ridge_ic_ir},
    }


def walk_forward_validation(data, final_factors, label_col='fwd_return_20d'):
    """Walk-Forward验证"""
    print("\n" + "="*70)
    print("Stage 6: Walk-Forward OOS验证")
    print("="*70)
    
    from sklearn.linear_model import Ridge
    
    # 准备数据
    data = data[data['is_tradable'] == True].dropna(subset=final_factors + [label_col]).copy()
    data['trade_date'] = pd.to_datetime(data['trade_date'])
    data = data.sort_values(['trade_date', 'symbol'])
    
    dates = sorted(data['trade_date'].unique())
    
    # 配置
    train_window = 500
    test_window = 60
    step = 30
    top_n = 15
    
    results = []
    
    idx = train_window - 1
    window_count = 0
    
    while idx < len(dates) - test_window:
        train_end = dates[idx]
        test_start = dates[idx + 1]
        test_end = dates[idx + test_window]
        
        train_data = data[data['trade_date'] <= train_end].copy()
        test_data = data[(data['trade_date'] >= test_start) & (data['trade_date'] <= test_end)].copy()
        
        if len(train_data) < 100:
            idx += step
            continue
        
        # Ridge回归
        X_train = train_data[final_factors].values
        y_train = train_data[label_col].values
        
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - X_mean) / X_std
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_norm, y_train)
        
        # 预测
        test_data['score'] = (test_data[final_factors].values - X_mean) / X_std @ model.coef_
        
        # 调仓日
        rebalance_dates = dates[idx + 1 : idx + 11]
        if len(rebalance_dates) == 0:
            idx += step
            continue
        
        rebalance_date = rebalance_dates[0]
        rebalance_preds = test_data[test_data['trade_date'] == rebalance_date].copy()
        
        if len(rebalance_preds) < top_n:
            idx += step
            continue
        
        # 选股
        rebalance_preds = rebalance_preds.sort_values('score', ascending=False)
        selected_stocks = rebalance_preds.head(top_n)['symbol'].tolist()
        
        # 计算收益
        selected_returns = []
        for stock in selected_stocks:
            stock_data = test_data[test_data['symbol'] == stock].sort_values('trade_date')
            if len(stock_data) >= 2:
                ret = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[0]) - 1
                selected_returns.append(ret)
        
        portfolio_return = np.mean(selected_returns) if selected_returns else 0
        
        # 市场收益
        market_data = test_data.groupby('trade_date')['close'].last()
        market_return = (market_data.iloc[-1] / market_data.iloc[0]) - 1 if len(market_data) >= 2 else 0
        
        # IC
        daily_ics = []
        for date, group in test_data.groupby('trade_date'):
            if len(group) < 20:
                continue
            score_rank = group['score'].rank(pct=True)
            label_rank = group[label_col].rank(pct=True)
            if score_rank.std() > 0 and label_rank.std() > 0:
                ic = score_rank.corr(label_rank)
                daily_ics.append(ic)
        
        mean_ic = np.mean(daily_ics) if daily_ics else 0
        ic_ir = mean_ic / max(np.std(daily_ics), 0.001) if len(daily_ics) > 1 else 0
        
        results.append({
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'portfolio_return': portfolio_return,
            'market_return': market_return,
            'excess_return': portfolio_return - market_return,
            'test_ic': mean_ic,
            'test_ic_ir': ic_ir,
            'is_bull': market_return > 0,
            'is_bear': market_return < -0.05,
        })
        
        window_count += 1
        if window_count % 5 == 0:
            print(f"  Window {window_count}: IC={mean_ic:.4f}, Return={portfolio_return:.2%}")
        
        idx += step
    
    results_df = pd.DataFrame(results)
    
    # 汇总
    n_windows = len(results_df)
    annual_factor = 12 / n_windows if n_windows > 0 else 1
    
    mean_return = results_df['portfolio_return'].mean() if n_windows > 0 else 0
    std_return = results_df['portfolio_return'].std() if n_windows > 0 else 0
    annual_return = mean_return * annual_factor
    annual_vol = std_return * np.sqrt(annual_factor)
    sharpe = annual_return / max(annual_vol, 0.001)
    
    win_rate = (results_df['portfolio_return'] > 0).mean() if n_windows > 0 else 0
    excess_win_rate = (results_df['excess_return'] > 0).mean() if n_windows > 0 else 0
    
    bull_mask = results_df['is_bull'].values
    bear_mask = results_df['is_bear'].values
    bull_return = results_df.loc[bull_mask, 'portfolio_return'].mean() if bull_mask.any() else 0
    bear_return = results_df.loc[bear_mask, 'portfolio_return'].mean() if bear_mask.any() else 0
    
    print(f"\n  Walk-Forward OOS结果:")
    print(f"    窗口数: {n_windows}")
    print(f"    OOS Sharpe: {sharpe:.3f}")
    print(f"    OOS Annual Return: {annual_return:.2%}")
    print(f"    Win Rate: {win_rate:.1%}")
    print(f"    Excess Win Rate: {excess_win_rate:.1%}")
    print(f"    Bull Return: {bull_return:.2%}")
    print(f"    Bear Return: {bear_return:.2%}")
    
    return {
        'n_windows': n_windows,
        'sharpe': sharpe,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'win_rate': win_rate,
        'excess_win_rate': excess_win_rate,
        'bull_return': bull_return,
        'bear_return': bear_return,
        'mean_ic': results_df['test_ic'].mean() if n_windows > 0 else 0,
        'mean_ic_ir': results_df['test_ic_ir'].mean() if n_windows > 0 else 0,
    }


def main():
    """主流程"""
    print("="*70)
    print("完整因子筛选与策略发现流程")
    print("="*70)
    print(f"开始时间: {datetime.now()}")
    
    # Stage 1: 加载数据
    data = load_data()
    
    # Stage 2: 计算因子
    data, factor_cols = compute_factors(data)
    
    # Stage 3: 单因子IC测试
    ic_results = compute_factor_ic(data, factor_cols)
    
    # 保存IC结果
    ic_results.to_csv('/Users/leolee/Desktop/qmt_investment_assistant/artifacts/factor_ic_results.csv', index=False)
    
    # Stage 4: 筛选与去冗余
    final_factors = filter_and_deduplicate(ic_results, data, factor_cols)
    
    # Stage 5: 多模型实验
    experiment_results = run_experiments(data, final_factors)
    
    # Stage 6: Walk-Forward验证
    wf_results = walk_forward_validation(data, final_factors)
    
    # 汇总
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_total_factors': len(factor_cols),
        'n_selected_factors': len(final_factors),
        'selected_factors': final_factors,
        'ic_results': ic_results.head(30).to_dict('records'),
        'experiment_results': experiment_results,
        'wf_results': wf_results,
    }
    
    # 保存
    output_dir = Path('/Users/leolee/Desktop/qmt_investment_assistant/artifacts/strategy_discovery')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("流程完成!")
    print("="*70)
    print(f"  最终因子数: {len(final_factors)}")
    print(f"  因子列表: {final_factors}")
    print(f"  结果保存至: {output_dir}")
    
    return summary


if __name__ == "__main__":
    summary = main()
