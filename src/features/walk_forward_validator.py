"""
Walk-Forward 滚动样本外验证

不是一次切分 train/test，而是:
1. 滚动窗口验证
2. Expanding/Sliding窗口
3. 多市场周期检验
4. 稳定性分析
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
import pandas as pd
import numpy as np


@dataclass
class WindowResult:
    """单个窗口结果"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # 样本内指标
    is_total_return: float = 0.0
    is_sharpe: float = 0.0
    is_ic_mean: float = 0.0
    is_ic_ir: float = 0.0
    
    # 样本外指标
    oos_total_return: float = 0.0
    oos_sharpe: float = 0.0
    oos_ic_mean: float = 0.0
    oos_ic_ir: float = 0.0
    oos_max_drawdown: float = 0.0
    
    # 成本
    oos_turnover: float = 0.0
    oos_transaction_cost: float = 0.0
    
    # 稳定性
    oos_daily_returns_std: float = 0.0
    oos_hit_ratio: float = 0.0  # 正收益天数占比
    
    def to_dict(self) -> dict:
        return {
            'window_id': self.window_id,
            'train_start': self.train_start.isoformat() if isinstance(self.train_start, datetime) else self.train_start,
            'train_end': self.train_end.isoformat() if isinstance(self.train_end, datetime) else self.train_end,
            'test_start': self.test_start.isoformat() if isinstance(self.test_start, datetime) else self.test_start,
            'test_end': self.test_end.isoformat() if isinstance(self.test_end, datetime) else self.test_end,
            'is_total_return': self.is_total_return,
            'is_sharpe': self.is_sharpe,
            'is_ic_mean': self.is_ic_mean,
            'is_ic_ir': self.is_ic_ir,
            'oos_total_return': self.oos_total_return,
            'oos_sharpe': self.oos_sharpe,
            'oos_ic_mean': self.oos_ic_mean,
            'oos_ic_ir': self.oos_ic_ir,
            'oos_max_drawdown': self.oos_max_drawdown,
            'oos_turnover': self.oos_turnover,
            'oos_transaction_cost': self.oos_transaction_cost,
        }


@dataclass
class WalkForwardConfig:
    """Walk-Forward配置"""
    window_type: Literal["expanding", "sliding"] = "expanding"
    
    # 窗口长度 (天)
    train_window_days: int = 500
    test_window_days: int = 60
    
    # 步长 (天)
    step_days: int = 30
    
    # 最小样本要求
    min_train_samples: int = 100
    min_test_samples: int = 20
    
    # 成本假设
    commission_bps: float = 0.75
    stamp_duty_bps: float = 10.0
    slippage_bps: float = 5.0
    
    total_cost_bps: float = field(init=False)
    
    def __post_init__(self):
        self.total_cost_bps = self.commission_bps + self.stamp_duty_bps + self.slippage_bps


@dataclass
class WalkForwardReport:
    """Walk-Forward验证报告"""
    config: WalkForwardConfig
    windows: list[WindowResult]
    start_date: datetime
    end_date: datetime
    
    # 聚合统计
    oos_mean_return: float = 0.0
    oos_std_return: float = 0.0
    oos_min_return: float = 0.0
    oos_max_return: float = 0.0
    
    oos_mean_sharpe: float = 0.0
    oos_mean_ic: float = 0.0
    oos_mean_ic_ir: float = 0.0
    
    # 稳定性
    return_stability: float = 0.0  # 收益标准差/均值
    sharpe_consistency: float = 0.0  # 正Sharpe窗口占比
    
    # 样本内外对比
    is_oos_correlation: float = 0.0  # 样本内外收益相关性
    degradation_ratio: float = 0.0  # OOS/IS比值
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([w.to_dict() for w in self.windows])
    
    def to_summary(self) -> dict:
        return {
            'config': {
                'window_type': self.config.window_type,
                'train_window_days': self.config.train_window_days,
                'test_window_days': self.config.test_window_days,
                'step_days': self.config.step_days,
            },
            'windows_count': len(self.windows),
            'date_range': f'{self.start_date} ~ {self.end_date}',
            'oos_metrics': {
                'mean_return': f'{self.oos_mean_return:.2%}',
                'std_return': f'{self.oos_std_return:.2%}',
                'min_return': f'{self.oos_min_return:.2%}',
                'max_return': f'{self.oos_max_return:.2%}',
                'mean_sharpe': f'{self.oos_mean_sharpe:.2f}',
                'mean_ic': f'{self.oos_mean_ic:.4f}',
                'mean_ic_ir': f'{self.oos_mean_ic_ir:.2f}',
            },
            'stability': {
                'return_stability': f'{self.return_stability:.2f}',
                'sharpe_consistency': f'{self.sharpe_consistency:.1%}',
                'is_oos_correlation': f'{self.is_oos_correlation:.2f}',
                'degradation_ratio': f'{self.degradation_ratio:.2%}',
            },
            'verdict': self._get_verdict(),
        }
    
    def _get_verdict(self) -> str:
        """根据结果给出判断"""
        issues = []
        
        if self.sharpe_consistency < 0.6:
            issues.append(f"Sharpe一致性低 ({self.sharpe_consistency:.1%})")
        
        if self.degradation_ratio < 0.5:
            issues.append(f"样本外退化严重 (OOS/IS={self.degradation_ratio:.1%})")
        
        if self.oos_mean_sharpe < 0.3:
            issues.append(f"样本外Sharpe低 ({self.oos_mean_sharpe:.2f})")
        
        if self.return_stability > 2.0:
            issues.append(f"收益不稳定 (波动/均值={self.return_stability:.2f})")
        
        if issues:
            return f"⚠️ 存在问题: {'; '.join(issues)}"
        else:
            return "✅ 验证通过"


class WalkForwardValidator:
    """
    Walk-Forward验证器
    
    用法:
    validator = WalkForwardValidator(
        config=WalkForwardConfig(
            train_window_days=500,
            test_window_days=60,
            step_days=30,
        )
    )
    
    report = validator.run(
        signals,  # 信号DataFrame
        returns,  # 收益DataFrame
        dates,    # 日期列表
    )
    
    print(report.to_summary())
    """
    
    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()
    
    def run(
        self,
        factor_data: pd.DataFrame,
        label_col: str = 'fwd_return_20d',
        date_col: str = 'trade_date',
    ) -> WalkForwardReport:
        """
        运行Walk-Forward验证
        
        Args:
            factor_data: 包含日期、因子值、标签的DataFrame
            label_col: 标签列名
            date_col: 日期列名
            
        Returns:
            WalkForwardReport
        """
        # 确保日期排序
        factor_data = factor_data.sort_values(date_col)
        dates = factor_data[date_col].unique()
        dates = sorted(dates)
        
        windows = []
        window_id = 0
        current_date_idx = self.config.train_window_days
        
        while current_date_idx + self.config.test_window_days <= len(dates):
            # 训练期
            train_end_idx = current_date_idx
            train_start_idx = max(0, train_end_idx - self.config.train_window_days)
            train_dates = dates[train_start_idx:train_end_idx]
            
            # 测试期
            test_start_idx = current_date_idx
            test_end_idx = min(len(dates), test_start_idx + self.config.test_window_days)
            test_dates = dates[test_start_idx:test_end_idx]
            
            if len(train_dates) < self.config.min_train_samples:
                current_date_idx += self.config.step_days
                continue
            
            if len(test_dates) < self.config.min_test_samples:
                break
            
            # 计算窗口结果
            result = self._evaluate_window(
                factor_data,
                train_dates,
                test_dates,
                label_col,
                date_col,
                window_id,
            )
            windows.append(result)
            
            current_date_idx += self.config.step_days
            window_id += 1
        
        # 生成报告
        report = self._generate_report(windows, dates[0], dates[-1])
        return report
    
    def _evaluate_window(
        self,
        factor_data: pd.DataFrame,
        train_dates: list,
        test_dates: list,
        label_col: str,
        date_col: str,
        window_id: int,
    ) -> WindowResult:
        """评估单个窗口"""
        # 分割数据
        train_data = factor_data[factor_data[date_col].isin(train_dates)]
        test_data = factor_data[factor_data[date_col].isin(test_dates)]
        
        result = WindowResult(
            window_id=window_id,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            test_start=test_dates[0],
            test_end=test_dates[-1],
        )
        
        # 样本内IC
        if len(train_data) > 0:
            factor_col = [c for c in train_data.columns if c not in [date_col, label_col, 'symbol']]
            if factor_col:
                is_ic = train_data.groupby(date_col).apply(
                    lambda x: x[factor_col[0]].corr(x[label_col], method='spearman')
                ).mean()
                result.is_ic_mean = is_ic
        
        # 样本外IC
        if len(test_data) > 0 and factor_col:
            oos_ic_series = test_data.groupby(date_col).apply(
                lambda x: x[factor_col[0]].corr(x[label_col], method='spearman')
            ).dropna()
            
            if len(oos_ic_series) > 0:
                result.oos_ic_mean = oos_ic_series.mean()
                result.oos_ic_ir = oos_ic_series.mean() / oos_ic_series.std() if oos_ic_series.std() > 0 else 0
            
            # 样本外收益 (简化模拟)
            test_data = test_data.copy()
            test_data['signal'] = test_data[factor_col[0]].rank(ascending=False)
            test_data['weight'] = np.where(
                test_data['signal'] <= 40,  # 持仓40只
                1.0 / 40,
                0
            )
            
            test_data['position_return'] = test_data[label_col] * test_data['weight']
            portfolio_return = test_data.groupby(date_col)['position_return'].sum()
            
            if len(portfolio_return) > 0:
                # 估算成本
                turnover_estimate = 0.02  # 假设2%换手
                cost = turnover_estimate * (self.config.total_cost_bps / 10000)
                portfolio_return = portfolio_return - cost
                
                result.oos_total_return = (1 + portfolio_return).prod() - 1
                result.oos_sharpe = portfolio_return.mean() / portfolio_return.std() * np.sqrt(252) if portfolio_return.std() > 0 else 0
                result.oos_max_drawdown = self._calc_max_drawdown(portfolio_return)
                result.oos_hit_ratio = (portfolio_return > 0).mean()
        
        return result
    
    def _calc_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        nav = (1 + returns).cumprod()
        peak = nav.expanding().max()
        drawdown = (nav - peak) / peak
        return drawdown.min()
    
    def _generate_report(
        self,
        windows: list[WindowResult],
        start_date,
        end_date,
    ) -> WalkForwardReport:
        """生成汇总报告"""
        report = WalkForwardReport(
            config=self.config,
            windows=windows,
            start_date=start_date,
            end_date=end_date,
        )
        
        if not windows:
            return report
        
        # 聚合样本外指标
        oos_returns = [w.oos_total_return for w in windows]
        oos_sharpes = [w.oos_sharpe for w in windows]
        oos_ics = [w.oos_ic_mean for w in windows]
        oos_ic_irs = [w.oos_ic_ir for w in windows]
        
        report.oos_mean_return = np.mean(oos_returns)
        report.oos_std_return = np.std(oos_returns)
        report.oos_min_return = np.min(oos_returns)
        report.oos_max_return = np.max(oos_returns)
        report.oos_mean_sharpe = np.mean(oos_sharpes)
        report.oos_mean_ic = np.mean(oos_ics)
        report.oos_mean_ic_ir = np.mean(oos_ic_irs)
        
        # 稳定性
        report.return_stability = report.oos_std_return / abs(report.oos_mean_return) if report.oos_mean_return != 0 else float('inf')
        report.sharpe_consistency = np.mean([s > 0 for s in oos_sharpes])
        
        # 样本内外对比
        is_returns = [w.is_total_return for w in windows]
        if is_returns and oos_returns:
            if np.std(is_returns) > 0 and np.std(oos_returns) > 0:
                report.is_oos_correlation = np.corrcoef(is_returns, oos_returns)[0, 1]
            
            mean_is = np.mean(is_returns)
            if mean_is != 0:
                report.degradation_ratio = report.oos_mean_return / mean_is
        
        return report


def run_walk_forward_experiment(
    factor_names: list[str],
    data: pd.DataFrame,
    config: Optional[WalkForwardConfig] = None,
) -> WalkForwardReport:
    """
    便捷函数: 运行Walk-Forward实验
    
    用法:
    report = run_walk_forward_experiment(
        factor_names=['roe', 'earnings_yield'],
        data=feature_panel.merge(label_panel),
    )
    """
    validator = WalkForwardValidator(config or WalkForwardConfig())
    return validator.run(data)
