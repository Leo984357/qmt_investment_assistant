"""因子研究和筛选工具"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FactorReport:
    """单因子分析报告"""
    name: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_win_rate: float
    ic_positive_years: int
    ic_negative_years: int
    long_short_return: float
    quantile_returns: dict[int, float]
    cost_sensitivity: dict[float, float]
    correlation_with_other_factors: dict[str, float]
    recommendation: str = "unclear"


@dataclass
class FactorSelectionResult:
    """因子筛选结果"""
    all_factors: list[FactorReport]
    selected_factors: list[str]
    rejected_factors: dict[str, str]
    summary: str


class FactorResearcher:
    """因子研究器 - 用于评估和筛选因子"""
    
    def __init__(
        self,
        min_ic_mean: float = 0.005,
        min_ic_ir: float = 0.1,
        min_win_rate: float = 0.50,
        min_positive_years: int = 3,
        min_long_short_return: float = 0.0,
    ):
        self.min_ic_mean = min_ic_mean
        self.min_ic_ir = min_ic_ir
        self.min_win_rate = min_win_rate
        self.min_positive_years = min_positive_years
        self.min_long_short_return = min_long_short_return
    
    def analyze_factor(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        factor_name: str,
        label_name: str = 'fwd_return_20d',
        n_quantiles: int = 5,
    ) -> FactorReport:
        """分析单个因子"""
        merged = feature_panel.merge(
            label_panel[['trade_date', 'symbol', label_name]],
            on=['trade_date', 'symbol']
        )
        merged['trade_date'] = pd.to_datetime(merged['trade_date'])
        
        df = merged.dropna(subset=[factor_name, label_name]).copy()
        
        # 计算每日IC
        def calc_ic(group):
            if len(group) < 20:
                return np.nan
            return group[factor_name].corr(group[label_name], method='spearman')
        
        daily_ic = df.groupby('trade_date').apply(calc_ic, include_groups=False).dropna()
        
        # IC统计
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_win_rate = (daily_ic > 0).mean()
        
        # 年IC
        yearly_ic = daily_ic.groupby(daily_ic.index.year).mean()
        ic_positive_years = (yearly_ic > 0).sum()
        ic_negative_years = (yearly_ic <= 0).sum()
        
        # 分组回测
        df['quantile'] = df.groupby('trade_date')[factor_name].transform(
            lambda x: pd.qcut(x.rank(method='first'), n_quantiles, labels=False, duplicates='drop') + 1
        )
        group_returns = df.groupby(['trade_date', 'quantile'])[label_name].mean()
        avg_returns = group_returns.groupby('quantile').mean()
        
        long_short = avg_returns.iloc[-1] - avg_returns.iloc[0]
        
        # 成本敏感性
        cost_sensitivity = {}
        for cost in [0, 0.05, 0.10, 0.20, 0.30, 0.50]:
            net = long_short - cost / 100
            cost_sensitivity[cost * 100] = net
        
        # 推荐
        recommendation = self._make_recommendation(ic_mean, ic_ir, ic_win_rate, ic_positive_years, long_short)
        
        return FactorReport(
            name=factor_name,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_win_rate=ic_win_rate,
            ic_positive_years=int(ic_positive_years),
            ic_negative_years=int(ic_negative_years),
            long_short_return=long_short,
            quantile_returns={int(k): v for k, v in avg_returns.to_dict().items()},
            cost_sensitivity=cost_sensitivity,
            correlation_with_other_factors={},
            recommendation=recommendation,
        )
    
    def _make_recommendation(
        self,
        ic_mean: float,
        ic_ir: float,
        ic_win_rate: float,
        positive_years: int,
        long_short: float,
    ) -> str:
        """生成推荐建议"""
        scores = []
        
        # IC均值
        if ic_mean > 0.01:
            scores.append('strong_positive')
        elif ic_mean > 0.005:
            scores.append('moderate_positive')
        elif ic_mean > 0:
            scores.append('weak_positive')
        elif ic_mean > -0.005:
            scores.append('weak_negative')
        else:
            scores.append('strong_negative')
        
        # IC_IR
        if abs(ic_ir) > 0.3:
            scores.append('high_ir')
        elif abs(ic_ir) > 0.1:
            scores.append('moderate_ir')
        else:
            scores.append('low_ir')
        
        # 综合判断
        if ic_mean > 0.005 and ic_ir > 0.1 and positive_years >= 4:
            return "keep_strong"
        elif ic_mean > 0.002 and ic_ir > 0.05 and positive_years >= 3:
            return "keep_moderate"
        elif ic_mean > 0 and positive_years >= 2:
            return "keep_weak"
        elif long_short > 0.005:
            return "marginal_long_short"
        else:
            return "reject"
    
    def select_factors(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        factor_names: list[str],
    ) -> FactorSelectionResult:
        """筛选因子"""
        reports = []
        selected = []
        rejected = {}
        
        for name in factor_names:
            try:
                report = self.analyze_factor(feature_panel, label_panel, name)
                reports.append(report)
                
                if report.recommendation in ['keep_strong', 'keep_moderate', 'keep_weak']:
                    selected.append(name)
                else:
                    rejected[name] = report.recommendation
            except Exception as e:
                rejected[name] = f"error: {str(e)}"
        
        summary = f"筛选完成: {len(selected)}/{len(factor_names)} 因子被保留"
        
        return FactorSelectionResult(
            all_factors=reports,
            selected_factors=selected,
            rejected_factors=rejected,
            summary=summary,
        )
    
    def print_report(self, result: FactorSelectionResult, feature_panel: pd.DataFrame = None, label_panel: pd.DataFrame = None):
        """打印报告"""
        print("=" * 80)
        print("因子筛选报告")
        print("=" * 80)
        
        # 统计
        print(f"\n筛选条件:")
        print(f"  - 最小IC均值: {self.min_ic_mean}")
        print(f"  - 最小IC_IR: {self.min_ic_ir}")
        print(f"  - 最小IC胜率: {self.min_win_rate}")
        print(f"  - 最小正IC年数: {self.min_positive_years}")
        
        print(f"\n结果: {len(result.selected_factors)}/{len(result.all_factors) + len(result.rejected_factors)} 因子被保留")
        
        # 推荐保留的因子
        print("\n" + "=" * 80)
        print("✅ 推荐保留的因子")
        print("=" * 80)
        kept = [r for r in result.all_factors if r.name in result.selected_factors]
        kept.sort(key=lambda x: x.ic_mean, reverse=True)
        
        print(f"\n{'因子':<20} {'IC均值':>10} {'IC_IR':>10} {'IC胜率':>10} {'多空收益':>12} {'推荐':>15}")
        print("-" * 80)
        for r in kept:
            print(f"{r.name:<20} {r.ic_mean:>10.4f} {r.ic_ir:>10.3f} {r.ic_win_rate:>10.1%} {r.long_short_return:>12.4f} {r.recommendation:>15}")
        
        # 被拒绝的因子
        if result.rejected_factors:
            print("\n" + "=" * 80)
            print("❌ 被拒绝的因子")
            print("=" * 80)
            
            rejected_reasons = {}
            for name, reason in result.rejected_factors.items():
                if reason not in rejected_reasons:
                    rejected_reasons[reason] = []
                rejected_reasons[reason].append(name)
            
            for reason, names in rejected_reasons.items():
                print(f"\n{reason}: {len(names)}个")
                for name in names[:10]:
                    print(f"  - {name}")
                if len(names) > 10:
                    print(f"  ... 还有{len(names) - 10}个")
        
        # 年度IC矩阵
        print("\n" + "=" * 80)
        print("年度IC热力图")
        print("=" * 80)
        
        if kept:
            # 计算年度IC
            merged = feature_panel.merge(label_panel[['trade_date', 'symbol', 'fwd_return_20d']], on=['trade_date', 'symbol'])
            merged['trade_date'] = pd.to_datetime(merged['trade_date'])
            
            yearly_data = []
            for factor in kept[:20]:  # 只显示前20个
                df = merged.dropna(subset=[factor.name, 'fwd_return_20d'])
                ic_by_date = df.groupby('trade_date').apply(
                    lambda g: g[factor.name].corr(g['fwd_return_20d'], method='spearman'),
                    include_groups=False
                ).dropna()
                yearly_ic = ic_by_date.groupby(ic_by_date.index.year).mean()
                yearly_data.append(yearly_ic)
            
            yearly_df = pd.DataFrame(yearly_data, index=[r.name for r in kept[:20]])
            print(yearly_df.round(3).to_string())
        
        print("\n" + "=" * 80)
        print(result.summary)
        print("=" * 80)


def run_factor_research(
    feature_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    factor_names: list[str],
    **kwargs
) -> FactorSelectionResult:
    """运行因子研究的主函数"""
    researcher = FactorResearcher(**kwargs)
    result = researcher.select_factors(feature_panel, label_panel, factor_names)
    researcher.print_report(result)
    return result
