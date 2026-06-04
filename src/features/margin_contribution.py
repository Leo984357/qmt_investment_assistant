"""
因子边际贡献分析器

分析新因子加入因子池后的边际价值:
1. OOS表现变化
2. 成本变化
3. 暴露变化
4. 因子角色判断
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarginContributionResult:
    """边际贡献分析结果"""
    factor_name: str
    baseline_metrics: dict      # 基准池的指标
    new_pool_metrics: dict      # 新池的指标
    marginal_metrics: dict       # 边际变化
    recommendation: str        # 建议
    priority: str              # HIGH/MEDIUM/LOW


class MarginContributionAnalyzer:
    """
    边际贡献分析器
    
    用法:
    analyzer = MarginContributionAnalyzer()
    
    result = analyzer.analyze(
        factor_name='new_factor',
        baseline_pool=['roe', 'earnings_yield'],
        new_pool=['roe', 'earnings_yield', 'new_factor'],
        factor_data=factor_data,
    )
    """

    def __init__(self):
        pass

    def analyze(
        self,
        factor_name: str,
        baseline_pool: list[str],
        new_pool: list[str],
        factor_data: pd.DataFrame,
        label_col: str = 'fwd_return_20d',
        date_col: str = 'trade_date',
        symbol_col: str = 'symbol',
    ) -> MarginContributionResult:
        """
        分析因子加入后的边际贡献
        """
        # 计算基准池指标
        baseline_metrics = self._calculate_pool_metrics(
            factor_data, baseline_pool, label_col, date_col
        )

        # 计算新池指标
        new_pool_metrics = self._calculate_pool_metrics(
            factor_data, new_pool, label_col, date_col
        )

        # 计算边际变化
        marginal_metrics = {
            'ic_change': new_pool_metrics['ic_mean'] - baseline_metrics['ic_mean'],
            'ic_ir_change': new_pool_metrics['ic_ir'] - baseline_metrics['ic_ir'],
            'turnover_change': new_pool_metrics['turnover'] - baseline_metrics['turnover'],
            'pool_diversity_change': new_pool_metrics['diversity'] - baseline_metrics['diversity'],
        }

        # 判断边际价值
        recommendation, priority = self._make_recommendation(
            marginal_metrics, baseline_metrics, new_pool_metrics
        )

        return MarginContributionResult(
            factor_name=factor_name,
            baseline_metrics=baseline_metrics,
            new_pool_metrics=new_pool_metrics,
            marginal_metrics=marginal_metrics,
            recommendation=recommendation,
            priority=priority,
        )

    def _calculate_pool_metrics(
        self,
        data: pd.DataFrame,
        pool: list[str],
        label_col: str,
        date_col: str,
    ) -> dict:
        """计算因子池的综合指标"""
        # 只保留池中存在的因子
        available_factors = [f for f in pool if f in data.columns]

        if not available_factors:
            return {
                'ic_mean': 0,
                'ic_ir': 0,
                'turnover': 0,
                'diversity': 0,
            }

        # 计算每个因子的IC
        valid_data = data.dropna(subset=available_factors + [label_col])

        ic_series_dict = {}
        for f in available_factors:
            ic_series = valid_data.groupby(date_col).apply(
                lambda x: x[f].corr(x[label_col], method='spearman')
            ).dropna()
            ic_series_dict[f] = ic_series

        # 池的平均IC
        all_ics = []
        for ic_s in ic_series_dict.values():
            all_ics.extend(ic_s.values)

        ic_mean = np.mean(all_ics) if all_ics else 0
        ic_std = np.std(all_ics) if all_ics else 1
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0

        # 因子间相关性 (池的多样性)
        diversity = 1 - self._average_pool_correlation(data, available_factors, date_col)

        # 估计换手率 (因子值变化率的均值)
        turnover = self._estimate_pool_turnover(data, available_factors, date_col)

        return {
            'ic_mean': ic_mean,
            'ic_ir': ic_ir,
            'turnover': turnover,
            'diversity': diversity,
            'factor_count': len(available_factors),
        }

    def _average_pool_correlation(
        self,
        data: pd.DataFrame,
        factors: list[str],
        date_col: str,
    ) -> float:
        """计算池内因子平均相关性"""
        if len(factors) < 2:
            return 0

        corr_matrix = data[factors].corr()
        n = len(factors)

        # 取上三角 (不含对角线)
        total_corr = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                total_corr += abs(corr_matrix.iloc[i, j])
                count += 1

        return total_corr / count if count > 0 else 0

    def _estimate_pool_turnover(
        self,
        data: pd.DataFrame,
        factors: list[str],
        date_col: str,
    ) -> float:
        """估计池的平均换手率"""
        if len(factors) < 1:
            return 0

        turnovers = []
        for f in factors:
            valid = data.dropna(subset=[f])
            if len(valid) > 1:
                # 因子值变化率
                factor_change = valid[f].pct_change().abs().mean()
                turnovers.append(factor_change)

        return np.mean(turnovers) if turnovers else 0

    def _make_recommendation(
        self,
        marginal: dict,
        baseline: dict,
        new_pool: dict,
    ) -> tuple[str, str]:
        """生成建议"""
        ic_gain = marginal['ic_change']
        diversity_gain = marginal['pool_diversity_change']
        turnover_penalty = marginal['turnover_change']

        # 计算综合得分
        # IC提升权重高，多样性提升中等，换手率增加扣分
        score = ic_gain * 10 + diversity_gain * 5 - turnover_penalty * 100

        if score > 0.5:
            return "ADD", "HIGH"
        elif score > 0:
            return "CONDITIONAL_ADD", "MEDIUM"
        elif score > -0.5:
            return "WATCH", "LOW"
        else:
            return "REJECT", "LOW"

    def batch_analyze(
        self,
        candidate_factors: list[str],
        current_pool: list[str],
        factor_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """批量分析候选因子"""
        results = []

        for f in candidate_factors:
            if f not in factor_data.columns:
                continue

            new_pool = current_pool + [f]
            result = self.analyze(
                factor_name=f,
                baseline_pool=current_pool,
                new_pool=new_pool,
                factor_data=factor_data,
            )

            results.append({
                'factor': result.factor_name,
                'baseline_ic': result.baseline_metrics['ic_mean'],
                'new_pool_ic': result.new_pool_metrics['ic_mean'],
                'ic_marginal': result.marginal_metrics['ic_change'],
                'diversity_change': result.marginal_metrics['pool_diversity_change'],
                'turnover_change': result.marginal_metrics['turnover_change'],
                'recommendation': result.recommendation,
                'priority': result.priority,
            })

        return pd.DataFrame(results).sort_values('ic_marginal', ascending=False)


class FactorRoleClassifier:
    """
    因子角色分类器
    
    判断因子在池中的角色:
    1. PRIMARY: 主因子 - 直接贡献收益
    2. AUXILIARY: 辅助因子 - 提供条件信息
    3. CONDITIONAL: 条件因子 - 仅在特定市场有效
    4. REDUNDANT: 冗余因子 - 与其他因子高度相关
    """

    @staticmethod
    def classify(
        factor_name: str,
        pool_metrics: dict,
        regime_analysis: dict | None = None,
    ) -> str:
        """
        分类因子角色
        
        Args:
            factor_name: 因子名
            pool_metrics: 包含IC_IR, IC_mean等指标
            regime_analysis: 可选，各regime下的IC表现
            
        Returns:
            因子角色
        """
        ic_ir = pool_metrics.get('ic_ir', 0)
        ic_mean = pool_metrics.get('ic_mean', 0)
        turnover = pool_metrics.get('turnover', 0)

        # 冗余检查
        highly_correlated = pool_metrics.get('highly_correlated_factors', [])
        if highly_correlated:
            return 'REDUNDANT'

        # 条件因子检查 (只在某些regime有效)
        if regime_analysis:
            regime_ics = regime_analysis.get('by_trend_regime', {})
            if regime_ics:
                up_ic = regime_ics.get('uptrend', 0)
                down_ic = regime_ics.get('downtrend', 0)

                # 在一个方向显著正，另一个方向负或零
                if (up_ic > 0.02 and down_ic < 0) or (down_ic > 0.02 and up_ic < 0):
                    return 'CONDITIONAL'

        # 主因子判断
        if ic_ir > 0.3 and ic_mean > 0.02:
            return 'PRIMARY'

        # 辅助因子判断
        if ic_ir > 0.1 or ic_mean > 0.01:
            return 'AUXILIARY'

        return 'UNCLASSIFIED'

    @staticmethod
    def suggest_pool_composition(
        pool_metrics: dict[str, dict],
        target_primary_count: int = 3,
        target_auxiliary_count: int = 5,
    ) -> dict:
        """
        建议因子池组成
        
        Returns:
            {
                'primary': ['roe', 'earnings_yield', ...],
                'auxiliary': ['margin_factor', ...],
                'conditional': [...],
            }
        """
        composition = {
            'primary': [],
            'auxiliary': [],
            'conditional': [],
            'redundant': [],
            'unclassified': [],
        }

        for factor_name, metrics in pool_metrics.items():
            role = FactorRoleClassifier.classify(factor_name, metrics)

            if role == 'PRIMARY':
                composition['primary'].append(factor_name)
            elif role == 'AUXILIARY':
                composition['auxiliary'].append(factor_name)
            elif role == 'CONDITIONAL':
                composition['conditional'].append(factor_name)
            elif role == 'REDUNDANT':
                composition['redundant'].append(factor_name)
            else:
                composition['unclassified'].append(factor_name)

        # 按IC排序
        for key in ['primary', 'auxiliary', 'conditional']:
            metrics_list = [(f, pool_metrics[f].get('ic_ir', 0))
                          for f in composition[key]]
            metrics_list.sort(key=lambda x: -x[1])
            composition[key] = [f for f, _ in metrics_list]

        return composition
