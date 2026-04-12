"""
风控约束引擎

核心功能：把风控检查结果反向作用到持仓构建

之前只是"体检"，现在是"手术刀"

Usage:
    from src.risk.risk_engine import RiskConstraintEngine, RiskConstraints
    
    engine = RiskConstraintEngine(constraints)
    
    # 检查并修改持仓
    constrained_weights = engine.apply_constraints(
        target_weights,
        current_positions,
        prices,
        as_of_date,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class RiskConstraints:
    """风控约束配置"""
    # Layer 1: 模型健康
    min_score_dispersion: float = 0.5
    max_missing_ratio: float = 0.3
    min_samples: int = 50
    
    # Layer 2: 组合结构
    max_single_weight: float = 0.05
    max_industry_weight: float = 0.25
    max_style_beta: float = 2.0
    min_positions: int = 10
    max_positions: int = 30
    
    # Layer 3: 总仓位
    min_gross_exposure: float = 0.0
    max_gross_exposure: float = 1.0
    
    # Layer 4: 成本
    max_turnover: float = 0.5
    max_cost_ratio: float = 0.2  # 成本/收益比


@dataclass
class RiskCheckResult:
    """风控检查结果"""
    layer: str
    check_name: str
    passed: bool
    value: float
    threshold: float
    action: str = "pass"  # pass, warn, reject, force
    message: str = ""


class RiskConstraintEngine:
    """
    风控约束引擎
    
    把风控检查结果反向作用到持仓
    
    约束链:
    1. 检查 → 2. 判断 → 3. 修改权重 → 4. 验证
    """
    
    def __init__(
        self,
        constraints: Optional[RiskConstraints] = None,
    ):
        self.constraints = constraints or RiskConstraints()
        self._history: List[RiskCheckResult] = []
    
    def apply_constraints(
        self,
        target_weights: pd.DataFrame,
        current_positions: dict[str, float],
        prices: dict[str, float],
        as_of_date: datetime,
        industry_weights: Optional[dict[str, float]] = None,
        style_exposures: Optional[dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, List[RiskCheckResult]]:
        """
        应用风控约束
        
        Args:
            target_weights: 原始目标权重
            current_positions: 当前持仓
            prices: 当前价格
            as_of_date: 日期
            industry_weights: 行业权重 {industry: weight}
            style_exposures: 风格暴露 {style: beta}
        
        Returns:
            (constrained_weights, check_results)
        """
        result = target_weights.copy()
        all_checks: List[RiskCheckResult] = []
        
        # Layer 2: 组合结构约束
        result, checks = self._apply_position_constraints(result)
        all_checks.extend(checks)
        
        # Layer 3: 总仓位约束
        result, checks = self._apply_exposure_constraints(result)
        all_checks.extend(checks)
        
        # Layer 4: 成本约束（只对调仓生效）
        if current_positions:
            result, checks = self._apply_cost_constraints(
                result, current_positions, prices
            )
            all_checks.extend(checks)
        
        # 归一化确保总和正确
        if 'gross_exposure' in result.columns:
            exposure = result['gross_exposure'].iloc[0]
        else:
            exposure = 1.0
        
        total_weight = result['target_weight'].sum()
        if total_weight > 0:
            result['target_weight'] = result['target_weight'] / total_weight * exposure
        
        self._history.extend(all_checks)
        
        return result, all_checks
    
    def _apply_position_constraints(
        self,
        weights: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[RiskCheckResult]]:
        """应用持仓结构约束"""
        checks = []
        result = weights.copy()
        
        # 1. 单票上限
        max_w = self.constraints.max_single_weight
        over_limit = result['target_weight'] > max_w
        
        if over_limit.any():
            checks.append(RiskCheckResult(
                layer="Layer 2",
                check_name="max_single_weight",
                passed=False,
                value=result.loc[over_limit, 'target_weight'].max(),
                threshold=max_w,
                action="clamp",
                message=f"{over_limit.sum()} stocks exceed max weight {max_w}",
            ))
            result.loc[over_limit, 'target_weight'] = max_w
        
        # 2. 持仓数量约束
        n_positions = len(result)
        min_pos = self.constraints.min_positions
        max_pos = self.constraints.max_positions
        
        if n_positions < min_pos:
            checks.append(RiskCheckResult(
                layer="Layer 2",
                check_name="min_positions",
                passed=False,
                value=n_positions,
                threshold=min_pos,
                action="warn",
                message=f"Too few positions: {n_positions}",
            ))
        elif n_positions > max_pos:
            checks.append(RiskCheckResult(
                layer="Layer 2",
                check_name="max_positions",
                passed=False,
                value=n_positions,
                threshold=max_pos,
                action="reject",
                message=f"Too many positions: {n_positions}",
            ))
            # 保留top N
            result = result.nlargest(max_pos, 'target_weight')
        
        # 3. 行业权重约束（如果提供）
        # 这里简化处理，实际需要行业权重数据
        
        return result, checks
    
    def _apply_exposure_constraints(
        self,
        weights: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[RiskCheckResult]]:
        """应用仓位约束"""
        checks = []
        result = weights.copy()
        
        # 总权重
        total_weight = result['target_weight'].sum()
        
        min_exp = self.constraints.min_gross_exposure
        max_exp = self.constraints.max_gross_exposure
        
        if total_weight < min_exp:
            checks.append(RiskCheckResult(
                layer="Layer 3",
                check_name="min_exposure",
                passed=False,
                value=total_weight,
                threshold=min_exp,
                action="force",
                message=f"Exposure too low: {total_weight:.2%}",
            ))
            # 归一化到最小暴露
            if total_weight > 0:
                result['target_weight'] = result['target_weight'] / total_weight * min_exp
        
        elif total_weight > max_exp:
            checks.append(RiskCheckResult(
                layer="Layer 3",
                check_name="max_exposure",
                passed=False,
                value=total_weight,
                threshold=max_exp,
                action="clamp",
                message=f"Exposure too high: {total_weight:.2%}",
            ))
            # 归一化到最大暴露
            result['target_weight'] = result['target_weight'] / total_weight * max_exp
        
        return result, checks
    
    def _apply_cost_constraints(
        self,
        weights: pd.DataFrame,
        current_positions: dict[str, float],
        prices: dict[str, float],
        lot_size: int = 100,
        min_trade_value: float = 2000.0,
    ) -> Tuple[pd.DataFrame, List[RiskCheckResult]]:
        """应用成本约束"""
        checks = []
        result = weights.copy()
        
        # 计算换手
        turnover_list = []
        
        for _, row in result.iterrows():
            symbol = row['symbol']
            target_w = row['target_weight']
            current_w = current_positions.get(symbol, 0)
            price = prices.get(symbol, 0)
            
            if price <= 0:
                continue
            
            weight_diff = abs(target_w - current_w)
            turnover_list.append(weight_diff)
        
        if turnover_list:
            estimated_turnover = sum(turnover_list) / 2  # 单边换手
            
            max_turn = self.constraints.max_turnover
            if estimated_turnover > max_turn:
                checks.append(RiskCheckResult(
                    layer="Layer 4",
                    check_name="max_turnover",
                    passed=False,
                    value=estimated_turnover,
                    threshold=max_turn,
                    action="reject",
                    message=f"Turnover too high: {estimated_turnover:.2%}",
                ))
                
                # 降低换手：减少调仓幅度
                scale = max_turn / estimated_turnover
                result['target_weight'] = result['target_weight'] * scale
                
                # 归一化
                total = result['target_weight'].sum()
                if total > 0:
                    result['target_weight'] = result['target_weight'] / total * result['target_weight'].sum()
        
        return result, checks
    
    def check_model_health(
        self,
        scores: pd.Series,
        n_samples: int,
        missing_ratio: float = 0.0,
    ) -> List[RiskCheckResult]:
        """检查模型健康状态"""
        checks = []
        
        # 1. 离散度检查
        score_std = scores.std()
        score_mean = scores.mean()
        dispersion = score_std / abs(score_mean) if score_mean != 0 else 0
        
        min_disp = self.constraints.min_score_dispersion
        if dispersion < min_disp:
            checks.append(RiskCheckResult(
                layer="Layer 1",
                check_name="score_dispersion",
                passed=False,
                value=dispersion,
                threshold=min_disp,
                action="fallback",
                message=f"Score dispersion too low: {dispersion:.3f}",
            ))
        
        # 2. 样本量检查
        min_samples = self.constraints.min_samples
        if n_samples < min_samples:
            checks.append(RiskCheckResult(
                layer="Layer 1",
                check_name="min_samples",
                passed=False,
                value=n_samples,
                threshold=min_samples,
                action="fallback",
                message=f"Too few samples: {n_samples}",
            ))
        
        # 3. 缺失率检查
        max_missing = self.constraints.max_missing_ratio
        if missing_ratio > max_missing:
            checks.append(RiskCheckResult(
                layer="Layer 1",
                check_name="missing_ratio",
                passed=False,
                value=missing_ratio,
                threshold=max_missing,
                action="warn",
                message=f"High missing ratio: {missing_ratio:.1%}",
            ))
        
        return checks
    
    def should_fallback(self) -> bool:
        """判断是否应该回退到基线"""
        fallback_checks = [c for c in self._history if c.action == "fallback"]
        return len(fallback_checks) >= 2
    
    def get_constraint_summary(self) -> pd.DataFrame:
        """获取约束执行摘要"""
        if not self._history:
            return pd.DataFrame()
        
        rows = []
        for c in self._history:
            rows.append({
                'layer': c.layer,
                'check': c.check_name,
                'passed': c.passed,
                'value': c.value,
                'threshold': c.threshold,
                'action': c.action,
                'message': c.message,
            })
        
        return pd.DataFrame(rows)
    
    def reset(self):
        """重置历史"""
        self._history = []


class CostSensitivityAnalyzer:
    """
    成本敏感性分析
    
    在多个成本情景下测试策略是否仍能盈利
    """
    
    def __init__(
        self,
        commission_bps_range: List[float] = [0.75, 1.0, 1.5, 2.0],
        slippage_bps_range: List[float] = [5, 10, 20, 30],
        scenarios: Optional[List[dict]] = None,
    ):
        self.commission_bps_range = commission_bps_range
        self.slippage_bps_range = slippage_bps_range
        
        if scenarios:
            self.scenarios = scenarios
        else:
            # 生成标准情景
            self.scenarios = [
                {'name': 'optimistic', 'commission': 0.75, 'slippage': 5},
                {'name': 'baseline', 'commission': 1.0, 'slippage': 10},
                {'name': 'conservative', 'commission': 1.5, 'slippage': 20},
                {'name': 'pessimistic', 'commission': 2.0, 'slippage': 30},
            ]
    
    def analyze(
        self,
        nav: pd.DataFrame,
        trades: pd.DataFrame,
        base_scenario: dict,
    ) -> pd.DataFrame:
        """
        分析不同成本情景下的表现
        
        Args:
            nav: 净值数据
            trades: 交易数据
            base_scenario: 基准情景 {'commission': 0.75, 'slippage': 5}
        
        Returns:
            各情景下的指标
        """
        results = []
        
        for scenario in self.scenarios:
            comm = scenario['commission']
            slip = scenario['slippage']
            
            # 重新计算成本
            scale = (comm + slip) / (base_scenario['commission'] + base_scenario['slippage'])
            
            # 估算收益变化
            total_return = nav['nav'].iloc[-1] / nav['nav'].iloc[0] - 1
            base_cost = trades['fee'].sum()
            scaled_cost = base_cost * scale
            
            # 调整后收益
            adjusted_return = total_return - scaled_cost / nav['nav'].iloc[0]
            
            # Sharpe调整（简化）
            sharpe = (nav['daily_return'].mean() / nav['daily_return'].std() 
                     * np.sqrt(252)) if 'daily_return' in nav.columns else 0
            
            results.append({
                'scenario': scenario['name'],
                'commission_bps': comm,
                'slippage_bps': slip,
                'total_cost': scaled_cost,
                'total_return': total_return,
                'adjusted_return': adjusted_return,
                'cost_impact': scaled_cost - base_cost,
                'return_reduction': total_return - adjusted_return,
                'still_profitable': adjusted_return > 0,
            })
        
        return pd.DataFrame(results)
    
    def get_survival_threshold(
        self,
        nav: pd.DataFrame,
        trades: pd.DataFrame,
        required_return: float = 0.0,
    ) -> Tuple[float, float]:
        """
        获取策略能承受的最大成本
        
        Returns:
            (max_commission, max_slippage)
        """
        base_return = nav['nav'].iloc[-1] / nav['nav'].iloc[0] - 1
        base_cost = trades['fee'].sum()
        
        max_total_cost = base_return - required_return
        
        # 假设佣金:滑点 = 1:10
        max_commission = max_total_cost / 11
        max_slippage = max_commission * 10
        
        return max_commission, max_slippage
