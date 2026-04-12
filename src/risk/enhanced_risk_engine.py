"""
增强风控引擎

修复：
1. 从"检查表"升级为真正的"约束引擎"
2. 添加小资金实盘特有的风险约束
3. 所有约束都有明确的触发条件和行动
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskCheck:
    """风控检查结果"""
    name: str
    status: RiskLevel
    value: float
    threshold: float
    action: str
    details: Dict


@dataclass
class RiskConstraints:
    """风控约束配置"""
    # 单票约束
    max_single_position: float = 0.08  # 单票上限8%
    max_single_position_emergency: float = 0.10  # 紧急上限10%
    
    # 行业约束
    max_industry_exposure: float = 0.30  # 单行业上限30%
    max_industry_concentration: float = 0.50  # 最大行业集中度
    
    # 风格约束
    max_size_beta: float = 0.3  # Size暴露上限
    max_leverage_beta: float = 0.2  # 杠杆暴露上限
    
    # 流动性约束 (小资金)
    max_daily_trade_pct: float = 0.01  # 单日交易额不超过组合1%
    min_avg_volume: float = 1000000  # 最低日均成交量
    
    # 成本约束
    max_turnover_per_rebalance: float = 0.30  # 单次调仓换手上限30%
    max_cost_ratio: float = 0.20  # 成本占比上限20%
    
    # 回撤约束
    max_drawdown_emergency: float = -0.15  # 回撤达到-15%触发紧急风控
    max_drawdown_stop: float = -0.20  # 回撤达到-20%停止交易
    
    # IC约束
    min_ic_rolling: float = -0.02  # 滚动IC低于-2%触发警告
    min_ic_ir: float = 0.05  # 最低IC IR要求
    
    # 信号约束
    min_score_spread: float = 0.1  # 最低分数差距
    min_coverage: float = 0.80  # 最低覆盖率


class EnhancedRiskEngine:
    """
    增强风控引擎
    
    从"检查表"升级为"约束引擎"：
    1. 所有检查都有明确阈值
    2. 触发条件明确
    3. 行动方案具体
    """
    
    def __init__(self, constraints: Optional[RiskConstraints] = None):
        self.constraints = constraints or RiskConstraints()
        self.checks_history: List[RiskCheck] = []
        self.emergency_mode = False
        self.blocked_factors: List[str] = []
    
    def check_all(
        self,
        weights: pd.Series,
        scores: pd.Series,
        positions: pd.DataFrame,
        nav_history: pd.DataFrame,
        ic_rolling: float,
        turnover: float,
        cost: float,
        industry: Optional[pd.Series] = None,
    ) -> List[RiskCheck]:
        """
        执行所有风控检查
        
        Returns:
            风控检查结果列表
        """
        checks = []
        
        # Layer 1: 模型健康检查
        checks.extend(self._check_model_health(scores, ic_rolling))
        
        # Layer 2: 组合结构检查
        checks.extend(self._check_portfolio_structure(weights, positions))
        
        # Layer 3: 总仓位检查
        checks.extend(self._check_exposure(weights, industry))
        
        # Layer 4: 成本检查
        checks.extend(self._check_cost(turnover, cost, nav_history))
        
        # Layer 5: 回撤检查
        checks.extend(self._check_drawdown(nav_history))
        
        # Layer 6: 流动性检查
        checks.extend(self._check_liquidity(positions, weights, nav_history))
        
        # 更新状态
        self.checks_history.extend(checks)
        self._update_emergency_mode(checks)
        
        return checks
    
    def _check_model_health(self, scores: pd.Series, ic_rolling: float) -> List[RiskCheck]:
        """Layer 1: 模型健康检查"""
        checks = []
        
        # IC检查
        if ic_rolling < self.constraints.min_ic_rolling:
            status = RiskLevel.HIGH
            action = "降权50%，观察3天"
        elif ic_rolling < self.constraints.min_ic_rolling * 2:
            status = RiskLevel.MEDIUM
            action = "观察"
        else:
            status = RiskLevel.LOW
            action = "正常"
        
        checks.append(RiskCheck(
            name="rolling_ic",
            status=status,
            value=ic_rolling,
            threshold=self.constraints.min_ic_rolling,
            action=action,
            details={'min_ir': self.constraints.min_ic_ir}
        ))
        
        # 分数离散度
        score_spread = scores.max() - scores.min()
        if score_spread < self.constraints.min_score_spread:
            checks.append(RiskCheck(
                name="score_spread",
                status=RiskLevel.MEDIUM,
                value=score_spread,
                threshold=self.constraints.min_score_spread,
                action="检查模型是否失灵",
                details={'current_spread': score_spread}
            ))
        
        # 覆盖率
        coverage = scores.notna().mean()
        if coverage < self.constraints.min_coverage:
            checks.append(RiskCheck(
                name="signal_coverage",
                status=RiskLevel.HIGH,
                value=coverage,
                threshold=self.constraints.min_coverage,
                action="暂停交易，待数据恢复",
                details={}
            ))
        
        return checks
    
    def _check_portfolio_structure(self, weights: pd.Series, positions: pd.DataFrame) -> List[RiskCheck]:
        """Layer 2: 组合结构检查"""
        checks = []
        
        # 单票集中度
        max_weight = weights.max() if len(weights) > 0 else 0
        max_limit = (self.constraints.max_single_position_emergency 
                     if self.emergency_mode 
                     else self.constraints.max_single_position)
        
        if max_weight > max_limit:
            status = RiskLevel.CRITICAL if max_weight > max_limit * 1.2 else RiskLevel.HIGH
            action = f"强制平仓至{max_limit*100:.0f}%"
        elif max_weight > self.constraints.max_single_position * 0.9:
            status = RiskLevel.MEDIUM
            action = "监控，暂不行动"
        else:
            status = RiskLevel.LOW
            action = "正常"
        
        checks.append(RiskCheck(
            name="position_concentration",
            status=status,
            value=max_weight,
            threshold=self.constraints.max_single_position,
            action=action,
            details={'emergency_limit': max_limit}
        ))
        
        # 持仓数量
        n_positions = len(weights[weights > 0.01])
        if n_positions < 10:
            checks.append(RiskCheck(
                name="position_count",
                status=RiskLevel.MEDIUM,
                value=n_positions,
                threshold=10,
                action="增加持仓分散",
                details={}
            ))
        
        return checks
    
    def _check_exposure(
        self, 
        weights: pd.Series, 
        industry: Optional[pd.Series] = None
    ) -> List[RiskCheck]:
        """Layer 3: 总仓位检查"""
        checks = []
        
        # 总暴露
        total_exposure = weights.sum()
        if total_exposure < 0.5:
            checks.append(RiskCheck(
                name="total_exposure",
                status=RiskLevel.HIGH,
                value=total_exposure,
                threshold=0.5,
                action="增加仓位或退出市场",
                details={}
            ))
        elif total_exposure > 1.1:
            checks.append(RiskCheck(
                name="total_exposure",
                status=RiskLevel.CRITICAL,
                value=total_exposure,
                threshold=1.1,
                action="立即减仓至100%",
                details={}
            ))
        
        # 行业暴露
        if industry is not None:
            weights_with_industry = pd.concat([weights, industry], axis=1)
            industry_exposure = weights_with_industry.groupby(industry.name).sum()[weights.name]
            
            max_industry = industry_exposure.max()
            if max_industry > self.constraints.max_industry_exposure:
                checks.append(RiskCheck(
                    name="industry_exposure",
                    status=RiskLevel.HIGH,
                    value=max_industry,
                    threshold=self.constraints.max_industry_exposure,
                    action=f"减仓最大行业至{self.constraints.max_industry_exposure*100:.0f}%",
                    details={'max_industry_weight': max_industry}
                ))
        
        return checks
    
    def _check_cost(
        self, 
        turnover: float, 
        cost: float, 
        nav_history: pd.DataFrame
    ) -> List[RiskCheck]:
        """Layer 4: 成本检查"""
        checks = []
        
        # 单次调仓换手
        if turnover > self.constraints.max_turnover_per_rebalance:
            checks.append(RiskCheck(
                name="turnover",
                status=RiskLevel.HIGH,
                value=turnover,
                threshold=self.constraints.max_turnover_per_rebalance,
                action="启用成本过滤",
                details={}
            ))
        
        # 成本占比
        nav_value = nav_history['nav'].iloc[-1] if len(nav_history) > 0 else 1
        cost_ratio = cost / nav_value if nav_value > 0 else 0
        
        if cost_ratio > self.constraints.max_cost_ratio:
            checks.append(RiskCheck(
                name="cost_ratio",
                status=RiskLevel.CRITICAL,
                value=cost_ratio,
                threshold=self.constraints.max_cost_ratio,
                action="停止交易，重新评估策略",
                details={}
            ))
        
        return checks
    
    def _check_drawdown(self, nav_history: pd.DataFrame) -> List[RiskCheck]:
        """Layer 5: 回撤检查"""
        checks = []
        
        if len(nav_history) < 20:
            return checks
        
        nav = nav_history['nav']
        peak = nav.cummax()
        drawdown = (nav - peak) / peak
        
        current_dd = drawdown.iloc[-1]
        max_dd = drawdown.min()
        
        # 当前回撤
        if current_dd < self.constraints.max_drawdown_stop:
            checks.append(RiskCheck(
                name="current_drawdown",
                status=RiskLevel.CRITICAL,
                value=current_dd,
                threshold=self.constraints.max_drawdown_stop,
                action="停止所有交易",
                details={'peak': peak.iloc[-1], 'current': nav.iloc[-1]}
            ))
        elif current_dd < self.constraints.max_drawdown_emergency:
            checks.append(RiskCheck(
                name="current_drawdown",
                status=RiskLevel.HIGH,
                value=current_dd,
                threshold=self.constraints.max_drawdown_emergency,
                action="降仓50%，启用紧急风控",
                details={}
            ))
        
        return checks
    
    def _check_liquidity(
        self, 
        positions: pd.DataFrame, 
        weights: pd.Series,
        nav_history: pd.DataFrame
    ) -> List[RiskCheck]:
        """Layer 6: 流动性检查 (小资金特有)"""
        checks = []
        
        nav_value = nav_history['nav'].iloc[-1] if len(nav_history) > 0 else 1
        
        # 检查每只股票的日均成交量
        if 'volume' in positions.columns and 'close' in positions.columns:
            positions['daily_volume'] = positions['volume'] * positions['close']
            avg_daily_volume = positions.groupby('symbol')['daily_volume'].mean()
            
            for sym, w in weights.items():
                if sym in avg_daily_volume.index:
                    vol = avg_daily_volume[sym]
                    trade_value = w * nav_value
                    
                    if vol > 0:
                        trade_pct = trade_value / vol
                        if trade_pct > self.constraints.max_daily_trade_pct:
                            checks.append(RiskCheck(
                                name="liquidity",
                                status=RiskLevel.MEDIUM,
                                value=trade_pct,
                                threshold=self.constraints.max_daily_trade_pct,
                                action=f"减少{sym}持仓",
                                details={'symbol': sym, 'avg_volume': vol}
                            ))
        
        return checks
    
    def _update_emergency_mode(self, checks: List[RiskCheck]):
        """更新紧急风控模式"""
        critical_checks = [c for c in checks if c.status == RiskLevel.CRITICAL]
        high_checks = [c for c in checks if c.status == RiskLevel.HIGH]
        
        if len(critical_checks) > 0:
            self.emergency_mode = True
        elif len(high_checks) > 3:
            self.emergency_mode = True
        elif len(high_checks) == 0:
            self.emergency_mode = False
    
    def get_risk_status(self) -> str:
        """获取整体风险状态"""
        if self.emergency_mode:
            return "⚠️ 紧急风控"
        
        if len(self.checks_history) == 0:
            return "✓ 正常"
        
        recent = self.checks_history[-10:]
        high_or_above = [c for c in recent if c.status in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        
        if len(high_or_above) > 5:
            return "⚠️ 高风险"
        elif len(high_or_above) > 0:
            return "△ 中风险"
        else:
            return "✓ 正常"
    
    def get_action_plan(self, checks: List[RiskCheck]) -> str:
        """生成行动方案"""
        if len(checks) == 0:
            return "继续正常交易"
        
        critical = [c for c in checks if c.status == RiskLevel.CRITICAL]
        high = [c for c in checks if c.status == RiskLevel.HIGH]
        
        actions = []
        
        if critical:
            actions.append(f"【紧急】立即处理 {len(critical)} 项:")
            for c in critical:
                actions.append(f"  - {c.name}: {c.action}")
        
        if high:
            actions.append(f"【重要】今日处理 {len(high)} 项:")
            for c in high[:5]:  # 最多显示5项
                actions.append(f"  - {c.name}: {c.action}")
            if len(high) > 5:
                actions.append(f"  ... 还有 {len(high)-5} 项")
        
        return "\n".join(actions) if actions else "正常"


def run_risk_check(
    weights: pd.Series,
    scores: pd.Series,
    positions: pd.DataFrame,
    nav_history: pd.DataFrame,
    ic_rolling: float = 0.0,
    turnover: float = 0.0,
    cost: float = 0.0,
    industry: Optional[pd.Series] = None,
) -> Tuple[str, str]:
    """
    运行风控检查的便捷函数
    
    Returns:
        (风险状态, 行动方案)
    """
    engine = EnhancedRiskEngine()
    checks = engine.check_all(
        weights, scores, positions, nav_history,
        ic_rolling, turnover, cost, industry
    )
    
    return engine.get_risk_status(), engine.get_action_plan(checks)


if __name__ == '__main__':
    print("增强风控引擎测试")
    print("="*50)
    
    engine = EnhancedRiskEngine()
    
    # 模拟数据
    weights = pd.Series({
        '000001.SZ': 0.08,
        '000002.SZ': 0.07,
        '600000.SH': 0.06,
    })
    
    scores = pd.Series({
        '000001.SZ': 1.5,
        '000002.SZ': 1.2,
        '600000.SH': 0.8,
    })
    
    positions = pd.DataFrame({
        'symbol': ['000001.SZ', '000002.SZ', '600000.SH'],
        'volume': [1000000, 800000, 600000],
        'close': [10, 20, 15],
    })
    
    nav_history = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=60),
        'nav': np.linspace(1.0, 1.05, 60),
    })
    
    # 运行检查
    checks = engine.check_all(
        weights=weights,
        scores=scores,
        positions=positions,
        nav_history=nav_history,
        ic_rolling=0.02,
        turnover=0.15,
        cost=100,
    )
    
    print(f"风险状态: {engine.get_risk_status()}")
    print(f"\n行动方案:\n{engine.get_action_plan(checks)}")
