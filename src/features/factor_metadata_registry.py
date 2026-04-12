"""
因子元数据注册表 (Factor Metadata Registry)

每个因子必须登记:
1. 基本信息: 名称、描述、家族
2. 计算信息: 公式、数据依赖、lookback
3. 时点信息: 更新频率、lag规则
4. 状态信息: 当前状态(候选/研究/生产/退役)
5. 评估信息: IC、单调性、换手率、成本等
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np


class FactorFamily(Enum):
    """因子家族"""
    VALUE = "value"                      # 价值因子
    PROFITABILITY = "profitability"      # 盈利因子
    GROWTH = "growth"                   # 成长因子
    LEVERAGE = "leverage"              # 杠杆因子
    EFFICIENCY = "efficiency"           # 效率因子
    MOMENTUM = "momentum"               # 动量因子
    REVERSAL = "reversal"              # 反转因子
    VOLATILITY = "volatility"           # 波动率因子
    LIQUIDITY = "liquidity"             # 流动性因子
    DISTRIBUTION = "distribution"       # 分布因子
    PATTERN = "pattern"                # 形态因子
    SECTOR = "sector"                  # 行业因子
    SECTOR_ADJ_PATTERN = "sector_adj_pattern"  # 行业调整形态因子


class FactorStatus(Enum):
    """因子状态"""
    CANDIDATE = "candidate"            # 候选
    RESEARCH = "research"               # 研究中
    PRODUCTION = "production"          # 生产中
    OBSERVATION = "observation"        # 观察中
    RETIRED = "retired"               # 已退役


class FactorRole(Enum):
    """因子角色"""
    PRIMARY = "primary"                # 主因子 - 直接贡献收益
    AUXILIARY = "auxiliary"            # 辅助因子 - 提供条件信息
    REDUNDANT = "redundant"            # 冗余因子 - 与其他因子高度相关
    CONDITIONAL = "conditional"        # 条件因子 - 仅在特定市场有效


@dataclass
class FactorMetadata:
    """因子元数据"""
    name: str
    description: str = ""
    family: FactorFamily = FactorFamily.VALUE
    
    # 计算信息
    formula: str = ""                   # 计算公式
    data_dependencies: list[str] = field(default_factory=list)
    lookback: int = 1
    preprocessing: list[str] = field(default_factory=list)
    
    # 时点信息
    update_frequency: str = "daily"     # daily / weekly / quarterly
    timing_type: str = "end_of_period"  # 时点类型
    
    # 评估指标 (动态更新)
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    ic_positive_rate: float = 0.0
    monotonicity: float = 0.0
    turnover: float = 0.0
    coverage: float = 0.0
    
    # IC decay (不同持有期的IC)
    ic_decay: dict = field(default_factory=dict)  # {5: 0.02, 10: 0.03, 20: 0.025}
    half_life: int = 0                    # 半衰期(天)
    
    # 状态信息
    status: FactorStatus = FactorStatus.CANDIDATE
    role: FactorRole = FactorRole.PRIMARY
    
    # 相关性信息
    highly_correlated_with: list[str] = field(default_factory=list)
    substitute_factor: Optional[str] = None  # 冗余时推荐使用的替代因子
    
    # 极端行情稳定性
    regime_stability: dict = field(default_factory=dict)
    # {'high_vol': 0.02, 'low_vol': 0.03, 'uptrend': 0.04, 'downtrend': 0.01}
    
    # 中性化后表现
    neutralized_ic: float = 0.0
    industry_neutral_ic: float = 0.0
    size_neutral_ic: float = 0.0
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_ic_update: Optional[datetime] = None
    last_full_audit: Optional[datetime] = None
    
    # 备注
    notes: str = ""
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'family': self.family.value,
            'formula': self.formula,
            'data_dependencies': self.data_dependencies,
            'lookback': self.lookback,
            'preprocessing': self.preprocessing,
            'update_frequency': self.update_frequency,
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'ic_ir': self.ic_ir,
            'ic_positive_rate': self.ic_positive_rate,
            'monotonicity': self.monotonicity,
            'turnover': self.turnover,
            'coverage': self.coverage,
            'ic_decay': self.ic_decay,
            'half_life': self.half_life,
            'status': self.status.value,
            'role': self.role.value,
            'highly_correlated_with': self.highly_correlated_with,
            'substitute_factor': self.substitute_factor,
            'regime_stability': self.regime_stability,
            'neutralized_ic': self.neutralized_ic,
            'industry_neutral_ic': self.industry_neutral_ic,
            'size_neutral_ic': self.size_neutral_ic,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_ic_update': self.last_ic_update.isoformat() if self.last_ic_update else None,
            'last_full_audit': self.last_full_audit.isoformat() if self.last_full_audit else None,
            'notes': self.notes,
        }


class FactorMetadataRegistry:
    """
    因子元数据注册表
    
    用法:
    registry = FactorMetadataRegistry()
    
    # 注册因子
    registry.register('roe', family=FactorFamily.PROFITABILITY, ...)
    
    # 查询
    registry.get('roe')
    registry.get_by_family(FactorFamily.PROFITABILITY)
    registry.get_by_status(FactorStatus.PRODUCTION)
    
    # 更新评估
    registry.update_ic('roe', ic_mean=0.035, ic_ir=0.5)
    
    # 导出报告
    registry.to_dataframe()
    """
    
    def __init__(self):
        self._metadata: dict[str, FactorMetadata] = {}
        self._family_index: dict[FactorFamily, list[str]] = {}
        self._status_index: dict[FactorStatus, list[str]] = {}
        self._role_index: dict[FactorRole, list[str]] = {}
    
    def register(
        self,
        name: str,
        description: str = "",
        family: FactorFamily = FactorFamily.VALUE,
        formula: str = "",
        data_dependencies: list[str] = None,
        lookback: int = 1,
        preprocessing: list[str] = None,
        update_frequency: str = "daily",
        timing_type: str = "end_of_period",
    ) -> FactorMetadata:
        """注册新因子"""
        if data_dependencies is None:
            data_dependencies = []
        if preprocessing is None:
            preprocessing = []
        
        metadata = FactorMetadata(
            name=name,
            description=description,
            family=family,
            formula=formula,
            data_dependencies=data_dependencies,
            lookback=lookback,
            preprocessing=preprocessing,
            update_frequency=update_frequency,
            timing_type=timing_type,
        )
        
        self._metadata[name] = metadata
        self._update_indices(name, family, FactorStatus.CANDIDATE, FactorRole.PRIMARY)
        
        return metadata
    
    def _update_indices(
        self, 
        name: str, 
        family: FactorFamily,
        status: FactorStatus,
        role: FactorRole,
    ):
        """更新索引"""
        # Family index
        if family not in self._family_index:
            self._family_index[family] = []
        if name not in self._family_index[family]:
            self._family_index[family].append(name)
        
        # Status index
        if status not in self._status_index:
            self._status_index[status] = []
        if name not in self._status_index[status]:
            self._status_index[status].append(name)
        
        # Role index
        if role not in self._role_index:
            self._role_index[role] = []
        if name not in self._role_index[role]:
            self._role_index[role].append(name)
    
    def get(self, name: str) -> Optional[FactorMetadata]:
        """获取因子元数据"""
        return self._metadata.get(name)
    
    def get_by_family(self, family: FactorFamily) -> list[FactorMetadata]:
        """获取指定家族的因子"""
        names = self._family_index.get(family, [])
        return [self._metadata[n] for n in names if n in self._metadata]
    
    def get_by_status(self, status: FactorStatus) -> list[FactorMetadata]:
        """获取指定状态的因子"""
        names = self._status_index.get(status, [])
        return [self._metadata[n] for n in names if n in self._metadata]
    
    def get_by_role(self, role: FactorRole) -> list[FactorMetadata]:
        """获取指定角色的因子"""
        names = self._role_index.get(role, [])
        return [self._metadata[n] for n in names if n in self._metadata]
    
    def get_production_factors(self) -> list[FactorMetadata]:
        """获取生产中的因子"""
        return self.get_by_status(FactorStatus.PRODUCTION)
    
    def get_candidate_factors(self) -> list[FactorMetadata]:
        """获取候选因子"""
        return self.get_by_status(FactorStatus.CANDIDATE)
    
    def update_ic(
        self,
        name: str,
        ic_mean: float,
        ic_std: float,
        ic_ir: float,
        ic_positive_rate: float,
        monotonicity: float,
        turnover: float,
        coverage: float,
    ):
        """更新IC评估"""
        metadata = self._metadata.get(name)
        if metadata:
            metadata.ic_mean = ic_mean
            metadata.ic_std = ic_std
            metadata.ic_ir = ic_ir
            metadata.ic_positive_rate = ic_positive_rate
            metadata.monotonicity = monotonicity
            metadata.turnover = turnover
            metadata.coverage = coverage
            metadata.last_ic_update = datetime.now()
            metadata.updated_at = datetime.now()
    
    def update_decay(
        self,
        name: str,
        ic_decay: dict,
        half_life: int,
    ):
        """更新IC decay分析"""
        metadata = self._metadata.get(name)
        if metadata:
            metadata.ic_decay = ic_decay
            metadata.half_life = half_life
            metadata.updated_at = datetime.now()
    
    def update_regime_stability(self, name: str, regime_stability: dict):
        """更新极端行情稳定性"""
        metadata = self._metadata.get(name)
        if metadata:
            metadata.regime_stability = regime_stability
            metadata.updated_at = datetime.now()
    
    def update_neutralization(self, name: str, industry_neutral_ic: float, size_neutral_ic: float):
        """更新中性化后IC"""
        metadata = self._metadata.get(name)
        if metadata:
            metadata.industry_neutral_ic = industry_neutral_ic
            metadata.size_neutral_ic = size_neutral_ic
            metadata.neutralized_ic = max(industry_neutral_ic, size_neutral_ic)
            metadata.updated_at = datetime.now()
    
    def update_status(self, name: str, status: FactorStatus):
        """更新因子状态"""
        metadata = self._metadata.get(name)
        if metadata:
            old_status = metadata.status
            metadata.status = status
            metadata.updated_at = datetime.now()
            
            # 更新索引
            if old_status in self._status_index:
                if name in self._status_index[old_status]:
                    self._status_index[old_status].remove(name)
            if status not in self._status_index:
                self._status_index[status] = []
            if name not in self._status_index[status]:
                self._status_index[status].append(name)
    
    def update_role(
        self, 
        name: str, 
        role: FactorRole,
        highly_correlated_with: list[str] = None,
        substitute_factor: str = None,
    ):
        """更新因子角色"""
        metadata = self._metadata.get(name)
        if metadata:
            old_role = metadata.role
            metadata.role = role
            if highly_correlated_with:
                metadata.highly_correlated_with = highly_correlated_with
            if substitute_factor:
                metadata.substitute_factor = substitute_factor
            metadata.updated_at = datetime.now()
            
            # 更新索引
            if old_role in self._role_index:
                if name in self._role_index[old_role]:
                    self._role_index[old_role].remove(name)
            if role not in self._role_index:
                self._role_index[role] = []
            if name not in self._role_index[role]:
                self._role_index[role].append(name)
    
    def mark_for_retirement(self, name: str, reason: str = ""):
        """标记因子退役"""
        metadata = self._metadata.get(name)
        if metadata:
            metadata.status = FactorStatus.RETIRED
            metadata.notes = f"退役原因: {reason}"
            metadata.updated_at = datetime.now()
    
    def to_dataframe(self) -> pd.DataFrame:
        """导出为DataFrame"""
        records = []
        for m in self._metadata.values():
            records.append(m.to_dict())
        return pd.DataFrame(records)
    
    def get_pool_summary(self) -> dict:
        """获取因子池汇总"""
        return {
            'total': len(self._metadata),
            'by_family': {
                f.value: len(names) 
                for f, names in self._family_index.items()
            },
            'by_status': {
                s.value: len(names)
                for s, names in self._status_index.items()
            },
            'by_role': {
                r.value: len(names)
                for r, names in self._role_index.items()
            },
        }
    
    def get_recommendations(self) -> pd.DataFrame:
        """获取因子池优化建议"""
        recommendations = []
        
        # 检查冗余
        for name, metadata in self._metadata.items():
            if metadata.role == FactorRole.REDUNDANT:
                recommendations.append({
                    'action': 'REMOVE',
                    'factor': name,
                    'reason': f'与 {metadata.highly_correlated_with} 高度冗余',
                    'suggestion': f'保留 {metadata.substitute_factor}' if metadata.substitute_factor else '考虑移除',
                    'priority': 'HIGH',
                })
        
        # 检查IC下降
        for name, metadata in self._metadata.items():
            if metadata.status == FactorStatus.PRODUCTION:
                if metadata.ic_ir < 0.1:
                    recommendations.append({
                        'action': 'OBSERVE',
                        'factor': name,
                        'reason': f'IC IR 下降到 {metadata.ic_ir:.3f}',
                        'suggestion': '持续监控，如持续下降考虑降权',
                        'priority': 'MEDIUM',
                    })
        
        # 检查长期未更新
        for name, metadata in self._metadata.items():
            if metadata.last_ic_update:
                days_since_update = (datetime.now() - metadata.last_ic_update).days
                if days_since_update > 90:
                    recommendations.append({
                        'action': 'UPDATE',
                        'factor': name,
                        'reason': f'IC数据已 {days_since_update} 天未更新',
                        'suggestion': '重新计算IC进行审计',
                        'priority': 'LOW',
                    })
        
        return pd.DataFrame(recommendations) if recommendations else pd.DataFrame()


# 全局实例
_factor_metadata_registry: Optional[FactorMetadataRegistry] = None

def get_factor_metadata_registry() -> FactorMetadataRegistry:
    """获取全局因子元数据注册表"""
    global _factor_metadata_registry
    if _factor_metadata_registry is None:
        _factor_metadata_registry = FactorMetadataRegistry()
    return _factor_metadata_registry
