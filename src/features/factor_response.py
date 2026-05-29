"""
因子响应监控系统 - 实时监控因子健康状态

基于AGENTS.md的快速响应规则：
- IC连续5天<0 → 进入观察
- IC连续10天<0 → 降权50%
- IC连续15天<0 → 下线，从备用池替换
- IC连续20天>0 → 可重新上线

止损规则：
- IC<-0.03 → 立即下线
- 单因子回撤>2% → 强制降权
- 贡献为负 → 权重归零
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta
from enum import Enum


class FactorStatus(Enum):
    """因子状态"""
    ACTIVE = "active"           # 活跃
    WATCH = "watch"             # 观察中
    REDUCED = "reduced"         # 降权
    OFFLINE = "offline"         # 已下线
    RESERVE = "reserve"         # 备用池


@dataclass
class FactorResponseRecord:
    """单次响应记录"""
    trade_date: pd.Timestamp
    factor_name: str
    ic: float
    rank_ic: float
    status: FactorStatus
    consecutive_negative: int
    action_taken: str
    notes: str = ""


@dataclass
class FactorHealthSnapshot:
    """因子健康快照"""
    factor_name: str
    status: FactorStatus
    
    # IC相关
    ic_latest: float = 0.0
    ic_5d_avg: float = 0.0
    ic_20d_avg: float = 0.0
    ic_ytd_avg: float = 0.0
    
    # 连续性
    consecutive_negative: int = 0
    consecutive_positive: int = 0
    
    # 稳定性
    ic_std_20d: float = 0.0
    ic_positive_rate_60d: float = 0.0
    
    # 贡献
    weight: float = 1.0
    contribution: float = 0.0
    
    # 动作
    action_required: str = "none"
    action_deadline: Optional[pd.Timestamp] = None


@dataclass
class FactorResponseMonitor:
    """因子响应监控器"""
    factor_names: list[str]
    ic_threshold_watch: float = 0.0
    ic_threshold_offline: float = -0.03
    consecutive_days_watch: int = 5
    consecutive_days_reduce: int = 10
    consecutive_days_offline: int = 15
    consecutive_days_recover: int = 20
    
    # 内部状态
    _ic_history: dict[str, list[float]] = field(default_factory=dict)
    _rank_ic_history: dict[str, list[float]] = field(default_factory=dict)
    _status_history: dict[str, list[tuple]] = field(default_factory=dict)
    _current_status: dict[str, FactorStatus] = field(default_factory=dict)
    _weights: dict[str, float] = field(default_factory=dict)
    _consecutive_negative: dict[str, int] = field(default_factory=dict)
    _consecutive_positive: dict[str, int] = field(default_factory=dict)
    _records: list[FactorResponseRecord] = field(default_factory=list)
    
    def __post_init__(self):
        for f in self.factor_names:
            self._ic_history[f] = []
            self._rank_ic_history[f] = []
            self._status_history[f] = []
            self._current_status[f] = FactorStatus.ACTIVE
            self._weights[f] = 1.0
            self._consecutive_negative[f] = 0
            self._consecutive_positive[f] = 0
    
    def update(self, trade_date: pd.Timestamp, ic_data: dict[str, float], rank_ic_data: Optional[dict[str, float]] = None) -> list[FactorResponseRecord]:
        """更新因子IC数据并返回需要采取的行动"""
        records = []
        
        for factor_name in self.factor_names:
            ic = ic_data.get(factor_name, np.nan)
            rank_ic = rank_ic_data.get(factor_name, np.nan) if rank_ic_data else np.nan
            
            if np.isnan(ic):
                continue
            
            self._ic_history[factor_name].append(ic)
            self._rank_ic_history[factor_name].append(rank_ic)
            
            prev_status = self._current_status[factor_name]
            
            self._update_consecutive_count(factor_name, ic)
            
            new_status, action, new_weight = self._determine_status_and_action(factor_name, ic)
            
            if new_status != prev_status:
                self._current_status[factor_name] = new_status
                self._status_history[factor_name].append((trade_date, new_status, action))
            
            # 更新权重
            self._weights[factor_name] = new_weight
            
            record = FactorResponseRecord(
                trade_date=trade_date,
                factor_name=factor_name,
                ic=ic,
                rank_ic=rank_ic,
                status=new_status,
                consecutive_negative=self._consecutive_negative[factor_name],
                action_taken=action,
            )
            self._records.append(record)
            records.append(record)
        
        return records
    
    def _update_consecutive_count(self, factor_name: str, ic: float):
        """更新连续正负计数"""
        if ic < 0:
            self._consecutive_negative[factor_name] += 1
            self._consecutive_positive[factor_name] = 0
        elif ic > 0:
            self._consecutive_positive[factor_name] += 1
            self._consecutive_negative[factor_name] = 0
        else:
            self._consecutive_negative[factor_name] = 0
            self._consecutive_positive[factor_name] = 0
    
    def _determine_status_and_action(self, factor_name: str, ic: float) -> tuple[FactorStatus, str, float]:
        """
        根据IC确定状态和行动
        
        Returns:
            tuple[FactorStatus, action_description, weight]
        
        核心原则：
        1. 默认保持当前状态（慢速变化）
        2. 恶化条件满足 → 立即降级
        3. 恢复条件满足 → 逐步升级（不能跳级）
        """
        prev_status = self._current_status[factor_name]
        prev_weight = self._weights[factor_name]
        
        # 1. 检查立即下线条件（最高优先级）
        if ic < self.ic_threshold_offline:
            return FactorStatus.OFFLINE, "IMMEDIATE OFFLINE - IC below threshold", 0.0
        
        # 2. 检查连续负IC导致的下线
        if self._consecutive_negative[factor_name] >= self.consecutive_days_offline:
            if prev_status != FactorStatus.OFFLINE:
                return FactorStatus.OFFLINE, f"OFFLINE after {self._consecutive_negative[factor_name]} consecutive negative IC", 0.0
        
        # 3. 检查连续负IC导致的降权
        if self._consecutive_negative[factor_name] >= self.consecutive_days_reduce:
            if prev_status not in [FactorStatus.OFFLINE, FactorStatus.REDUCED]:
                return FactorStatus.REDUCED, f"REDUCED 50% after {self._consecutive_negative[factor_name]} consecutive negative IC", 0.5
        
        # 4. 检查进入观察
        if self._consecutive_negative[factor_name] >= self.consecutive_days_watch:
            if prev_status == FactorStatus.ACTIVE:
                return FactorStatus.WATCH, f"WATCH after {self._consecutive_negative[factor_name]} consecutive negative IC", prev_weight
        
        # 5. 检查恢复条件（只有WATCH状态可以直接恢复）
        if self._consecutive_positive[factor_name] >= self.consecutive_days_recover:
            if prev_status == FactorStatus.WATCH:
                return FactorStatus.ACTIVE, f"RECOVERED after {self._consecutive_positive[factor_name]} consecutive positive IC", 1.0
        
        # 6. OFFLINE/RESERVE/REDUCED 状态保持，不自动恢复
        if prev_status == FactorStatus.OFFLINE:
            return FactorStatus.RESERVE, "Reserve pool - awaiting recovery", 0.0
        
        if prev_status == FactorStatus.RESERVE:
            return FactorStatus.RESERVE, "Still in reserve", 0.0
        
        if prev_status == FactorStatus.REDUCED:
            return FactorStatus.REDUCED, "Still reduced - need full recovery period", 0.5
        
        # 7. WATCH 状态保持（除非满足恢复条件）
        if prev_status == FactorStatus.WATCH:
            return FactorStatus.WATCH, f"Still watching - {self._consecutive_positive[factor_name]}/{self.consecutive_days_recover} positive IC to recover", prev_weight
        
        # 8. ACTIVE 状态保持
        return FactorStatus.ACTIVE, "none", 1.0
    
    def get_health_snapshot(self) -> pd.DataFrame:
        """获取当前健康快照"""
        snapshots = []
        
        for factor_name in self.factor_names:
            ic_hist = np.array(self._ic_history.get(factor_name, [np.nan]))
            rank_ic_hist = np.array(self._rank_ic_history.get(factor_name, [np.nan]))
            
            ic_5d = ic_hist[-5:].mean() if len(ic_hist) >= 5 else np.nan
            ic_20d = ic_hist[-20:].mean() if len(ic_hist) >= 20 else np.nan
            ic_60d = ic_hist[-60:].mean() if len(ic_hist) >= 60 else np.nan
            ic_std_20d = ic_hist[-20:].std() if len(ic_hist) >= 20 else np.nan
            positive_rate_60d = (ic_hist[-60:] > 0).mean() if len(ic_hist) >= 60 else np.nan
            
            snapshot = FactorHealthSnapshot(
                factor_name=factor_name,
                status=self._current_status[factor_name],
                ic_latest=ic_hist[-1] if len(ic_hist) > 0 else np.nan,
                ic_5d_avg=ic_5d,
                ic_20d_avg=ic_20d,
                ic_ytd_avg=ic_60d,
                consecutive_negative=self._consecutive_negative[factor_name],
                consecutive_positive=self._consecutive_positive[factor_name],
                ic_std_20d=ic_std_20d,
                ic_positive_rate_60d=positive_rate_60d,
                weight=self._weights[factor_name],
            )
            
            snapshots.append({
                'factor_name': snapshot.factor_name,
                'status': snapshot.status.value,
                'ic_latest': snapshot.ic_latest,
                'ic_5d_avg': snapshot.ic_5d_avg,
                'ic_20d_avg': snapshot.ic_20d_avg,
                'ic_60d_avg': snapshot.ic_ytd_avg,
                'ic_std_20d': snapshot.ic_std_20d,
                'positive_rate_60d': snapshot.ic_positive_rate_60d,
                'consecutive_negative': snapshot.consecutive_negative,
                'consecutive_positive': snapshot.consecutive_positive,
                'weight': snapshot.weight,
            })
        
        return pd.DataFrame(snapshots)
    
    def get_weights(self) -> dict[str, float]:
        """获取当前因子权重"""
        return self._weights.copy()
    
    def get_records(self) -> pd.DataFrame:
        """获取所有响应记录"""
        if not self._records:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'trade_date': r.trade_date,
                'factor_name': r.factor_name,
                'ic': r.ic,
                'rank_ic': r.rank_ic,
                'status': r.status.value,
                'consecutive_negative': r.consecutive_negative,
                'action': r.action_taken,
                'notes': r.notes,
            }
            for r in self._records
        ])
    
    def get_action_summary(self) -> pd.DataFrame:
        """获取需要采取的行动摘要"""
        health = self.get_health_snapshot()
        
        actions = health[health['status'] != FactorStatus.ACTIVE.value].copy()
        actions = actions.sort_values(['status', 'ic_latest'])
        
        return actions
    
    def export_config(self) -> dict:
        """导出监控配置"""
        return {
            'monitor_config': {
                'ic_threshold_watch': self.ic_threshold_watch,
                'ic_threshold_offline': self.ic_threshold_offline,
                'consecutive_days_watch': self.consecutive_days_watch,
                'consecutive_days_reduce': self.consecutive_days_reduce,
                'consecutive_days_offline': self.consecutive_days_offline,
                'consecutive_days_recover': self.consecutive_days_recover,
            },
            'current_status': {k: v.value for k, v in self._current_status.items()},
            'current_weights': self._weights.copy(),
        }


def create_monitor(factor_names: list[str], **kwargs) -> FactorResponseMonitor:
    """创建监控器"""
    return FactorResponseMonitor(factor_names=factor_names, **kwargs)


if __name__ == "__main__":
    monitor = create_monitor(
        factor_names=['mom250', 'close_to_high250', 'alpha_006'],
        ic_threshold_watch=0.0,
        ic_threshold_offline=-0.03,
    )
    
    dates = pd.date_range('2024-01-01', periods=30, freq='B')
    ic_data = {
        'mom250': [0.02, 0.01, 0.03, -0.01, -0.02, -0.01, -0.01, -0.01, -0.01, -0.01] * 3,
        'close_to_high250': [0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01] * 3,
        'alpha_006': [0.03, 0.02, -0.01, -0.02, -0.03, -0.04, -0.05, -0.04, -0.03, -0.02][:10] * 3,
    }
    
    for i, date in enumerate(dates[:30]):
        ic_slice = {k: v[i] if i < len(v) else v[-1] for k, v in ic_data.items()}
        monitor.update(date, ic_slice)
    
    print("Health Snapshot:")
    print(monitor.get_health_snapshot())
    print("\nWeights:", monitor.get_weights())
    print("\nActions Needed:")
    print(monitor.get_action_summary())
