"""
因子频率管理器

解决核心问题：财务因子季度更新，不应该日频重训

核心逻辑：
1. 财务因子只在财报发布后更新
2. 日频模型训练时，使用最近可用的财务数据
3. 模型不需要每5天重训，可以用同一套权重更久

Usage:
    from src.features.factor_frequency_manager import FactorFrequencyManager
    
    manager = FactorFrequencyManager()
    manager.register_factor('roe', 'quarterly', update_months=[4, 8, 10])
    
    # 获取某日期应该使用的因子值
    available_factors = manager.get_available_factors(as_of_date)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class FactorUpdateFrequency(Enum):
    """因子更新频率"""
    DAILY = "daily"           # 日频：价格、成交量、动量
    WEEKLY = "weekly"        # 周频：资金流
    MONTHLY = "monthly"      # 月频：宏观数据
    QUARTERLY = "quarterly"  # 季度：财务数据
    ANNOUNCEMENT = "announcement"  # 公告发布时


@dataclass
class FactorFrequency:
    """因子频率配置"""
    name: str
    frequency: FactorUpdateFrequency
    update_months: list[int] = field(default_factory=list)  # 财报发布月份
    lookback_days: int = 1  # 使用多少天前的数据（财报滞后）
    is_stale_after_days: int = 90  # 多少天后数据过时

    def get_latest_update_date(self, as_of: datetime) -> datetime:
        """获取最近一次应该更新的日期"""
        if self.frequency == FactorUpdateFrequency.DAILY:
            return as_of
        elif self.frequency == FactorUpdateFrequency.QUARTERLY:
            # 找最近一个财报发布月份
            months = self.update_months or [3, 4, 8, 10]  # 财报季
            current_month = as_of.month

            # 找本季度或上一季度
            for m in sorted(months):
                if m <= current_month:
                    return as_of.replace(month=m, day=1)

            # 上一年的最后一个财报月
            return as_of.replace(year=as_of.year - 1, month=months[-1], day=1)

        return as_of


@dataclass
class FactorStaleness:
    """因子时效状态"""
    factor_name: str
    last_update_date: datetime
    is_stale: bool
    days_since_update: int
    recommended_action: str  # 'use_current', 'use_lagged', 'skip'


class FactorFrequencyManager:
    """
    因子频率管理器
    
    作用：
    1. 跟踪每个因子的更新频率
    2. 判断因子在某日期是否过时
    3. 决定是使用当前值还是滞后值
    """

    def __init__(self):
        self._factors: dict[str, FactorFrequency] = {}
        self._last_update_cache: dict[str, datetime] = {}

    def register_factor(
        self,
        name: str,
        frequency: FactorUpdateFrequency,
        update_months: list[int] | None = None,
        lookback_days: int = 1,
        is_stale_after_days: int = 90,
    ):
        """注册因子频率"""
        self._factors[name] = FactorFrequency(
            name=name,
            frequency=frequency,
            update_months=update_months or [4, 8, 10],  # 默认财报季
            lookback_days=lookback_days,
            is_stale_after_days=is_stale_after_days,
        )

    def register_financial_factors(self):
        """注册所有财务因子"""
        quarterly_factors = [
            'roe', 'roe_weighted', 'earnings_yield', 'earnings_yield_weighted',
            'operating_margin', 'net_margin', 'gross_margin',
            'equity_growth', 'asset_growth', 'revenue_growth', 'profit_growth',
            'debt_ratio', 'current_ratio', 'cash_ratio',
            'ocf_per_share', 'asset_turnover', 'inv_turnover',
            'book_to_price', 'roa', 'total_roa',
        ]

        for f in quarterly_factors:
            self.register_factor(
                name=f,
                frequency=FactorUpdateFrequency.QUARTERLY,
                update_months=[4, 8, 10],  # 年报(4月), 中报(8月), 三季报(10月)
                lookback_days=30,  # 财报滞后约30天
                is_stale_after_days=95,
            )

    def register_price_factors(self):
        """注册价格因子"""
        daily_factors = [
            'mom5', 'mom10', 'mom20', 'mom60', 'mom120', 'mom250',
            'vol5', 'vol20', 'vol60',
            'rev5', 'rev20',
            'close_to_high', 'high_low_pos',
            'amount', 'turnover_rate',
        ]

        for f in daily_factors:
            self.register_factor(
                name=f,
                frequency=FactorUpdateFrequency.DAILY,
                is_stale_after_days=1,
            )

    def check_staleness(
        self,
        factor_name: str,
        last_value_date: datetime,
        as_of: datetime,
    ) -> FactorStaleness:
        """检查因子是否过时"""
        config = self._factors.get(factor_name)

        if config is None:
            # 未知因子，假设日频
            return FactorStaleness(
                factor_name=factor_name,
                last_update_date=last_value_date,
                is_stale=False,
                days_since_update=(as_of - last_value_date).days,
                recommended_action='use_current',
            )

        days_since = (as_of - last_value_date).days
        is_stale = days_since > config.is_stale_after_days

        if config.frequency == FactorUpdateFrequency.DAILY:
            action = 'use_current'
        elif is_stale:
            action = 'skip'  # 数据太旧，应该跳过或用行业平均
        else:
            action = 'use_lagged'  # 使用滞后值

        return FactorStaleness(
            factor_name=factor_name,
            last_update_date=last_value_date,
            is_stale=is_stale,
            days_since_update=days_since,
            recommended_action=action,
        )

    def get_training_frequency(
        self,
        factor_names: list[str],
        as_of: datetime,
    ) -> str:
        """
        获取基于最慢因子的训练频率
        
        Returns:
            'daily': 每天都应该重训
            'weekly': 每周重训
            'monthly': 每月重训
            'quarterly': 每季度重训
        """
        frequencies = []

        for name in factor_names:
            config = self._factors.get(name)
            if config:
                frequencies.append(config.frequency)
            else:
                frequencies.append(FactorUpdateFrequency.DAILY)

        # 返回最慢的频率
        priority = {
            FactorUpdateFrequency.DAILY: 1,
            FactorUpdateFrequency.WEEKLY: 2,
            FactorUpdateFrequency.MONTHLY: 3,
            FactorUpdateFrequency.QUARTERLY: 4,
            FactorUpdateFrequency.ANNOUNCEMENT: 5,
        }

        slowest = max(frequencies, key=lambda f: priority.get(f, 1))
        return slowest.value

    def should_retrain_model(
        self,
        factor_names: list[str],
        last_train_date: datetime,
        current_date: datetime,
    ) -> tuple[bool, str]:
        """
        判断是否应该重训模型
        
        Returns:
            (should_retrain, reason)
        """
        training_freq = self.get_training_frequency(factor_names, current_date)

        days_since_train = (current_date - last_train_date).days

        if training_freq == 'daily':
            threshold = 5  # 至少5天
        elif training_freq == 'weekly':
            threshold = 7
        elif training_freq == 'monthly':
            threshold = 30
        elif training_freq == 'quarterly':
            threshold = 90
        else:
            threshold = 60

        should = days_since_train >= threshold

        return should, f"days_since_train={days_since_train}, threshold={threshold}"

    def filter_stale_factors(
        self,
        factor_last_dates: dict[str, datetime],
        as_of: datetime,
        threshold_days: int = 60,
    ) -> tuple[list[str], list[str]]:
        """
        过滤过时因子
        
        Returns:
            (fresh_factors, stale_factors)
        """
        fresh = []
        stale = []

        for name, last_date in factor_last_dates.items():
            days_since = (as_of - last_date).days
            if days_since <= threshold_days:
                fresh.append(name)
            else:
                stale.append(name)

        return fresh, stale

    def get_recommended_preprocessing(
        self,
        factor_name: str,
    ) -> dict:
        """获取因子推荐的预处理方式"""
        config = self._factors.get(factor_name)

        if config is None:
            return {'fill_missing': 'cross_sectional_median'}

        if config.frequency == FactorUpdateFrequency.QUARTERLY:
            return {
                'fill_missing': 'industry_median',
                'add_missing_flag': True,  # 缺失本身是信息
                'clip_extreme': True,
                'neutralize': 'industry',
            }
        else:
            return {
                'fill_missing': 'cross_sectional_median',
                'add_missing_flag': False,
                'clip_extreme': True,
                'neutralize': None,
            }


def create_frequency_manager() -> FactorFrequencyManager:
    """创建并配置好的频率管理器"""
    manager = FactorFrequencyManager()
    manager.register_financial_factors()
    manager.register_price_factors()
    return manager
