"""
数据时点审计层 (Data Timing Audit Layer)

每个字段必须登记:
1. 原始发布日期 (pub_date)
2. 可交易使用日期 (tradeable_date)
3. Lag规则 (lag_days)
4. 是否Point-in-Time (PIT)

这是防止未来函数的第一道门。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd


class DataTimingType(Enum):
    """数据时点类型"""
    END_OF_PERIOD = "end_of_period"      # 期末数据 (如季报)
    POINT_IN_TIME = "point_in_time"      # 时点数据 (如当日收盘)
    ANNOUNCEMENT = "announcement"         # 公告数据 (如财报发布)
    ESTIMATE = "estimate"                # 估算数据 (如分析师预期)


@dataclass
class DataTimingSpec:
    """数据时点规格"""
    field_name: str
    data_source: str                      # akshare / baostock / 计算
    timing_type: DataTimingType

    # 报告期 vs 发布期
    report_period_lag: int = 0            # 相对报告期末的滞后天数
    announcement_lag: int = 0             # 相对公告日的滞后天数

    # 典型财报发布规则 (A股)
    # Q1: 4月底前
    # Q2: 8月底前
    # Q3: 10月底前
    # Q4/年报: 次年4月底前

    def get_tradeable_date(self, report_date: datetime) -> datetime:
        """计算可交易日期
        
        Args:
            report_date: 报告期/数据日期
            
        Returns:
            可用于交易的日期
        """
        if self.timing_type == DataTimingType.END_OF_PERIOD:
            # 财报类数据，需要等公告发布
            pub_date = self._get_announcement_date(report_date)
            return pub_date + timedelta(days=self.announcement_lag)
        elif self.timing_type == DataTimingType.POINT_IN_TIME:
            # 当日数据，收盘后可用
            return report_date + timedelta(days=self.report_period_lag)
        elif self.timing_type == DataTimingType.ANNOUNCEMENT:
            # 已经是公告数据
            return report_date + timedelta(days=self.announcement_lag)
        else:
            return report_date

    def _get_announcement_date(self, report_date: datetime) -> datetime:
        """根据报告期推算公告发布日期"""
        month = report_date.month

        if month == 3:
            # Q1: 4月底前
            return datetime(report_date.year, 4, 30)
        elif month == 6:
            # Q2: 8月底前
            return datetime(report_date.year, 8, 31)
        elif month == 9:
            # Q3: 10月底前
            return datetime(report_date.year, 10, 31)
        elif month == 12:
            # Q4/年报: 次年4月底前
            return datetime(report_date.year + 1, 4, 30)
        else:
            return report_date


class DataTimingAudit:
    """
    数据时点审计器
    
    用法:
    audit = DataTimingAudit()
    audit.register('roe', 'akshare', DataTimingType.END_OF_PERIOD, report_period_lag=0)
    audit.get_tradeable_date('roe', '2024-03-31')  # 返回 2024-04-30
    """

    def __init__(self):
        self._specs: dict[str, DataTimingSpec] = {}
        self._register_default_specs()

    def _register_default_specs(self):
        """注册默认规格"""

        # ===== 财务因子 =====
        financial_specs = [
            # 盈利因子 - 季报披露，滞后发布
            ('roe', DataTimingType.END_OF_PERIOD, 0),
            ('roe_weighted', DataTimingType.END_OF_PERIOD, 0),
            ('roa', DataTimingType.END_OF_PERIOD, 0),
            ('total_roa', DataTimingType.END_OF_PERIOD, 0),
            ('net_margin', DataTimingType.END_OF_PERIOD, 0),
            ('operating_margin', DataTimingType.END_OF_PERIOD, 0),
            ('gross_margin', DataTimingType.END_OF_PERIOD, 0),

            # 价值因子
            ('earnings_yield', DataTimingType.END_OF_PERIOD, 0),
            ('book_to_price', DataTimingType.END_OF_PERIOD, 0),

            # 成长因子
            ('revenue_growth', DataTimingType.END_OF_PERIOD, 0),
            ('profit_growth', DataTimingType.END_OF_PERIOD, 0),
            ('equity_growth', DataTimingType.END_OF_PERIOD, 0),
            ('asset_growth', DataTimingType.END_OF_PERIOD, 0),

            # 杠杆因子
            ('debt_ratio', DataTimingType.END_OF_PERIOD, 0),
            ('current_ratio', DataTimingType.END_OF_PERIOD, 0),
            ('quick_ratio', DataTimingType.END_OF_PERIOD, 0),
            ('cash_ratio', DataTimingType.END_OF_PERIOD, 0),

            # 效率因子
            ('asset_turnover', DataTimingType.END_OF_PERIOD, 0),
            ('inv_turnover', DataTimingType.END_OF_PERIOD, 0),
            ('ar_turnover', DataTimingType.END_OF_PERIOD, 0),

            # 每股因子
            ('ocf_per_share', DataTimingType.END_OF_PERIOD, 0),
            ('retained_earnings_per_share', DataTimingType.END_OF_PERIOD, 0),
            ('capital_reserve_per_share', DataTimingType.END_OF_PERIOD, 0),
        ]

        for field_name, timing_type, lag in financial_specs:
            self.register(
                field_name=field_name,
                data_source='akshare_financial',
                timing_type=timing_type,
                report_period_lag=lag
            )

        # ===== 价格因子 - 当日收盘后可用 =====
        price_specs = [
            'open', 'high', 'low', 'close', 'volume',
            'pct_chg', 'amount'
        ]
        for field_name in price_specs:
            self.register(
                field_name=field_name,
                data_source='baostock_daily',
                timing_type=DataTimingType.POINT_IN_TIME,
                report_period_lag=0
            )

        # ===== 技术因子 - 计算得到 =====
        tech_specs = [
            'mom1', 'mom5', 'mom10', 'mom20', 'mom60', 'mom120', 'mom250',
            'vol5', 'vol20', 'vol60',
            'close_to_high', 'close_to_low',
        ]
        for field_name in tech_specs:
            self.register(
                field_name=field_name,
                data_source='calculated',
                timing_type=DataTimingType.POINT_IN_TIME,
                report_period_lag=0
            )

        # ===== 形态因子 - 当日可用 =====
        pattern_specs = [
            'candle_body_ratio', 'candle_upper_shadow', 'candle_lower_shadow',
            'volume_trend', 'volume_momentum', 'close_position',
            'trend_strength', 'volatility_ratio', 'gap_size',
        ]
        for field_name in pattern_specs:
            self.register(
                field_name=field_name,
                data_source='calculated',
                timing_type=DataTimingType.POINT_IN_TIME,
                report_period_lag=0
            )

        # ===== 行业因子 - 次日可用 =====
        sector_specs = [
            'sector_mom_20d', 'sector_mom_60d', 'sector_rs_20d', 'sector_regime',
        ]
        for field_name in sector_specs:
            self.register(
                field_name=field_name,
                data_source='calculated',
                timing_type=DataTimingType.POINT_IN_TIME,
                report_period_lag=1  # 需要T日数据，T+1才能计算
            )

        # ===== 行业调整形态因子 =====
        sector_adj_specs = [
            'sector_adj_trend_strength', 'sector_adj_close_position',
            'sector_adj_volume_momentum', 'sector_adj_up_day_ratio',
            'sector_adj_consecutive_up',
        ]
        for field_name in sector_adj_specs:
            self.register(
                field_name=field_name,
                data_source='calculated',
                timing_type=DataTimingType.POINT_IN_TIME,
                report_period_lag=1
            )

    def register(
        self,
        field_name: str,
        data_source: str,
        timing_type: DataTimingType,
        report_period_lag: int = 0,
        announcement_lag: int = 0,
    ):
        """注册数据时点规格"""
        spec = DataTimingSpec(
            field_name=field_name,
            data_source=data_source,
            timing_type=timing_type,
            report_period_lag=report_period_lag,
            announcement_lag=announcement_lag,
        )
        self._specs[field_name] = spec

    def get_spec(self, field_name: str) -> DataTimingSpec | None:
        """获取字段规格"""
        return self._specs.get(field_name)

    def get_tradeable_date(self, field_name: str, data_date: datetime) -> datetime:
        """获取字段在指定日期数据的可交易日期"""
        spec = self._specs.get(field_name)
        if spec is None:
            # 未注册的字段，假设当日可用
            return data_date
        return spec.get_tradeable_date(data_date)

    def get_lag_days(self, field_name: str) -> int:
        """获取字段的滞后天数"""
        spec = self._specs.get(field_name)
        if spec is None:
            return 0
        return spec.report_period_lag + spec.announcement_lag

    def audit_factor_timing(self, factor_name: str, bar_date: datetime) -> dict:
        """审计因子的时点合规性
        
        Returns:
            dict: 包含审计结果
        """
        spec = self._specs.get(factor_name)
        if spec is None:
            return {
                'factor': factor_name,
                'registered': False,
                'timing_type': 'unknown',
                'lag_days': 0,
                'tradeable_date': bar_date,
                'is_pit': True,
                'warning': '因子未注册，假设当日可用'
            }

        tradeable_date = spec.get_tradeable_date(bar_date)
        is_pit = tradeable_date <= bar_date

        return {
            'factor': factor_name,
            'registered': True,
            'timing_type': spec.timing_type.value,
            'data_source': spec.data_source,
            'report_period_lag': spec.report_period_lag,
            'announcement_lag': spec.announcement_lag,
            'lag_days': spec.report_period_lag + spec.announcement_lag,
            'bar_date': bar_date,
            'tradeable_date': tradeable_date,
            'is_pit': is_pit,
            'warning': None if is_pit else f'数据最早在 {tradeable_date} 可用，但当前日期是 {bar_date}'
        }

    def batch_audit(
        self,
        factor_names: list[str],
        bar_date: datetime
    ) -> pd.DataFrame:
        """批量审计多个因子的时点合规性"""
        results = []
        for fn in factor_names:
            result = self.audit_factor_timing(fn, bar_date)
            results.append(result)
        return pd.DataFrame(results)

    def get_max_lag(self, factor_names: list[str]) -> int:
        """获取因子列表中的最大滞后"""
        lags = [self.get_lag_days(fn) for fn in factor_names]
        return max(lags) if lags else 0

    def get_factor_timing_report(self) -> pd.DataFrame:
        """生成因子时点报告"""
        records = []
        for name, spec in sorted(self._specs.items()):
            records.append({
                '因子名': name,
                '数据源': spec.data_source,
                '时点类型': spec.timing_type.value,
                '报告期滞后': spec.report_period_lag,
                '公告期滞后': spec.announcement_lag,
                '总滞后天数': spec.report_period_lag + spec.announcement_lag,
            })
        return pd.DataFrame(records)


# 全局实例
_data_timing_audit: DataTimingAudit | None = None

def get_data_timing_audit() -> DataTimingAudit:
    """获取全局数据时点审计器"""
    global _data_timing_audit
    if _data_timing_audit is None:
        _data_timing_audit = DataTimingAudit()
    return _data_timing_audit
