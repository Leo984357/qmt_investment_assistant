"""
因子退化分析模块 - 专业的因子生命周期管理
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class DegradationType(Enum):
    """退化类型"""
    HEALTHY = "healthy"              # 健康
    ECONOMIC_DECAY = "economic"       # 经济机制退化
    TRADING_DECAY = "trading"        # 交易层退化
    EXPRESSION_DECAY = "expression"   # 表达层退化
    DEAD = "dead"                    # 死亡


@dataclass
class SignalMetrics:
    """信号层指标"""
    ic_mean: float
    ic_ir: float
    ic_win_rate: float
    rank_ic_mean: float
    rank_ic_ir: float
    monotonicity: float
    yearly_ic: dict[int, float] = field(default_factory=dict)
    recent_ic_decay: float = 0.0


@dataclass
class PortfolioMetrics:
    """组合层指标"""
    gross_return: float
    net_return_after_cost: float
    turnover: float
    win_rate: float
    max_drawdown: float
    concentration_top10: float
    cost_per_trade: float


@dataclass
class ExposureMetrics:
    """暴露层指标"""
    style_correlation: dict[str, float] = field(default_factory=dict)
    industry_exposure: dict[str, float] = field(default_factory=dict)
    year_concentration: dict[int, float] = field(default_factory=dict)
    stock_concentration: float = 0.0
    extreme_stock_ratio: float = 0.0


@dataclass
class DegradationReport:
    """因子退化报告"""
    name: str

    # 三层指标
    signal: SignalMetrics
    portfolio: PortfolioMetrics
    exposure: ExposureMetrics

    # 退化诊断
    degradation_type: DegradationType
    degradation_score: float  # 0-100, 越高越差

    # 具体问题
    issues: list[str]

    # 建议
    action: str  # keep, reduce_weight, reformulate, offline
    target_weight: float  # 建议权重调整


class FactorDecayAnalyzer:
    """因子退化分析器"""

    def __init__(
        self,
        # 信号层阈值
        min_ic_mean: float = 0.005,
        min_ic_ir: float = 0.1,
        min_monotonicity: float = 0.5,
        min_industry_coverage: float = 0.5,

        # 组合层阈值
        min_net_return: float = 0.0,
        max_turnover: float = 0.5,
        max_concentration: float = 0.3,

        # 退化检测
        decay_detection_window: int = 3,  # 连续N年
        decay_threshold: float = 0.3,     # IC下降30%触发
    ):
        self.min_ic_mean = min_ic_mean
        self.min_ic_ir = min_ic_ir
        self.min_monotonicity = min_monotonicity
        self.min_industry_coverage = min_industry_coverage
        self.min_net_return = min_net_return
        self.max_turnover = max_turnover
        self.max_concentration = max_concentration
        self.decay_detection_window = decay_detection_window
        self.decay_threshold = decay_threshold

    def analyze(
        self,
        factor_name: str,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        returns_panel: pd.DataFrame = None,
        industry_panel: pd.DataFrame = None,
    ) -> DegradationReport:
        """完整分析一个因子"""
        # 合并数据
        df = feature_panel.merge(
            label_panel[['trade_date', 'symbol', 'fwd_return_20d']],
            on=['trade_date', 'symbol']
        )
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.dropna(subset=[factor_name, 'fwd_return_20d'])

        # === 信号层分析 ===
        signal = self._analyze_signal(df, factor_name)

        # === 组合层分析 ===
        portfolio = self._analyze_portfolio(df, factor_name)

        # === 暴露层分析 ===
        exposure = self._analyze_exposure(df, factor_name, industry_panel)

        # === 退化诊断 ===
        degradation_type, degradation_score, issues = self._diagnose_degradation(
            signal, portfolio, exposure
        )

        # === 建议 ===
        action, target_weight = self._suggest_action(degradation_type, degradation_score)

        return DegradationReport(
            name=factor_name,
            signal=signal,
            portfolio=portfolio,
            exposure=exposure,
            degradation_type=degradation_type,
            degradation_score=degradation_score,
            issues=issues,
            action=action,
            target_weight=target_weight,
        )

    def _analyze_signal(self, df: pd.DataFrame, factor: str) -> SignalMetrics:
        """信号层分析"""
        # 每日IC
        daily_ic = df.groupby('trade_date').apply(
            lambda g: g[factor].corr(g['fwd_return_20d'], method='spearman'),
            include_groups=False
        ).dropna()

        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_win_rate = (daily_ic > 0).mean()

        # 年IC
        yearly_ic = daily_ic.groupby(daily_ic.index.year).mean().to_dict()

        # IC衰减
        years = sorted(yearly_ic.keys())
        recent_decay = 0
        if len(years) >= 2:
            recent_decay = yearly_ic[years[-1]] - yearly_ic[years[-2]]

        # 单调性
        df['q'] = df.groupby('trade_date')[factor].transform(
            lambda x: pd.qcut(x.rank(method='first'), 5, labels=False, duplicates='drop') + 1
        )
        avg_returns = df.groupby(['trade_date', 'q'])['fwd_return_20d'].mean().groupby('q').mean()
        q_values = np.array(list(avg_returns.index))
        r_values = np.array(list(avg_returns.values))
        monotonicity = np.corrcoef(q_values, r_values)[0, 1] if len(q_values) > 1 else 0

        return SignalMetrics(
            ic_mean=ic_mean,
            ic_ir=ic_ir,
            ic_win_rate=ic_win_rate,
            rank_ic_mean=ic_mean,  # 简化，用IC替代
            rank_ic_ir=ic_ir,
            monotonicity=monotonicity,
            yearly_ic=yearly_ic,
            recent_ic_decay=recent_decay,
        )

    def _analyze_portfolio(self, df: pd.DataFrame, factor: str) -> PortfolioMetrics:
        """组合层分析"""
        df['q'] = df.groupby('trade_date')[factor].transform(
            lambda x: pd.qcut(x.rank(method='first'), 5, labels=False, duplicates='drop') + 1
        )

        # 分组收益
        group_returns = df.groupby(['trade_date', 'q'])['fwd_return_20d'].mean()
        avg_returns = group_returns.groupby('q').mean()

        # 多空组合
        long_short = avg_returns.iloc[-1] - avg_returns.iloc[0]

        # 成本后收益 (单边15bp)
        trading_cost = 0.0015
        net_return = long_short - trading_cost

        # 换手率估计
        turnover = 0.1  # 简化估计

        # 胜率
        daily_ls = df.groupby('trade_date').apply(
            lambda g: g[g['q']==5]['fwd_return_20d'].mean() - g[g['q']==1]['fwd_return_20d'].mean(),
            include_groups=False
        ).dropna()
        win_rate = (daily_ls > 0).mean()

        return PortfolioMetrics(
            gross_return=long_short,
            net_return_after_cost=net_return,
            turnover=turnover,
            win_rate=win_rate,
            max_drawdown=0,
            concentration_top10=0.2,
            cost_per_trade=trading_cost,
        )

    def _analyze_exposure(
        self,
        df: pd.DataFrame,
        factor: str,
        industry_panel: pd.DataFrame = None
    ) -> ExposureMetrics:
        """暴露层分析"""
        # 年份集中度
        df['year'] = df['trade_date'].dt.year
        yearly_returns = df.groupby([df['trade_date'], 'q'])['fwd_return_20d'].mean().unstack()
        yearly_ic = {}
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            ic = year_data.groupby('trade_date').apply(
                lambda g: g[factor].corr(g['fwd_return_20d'], method='spearman'),
                include_groups=False
            ).dropna().mean()
            yearly_ic[year] = ic

        # 检测集中年份
        total_ic = sum(yearly_ic.values())
        year_concentration = {y: ic / total_ic if total_ic != 0 else 0 for y, ic in yearly_ic.items()}

        return ExposureMetrics(
            year_concentration=year_concentration,
            stock_concentration=0.0,
            extreme_stock_ratio=0.0,
        )

    def _diagnose_degradation(
        self,
        signal: SignalMetrics,
        portfolio: PortfolioMetrics,
        exposure: ExposureMetrics,
    ) -> tuple[DegradationType, float, list[str]]:
        """退化诊断"""
        issues = []
        degradation_score = 0

        # === 检测交易层退化 ===
        # IC还行但成本后收益为负
        if signal.ic_mean > self.min_ic_mean and portfolio.net_return_after_cost < 0:
            issues.append(f"交易退化: IC={signal.ic_mean:.4f}但成本后收益={portfolio.net_return_after_cost:.4f}")
            degradation_score += 30

        # === 检测表达层退化 ===
        # 整体IC差但部分年份好
        recent_years = [y for y in signal.yearly_ic.keys() if y >= 2024]
        if recent_years:
            recent_ic = np.mean([signal.yearly_ic[y] for y in recent_years])
            if recent_ic < 0 and signal.ic_mean > 0:
                issues.append(f"表达退化: 全期IC正但近期({recent_years})转负")
                degradation_score += 25

        # === 检测经济机制退化 ===
        # IC持续下降
        if len(signal.yearly_ic) >= 3:
            years = sorted(signal.yearly_ic.keys())
            ic_trend = signal.yearly_ic[years[-1]] - signal.yearly_ic[years[0]]
            if ic_trend < -0.02:  # 下降超过2%
                issues.append(f"机制退化: IC从{signal.yearly_ic[years[0]]:.4f}降至{signal.yearly_ic[years[-1]]:.4f}")
                degradation_score += 40

        # === 检测单调性丧失 ===
        if signal.monotonicity < self.min_monotonicity:
            issues.append(f"单调性丧失: {signal.monotonicity:.3f} < {self.min_monotonicity}")
            degradation_score += 20

        # === 综合判断 ===
        if degradation_score >= 60:
            degradation_type = DegradationType.ECONOMIC_DECAY
        elif degradation_score >= 40:
            degradation_type = DegradationType.TRADING_DECAY
        elif degradation_score >= 20:
            degradation_type = DegradationType.EXPRESSION_DECAY
        elif signal.ic_mean > 0 and signal.ic_ir > 0:
            degradation_type = DegradationType.HEALTHY
        else:
            degradation_type = DegradationType.DEAD

        return degradation_type, degradation_score, issues

    def _suggest_action(
        self,
        degradation_type: DegradationType,
        degradation_score: float
    ) -> tuple[str, float]:
        """建议操作"""
        if degradation_type == DegradationType.HEALTHY:
            return "keep", 1.0
        elif degradation_type == DegradationType.EXPRESSION_DECAY:
            return "reformulate", 0.5
        elif degradation_type == DegradationType.TRADING_DECAY:
            return "reduce_weight", 0.3
        elif degradation_type == DegradationType.ECONOMIC_DECAY:
            return "offline", 0.0
        else:
            return "offline", 0.0

    def analyze_factors(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        factor_names: list[str],
        **kwargs
    ) -> list[DegradationReport]:
        """批量分析"""
        reports = []
        for name in factor_names:
            try:
                r = self.analyze(name, feature_panel, label_panel, **kwargs)
                reports.append(r)
            except Exception as e:
                print(f"  因子 {name} 分析失败: {e}")
        return reports

    def print_report(self, reports: list[DegradationReport]):
        """打印报告"""
        print("=" * 100)
        print("因子退化分析报告")
        print("=" * 100)

        # 按退化程度排序
        sorted_reports = sorted(reports, key=lambda x: x.degradation_score, reverse=True)

        # === 汇总 ===
        print("\n【因子状态汇总】")
        status_counts = {}
        for r in sorted_reports:
            status_counts[r.degradation_type.value] = status_counts.get(r.degradation_type.value, 0) + 1

        print(f"  健康: {status_counts.get('healthy', 0)}个")
        print(f"  表达退化: {status_counts.get('expression', 0)}个")
        print(f"  交易退化: {status_counts.get('trading', 0)}个")
        print(f"  机制退化: {status_counts.get('economic', 0)}个")
        print(f"  死亡: {status_counts.get('dead', 0)}个")

        # === 详细分析 ===
        print("\n【因子详情】")
        print(f"{'因子':<25} {'类型':<12} {'评分':>8} {'IC':>10} {'IR':>10} {'单调性':>10} {'建议':>15}")
        print("-" * 100)

        for r in sorted_reports:
            type_emoji = {
                DegradationType.HEALTHY: "✅",
                DegradationType.EXPRESSION_DECAY: "⚠️表达",
                DegradationType.TRADING_DECAY: "⚠️交易",
                DegradationType.ECONOMIC_DECAY: "❌机制",
                DegradationType.DEAD: "💀",
            }.get(r.degradation_type, "?")

            print(f"{type_emoji}{r.name:<23} {r.degradation_type.value:<12} {r.degradation_score:>8.0f} "
                  f"{r.signal.ic_mean:>10.4f} {r.signal.ic_ir:>10.3f} "
                  f"{r.signal.monotonicity:>10.3f} {r.action:>15}")

        # === 问题诊断 ===
        print("\n【退化问题诊断】")
        has_issues = [r for r in sorted_reports if r.issues]
        for r in has_issues[:10]:
            print(f"\n{r.name}:")
            for issue in r.issues:
                print(f"  - {issue}")

        # === 建议操作 ===
        print("\n【操作建议】")
        keep = [r for r in sorted_reports if r.action == "keep"]
        reformulate = [r for r in sorted_reports if r.action == "reformulate"]
        reduce_weight = [r for r in sorted_reports if r.action == "reduce_weight"]
        offline = [r for r in sorted_reports if r.action == "offline"]

        print(f"\n✅ 保持使用 ({len(keep)}个):")
        for r in keep:
            print(f"   {r.name}")

        print(f"\n🔄 需要重构 ({len(reformulate)}个):")
        for r in reformulate:
            print(f"   {r.name}: {r.target_weight:.0%}权重")

        print(f"\n⚠️ 降低权重 ({len(reduce_weight)}个):")
        for r in reduce_weight:
            print(f"   {r.name}: → {r.target_weight:.0%}权重")

        print(f"\n❌ 下线 ({len(offline)}个):")
        for r in offline:
            print(f"   {r.name}")

        print("\n" + "=" * 100)


def run_decay_analysis(
    feature_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    factor_names: list[str],
    **kwargs
) -> list[DegradationReport]:
    """运行退化分析的便捷函数"""
    analyzer = FactorDecayAnalyzer()
    reports = analyzer.analyze_factors(feature_panel, label_panel, factor_names, **kwargs)
    analyzer.print_report(reports)
    return reports
