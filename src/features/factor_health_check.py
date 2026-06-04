"""
单因子健康检查工具 - 因子池清洗协议

检查8项关键指标：
1. 横截面排序能力 (IC/RankIC)
2. 分组单调性 (Q5-Q1多空)
3. IC稳定性结构 (年度IC)
4. 年份集中度 (只在某几年有效)
5. 股票集中度 (只靠极少数股票贡献)
6. 成本敏感性 (成本后还剩多少)
7. 换手率 (是否太高)
8. 传统暴露检查 (市值/行业/Beta) - 可选

目标：淘汰掉
- 纯噪声因子
- 数据时点不干净的因子
- 高换手伪alpha
- 高度重复因子
- 只靠极端样本活着的因子
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class HealthStatus(Enum):
    """健康状态"""
    PASS = "pass"           # 通过
    CONDITIONAL = "conditional"  # 有条件通过
    FAIL = "fail"           # 失败
    UNKNOWN = "unknown"      # 无法判断


class CheckResult(Enum):
    """检查项结果"""
    GOOD = "good"
    WARNING = "warning"
    BAD = "bad"
    NA = "na"


@dataclass
class FactorHealthReport:
    """单因子健康报告"""
    name: str

    # 1. 横截面排序能力
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    ic_win_rate: float = 0.0
    rank_ic_mean: float = 0.0
    rank_ic_ir: float = 0.0
    ranking_ability: CheckResult = CheckResult.NA

    # 2. 单调性
    quantile_returns: dict[int, float] = field(default_factory=dict)
    long_short_return: float = 0.0
    monotonicity_score: float = 0.0
    monotonicity: CheckResult = CheckResult.NA

    # 3. IC稳定性
    yearly_ic: dict[int, float] = field(default_factory=dict)
    ic_stability: CheckResult = CheckResult.NA
    years_positive: int = 0
    years_negative: int = 0
    most_negative_year: tuple = None

    # 4. 年份集中度
    year_concentration: float = 0.0  # 0=均匀, 1=集中在单一年份
    year_concentration_check: CheckResult = CheckResult.NA
    dominant_year: int = 0

    # 5. 股票集中度
    top_stock_contribution: float = 0.0  # 前10股票贡献占比
    stock_concentration: CheckResult = CheckResult.NA

    # 6. 成本敏感性
    net_return_0bp: float = 0.0
    net_return_15bp: float = 0.0
    net_return_30bp: float = 0.0
    net_return_50bp: float = 0.0
    cost_sensitivity: CheckResult = CheckResult.NA

    # 7. 换手率
    avg_turnover: float = 0.0
    turnover_check: CheckResult = CheckResult.NA

    # 综合评估
    health_status: HealthStatus = HealthStatus.UNKNOWN
    verdict: str = ""
    reasons_to_reject: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class SingleFactorHealthChecker:
    """单因子健康检查器"""

    def __init__(
        self,
        trading_cost_bps: float = 15.0,
        min_positive_years: int = 3,
        max_concentration: float = 0.5,
        max_turnover: float = 1.5,
    ):
        self.trading_cost_bps = trading_cost_bps
        self.min_positive_years = min_positive_years
        self.max_concentration = max_concentration
        self.max_turnover = max_turnover

    def check(
        self,
        factor_data: pd.DataFrame,
        factor_name: str,
        label_col: str = 'fwd_return_20d',
        n_quantiles: int = 5,
        min_samples: int = 100,
    ) -> FactorHealthReport:
        """执行完整健康检查"""
        report = FactorHealthReport(name=factor_name)

        # 准备数据
        df = factor_data[['trade_date', 'symbol', factor_name, label_col]].copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.dropna(subset=[factor_name, label_col])

        if len(df) < min_samples:
            report.verdict = "数据不足"
            report.health_status = HealthStatus.UNKNOWN
            return report

        # ==== 1. 横截面排序能力 ====
        self._check_ranking_ability(df, factor_name, label_col, report)

        # ==== 2. 单调性 ====
        self._check_monotonicity(df, factor_name, label_col, n_quantiles, report)

        # ==== 3. IC稳定性 ====
        self._check_ic_stability(df, factor_name, label_col, report)

        # ==== 4. 年份集中度 ====
        self._check_year_concentration(df, factor_name, label_col, report)

        # ==== 5. 股票集中度 ====
        self._check_stock_concentration(df, factor_name, label_col, report)

        # ==== 6. 成本敏感性 ====
        self._check_cost_sensitivity(report)

        # ==== 7. 换手率 ====
        self._check_turnover(df, factor_name, report)

        # ==== 综合评估 ====
        self._make_verdict(report)

        return report

    def _check_ranking_ability(
        self,
        df: pd.DataFrame,
        factor_name: str,
        label_col: str,
        report: FactorHealthReport,
    ) -> None:
        """检查横截面排序能力"""
        # Pearson IC
        daily_ic = df.groupby('trade_date').apply(
            lambda g: g[factor_name].corr(g[label_col], method='pearson'),
            include_groups=False
        ).dropna()

        # Spearman Rank IC
        daily_rank_ic = df.groupby('trade_date').apply(
            lambda g: g[factor_name].corr(g[label_col], method='spearman'),
            include_groups=False
        ).dropna()

        report.ic_mean = daily_ic.mean()
        report.ic_std = daily_ic.std() if len(daily_ic) > 1 else 0
        report.ic_ir = report.ic_mean / report.ic_std if report.ic_std > 0 else 0
        report.ic_win_rate = (daily_ic > 0).mean()

        report.rank_ic_mean = daily_rank_ic.mean()
        report.rank_ic_ir = daily_rank_ic.mean() / daily_rank_ic.std() if daily_rank_ic.std() > 0 else 0

        # 判断
        if report.ic_mean > 0.005 and report.ic_ir > 0.05:
            report.ranking_ability = CheckResult.GOOD
        elif report.ic_mean > 0:
            report.ranking_ability = CheckResult.WARNING
        else:
            report.ranking_ability = CheckResult.BAD

    def _check_monotonicity(
        self,
        df: pd.DataFrame,
        factor_name: str,
        label_col: str,
        n_quantiles: int,
        report: FactorHealthReport,
    ) -> None:
        """检查分组单调性"""
        # 分组计算
        df['quantile'] = df.groupby('trade_date')[factor_name].transform(
            lambda x: pd.qcut(x.rank(method='first'), n_quantiles, labels=False, duplicates='drop') + 1
        )

        group_rets = df.groupby(['trade_date', 'quantile'])[label_col].mean()
        avg_returns = group_rets.groupby('quantile').mean()

        report.quantile_returns = {int(k): v for k, v in avg_returns.to_dict().items()}
        report.long_short_return = avg_returns.iloc[-1] - avg_returns.iloc[0]

        # 单调性得分 (Spearman相关性)
        q_values = list(avg_returns.index)
        r_values = list(avg_returns.values)
        if len(q_values) > 1:
            report.monotonicity_score = np.corrcoef(q_values, r_values)[0, 1]
        else:
            report.monotonicity_score = 0

        # 判断
        if report.monotonicity_score > 0.8 and report.long_short_return > 0.002:
            report.monotonicity = CheckResult.GOOD
        elif report.monotonicity_score > 0.5:
            report.monotonicity = CheckResult.WARNING
        else:
            report.monotonicity = CheckResult.BAD

    def _check_ic_stability(
        self,
        df: pd.DataFrame,
        factor_name: str,
        label_col: str,
        report: FactorHealthReport,
    ) -> None:
        """检查IC稳定性"""
        daily_ic = df.groupby('trade_date').apply(
            lambda g: g[factor_name].corr(g[label_col], method='spearman'),
            include_groups=False
        ).dropna()

        # 年度IC
        daily_ic.index = pd.to_datetime(daily_ic.index)
        report.yearly_ic = daily_ic.groupby(daily_ic.index.year).mean().to_dict()

        report.years_positive = sum(1 for v in report.yearly_ic.values() if v > 0)
        report.years_negative = sum(1 for v in report.yearly_ic.values() if v < 0)

        # 最差年份
        if report.yearly_ic:
            report.most_negative_year = min(report.yearly_ic.items(), key=lambda x: x[1])

        # 稳定性判断
        total_years = len(report.yearly_ic)
        if total_years >= 3:
            positive_ratio = report.years_positive / total_years
            if positive_ratio >= 0.7:
                report.ic_stability = CheckResult.GOOD
            elif positive_ratio >= 0.5:
                report.ic_stability = CheckResult.WARNING
            else:
                report.ic_stability = CheckResult.BAD
        else:
            report.ic_stability = CheckResult.NA

    def _check_year_concentration(
        self,
        df: pd.DataFrame,
        factor_name: str,
        label_col: str,
        report: FactorHealthReport,
    ) -> None:
        """检查年份集中度"""
        if not report.yearly_ic:
            return

        ic_values = np.array(list(report.yearly_ic.values()))
        total = ic_values.sum()

        if total <= 0:
            report.year_concentration = 1.0
            report.year_concentration_check = CheckResult.BAD
            report.reasons_to_reject.append("IC全为负")
            return

        # 计算集中度 (Herfindahl指数)
        weights = ic_values / total
        weights = weights[weights > 0]
        report.year_concentration = (weights ** 2).sum()

        # 找出主导年份
        if report.yearly_ic:
            report.dominant_year = max(report.yearly_ic.items(), key=lambda x: x[1])[0]

        # 判断: 如果前1-2年贡献>70%的IC，说明集中在少数年份
        if report.year_concentration > 0.5:
            report.year_concentration_check = CheckResult.WARNING
            report.warnings.append(f"IC集中在{report.dominant_year}年")
        else:
            report.year_concentration_check = CheckResult.GOOD

    def _check_stock_concentration(
        self,
        df: pd.DataFrame,
        factor_name: str,
        label_col: str,
        report: FactorHealthReport,
    ) -> None:
        """检查股票集中度"""
        # 计算每个日期的IC贡献分解
        daily_ic_contrib = []

        for date, group in df.groupby('trade_date'):
            if len(group) < 10:
                continue

            # 因子值排序
            sorted_group = group.sort_values(factor_name)

            # 取top和bottom各10%
            n = max(1, int(len(group) * 0.1))
            top_stocks = sorted_group.tail(n)
            bottom_stocks = sorted_group.head(n)

            # 计算两组的平均收益
            top_ret = top_stocks[label_col].mean()
            bottom_ret = bottom_stocks[label_col].mean()
            spread = top_ret - bottom_ret

            # 记录每只股票的贡献
            for _, row in top_stocks.iterrows():
                daily_ic_contrib.append({
                    'date': date,
                    'symbol': row['symbol'],
                    'contribution': (row[label_col] - top_ret) + spread / 2
                })
            for _, row in bottom_stocks.iterrows():
                daily_ic_contrib.append({
                    'date': date,
                    'symbol': row['symbol'],
                    'contribution': (row[label_col] - bottom_ret) - spread / 2
                })

        if not daily_ic_contrib:
            return

        contrib_df = pd.DataFrame(daily_ic_contrib)

        # 计算每只股票的总贡献
        stock_total = contrib_df.groupby('symbol')['contribution'].sum()
        total_contribution = stock_total.abs().sum()

        if total_contribution > 0:
            # 前10股票贡献占比
            top10_contrib = stock_total.abs().nlargest(10).sum()
            report.top_stock_contribution = top10_contrib / total_contribution

        # 判断
        if report.top_stock_contribution > 0.5:
            report.stock_concentration = CheckResult.WARNING
            report.warnings.append(f"前10股票贡献了{report.top_stock_contribution:.1%}的IC")
        else:
            report.stock_concentration = CheckResult.GOOD

    def _check_cost_sensitivity(self, report: FactorHealthReport) -> None:
        """检查成本敏感性"""
        report.net_return_0bp = report.long_short_return
        report.net_return_15bp = report.long_short_return - 0.0015  # 15bp单边
        report.net_return_30bp = report.long_short_return - 0.003   # 30bp单边
        report.net_return_50bp = report.long_short_return - 0.005   # 50bp单边

        # 判断
        if report.net_return_30bp > 0.002:
            report.cost_sensitivity = CheckResult.GOOD
        elif report.net_return_30bp > 0:
            report.cost_sensitivity = CheckResult.WARNING
        else:
            report.cost_sensitivity = CheckResult.BAD

    def _check_turnover(
        self,
        df: pd.DataFrame,
        factor_name: str,
        report: FactorHealthReport,
    ) -> None:
        """检查换手率"""
        # 计算每个日期的因子方向变化
        df = df.sort_values(['symbol', 'trade_date'])

        # 计算因子排名变化
        df['factor_rank'] = df.groupby('trade_date')[factor_name].rank(pct=True)

        # 计算相邻日期的排名变化
        df['prev_rank'] = df.groupby('symbol')['factor_rank'].shift(1)
        df['rank_change'] = (df['factor_rank'] - df['prev_rank']).abs()

        # 平均换手率 (用排名变化近似)
        avg_rank_change = df['rank_change'].mean()

        # 换手率估算 (假设5分位，rank变化0.2对应完整换手一次)
        report.avg_turnover = avg_rank_change * 5

        # 判断
        if report.avg_turnover < 0.3:
            report.turnover_check = CheckResult.GOOD
        elif report.avg_turnover < 0.5:
            report.turnover_check = CheckResult.WARNING
        else:
            report.turnover_check = CheckResult.BAD

    def _make_verdict(self, report: FactorHealthReport) -> None:
        """综合判断"""
        # 淘汰条件
        reject_reasons = []

        # 1. IC为负
        if report.ic_mean < 0:
            reject_reasons.append(f"IC均值负({report.ic_mean:.4f})")

        # 2. IC不稳定
        if report.ic_stability == CheckResult.BAD:
            reject_reasons.append(f"IC不稳定(正年份{report.years_positive}/{len(report.yearly_ic)})")

        # 3. 单调性差
        if report.monotonicity == CheckResult.BAD:
            reject_reasons.append(f"单调性差({report.monotonicity_score:.2f})")

        # 4. 成本后无效
        if report.net_return_30bp < 0:
            reject_reasons.append(f"成本30bp后无效({report.net_return_30bp:.4f})")

        # 5. 换手率过高
        if report.turnover_check == CheckResult.BAD:
            reject_reasons.append(f"换手率过高({report.avg_turnover:.2%})")

        report.reasons_to_reject = reject_reasons

        # 判断健康状态
        if reject_reasons:
            report.health_status = HealthStatus.FAIL
            report.verdict = "REJECT"
        elif any([
            report.ranking_ability == CheckResult.WARNING,
            report.monotonicity == CheckResult.WARNING,
            report.ic_stability == CheckResult.WARNING,
            report.stock_concentration == CheckResult.WARNING,
        ]):
            report.health_status = HealthStatus.CONDITIONAL
            report.verdict = "CONDITIONAL"
        else:
            report.health_status = HealthStatus.PASS
            report.verdict = "PASS"

        # 添加警告和建议
        if report.ic_mean > 0 and report.ic_mean < 0.005:
            report.warnings.append("IC较弱，需要更多验证")

        if report.ic_win_rate < 0.55:
            report.warnings.append(f"IC胜率低({report.ic_win_rate:.1%})")

        # 建议
        if report.health_status == HealthStatus.PASS:
            report.recommendations.append("可进入研究池")
        elif report.health_status == HealthStatus.CONDITIONAL:
            report.recommendations.append("可进入观察池，需持续监控")
        else:
            report.recommendations.append("建议淘汰")


def batch_check(
    data: pd.DataFrame,
    factor_names: list[str],
    label_col: str = 'fwd_return_20d',
    **kwargs
) -> list[FactorHealthReport]:
    """批量检查多个因子"""
    checker = SingleFactorHealthChecker(**kwargs)
    reports = []

    for name in factor_names:
        if name in data.columns:
            try:
                report = checker.check(data, name, label_col)
                reports.append(report)
            except Exception as e:
                print(f"因子 {name} 检查失败: {e}")

    return reports


def print_health_report(reports: list[FactorHealthReport]) -> None:
    """打印健康检查报告"""
    print("=" * 120)
    print("单因子健康检查报告 - 因子池清洗协议")
    print("=" * 120)

    # 按状态分组
    pass_reports = [r for r in reports if r.health_status == HealthStatus.PASS]
    conditional_reports = [r for r in reports if r.health_status == HealthStatus.CONDITIONAL]
    fail_reports = [r for r in reports if r.health_status == HealthStatus.FAIL]

    # 汇总
    print("\n【汇总】")
    print(f"  通过: {len(pass_reports)}个")
    print(f"  条件通过: {len(conditional_reports)}个")
    print(f"  失败: {len(fail_reports)}个")
    print(f"  总计: {len(reports)}个")

    # 通过的因子
    if pass_reports:
        print(f"\n{'='*120}")
        print(f"✅ PASS ({len(pass_reports)}个)")
        print("-" * 120)
        print(f"{'因子':<20} {'IC':>10} {'IR':>8} {'单调性':>10} {'多空%':>10} {'成本30bp':>12} {'换手率':>10}")
        print("-" * 120)
        for r in sorted(pass_reports, key=lambda x: x.ic_mean, reverse=True):
            print(f"{r.name:<20} {r.ic_mean:>10.4f} {r.ic_ir:>8.3f} {r.monotonicity_score:>10.2f} "
                  f"{r.long_short_return*100:>9.2f}% {r.net_return_30bp*100:>11.2f}% {r.avg_turnover:>10.2%}")

    # 条件通过的因子
    if conditional_reports:
        print(f"\n{'='*120}")
        print(f"⚠️ CONDITIONAL ({len(conditional_reports)}个) - 需监控")
        print("-" * 120)
        for r in sorted(conditional_reports, key=lambda x: x.ic_mean, reverse=True):
            print(f"\n  {r.name}:")
            print(f"    IC={r.ic_mean:.4f}, 单调性={r.monotonicity_score:.2f}")
            for w in r.warnings:
                print(f"    ⚠️ {w}")

    # 失败的因子
    if fail_reports:
        print(f"\n{'='*120}")
        print(f"❌ FAIL ({len(fail_reports)}个) - 建议淘汰")
        print("-" * 120)
        for r in sorted(fail_reports, key=lambda x: x.ic_mean):
            reasons = "; ".join(r.reasons_to_reject) if r.reasons_to_reject else "未知"
            print(f"  {r.name:<20}: {reasons}")

    print("\n" + "=" * 120)


def print_detailed_report(report: FactorHealthReport) -> None:
    """打印单因子详细报告"""
    print("=" * 100)
    print(f"因子健康报告: {report.name}")
    print("=" * 100)

    print("\n【综合评估】")
    print(f"  状态: {report.verdict}")
    print(f"  IC均值: {report.ic_mean:.4f}")
    print(f"  IC IR: {report.ic_ir:.3f}")
    print(f"  RankIC: {report.rank_ic_mean:.4f}")

    print("\n【1. 横截面排序能力】")
    print(f"  IC均值: {report.ic_mean:.4f}")
    print(f"  IC标准差: {report.ic_std:.4f}")
    print(f"  IC IR: {report.ic_ir:.3f}")
    print(f"  IC胜率: {report.ic_win_rate:.1%}")
    print(f"  结果: {report.ranking_ability.value}")

    print("\n【2. 单调性检验】")
    print("  分组收益:")
    for q, ret in sorted(report.quantile_returns.items()):
        print(f"    Q{q}: {ret*100:.2f}%")
    print(f"  多空收益: {report.long_short_return*100:.2f}%")
    print(f"  单调性得分: {report.monotonicity_score:.3f}")
    print(f"  结果: {report.monotonicity.value}")

    print("\n【3. IC稳定性】")
    print("  年度IC:")
    for year, ic in sorted(report.yearly_ic.items()):
        flag = "🟢" if ic > 0 else "🔴"
        print(f"    {year}: {flag}{ic:.4f}")
    print(f"  正年份: {report.years_positive}, 负年份: {report.years_negative}")
    print(f"  结果: {report.ic_stability.value}")

    print("\n【4. 年份集中度】")
    print(f"  集中度: {report.year_concentration:.2f} (0=均匀, 1=集中)")
    print(f"  主导年份: {report.dominant_year}")
    print(f"  结果: {report.year_concentration_check.value}")

    print("\n【5. 股票集中度】")
    print(f"  前10股票贡献: {report.top_stock_contribution:.1%}")
    print(f"  结果: {report.stock_concentration.value}")

    print("\n【6. 成本敏感性】")
    print(f"  0bp: {report.net_return_0bp*100:.2f}%")
    print(f"  15bp: {report.net_return_15bp*100:.2f}%")
    print(f"  30bp: {report.net_return_30bp*100:.2f}%")
    print(f"  50bp: {report.net_return_50bp*100:.2f}%")
    print(f"  结果: {report.cost_sensitivity.value}")

    print("\n【7. 换手率】")
    print(f"  平均换手率: {report.avg_turnover:.2%}")
    print(f"  结果: {report.turnover_check.value}")

    if report.reasons_to_reject:
        print("\n【淘汰原因】")
        for r in report.reasons_to_reject:
            print(f"  ❌ {r}")

    if report.warnings:
        print("\n【警告】")
        for w in report.warnings:
            print(f"  ⚠️ {w}")

    if report.recommendations:
        print("\n【建议】")
        for r in report.recommendations:
            print(f"  → {r}")

    print("\n" + "=" * 100)
