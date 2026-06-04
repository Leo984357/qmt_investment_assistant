"""
策略准入门控 (Strategy Gate)

策略必须通过所有门控才能标记为 recommended。

门控标准 (必须全部通过):
1. IC均值 > 0.02
2. IC IR > 0.15
3. 分组单调性 > 0.6 (Q5 > Q4 > Q3 > Q2 > Q1)
4. 成本后超额收益 > 0
5. 最大回撤 < 30%
6. 平均换手率 < 50%
7. 分年度稳定性: 至少60%年份正收益
8. 相对基线增量: Sharpe > 基线Sharpe

使用方式:
    from src.evaluation.strategy_gate import StrategyGate, GateResult
    
    gate = StrategyGate()
    result = gate.evaluate(
        nav=strategy_nav,
        rank_ic=rank_ic_df,
        quantile_summary=quantile_summary_df,
        trades=trades_df,
        benchmark_nav=benchmark_nav,
        yearly_breakdown=yearly_df,
    )
    
    if result.passed:
        print(f"Strategy {name} PASSED gate")
    else:
        print(f"Strategy {name} FAILED: {result.failed_gates}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from src.core.logging_utils import get_logger

logger = get_logger(__name__)


class GateStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class GateCheck:
    """单个门控检查结果"""
    name: str
    status: GateStatus
    value: float
    threshold: float
    message: str


@dataclass
class GateResult:
    """策略门控结果"""
    strategy_name: str
    passed: bool
    overall_score: float  # 0-100
    gate_checks: list[GateCheck] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed_gates(self) -> list[str]:
        return [g.name for g in self.gate_checks if g.status == GateStatus.PASSED]

    @property
    def failed_gates(self) -> list[str]:
        return [g.name for g in self.gate_checks if g.status == GateStatus.FAILED]

    def to_dict(self) -> dict:
        return {
            'strategy_name': self.strategy_name,
            'passed': self.passed,
            'overall_score': self.overall_score,
            'gate_checks': [
                {
                    'name': g.name,
                    'status': g.status.value,
                    'value': g.value,
                    'threshold': g.threshold,
                    'message': g.message,
                }
                for g in self.gate_checks
            ],
            'recommendations': self.recommendations,
            'warnings': self.warnings,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Strategy Gate: {self.strategy_name}",
            "",
            f"**Overall Status:** {'✅ PASSED' if self.passed else '❌ FAILED'}",
            f"**Score:** {self.overall_score:.1f}/100",
            "",
            "## Gate Checks",
            "",
        ]

        for g in self.gate_checks:
            icon = {
                GateStatus.PASSED: '✅',
                GateStatus.FAILED: '❌',
                GateStatus.WARNING: '⚠️',
                GateStatus.SKIPPED: '⏭️',
            }[g.status]
            lines.append(f"{icon} **{g.name}**: {g.value:.4f} (threshold: {g.threshold:.4f})")
            lines.append(f"   {g.message}")
            lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        if self.recommendations:
            lines.append("## Recommendations")
            for r in self.recommendations:
                lines.append(f"- {r}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class GateThresholds:
    """门控阈值配置"""
    ic_mean_min: float = 0.02
    ic_ir_min: float = 0.15
    monotonicity_min: float = 0.6
    excess_return_min: float = 0.0
    max_drawdown_max: float = 0.30
    avg_turnover_max: float = 0.50
    yearly_win_rate_min: float = 0.60
    sharpe_vs_baseline_min: float = 0.0


class StrategyGate:
    """
    策略准入门控
    
    评估策略是否满足推荐标准的正式门控。
    所有门控必须通过才能标记为 recommended。
    """

    def __init__(self, thresholds: GateThresholds | None = None):
        self.thresholds = thresholds or GateThresholds()

    def evaluate(
        self,
        strategy_name: str,
        nav: pd.DataFrame,
        rank_ic: pd.DataFrame | None = None,
        quantile_summary: pd.DataFrame | None = None,
        trades: pd.DataFrame | None = None,
        benchmark_nav: pd.DataFrame | None = None,
        yearly_breakdown: pd.DataFrame | None = None,
        baseline_sharpe: float = 0.0,
    ) -> GateResult:
        """
        评估策略是否通过所有门控。
        
        Args:
            strategy_name: 策略名称
            nav: 策略净值序列
            rank_ic: IC时间序列
            quantile_summary: 分组收益摘要
            trades: 交易记录
            benchmark_nav: 基准净值
            yearly_breakdown: 年度分解
            baseline_sharpe: 基线策略Sharpe (用于比较)
        """
        checks = []
        warnings = []
        recommendations = []

        # Gate 0: 零交易硬失败
        trade_check = self._check_trade_count(trades)
        checks.append(trade_check)

        # Gate 1: IC均值
        ic_mean, ic_check = self._check_ic_mean(rank_ic)
        checks.append(ic_check)

        # Gate 2: IC IR
        ic_ir, ir_check = self._check_ic_ir(rank_ic)
        checks.append(ir_check)

        # Gate 3: 分组单调性
        monotonicity, mono_check = self._check_monotonicity(quantile_summary)
        checks.append(mono_check)

        # Gate 4: 成本后超额收益
        excess_return, excess_check = self._check_excess_return(nav, benchmark_nav)
        checks.append(excess_check)

        # Gate 5: 最大回撤
        max_dd, dd_check = self._check_max_drawdown(nav)
        checks.append(dd_check)

        # Gate 6: 平均换手率
        avg_turnover, turnover_check = self._check_avg_turnover(trades, nav)
        checks.append(turnover_check)

        # Gate 7: 分年度稳定性
        yearly_win_rate, yearly_check = self._check_yearly_stability(yearly_breakdown)
        checks.append(yearly_check)

        # Gate 8: 相对基线增量
        sharpe_increment, strategy_sharpe, sharpe_check = self._check_sharpe_vs_baseline(nav, baseline_sharpe)
        checks.append(sharpe_check)

        # 计算总分
        passed_count = sum(1 for c in checks if c.status == GateStatus.PASSED)
        total_count = len(checks)
        overall_score = (passed_count / total_count) * 100 if total_count > 0 else 0

        # 判断是否通过: 9个门控必须全部PASSED，不允许FAILED或SKIPPED
        # Gate 0 (零交易检测) 是硬失败，任何失败都算失败
        failed_count = sum(1 for c in checks if c.status == GateStatus.FAILED)
        skipped_count = sum(1 for c in checks if c.status == GateStatus.SKIPPED)
        passed = failed_count == 0 and skipped_count == 0

        # 生成建议
        if ic_mean < self.thresholds.ic_mean_min * 0.5:
            recommendations.append("IC均值过低，考虑更换或增加有效因子")
        if ic_ir < self.thresholds.ic_ir_min * 0.5:
            recommendations.append("IC IR过低，策略信号不稳定")
        if monotonicity < self.thresholds.monotonicity_min:
            recommendations.append("分组单调性差，检查因子方向和分组逻辑")
        if max_dd > self.thresholds.max_drawdown_max * 0.8:
            recommendations.append("最大回撤较高，建议增加风控约束")
        if avg_turnover > self.thresholds.avg_turnover_max * 0.8:
            recommendations.append("换手率偏高，考虑增加持仓缓冲区或平滑")
        if yearly_win_rate < self.thresholds.yearly_win_rate_min:
            recommendations.append("年度胜率偏低，策略可能存在周期性")

        # 生成警告
        if ic_mean < self.thresholds.ic_mean_min:
            warnings.append(f"IC均值 {ic_mean:.4f} 接近阈值下限")
        if strategy_sharpe < baseline_sharpe:
            warnings.append(f"Sharpe {strategy_sharpe:.3f} 低于基线 {baseline_sharpe:.3f}")

        return GateResult(
            strategy_name=strategy_name,
            passed=passed,
            overall_score=overall_score,
            gate_checks=checks,
            recommendations=recommendations,
            warnings=warnings,
        )

    def _check_trade_count(self, trades: pd.DataFrame | None) -> GateCheck:
        """Gate 0: 零交易硬失败"""
        trade_count = len(trades) if trades is not None else 0
        passed = trade_count > 0

        return GateCheck(
            name="零交易检测",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=float(trade_count),
            threshold=1.0,
            message=f"交易数 {'≥ 1' if passed else '= 0 (硬失败)'}",
        )

    def _check_ic_mean(self, rank_ic: pd.DataFrame | None) -> tuple[float, GateCheck]:
        """Gate 1: IC均值 > 阈值"""
        if rank_ic is None or rank_ic.empty:
            return 0.0, GateCheck(
                name="IC均值",
                status=GateStatus.SKIPPED,
                value=0.0,
                threshold=self.thresholds.ic_mean_min,
                message="无IC数据，跳过检查",
            )

        ic_mean = rank_ic['rank_ic'].mean()
        passed = ic_mean >= self.thresholds.ic_mean_min

        return ic_mean, GateCheck(
            name="IC均值",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=ic_mean,
            threshold=self.thresholds.ic_mean_min,
            message=f"IC均值 {'≥' if passed else '<'} {self.thresholds.ic_mean_min}",
        )

    def _check_ic_ir(self, rank_ic: pd.DataFrame | None) -> tuple[float, GateCheck]:
        """Gate 2: IC IR > 阈值"""
        if rank_ic is None or rank_ic.empty or len(rank_ic) < 10:
            return 0.0, GateCheck(
                name="IC IR",
                status=GateStatus.SKIPPED,
                value=0.0,
                threshold=self.thresholds.ic_ir_min,
                message="IC数据不足，跳过检查",
            )

        ic_mean = rank_ic['rank_ic'].mean()
        ic_std = rank_ic['rank_ic'].std()
        ic_ir = ic_mean / ic_std if ic_std > 1e-8 else 0.0
        passed = ic_ir >= self.thresholds.ic_ir_min

        return ic_ir, GateCheck(
            name="IC IR",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=ic_ir,
            threshold=self.thresholds.ic_ir_min,
            message=f"IC IR {'≥' if passed else '<'} {self.thresholds.ic_ir_min}",
        )

    def _check_monotonicity(self, quantile_summary: pd.DataFrame | None) -> tuple[float, GateCheck]:
        """Gate 3: 分组单调性 > 阈值"""
        if quantile_summary is None or quantile_summary.empty:
            return 0.0, GateCheck(
                name="分组单调性",
                status=GateStatus.SKIPPED,
                value=0.0,
                threshold=self.thresholds.monotonicity_min,
                message="无分组收益数据，跳过检查",
            )

        # 提取各组收益
        quantile_returns = {}
        for _, row in quantile_summary.iterrows():
            bucket = str(row.get('bucket', ''))
            if bucket.isdigit():
                quantile_returns[int(bucket)] = row.get('mean_forward_return', 0)

        if len(quantile_returns) < 3:
            return 0.0, GateCheck(
                name="分组单调性",
                status=GateStatus.SKIPPED,
                value=0.0,
                threshold=self.thresholds.monotonicity_min,
                message="分组数据不足，跳过检查",
            )

        # 计算单调性: 正确的升序对数 / 总对数
        sorted_quantiles = sorted(quantile_returns.keys())
        correct_pairs = 0
        total_pairs = 0
        for i in range(len(sorted_quantiles) - 1):
            q1, q2 = sorted_quantiles[i], sorted_quantiles[i + 1]
            if quantile_returns[q1] <= quantile_returns[q2]:
                correct_pairs += 1
            total_pairs += 1

        monotonicity = correct_pairs / total_pairs if total_pairs > 0 else 0.0
        passed = monotonicity >= self.thresholds.monotonicity_min

        return monotonicity, GateCheck(
            name="分组单调性",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=monotonicity,
            threshold=self.thresholds.monotonicity_min,
            message=f"单调性 {'≥' if passed else '<'} {self.thresholds.monotonicity_min}",
        )

    def _check_excess_return(self, nav: pd.DataFrame, benchmark_nav: pd.DataFrame | None) -> tuple[float, GateCheck]:
        """Gate 4: 成本后超额收益 > 阈值"""
        if nav is None or nav.empty:
            return 0.0, GateCheck(
                name="成本后超额收益",
                status=GateStatus.SKIPPED,
                value=0.0,
                threshold=self.thresholds.excess_return_min,
                message="无净值数据，跳过检查",
            )

        strategy_return = nav['nav'].iloc[-1] / nav['nav'].iloc[0] - 1

        if benchmark_nav is not None and not benchmark_nav.empty:
            # 处理不同的列名: nav 或 benchmark_nav
            if 'nav' in benchmark_nav.columns:
                benchmark_col = 'nav'
            elif 'benchmark_nav' in benchmark_nav.columns:
                benchmark_col = 'benchmark_nav'
            else:
                return strategy_return, GateCheck(
                    name="成本后超额收益",
                    status=GateStatus.SKIPPED,
                    value=0.0,
                    threshold=self.thresholds.excess_return_min,
                    message="基准净值无有效列，跳过检查",
                )

            benchmark_return = benchmark_nav[benchmark_col].iloc[-1] / benchmark_nav[benchmark_col].iloc[0] - 1
            excess_return = strategy_return - benchmark_return
        else:
            excess_return = strategy_return

        passed = excess_return >= self.thresholds.excess_return_min

        return excess_return, GateCheck(
            name="成本后超额收益",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=excess_return,
            threshold=self.thresholds.excess_return_min,
            message=f"超额收益 {'≥' if passed else '<'} {self.thresholds.excess_return_min:.2%}",
        )

    def _check_max_drawdown(self, nav: pd.DataFrame) -> tuple[float, GateCheck]:
        """Gate 5: 最大回撤 < 阈值"""
        if nav is None or nav.empty:
            return 0.0, GateCheck(
                name="最大回撤",
                status=GateStatus.SKIPPED,
                value=0.0,
                threshold=self.thresholds.max_drawdown_max,
                message="无净值数据，跳过检查",
            )

        nav_series = nav.set_index('trade_date')['nav']
        cumulative_max = nav_series.cummax()
        drawdown = (nav_series - cumulative_max) / cumulative_max
        max_dd = drawdown.min()

        passed = max_dd >= -self.thresholds.max_drawdown_max

        return abs(max_dd), GateCheck(
            name="最大回撤",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=abs(max_dd),
            threshold=self.thresholds.max_drawdown_max,
            message=f"最大回撤 {'≤' if passed else '>'} {self.thresholds.max_drawdown_max:.2%}",
        )

    def _check_avg_turnover(self, trades: pd.DataFrame | None, nav: pd.DataFrame | None = None) -> tuple[float, GateCheck]:
        """Gate 6: 平均换手率 < 阈值"""
        if trades is None or trades.empty:
            return 0.0, GateCheck(
                name="平均换手率",
                status=GateStatus.SKIPPED,
                value=0.0,
                threshold=self.thresholds.avg_turnover_max,
                message="无交易数据，跳过检查",
            )

        # 估算换手率 (使用backtest实际字段: turnover 是百分比, notional需归一化)
        if 'turnover' in trades.columns and 'trade_date' in trades.columns:
            avg_turnover = trades.groupby('trade_date')['turnover'].mean().mean()
        elif 'notional' in trades.columns and 'trade_date' in trades.columns:
            if nav is not None and not nav.empty and 'equity' in nav.columns:
                equity = nav['equity'].iloc[-1] if len(nav) > 0 else 1e6
            elif nav is not None and not nav.empty and 'nav' in nav.columns:
                equity = nav['nav'].iloc[-1] if len(nav) > 0 else 1.0
            else:
                equity = 1e6
            avg_notional = trades.groupby('trade_date')['notional'].sum().mean()
            avg_turnover = avg_notional / max(equity, 1e-9)
        else:
            avg_turnover = 0.0

        passed = avg_turnover <= self.thresholds.avg_turnover_max

        return avg_turnover, GateCheck(
            name="平均换手率",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=avg_turnover,
            threshold=self.thresholds.avg_turnover_max,
            message=f"平均换手 {'≤' if passed else '>'} {self.thresholds.avg_turnover_max:.2%}",
        )

    def _check_yearly_stability(self, yearly_breakdown: pd.DataFrame | None) -> tuple[float, GateCheck]:
        """Gate 7: 分年度稳定性"""
        if yearly_breakdown is None or yearly_breakdown.empty or len(yearly_breakdown) < 2:
            return 0.0, GateCheck(
                name="年度稳定性",
                status=GateStatus.SKIPPED,
                value=0.0,
                threshold=self.thresholds.yearly_win_rate_min,
                message="年度数据不足，跳过检查",
            )

        positive_years = (yearly_breakdown['total_return'] > 0).sum()
        total_years = len(yearly_breakdown)
        win_rate = positive_years / total_years if total_years > 0 else 0.0

        passed = win_rate >= self.thresholds.yearly_win_rate_min

        return win_rate, GateCheck(
            name="年度稳定性",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=win_rate,
            threshold=self.thresholds.yearly_win_rate_min,
            message=f"年度胜率 {'≥' if passed else '<'} {self.thresholds.yearly_win_rate_min:.0%} ({positive_years}/{total_years}年正收益)",
        )

    def _check_sharpe_vs_baseline(self, nav: pd.DataFrame, baseline_sharpe: float) -> tuple[float, float, GateCheck]:
        """Gate 8: 相对基线增量"""
        if nav is None or nav.empty:
            return 0.0, 0.0, GateCheck(
                name="相对基线增量",
                status=GateStatus.SKIPPED,
                value=0.0,
                threshold=self.thresholds.sharpe_vs_baseline_min,
                message="无净值数据，跳过检查",
            )

        # 计算策略Sharpe
        nav_series = nav.set_index('trade_date')['nav']
        daily_returns = nav_series.pct_change().dropna()

        if len(daily_returns) < 60:
            return 0.0, 0.0, GateCheck(
                name="相对基线增量",
                status=GateStatus.SKIPPED,
                value=0.0,
                threshold=self.thresholds.sharpe_vs_baseline_min,
                message="数据不足，跳过检查",
            )

        annual_return = daily_returns.mean() * 252
        annual_vol = daily_returns.std() * (252 ** 0.5)
        strategy_sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

        sharpe_increment = strategy_sharpe - baseline_sharpe
        passed = sharpe_increment >= self.thresholds.sharpe_vs_baseline_min

        return sharpe_increment, strategy_sharpe, GateCheck(
            name="相对基线增量",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=sharpe_increment,
            threshold=self.thresholds.sharpe_vs_baseline_min,
            message=f"Sharpe增量 {'≥' if passed else '<'} {self.thresholds.sharpe_vs_baseline_min:.3f} (策略:{strategy_sharpe:.3f} vs 基线:{baseline_sharpe:.3f})",
        )


def default_gate() -> StrategyGate:
    """返回默认门控配置"""
    return StrategyGate(GateThresholds())
