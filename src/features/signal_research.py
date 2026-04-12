"""
信号研究工具 - 专注于因子评价和筛选
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FactorMetrics:
    """因子评价指标"""
    name: str
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_win_rate: float
    ic_t_stat: float
    ic_positive_ratio: float
    
    # 单调性
    quantile_returns: dict[int, float]
    long_short_return: float
    monotonicity_score: float
    
    # 稳定性
    yearly_ic: dict[int, float]
    ic_decay_1y: float
    ic_decay_2y: float
    
    # 成本
    cost_sensitivity: dict[float, float]
    
    # 综合评分
    composite_score: float
    recommendation: str


class SignalResearcher:
    """信号研究员 - 专门做因子评价"""
    
    def __init__(self, trading_cost_bps: float = 15):
        self.trading_cost_bps = trading_cost_bps
    
    def evaluate_factor(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        factor_name: str,
        label_col: str = 'fwd_return_20d',
        n_quantiles: int = 5,
    ) -> FactorMetrics:
        """评价单个因子"""
        # 合并数据
        df = feature_panel.merge(
            label_panel[['trade_date', 'symbol', label_col]],
            on=['trade_date', 'symbol']
        )
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.dropna(subset=[factor_name, label_col])
        
        if len(df) < 100:
            return None
        
        # ===== 1. 横截面IC =====
        daily_ic = df.groupby('trade_date').apply(
            lambda g: g[factor_name].corr(g[label_col], method='spearman'),
            include_groups=False
        ).dropna()
        
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_win_rate = (daily_ic > 0).mean()
        ic_t_stat = ic_mean / (ic_std / np.sqrt(len(daily_ic))) if ic_std > 0 else 0
        
        # ===== 2. 单调性检验 =====
        df['quantile'] = df.groupby('trade_date')[factor_name].transform(
            lambda x: pd.qcut(x.rank(method='first'), n_quantiles, labels=False, duplicates='drop') + 1
        )
        group_rets = df.groupby(['trade_date', 'quantile'])[label_col].mean()
        avg_returns = group_rets.groupby('quantile').mean()
        long_short = avg_returns.iloc[-1] - avg_returns.iloc[0]
        
        # 单调性得分 (Spearman相关性)
        q_values = list(avg_returns.index)
        r_values = list(avg_returns.values)
        monotonicity = np.corrcoef(q_values, r_values)[0, 1] if len(q_values) > 1 else 0
        
        # ===== 3. 稳定性检验 =====
        yearly_ic = daily_ic.groupby(daily_ic.index.year).mean().to_dict()
        
        # IC衰减
        years = sorted(yearly_ic.keys())
        ic_decay_1y = yearly_ic.get(years[-1], 0) - yearly_ic.get(years[-2], 0) if len(years) >= 2 else 0
        ic_decay_2y = yearly_ic.get(years[-1], 0) - yearly_ic.get(years[-3], 0) if len(years) >= 3 else 0
        
        # ===== 4. 成本敏感性 =====
        cost_sensitivity = {}
        for cost in [0, 5, 10, 15, 20, 30, 50]:
            net = long_short - cost / 10000
            cost_sensitivity[cost] = net
        
        # ===== 5. 综合评分 =====
        # IC得分 (0-40)
        ic_score = min(40, max(0, ic_mean * 1000 + 10))
        
        # IR得分 (0-20)
        ir_score = min(20, max(0, ic_ir * 50 + 10))
        
        # 单调性得分 (0-20)
        mono_score = min(20, max(0, monotonicity * 20))
        
        # 稳定性得分 (0-20)
        stability_score = 20 if ic_decay_1y > -0.02 else max(0, 20 + ic_decay_1y * 500)
        
        composite = ic_score + ir_score + mono_score + stability_score
        
        # ===== 6. 推荐 =====
        if composite >= 60 and ic_mean > 0:
            recommendation = "strong"
        elif composite >= 40 and ic_mean > 0:
            recommendation = "moderate"
        elif composite >= 20 and ic_mean > 0:
            recommendation = "weak"
        elif long_short > 0.002:
            recommendation = "marginal"
        else:
            recommendation = "reject"
        
        return FactorMetrics(
            name=factor_name,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_win_rate=ic_win_rate,
            ic_t_stat=ic_t_stat,
            ic_positive_ratio=(ic_mean > 0),
            quantile_returns={int(k): v for k, v in avg_returns.to_dict().items()},
            long_short_return=long_short,
            monotonicity_score=monotonicity,
            yearly_ic=yearly_ic,
            ic_decay_1y=ic_decay_1y,
            ic_decay_2y=ic_decay_2y,
            cost_sensitivity=cost_sensitivity,
            composite_score=composite,
            recommendation=recommendation,
        )
    
    def evaluate_factors(
        self,
        feature_panel: pd.DataFrame,
        label_panel: pd.DataFrame,
        factor_names: list[str],
        **kwargs
    ) -> list[FactorMetrics]:
        """批量评价因子"""
        results = []
        for name in factor_names:
            try:
                m = self.evaluate_factor(feature_panel, label_panel, name, **kwargs)
                if m:
                    results.append(m)
            except Exception as e:
                print(f"  因子 {name} 评价失败: {e}")
        return results
    
    def print_report(self, metrics: list[FactorMetrics]):
        """打印报告"""
        if not metrics:
            print("没有可用的因子评价结果")
            return
        
        # 按综合评分排序
        sorted_metrics = sorted(metrics, key=lambda x: x.composite_score, reverse=True)
        
        print("=" * 100)
        print("信号研究报告 - 因子评价与筛选")
        print("=" * 100)
        
        # ---- 汇总表 ----
        print("\n【一、横截面IC分析】")
        print(f"{'因子':<25} {'IC均值':>10} {'IC_IR':>10} {'IC胜率':>10} {'t统计量':>10} {'综合评分':>10}")
        print("-" * 85)
        for m in sorted_metrics:
            flag = "✅" if m.ic_mean > 0 else "❌"
            print(f"{flag}{m.name:<23} {m.ic_mean:>10.4f} {m.ic_ir:>10.3f} {m.ic_win_rate:>10.1%} {m.ic_t_stat:>10.2f} {m.composite_score:>10.1f}")
        
        # ---- 单调性 ----
        print("\n【二、单调性检验 (Q5-Q1多空收益)】")
        print(f"{'因子':<25} {'Q1收益':>10} {'Q3收益':>10} {'Q5收益':>10} {'多空':>12} {'单调性':>10}")
        print("-" * 85)
        for m in sorted_metrics[:15]:
            q1 = m.quantile_returns.get(1, 0)
            q3 = m.quantile_returns.get(3, 0)
            q5 = m.quantile_returns.get(5, 0)
            mono_flag = "✅" if m.monotonicity_score > 0.5 else "⚠️"
            print(f"{mono_flag}{m.name:<23} {q1*100:>9.2f}% {q3*100:>9.2f}% {q5*100:>9.2f}% {m.long_short_return*100:>11.2f}% {m.monotonicity_score:>10.3f}")
        
        # ---- 年度IC ----
        print("\n【三、年度IC稳定性】")
        all_years = set()
        for m in sorted_metrics:
            all_years.update(m.yearly_ic.keys())
        years = sorted(all_years, reverse=True)[:6]
        
        print(f"{'因子':<25}", end="")
        for y in years:
            print(f"{y:>10}", end="")
        print(f"{'近期衰减':>12}")
        print("-" * (25 + 10 * len(years) + 12))
        
        for m in sorted_metrics[:15]:
            print(f"{m.name:<25}", end="")
            for y in years:
                ic = m.yearly_ic.get(y, np.nan)
                if not np.isnan(ic):
                    color = "🟢" if ic > 0 else "🔴"
                    print(f"{color}{ic:>9.3f}", end="")
                else:
                    print(f"{'N/A':>10}", end="")
            print(f"{m.ic_decay_1y:>12.3f}")
        
        # ---- 成本敏感性 ----
        print("\n【四、成本敏感性 (单边bps)】")
        print(f"{'因子':<25} {'0bp':>10} {'15bp':>10} {'30bp':>10} {'50bp':>10} {'结论':>15}")
        print("-" * 80)
        for m in sorted_metrics[:15]:
            r0 = m.cost_sensitivity.get(0, 0)
            r15 = m.cost_sensitivity.get(15, 0)
            r30 = m.cost_sensitivity.get(30, 0)
            r50 = m.cost_sensitivity.get(50, 0)
            
            if r50 > 0:
                conclusion = "✅成本不敏感"
            elif r30 > 0:
                conclusion = "⚠️中等敏感"
            elif r15 > 0:
                conclusion = "⚠️较敏感"
            else:
                conclusion = "❌成本敏感"
            
            print(f"{m.name:<25} {r0*100:>9.2f}% {r15*100:>9.2f}% {r30*100:>9.2f}% {r50*100:>9.2f}% {conclusion:>15}")
        
        # ---- 推荐 ----
        print("\n【五、因子推荐】")
        strong = [m for m in sorted_metrics if m.recommendation == "strong"]
        moderate = [m for m in sorted_metrics if m.recommendation == "moderate"]
        weak = [m for m in sorted_metrics if m.recommendation == "weak"]
        marginal = [m for m in sorted_metrics if m.recommendation == "marginal"]
        reject = [m for m in sorted_metrics if m.recommendation == "reject"]
        
        print(f"\n✅ 强烈推荐 ({len(strong)}个):")
        for m in strong:
            print(f"   {m.name}: IC={m.ic_mean:.4f}, IR={m.ic_ir:.3f}, 多空={m.long_short_return*100:.2f}%")
        
        print(f"\n⚠️  可考虑 ({len(moderate)}个):")
        for m in moderate:
            print(f"   {m.name}: IC={m.ic_mean:.4f}, IR={m.ic_ir:.3f}")
        
        print(f"\n❌  不推荐 ({len(reject)}个):")
        for m in reject[:10]:
            print(f"   {m.name}: IC={m.ic_mean:.4f}, {m.recommendation}")
        if len(reject) > 10:
            print(f"   ... 还有{len(reject)-10}个")
        
        print("\n" + "=" * 100)
        print(f"总计: {len(sorted_metrics)}个因子, 推荐使用: {len(strong)+len(moderate)}个")
        print("=" * 100)
    
    def get_selected_factors(self, metrics: list[FactorMetrics]) -> list[str]:
        """获取推荐的因子列表"""
        return [
            m.name for m in metrics 
            if m.recommendation in ["strong", "moderate"]
        ]


def run_signal_research(
    feature_panel: pd.DataFrame,
    label_panel: pd.DataFrame,
    factor_names: list[str],
    **kwargs
) -> tuple[list[FactorMetrics], list[str]]:
    """运行信号研究的便捷函数"""
    researcher = SignalResearcher()
    metrics = researcher.evaluate_factors(feature_panel, label_panel, factor_names, **kwargs)
    researcher.print_report(metrics)
    selected = researcher.get_selected_factors(metrics)
    return metrics, selected
