"""
因子档案系统 - 因子池的"资产管理系统"

目标：
1. 为每个因子建立档案（定义、机制、观测时点、冗余性）
2. 按家族分类因子
3. 识别近似替代品
4. 支持Research Pool筛选
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class FactorFamily(Enum):
    """因子家族分类"""
    MOMENTUM = "momentum"           # 动量
    REVERSAL = "reversal"           # 反转
    VOLATILITY = "volatility"       # 波动率
    LIQUIDITY = "liquidity"         # 流动性
    TECHNICAL_PRICE_LEVEL = "technical_price_level"  # 价格位置
    TECHNICAL_OSCILLATOR = "technical_oscillator"    # 技术震荡
    TECHNICAL_PATTERN = "technical_pattern"          # 技术形态
    DISTRIBUTION = "distribution"    # 分布特征
    SENTIMENT = "sentiment"         # 情绪/资金流
    VALUE = "value"                 # 估值
    PROFITABILITY = "profitability" # 盈利能力
    GROWTH = "growth"              # 成长能力
    CASHFLOW = "cashflow"           # 现金流
    EFFICIENCY = "efficiency"       # 运营效率
    LEVERAGE = "leverage"          # 杠杆/偿债
    QUALITY = "quality"            # 质量
    BEHAVIORAL = "behavioral"       # 行为金融


class FactorStatus(Enum):
    """因子状态"""
    CORE = "core"                   # 核心池 - IC>0.01, IR>0.1, 稳定3年+
    BACKUP = "backup"              # 备用池 - IC>0.005
    OBSERVE = "observe"             # 观察池 - IC>0
    REJECT = "reject"               # 拒绝池 - IC<=0
    DEPRECATED = "deprecated"       # 已退役
    POOL = "pool"                   # 研究池 - 尚未评估


class FactorSignal(Enum):
    """因子信号方向"""
    LONG = "long"                  # 预期未来收益为正
    SHORT = "short"                # 预期未来收益为负
    POSITIVE = "positive"          # 预期正向收益
    NEGATIVE = "negative"         # 预期负向收益
    NEUTRAL = "neutral"            # 方向不确定
    CONTEXT_DEPENDENT = "context"  # 依赖于市场状态


@dataclass
class FactorProfile:
    """因子档案"""
    name: str
    family: FactorFamily
    description: str

    # 经济机制
    economic_mechanism: str
    hypothesis: str
    expected_signal: FactorSignal

    # 技术规格
    lookback: int
    inputs: tuple[str, ...]
    computation: str

    # 冗余性
    similar_to: list[str] = field(default_factory=list)  # 近似替代品
    dominates: list[str] = field(default_factory=list)    # 被此因子主导的因子
    noise_correlate: list[str] = field(default_factory=list)  # 常被误认为相关的噪音

    # 状态
    status: FactorStatus = FactorStatus.OBSERVE

    # 性能指标 (来自研究)
    ic_mean: float | None = None
    ic_std: float | None = None
    ic_ir: float | None = None
    ic_positive_ratio: float | None = None

    # 成本敏感性
    cost_threshold_30bp: bool = False  # 30bp单边成本下是否仍有效

    # 风险点
    failure_modes: list[str] = field(default_factory=list)
    regime_sensitivity: str = ""       # 对哪种市场状态敏感

    # 元数据
    version: str = "v1"
    owner: str = "research"
    last_review: str = ""
    notes: str = ""


class FactorCatalog:
    """因子档案管理器"""

    def __init__(self):
        self._profiles: dict[str, FactorProfile] = {}
        self._family_index: dict[FactorFamily, list[str]] = {}
        self._status_index: dict[FactorStatus, list[str]] = {}

    def register(self, profile: FactorProfile) -> None:
        """注册因子档案"""
        self._profiles[profile.name] = profile

        # 更新家族索引
        if profile.family not in self._family_index:
            self._family_index[profile.family] = []
        self._family_index[profile.family].append(profile.name)

        # 更新状态索引
        if profile.status not in self._status_index:
            self._status_index[profile.status] = []
        self._status_index[profile.status].append(profile.name)

    def get(self, name: str) -> FactorProfile | None:
        return self._profiles.get(name)

    def get_by_family(self, family: FactorFamily) -> list[FactorProfile]:
        names = self._family_index.get(family, [])
        return [self._profiles[n] for n in names]

    def get_by_status(self, status: FactorStatus) -> list[FactorProfile]:
        names = self._status_index.get(status, [])
        return [self._profiles[n] for n in names]

    def get_core_factors(self) -> list[FactorProfile]:
        return self.get_by_status(FactorStatus.CORE)

    def get_research_pool(self) -> list[FactorProfile]:
        """获取研究池 - CORE + BACKUP"""
        return self.get_by_status(FactorStatus.CORE) + self.get_by_status(FactorStatus.BACKUP)

    def inventory(self) -> pd.DataFrame:
        """导出档案清单"""
        rows = []
        for p in self._profiles.values():
            rows.append({
                'factor': p.name,
                'family': p.family.value,
                'status': p.status.value,
                'expected_signal': p.expected_signal.value,
                'lookback': p.lookback,
                'ic_mean': p.ic_mean,
                'ic_ir': p.ic_ir,
                'cost_30bp_ok': p.cost_threshold_30bp,
                'similar_to': ', '.join(p.similar_to),
                'dominates': ', '.join(p.dominates),
                'economic_mechanism': p.economic_mechanism,
                'failure_modes': '; '.join(p.failure_modes),
            })
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values(['status', 'family', 'ic_mean'], ascending=[True, True, False])
        return df

    def print_report(self):
        """打印因子池报告"""
        print("=" * 100)
        print("因子档案系统 - 因子池报告")
        print("=" * 100)

        # 状态汇总
        print("\n【一、因子池状态分布】")
        for status in FactorStatus:
            count = len(self._status_index.get(status, []))
            if count > 0:
                print(f"  {status.value}: {count}个")

        # 家族分布
        print("\n【二、因子家族分布】")
        for family in FactorFamily:
            count = len(self._family_index.get(family, []))
            if count > 0:
                print(f"  {family.value}: {count}个")

        # 核心池
        core = self.get_core_factors()
        if core:
            print("\n【三、核心池 (CORE) - IC>0.01, IR>0.1, 稳定3年+】")
            print(f"{'因子':<20} {'家族':<15} {'IC':>8} {'IR':>8} {'信号':<10} {'成本30bp':>10}")
            print("-" * 80)
            for p in sorted(core, key=lambda x: x.ic_mean or 0, reverse=True):
                cost = "✅" if p.cost_threshold_30bp else "❌"
                ic = f"{p.ic_mean:.4f}" if p.ic_mean else "N/A"
                ir = f"{p.ic_ir:.3f}" if p.ic_ir else "N/A"
                print(f"{p.name:<20} {p.family.value:<15} {ic:>8} {ir:>8} {p.expected_signal.value:<10} {cost:>10}")

        # 备用池
        backup = self.get_by_status(FactorStatus.BACKUP)
        if backup:
            print("\n【四、备用池 (BACKUP) - IC>0.005】")
            for p in sorted(backup, key=lambda x: x.ic_mean or 0, reverse=True):
                print(f"  {p.name}: {p.economic_mechanism}")

        # 冗余关系
        print("\n【五、近似替代品关系】")
        for p in self._profiles.values():
            if p.similar_to or p.dominates:
                print(f"\n  {p.name}:")
                if p.similar_to:
                    print(f"    ≈ 类似: {', '.join(p.similar_to)}")
                if p.dominates:
                    print(f"    > 主导: {', '.join(p.dominates)}")

        print("\n" + "=" * 100)


def build_default_catalog() -> FactorCatalog:
    """构建默认因子档案"""
    catalog = FactorCatalog()

    # ============ 动量家族 ============
    # 长周期动量 - 核心有效
    catalog.register(FactorProfile(
        name="mom250",
        family=FactorFamily.MOMENTUM,
        description="250日累计收益率",
        economic_mechanism="趋势延续 + 机构投资者行为惰性",
        hypothesis="过去250日涨幅大的股票未来表现更好",
        expected_signal=FactorSignal.LONG,
        lookback=251,
        inputs=('adj_close',),
        computation="_pct_change(adj_close, 250)",
        similar_to=["mom120", "mom90", "close_to_high250"],
        dominates=["mom3", "mom5", "mom10", "mom20"],
        status=FactorStatus.CORE,
        ic_mean=0.035,
        ic_ir=0.16,
        ic_positive_ratio=0.65,
        cost_threshold_30bp=True,
        failure_modes=["市场风格切换", "长期横盘后的补跌"],
        regime_sensitivity="牛市动量更强，熊市初期有效",
    ))

    # 中周期动量 - 备用
    catalog.register(FactorProfile(
        name="mom120",
        family=FactorFamily.MOMENTUM,
        description="120日累计收益率",
        economic_mechanism="中期趋势延续",
        hypothesis="过去120日涨幅大的股票未来表现更好",
        expected_signal=FactorSignal.LONG,
        lookback=121,
        inputs=('adj_close',),
        computation="_pct_change(adj_close, 120)",
        similar_to=["mom90", "mom250"],
        dominates=["mom20", "mom30", "mom60"],
        status=FactorStatus.BACKUP,
        ic_mean=0.010,
        ic_ir=0.05,
        cost_threshold_30bp=True,
        failure_modes=["中期反转效应干扰"],
        regime_sensitivity="震荡市有效",
    ))

    # 短周期动量 - 拒绝/备份
    for w in [3, 5, 10, 20, 30, 60, 90]:
        status = FactorStatus.REJECT if w <= 60 else FactorStatus.BACKUP
        ic_mean = -0.02 if w <= 30 else 0.0 if w <= 60 else 0.001
        catalog.register(FactorProfile(
            name=f"mom{w}",
            family=FactorFamily.MOMENTUM,
            description=f"{w}日累计收益率",
            economic_mechanism="短中期反转效应主导" if w <= 30 else "趋势延续",
            hypothesis="短期反转，中期趋势",
            expected_signal=FactorSignal.CONTEXT_DEPENDENT,
            lookback=w + 1,
            inputs=('adj_close',),
            computation="_pct_change(adj_close, w)",
            similar_to=[f"mom{x}" for x in [5, 10, 20, 30, 60, 90, 120] if x != w],
            dominates=["mom250"] if w < 250 else [],
            status=status,
            ic_mean=ic_mean,
            failure_modes=["噪音过大", "短期反转干扰"],
        ))

    # ============ 反转家族 ============
    # 20日反转 - 备用有效
    catalog.register(FactorProfile(
        name="rev20",
        family=FactorFamily.REVERSAL,
        description="20日反转因子（取负收益）",
        economic_mechanism="短期超买超卖回归",
        hypothesis="过去20日跌幅大的股票会反弹",
        expected_signal=FactorSignal.LONG,
        lookback=21,
        inputs=('adj_close',),
        computation="-_pct_change(adj_close, 20)",
        similar_to=["rev10", "rev5", "rsi6"],
        dominates=["rev1", "rev3"],
        status=FactorStatus.BACKUP,
        ic_mean=0.019,
        ic_ir=0.08,
        cost_threshold_30bp=True,
        failure_modes=["强趋势市场中反转失效"],
        regime_sensitivity="震荡市反转更强，趋势市失效",
    ))

    for w in [1, 3, 5, 10]:
        ic_mean = 0.01 if w == 10 else 0.005 if w == 5 else 0.0
        catalog.register(FactorProfile(
            name=f"rev{w}",
            family=FactorFamily.REVERSAL,
            description=f"{w}日反转因子",
            economic_mechanism="短期超买超卖回归",
            hypothesis="短期跌幅大则反弹",
            expected_signal=FactorSignal.LONG if w >= 5 else FactorSignal.NEUTRAL,
            lookback=w + 1,
            inputs=('adj_close',),
            computation="-_pct_change(adj_close, w)",
            similar_to=[f"rev{x}" for x in [1, 3, 5, 10, 20] if x != w],
            dominates=["rev20"] if w < 20 else [],
            status=FactorStatus.BACKUP if w >= 5 else FactorStatus.REJECT,
            ic_mean=ic_mean,
            failure_modes=["噪音过大"],
        ))

    # ============ 波动率家族 ============
    # 高波动率因子 - 拒绝 (IC显著为负)
    for w in [5, 10, 20, 30, 60, 120]:
        ic_mean = -0.03 if w >= 60 else -0.02 if w >= 30 else -0.01
        catalog.register(FactorProfile(
            name=f"vol{w}",
            family=FactorFamily.VOLATILITY,
            description=f"{w}日收益率标准差",
            economic_mechanism="高波动股票风险溢价(但A股实证IC为负)",
            hypothesis="高波动股票未来收益更低(风险溢价理论)",
            expected_signal=FactorSignal.SHORT,
            lookback=w,
            inputs=('adj_close',),
            computation="_rolling_std(adj_close, w)",
            similar_to=[f"vol{x}" for x in [5, 10, 20, 30, 60, 120] if x != w] + [f"atr{x}" for x in [14, 20]],
            status=FactorStatus.REJECT,
            ic_mean=ic_mean,
            ic_ir=-0.15,
            cost_threshold_30bp=False,
            failure_modes=["波动率被机构用于择时止损", "高频交易干扰", "LightGBM高权重误导"],
            notes="⚠️ LightGBM常给此因子高权重但IC为负，是误导因子",
        ))

    # ATR标准化
    for w in [14, 20]:
        catalog.register(FactorProfile(
            name=f"atr{w}",
            family=FactorFamily.VOLATILITY,
            description=f"{w}日ATR标准化",
            economic_mechanism="真实波动范围",
            hypothesis="波动率效应的另一种表达",
            expected_signal=FactorSignal.SHORT,
            lookback=w + 1,
            inputs=('high', 'low', 'close'),
            computation="_atr_normalized(w)",
            similar_to=[f"vol{w}"],
            status=FactorStatus.REJECT,
            ic_mean=-0.03,
            cost_threshold_30bp=False,
            failure_modes=["与vol类因子同质化"],
        ))

    # 波动率比
    for s, l in [(5, 20), (10, 60), (20, 60)]:
        catalog.register(FactorProfile(
            name=f"vol_std_ratio{s}_{l}",
            family=FactorFamily.VOLATILITY,
            description=f"波动率比 {s}/{l}",
            economic_mechanism="短期波动相对长期变化",
            hypothesis="波动率比变化预测收益",
            expected_signal=FactorSignal.NEUTRAL,
            lookback=l,
            inputs=('adj_close',),
            computation="_vol_std_ratio(s, l)",
            status=FactorStatus.REJECT,
            ic_mean=-0.02,
        ))

    # ============ 流动性家族 ============
    for short, long in [(5, 20), (10, 60), (20, 60), (5, 60)]:
        catalog.register(FactorProfile(
            name=f"vol_ratio_{short}_{long}",
            family=FactorFamily.LIQUIDITY,
            description=f"成交量比 {short}/{long}",
            economic_mechanism="量能变化趋势",
            hypothesis="放量预示趋势延续",
            expected_signal=FactorSignal.CONTEXT_DEPENDENT,
            lookback=long,
            inputs=('volume',),
            computation="_volume_ratio(short, long)",
            status=FactorStatus.BACKUP,
            ic_mean=0.001,
            failure_modes=["放量可能是出货信号"],
        ))

    for w in [5, 20, 60]:
        catalog.register(FactorProfile(
            name=f"amount_growth{w}",
            family=FactorFamily.LIQUIDITY,
            description=f"{w}日成交额增长",
            economic_mechanism="资金流入流出",
            hypothesis="成交额增长预示上涨",
            expected_signal=FactorSignal.LONG if w <= 20 else FactorSignal.NEUTRAL,
            lookback=w + 1,
            inputs=('adj_close', 'volume'),
            computation="_amount_growth(w)",
            status=FactorStatus.BACKUP,
            ic_mean=0.001,
        ))

    for w in [5, 10, 20]:
        catalog.register(FactorProfile(
            name=f"vol_growth{w}",
            family=FactorFamily.LIQUIDITY,
            description=f"{w}日成交量增长",
            economic_mechanism="量能变化",
            hypothesis="量增价涨",
            expected_signal=FactorSignal.CONTEXT_DEPENDENT,
            lookback=w + 1,
            inputs=('volume',),
            computation="_pct_change(volume, w)",
            similar_to=[f"vol_ratio_{5}_{20}"],
            status=FactorStatus.BACKUP,
            ic_mean=0.001,
        ))

    # ============ 价格位置家族 ============
    # 创新高 - 核心有效
    catalog.register(FactorProfile(
        name="close_to_high250",
        family=FactorFamily.TECHNICAL_PRICE_LEVEL,
        description="收盘价/250日最高价",
        economic_mechanism="趋势确认 + 阻力位突破",
        hypothesis="接近250日新高的股票会继续上涨",
        expected_signal=FactorSignal.LONG,
        lookback=250,
        inputs=('adj_close',),
        computation="_close_to_high(250)",
        similar_to=["mom250", "close_to_high120"],
        dominates=["close_to_high60", "close_to_high120"],
        status=FactorStatus.CORE,
        ic_mean=0.023,
        ic_ir=0.11,
        cost_threshold_30bp=True,
        failure_modes=["假突破"],
        regime_sensitivity="趋势市有效",
    ))

    # 120日高位 - 备用有效
    catalog.register(FactorProfile(
        name="close_to_high120",
        family=FactorFamily.TECHNICAL_PRICE_LEVEL,
        description="收盘价/120日最高价",
        economic_mechanism="中期趋势位置",
        hypothesis="接近120日新高的股票中期趋势向上",
        expected_signal=FactorSignal.LONG,
        lookback=120,
        inputs=('adj_close',),
        computation="_close_to_high(120)",
        similar_to=["close_to_high250", "high_low_pos120"],
        dominates=["close_to_high60", "close_to_high20"],
        status=FactorStatus.BACKUP,
        ic_mean=0.007,
        cost_threshold_30bp=True,
    ))

    for w in [20, 60, 120]:
        if w not in [120, 250]:
            catalog.register(FactorProfile(
                name=f"close_to_high{w}",
                family=FactorFamily.TECHNICAL_PRICE_LEVEL,
                description=f"收盘价/{w}日最高价",
                economic_mechanism="趋势位置",
                hypothesis="接近新高预示上涨",
                expected_signal=FactorSignal.LONG,
                lookback=w,
                inputs=('adj_close',),
                computation="_close_to_high(w)",
                similar_to=["close_to_high120", "close_to_high250"],
                dominates=[f"close_to_high{x}" for x in [120, 250] if x > w],
                status=FactorStatus.BACKUP if w >= 60 else FactorStatus.REJECT,
                ic_mean=0.005 if w >= 60 else 0.001,
            ))

    # 创新低
    for w in [20, 60, 120, 250]:
        catalog.register(FactorProfile(
            name=f"close_to_low{w}",
            family=FactorFamily.TECHNICAL_PRICE_LEVEL,
            description=f"收盘价/{w}日最低价",
            economic_mechanism="超卖反弹",
            hypothesis="接近新低可能反弹",
            expected_signal=FactorSignal.NEUTRAL,  # A股实证不显著
            lookback=w,
            inputs=('adj_close',),
            computation="_close_to_low(w)",
            status=FactorStatus.REJECT,
            ic_mean=-0.01 if w <= 60 else 0.0,
            failure_modes=["跌跌不休股票不适合抄底"],
        ))

    # 高低价位置 - 备用有效
    catalog.register(FactorProfile(
        name="high_low_pos120",
        family=FactorFamily.TECHNICAL_PRICE_LEVEL,
        description="(收盘-最低)/(最高-最低) 120日",
        economic_mechanism="价格在高低点的相对位置",
        hypothesis="价格位置高预示上涨趋势",
        expected_signal=FactorSignal.LONG,
        lookback=120,
        inputs=('adj_close',),
        computation="_high_low_position(120)",
        similar_to=["close_to_high120", "price_to_ma120"],
        status=FactorStatus.BACKUP,
        ic_mean=0.005,
        cost_threshold_30bp=True,
    ))

    for w in [20, 60]:
        catalog.register(FactorProfile(
            name=f"high_low_pos{w}",
            family=FactorFamily.TECHNICAL_PRICE_LEVEL,
            description=f"(收盘-最低)/(最高-最低) {w}日",
            economic_mechanism="短期价格位置",
            hypothesis="价格位置高预示上涨",
            expected_signal=FactorSignal.LONG,
            lookback=w,
            inputs=('adj_close',),
            computation="_high_low_position(w)",
            dominates=[f"high_low_pos{x}" for x in [60, 120] if x > w],
            status=FactorStatus.BACKUP,
            ic_mean=0.001,
        ))

    # ============ 技术震荡指标家族 ============
    # RSI - 备用有效
    catalog.register(FactorProfile(
        name="rsi6",
        family=FactorFamily.TECHNICAL_OSCILLATOR,
        description="6日RSI",
        economic_mechanism="超买超卖",
        hypothesis="RSI低(超卖)预示反弹",
        expected_signal=FactorSignal.LONG,
        lookback=12,
        inputs=('adj_close',),
        computation="_rsi(6)",
        similar_to=["rev5", "rev10", "williams_r14"],
        status=FactorStatus.BACKUP,
        ic_mean=0.003,
        cost_threshold_30bp=True,
        failure_modes=["趋势市RSI失效"],
        regime_sensitivity="震荡市有效",
    ))

    for w in [12, 14, 24]:
        catalog.register(FactorProfile(
            name=f"rsi{w}",
            family=FactorFamily.TECHNICAL_OSCILLATOR,
            description=f"{w}日RSI",
            economic_mechanism="超买超卖",
            hypothesis="RSI低预示反弹",
            expected_signal=FactorSignal.NEUTRAL,
            lookback=w * 2,
            inputs=('adj_close',),
            computation="_rsi(w)",
            dominates=["rsi6"],
            status=FactorStatus.BACKUP if w == 14 else FactorStatus.OBSERVE,
            ic_mean=0.002 if w == 14 else 0.001,
        ))

    # Williams %R
    for w in [14, 28]:
        catalog.register(FactorProfile(
            name=f"williams_r{w}",
            family=FactorFamily.TECHNICAL_OSCILLATOR,
            description=f"{w}日Williams %R",
            economic_mechanism="超买超卖",
            hypothesis="低值超卖",
            expected_signal=FactorSignal.NEUTRAL,
            lookback=w,
            inputs=('high', 'low', 'close'),
            computation="_williams_r(w)",
            similar_to=["rsi6"],
            status=FactorStatus.REJECT,
            ic_mean=-0.01,
            failure_modes=["与RSI同质化"],
        ))

    # KDJ - 拒绝
    for w in [9, 14]:
        for sub in ['k', 'd']:
            catalog.register(FactorProfile(
                name=f"kdj_{sub}{w}",
                family=FactorFamily.TECHNICAL_OSCILLATOR,
                description=f"KDJ {sub.upper()} {w}日",
                economic_mechanism="随机指标",
                hypothesis="低金叉买",
                expected_signal=FactorSignal.NEUTRAL,
                lookback=w,
                inputs=('high', 'low', 'close'),
                computation=f"_kdj_{sub}(w)",
                status=FactorStatus.REJECT,
                ic_mean=-0.06,
                cost_threshold_30bp=False,
                failure_modes=["参数敏感", "过度拟合历史"],
                notes="⚠️ IC显著为负，不适合作为因子",
            ))

    # CCI
    for w in [14, 20]:
        catalog.register(FactorProfile(
            name=f"cci{w}",
            family=FactorFamily.TECHNICAL_OSCILLATOR,
            description=f"{w}日CCI",
            economic_mechanism="顺势指标",
            hypothesis="超买超卖",
            expected_signal=FactorSignal.NEUTRAL,
            lookback=w,
            inputs=('high', 'low', 'close'),
            computation="_cci(w)",
            status=FactorStatus.REJECT,
            ic_mean=-0.01,
        ))

    # MACD
    catalog.register(FactorProfile(
        name="macd_diff",
        family=FactorFamily.TECHNICAL_OSCILLATOR,
        description="MACD DIF线",
        economic_mechanism="快慢均线差",
        hypothesis="DIF上穿DEA买",
        expected_signal=FactorSignal.NEUTRAL,
        lookback=26,
        inputs=('adj_close',),
        computation="_macd_diff(12, 26)",
        status=FactorStatus.REJECT,
        ic_mean=-0.01,
        failure_modes=["趋势滞后", "噪音大"],
    ))

    catalog.register(FactorProfile(
        name="macd_dea",
        family=FactorFamily.TECHNICAL_OSCILLATOR,
        description="MACD DEA线",
        economic_mechanism="MACD信号线",
        hypothesis="趋势确认",
        expected_signal=FactorSignal.NEUTRAL,
        lookback=35,
        inputs=('adj_close',),
        computation="_dea(12, 26, 9)",
        status=FactorStatus.REJECT,
        ic_mean=-0.01,
    ))

    catalog.register(FactorProfile(
        name="macd_hist",
        family=FactorFamily.TECHNICAL_OSCILLATOR,
        description="MACD柱状图",
        economic_mechanism="MACD动能",
        hypothesis="柱变长趋势延续",
        expected_signal=FactorSignal.NEUTRAL,
        lookback=35,
        inputs=('adj_close',),
        computation="_macd_hist(12, 26, 9)",
        status=FactorStatus.REJECT,
        ic_mean=-0.01,
    ))

    # ============ 均线乖离家族 ============
    for short, long in [(5, 20), (10, 20), (5, 60), (20, 60)]:
        catalog.register(FactorProfile(
            name=f"ma_diff_{short}_{long}",
            family=FactorFamily.TECHNICAL_PRICE_LEVEL,
            description=f"MA{short}/MA{long}",
            economic_mechanism="均线收敛发散",
            hypothesis="比值>1强势",
            expected_signal=FactorSignal.LONG,
            lookback=long,
            inputs=('adj_close',),
            computation="_ma_diff(short, long)",
            similar_to=[f"price_to_ma{short}"],
            status=FactorStatus.BACKUP,
            ic_mean=0.001,
        ))

    for w in [20, 60, 120]:
        catalog.register(FactorProfile(
            name=f"price_to_ma{w}",
            family=FactorFamily.TECHNICAL_PRICE_LEVEL,
            description=f"收盘价/MA{w}",
            economic_mechanism="价格与均线关系",
            hypothesis="价格>均线强势",
            expected_signal=FactorSignal.LONG,
            lookback=w,
            inputs=('adj_close',),
            computation="_price_to_ma(w)",
            similar_to=[f"high_low_pos{w}", f"close_to_high{w}"],
            status=FactorStatus.BACKUP,
            ic_mean=0.001,
        ))

    # ============ 分布特征家族 ============
    for w in [20, 60]:
        catalog.register(FactorProfile(
            name=f"return_skew{w}",
            family=FactorFamily.DISTRIBUTION,
            description=f"{w}日收益率偏度",
            economic_mechanism="收益分布不对称",
            hypothesis="正偏(右尾长)可能预示上涨",
            expected_signal=FactorSignal.NEUTRAL,
            lookback=w,
            inputs=('adj_close',),
            computation="_skewness(w)",
            status=FactorStatus.BACKUP,
            ic_mean=0.001,
        ))

        catalog.register(FactorProfile(
            name=f"return_kurt{w}",
            family=FactorFamily.DISTRIBUTION,
            description=f"{w}日收益率峰度",
            economic_mechanism="收益分布尖峰",
            hypothesis="尖峰可能预示波动",
            expected_signal=FactorSignal.NEUTRAL,
            lookback=w,
            inputs=('adj_close',),
            computation="_kurtosis(w)",
            status=FactorStatus.BACKUP,
            ic_mean=0.001,
        ))

    # ============ 财务因子家族 ============
    # 基于 AGENTS.md 2026-04-12 策略发现结果
    # IC/IR 数据来自 scripts/strategy_discovery_v2.py 验证

    # ---- 估值因子 (Value) ----
    catalog.register(FactorProfile(
        name="earnings_yield",
        family=FactorFamily.VALUE,
        description="市盈率倒数 (E/P)",
        economic_mechanism="价值投资 - 低估值股票被低估，终将回归",
        hypothesis="低PE股票未来收益更高",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="1 / PE_ttm",
        similar_to=["book_to_price", "cashflow_yield", "dividend_yield"],
        dominates=[],
        status=FactorStatus.CORE,
        ic_mean=0.048,
        ic_ir=0.430,
        ic_positive_ratio=0.64,
        cost_threshold_30bp=True,
        failure_modes=["价值陷阱(低PE因低增长)"],
        regime_sensitivity="牛市初期有效，熊市价值防御",
        notes="= 1/PE_ttm，不是ROE类因子",
    ))

    catalog.register(FactorProfile(
        name="book_to_price",
        family=FactorFamily.VALUE,
        description="市净率倒数 (B/P)",
        economic_mechanism="清算价值保护",
        hypothesis="低PB股票有清算价值兜底",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="PB / 1",
        similar_to=["earnings_yield", "equity_ratio"],
        status=FactorStatus.BACKUP,
        ic_mean=0.020,
        ic_ir=0.180,
        cost_threshold_30bp=True,
    ))

    # ---- 盈利能力因子 (Profitability) ----
    catalog.register(FactorProfile(
        name="roe",
        family=FactorFamily.PROFITABILITY,
        description="净资产收益率 (ROE)",
        economic_mechanism="盈利能力 - 高ROE公司创造更多价值",
        hypothesis="高ROE股票未来收益更高",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="net_profit / equity",
        similar_to=["roa", "roe_weighted", "roe_assets"],
        dominates=["roe_weighted"],
        status=FactorStatus.CORE,
        ic_mean=0.041,
        ic_ir=0.363,
        ic_positive_ratio=0.66,
        cost_threshold_30bp=True,
        failure_modes=["高杠杆ROE", "盈利周期性"],
        regime_sensitivity="牛市进攻性强，熊市防御一般",
        notes="与earnings_yield低相关(0.1)，可叠加",
    ))

    catalog.register(FactorProfile(
        name="roe_weighted",
        family=FactorFamily.PROFITABILITY,
        description="加权ROE (TTM加权)",
        economic_mechanism="平滑盈利波动",
        hypothesis="加权ROE更稳定",
        expected_signal=FactorSignal.LONG,
        lookback=4,
        inputs=('financial_data',),
        computation="TTM加权ROE",
        similar_to=["roe", "total_roa"],
        status=FactorStatus.BACKUP,
        ic_mean=0.015,
        ic_ir=0.130,
        cost_threshold_30bp=True,
        notes="与roe高度冗余(0.85)，优先使用roe",
    ))

    catalog.register(FactorProfile(
        name="roa",
        family=FactorFamily.PROFITABILITY,
        description="总资产收益率",
        economic_mechanism="资产效率",
        hypothesis="高ROA公司资产利用效率高",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="net_profit / total_assets",
        similar_to=["roe", "roe_assets"],
        status=FactorStatus.BACKUP,
        ic_mean=0.029,
        ic_ir=0.260,
        cost_threshold_30bp=True,
    ))

    catalog.register(FactorProfile(
        name="total_roa",
        family=FactorFamily.PROFITABILITY,
        description="总资产净收益率",
        economic_mechanism="资产整体回报",
        hypothesis="高总资产净收益率公司更优质",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="net_profit_after_tax / total_assets",
        similar_to=["roa", "roe"],
        status=FactorStatus.BACKUP,
        ic_mean=0.020,
        ic_ir=0.180,
        cost_threshold_30bp=True,
    ))

    catalog.register(FactorProfile(
        name="operating_margin",
        family=FactorFamily.PROFITABILITY,
        description="营业利润率",
        economic_mechanism="主营业务盈利能力",
        hypothesis="高营业利润率公司定价能力强",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="operating_profit / revenue",
        similar_to=["gross_margin", "net_margin", "cost_profit_margin"],
        dominates=["net_margin"],
        status=FactorStatus.CORE,
        ic_mean=0.027,
        ic_ir=0.307,
        ic_positive_ratio=0.64,
        cost_threshold_30bp=True,
        notes="与gross_margin中等冗余(0.6)，都保留",
    ))

    catalog.register(FactorProfile(
        name="gross_margin",
        family=FactorFamily.PROFITABILITY,
        description="毛利率",
        economic_mechanism="定价权护城河",
        hypothesis="高毛利率公司有竞争优势",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="(revenue - cost) / revenue",
        similar_to=["operating_margin", "mainbiz_profit_margin"],
        status=FactorStatus.BACKUP,
        ic_mean=0.018,
        ic_ir=0.169,
        ic_positive_ratio=0.59,
        cost_threshold_30bp=True,
    ))

    # ---- 成长因子 (Growth) ----
    catalog.register(FactorProfile(
        name="equity_growth",
        family=FactorFamily.GROWTH,
        description="净资产增速",
        economic_mechanism="内生增长能力",
        hypothesis="高净资产增速公司成长性强",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="(equity_t - equity_t-4) / equity_t-4",
        similar_to=["asset_growth", "revenue_growth"],
        status=FactorStatus.BACKUP,
        ic_mean=0.019,
        ic_ir=0.190,
        ic_positive_ratio=0.56,
        cost_threshold_30bp=True,
        failure_modes=["外延并购增长"],
    ))

    catalog.register(FactorProfile(
        name="revenue_growth",
        family=FactorFamily.GROWTH,
        description="营收增速",
        economic_mechanism="收入规模扩张",
        hypothesis="高营收增速公司市场占有率提升",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="(revenue_t - revenue_t-4) / revenue_t-4",
        similar_to=["equity_growth", "profit_growth"],
        status=FactorStatus.BACKUP,
        ic_mean=0.011,
        ic_ir=0.092,
        cost_threshold_30bp=True,
        notes="已实现，可用于研究候选",
    ))

    catalog.register(FactorProfile(
        name="profit_growth",
        family=FactorFamily.GROWTH,
        description="利润增速",
        economic_mechanism="盈利增长质量",
        hypothesis="高利润增速公司被市场看好",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="(profit_t - profit_t-4) / profit_t-4",
        similar_to=["revenue_growth", "equity_growth"],
        status=FactorStatus.BACKUP,
        ic_mean=0.012,
        ic_ir=0.110,
        cost_threshold_30bp=True,
        notes="已实现，可用于研究候选",
    ))

    # ---- 现金流因子 (Cash Flow) ----
    catalog.register(FactorProfile(
        name="ocf_per_share",
        family=FactorFamily.CASHFLOW,
        description="每股经营现金流",
        economic_mechanism="盈利质量 - 现金流比利润更真实",
        hypothesis="高每股经营现金流公司盈利质量高",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="operating_cash_flow / shares",
        similar_to=["cash_ratio", "retained_earnings_per_share"],
        status=FactorStatus.CORE,
        ic_mean=0.032,
        ic_ir=0.266,
        ic_positive_ratio=0.60,
        cost_threshold_30bp=True,
        failure_modes=["重资产行业现金流天然低"],
    ))

    # ---- 运营效率因子 (Efficiency) ----
    catalog.register(FactorProfile(
        name="asset_turnover",
        family=FactorFamily.EFFICIENCY,
        description="资产周转率",
        economic_mechanism="资产利用效率",
        hypothesis="高资产周转率公司运营效率高",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="revenue / total_assets",
        similar_to=["fixasset_turnover", "inv_turnover"],
        status=FactorStatus.BACKUP,
        ic_mean=0.018,
        ic_ir=0.198,
        ic_positive_ratio=0.55,
        cost_threshold_30bp=True,
    ))

    # ---- 杠杆/偿债因子 (Leverage) ----
    catalog.register(FactorProfile(
        name="cash_ratio",
        family=FactorFamily.LEVERAGE,
        description="现金比率",
        economic_mechanism="短期偿债能力",
        hypothesis="高现金比率公司财务稳健",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="cash / current_liabilities",
        similar_to=["current_ratio", "quick_ratio"],
        status=FactorStatus.BACKUP,
        ic_mean=0.010,
        ic_ir=0.090,
        cost_threshold_30bp=True,
    ))

    catalog.register(FactorProfile(
        name="current_ratio",
        family=FactorFamily.LEVERAGE,
        description="流动比率",
        economic_mechanism="短期偿债能力",
        hypothesis="高流动比率公司流动性好",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="current_assets / current_liabilities",
        similar_to=["quick_ratio", "cash_ratio"],
        dominates=["quick_ratio"],
        status=FactorStatus.BACKUP,
        ic_mean=0.008,
        ic_ir=0.070,
        cost_threshold_30bp=True,
        notes="已实现，可用于研究候选",
    ))

    catalog.register(FactorProfile(
        name="debt_ratio",
        family=FactorFamily.LEVERAGE,
        description="资产负债率",
        economic_mechanism="杠杆水平",
        hypothesis="低资产负债率公司财务风险低",
        expected_signal=FactorSignal.SHORT,
        lookback=1,
        inputs=('financial_data',),
        computation="total_liabilities / total_assets",
        similar_to=["equity_ratio", "longterm_debt_ratio"],
        status=FactorStatus.BACKUP,
        ic_mean=-0.005,
        ic_ir=-0.040,
        cost_threshold_30bp=False,
        notes="已实现，可用于研究候选",
    ))

    # ---- 质量因子 (Quality) ----
    catalog.register(FactorProfile(
        name="roe_assets",
        family=FactorFamily.QUALITY,
        description="ROE × 总资产 (杜邦分解)",
        economic_mechanism="杜邦分析 - 盈利×资产×杠杆",
        hypothesis="ROE×总资产高的公司综合能力强",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="ROE × total_assets",
        similar_to=["roe", "roa"],
        status=FactorStatus.BACKUP,
        ic_mean=0.015,
        ic_ir=0.130,
        cost_threshold_30bp=True,
        notes="已实现，可用于研究候选",
    ))

    catalog.register(FactorProfile(
        name="roe_change",
        family=FactorFamily.QUALITY,
        description="ROE变化量",
        economic_mechanism="盈利边际改善",
        hypothesis="ROE提升的公司业绩好转",
        expected_signal=FactorSignal.LONG,
        lookback=4,
        inputs=('financial_data',),
        computation="ROE_t - ROE_t-4",
        similar_to=["profit_growth"],
        status=FactorStatus.BACKUP,
        ic_mean=0.015,
        ic_ir=0.130,
        ic_positive_ratio=0.55,
        cost_threshold_30bp=True,
        notes="已实现，可用于研究候选",
    ))

    # ---- 补充财务因子 (Observing) ----
    # 以下因子已实现，状态更新为 backup

    catalog.register(FactorProfile(
        name="net_margin",
        family=FactorFamily.PROFITABILITY,
        description="净利率",
        economic_mechanism="最终盈利能力",
        hypothesis="高净利率公司定价能力强",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="net_profit / revenue",
        similar_to=["operating_margin", "gross_margin"],
        status=FactorStatus.BACKUP,
        ic_mean=0.005,
        ic_ir=0.040,
        cost_threshold_30bp=True,
        notes="已实现，可用于研究候选",
    ))

    catalog.register(FactorProfile(
        name="quick_ratio",
        family=FactorFamily.LEVERAGE,
        description="速动比率",
        economic_mechanism="剔除存货的短期偿债能力",
        hypothesis="高速动比率公司流动性好",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="(current_assets - inventory) / current_liabilities",
        similar_to=["current_ratio", "cash_ratio"],
        status=FactorStatus.BACKUP,
        ic_mean=0.003,
        cost_threshold_30bp=True,
        notes="已实现，可用于研究候选",
    ))

    catalog.register(FactorProfile(
        name="inv_turnover",
        family=FactorFamily.EFFICIENCY,
        description="存货周转率",
        economic_mechanism="存货管理效率",
        hypothesis="高存货周转率公司运营效率高",
        expected_signal=FactorSignal.LONG,
        lookback=1,
        inputs=('financial_data',),
        computation="COGS / inventory",
        status=FactorStatus.BACKUP,
        ic_mean=0.003,
        cost_threshold_30bp=True,
    ))

    # ---- WorldQuant Alpha 基础档案 ----
    # 基于 scripts/full_factor_ic_test.py 2026-04 结果
    # 有效Alpha: alpha_006, alpha_003, alpha_016, alpha_014, alpha_036

    alpha_valid = {
        'alpha_006': {'ic': 0.027, 'ir': 0.261, 'desc': '开盘量价关系'},
        'alpha_003': {'ic': 0.024, 'ir': 0.245, 'desc': '开盘量rank相关'},
        'alpha_016': {'ic': 0.023, 'ir': 0.225, 'desc': '高价成交量关系'},
        'alpha_014': {'ic': 0.020, 'ir': 0.214, 'desc': '量价混合'},
        'alpha_036': {'ic': 0.017, 'ir': 0.184, 'desc': '成交量加权'},
    }

    for alpha_name, info in alpha_valid.items():
        catalog.register(FactorProfile(
            name=alpha_name,
            family=FactorFamily.TECHNICAL_OSCILLATOR,
            description=f"WorldQuant Alpha {alpha_name[6:]}: {info['desc']}",
            economic_mechanism="量价异质信息挖掘",
            hypothesis="量价组合关系预测未来收益",
            expected_signal=FactorSignal.LONG,
            lookback=16,
            inputs=('adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume'),
            computation="WorldQuant公式",
            similar_to=[k for k in alpha_valid.keys() if k != alpha_name],
            status=FactorStatus.BACKUP if info['ic'] >= 0.015 else FactorStatus.OBSERVE,
            ic_mean=info['ic'],
            ic_ir=info['ir'],
            ic_positive_ratio=0.61,
            cost_threshold_30bp=True,
            failure_modes=["参数敏感", "和市场状态相关"],
            regime_sensitivity="震荡市更有效",
        ))

    # 无效Alpha (观察/拒绝)
    alpha_invalid = {
        'alpha_013': {'ic': -0.003, 'desc': '负向'},
        'alpha_018': {'ic': -0.009, 'desc': '负向'},
        'alpha_031': {'ic': -0.012, 'desc': '显著负向'},
        'alpha_032': {'ic': -0.015, 'desc': '显著负向'},
        'alpha_039': {'ic': -0.022, 'desc': '强负向'},
        'alpha_087': {'ic': -0.018, 'desc': '负向'},
        'alpha_088': {'ic': -0.016, 'desc': '负向'},
    }

    for alpha_name, info in alpha_invalid.items():
        catalog.register(FactorProfile(
            name=alpha_name,
            family=FactorFamily.TECHNICAL_OSCILLATOR,
            description=f"WorldQuant Alpha {alpha_name[6:]}: {info['desc']}",
            economic_mechanism="量价关系",
            hypothesis="此Alpha组合预测能力为负",
            expected_signal=FactorSignal.SHORT,
            lookback=16,
            inputs=('adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume'),
            computation="WorldQuant公式",
            status=FactorStatus.REJECT,
            ic_mean=info['ic'],
            ic_ir=0.0,
            cost_threshold_30bp=False,
            failure_modes=["IC为负", "做空成本高"],
        ))

    # 已实现的Alpha (backup)
    for i in [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]:
        alpha_name = f'alpha_{i:03d}'
        catalog.register(FactorProfile(
            name=alpha_name,
            family=FactorFamily.TECHNICAL_OSCILLATOR,
            description=f"WorldQuant Alpha {i}: 已实现",
            economic_mechanism="量价关系",
            hypothesis="量价组合关系预测未来收益",
            expected_signal=FactorSignal.NEUTRAL,
            lookback=16,
            inputs=('adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume'),
            computation="WorldQuant公式",
            status=FactorStatus.BACKUP,
            ic_mean=None,
            notes="需运行 full_factor_ic_test.py 验证",
        ))

    # ---- Pattern 因子 (观察) ----
    pattern_factors = [
        ('ma_diff_5_20', 'TECHNICAL_OSCILLATOR', '5日/20日均线差', 0.001),
        ('ma_diff_20_60', 'TECHNICAL_OSCILLATOR', '20日/60日均线差', 0.001),
        ('price_to_ma20', 'TECHNICAL_PRICE_LEVEL', '价格/20日均线', 0.001),
        ('price_to_ma60', 'TECHNICAL_PRICE_LEVEL', '价格/60日均线', 0.001),
        ('rsi6', 'TECHNICAL_OSCILLATOR', '6日RSI', 0.003),
        ('rsi12', 'TECHNICAL_OSCILLATOR', '12日RSI', 0.001),
        ('rsi24', 'TECHNICAL_OSCILLATOR', '24日RSI', 0.001),
        ('macd_hist', 'TECHNICAL_OSCILLATOR', 'MACD柱', -0.001),
        ('kdj_k9', 'TECHNICAL_OSCILLATOR', 'KDJ K值', 0.001),
        ('cci14', 'TECHNICAL_OSCILLATOR', '14日CCI', 0.001),
    ]

    for name, family, desc, ic in pattern_factors:
        family_enum = FactorFamily[family]
        catalog.register(FactorProfile(
            name=name,
            family=family_enum,
            description=desc,
            economic_mechanism="技术指标",
            hypothesis="技术指标组合预测未来收益",
            expected_signal=FactorSignal.NEUTRAL if ic == 0 else FactorSignal.LONG if ic > 0 else FactorSignal.SHORT,
            lookback=30,
            inputs=('adj_close', 'volume'),
            computation="技术公式",
            status=FactorStatus.OBSERVE if ic == 0 else (FactorStatus.BACKUP if ic > 0.002 else FactorStatus.REJECT),
            ic_mean=ic,
            notes="需验证",
        ))

    # ---- Sentiment/Analyst 因子 ----
    sentiment_analyst_factors = [
        ('industry_money_flow_rank', 'SENTIMENT', '行业资金流排名', 0.002),
        ('money_flow_intensity', 'SENTIMENT', '资金流强度', 0.001),
        ('sector_inflow_rank', 'SENTIMENT', '行业净流入排名', 0.002),
        ('analyst_performance_rank', 'SENTIMENT', '分析师绩效排名', 0.003),
        ('avg_analyst_return', 'SENTIMENT', '分析师平均收益率', 0.002),
        ('sector_analyst_breadth', 'SENTIMENT', '行业分析师覆盖广度', 0.001),
    ]

    for name, family, desc, ic in sentiment_analyst_factors:
        family_enum = FactorFamily[family]
        catalog.register(FactorProfile(
            name=name,
            family=family_enum,
            description=desc,
            economic_mechanism="资金流/分析师行为",
            hypothesis="资金流向和分析师关注度预测股票表现",
            expected_signal=FactorSignal.NEUTRAL,
            lookback=1,
            inputs=('symbol',),
            computation="行业/分析师数据映射",
            status=FactorStatus.BACKUP if ic > 0.002 else FactorStatus.OBSERVE,
            ic_mean=ic,
            notes="需验证IC和稳定性",
        ))

    # ---- Stock-level Money Flow & Research Report 因子 ----
    stock_mf_report_factors = [
        ('main_flow_rank', 'LIQUIDITY', '主力资金流排名', 0.002),
        ('institutional_intensity', 'LIQUIDITY', '机构资金强度', 0.003),
        ('super_flow_mean', 'LIQUIDITY', '超大单资金均值', 0.002),
        ('research_report_count', 'SENTIMENT', '研报数量', 0.001),
        ('avg_pe_2026', 'VALUE', '2026预测PE均值', 0.001),
        ('institution_coverage', 'SENTIMENT', '机构覆盖数量', 0.002),
    ]

    for name, family, desc, ic in stock_mf_report_factors:
        family_enum = FactorFamily[family]
        catalog.register(FactorProfile(
            name=name,
            family=family_enum,
            description=desc,
            economic_mechanism="资金流/研报数据",
            hypothesis="资金流向和研报覆盖预测股票表现",
            expected_signal=FactorSignal.NEUTRAL,
            lookback=1,
            inputs=('symbol',),
            computation="个股资金流/研报数据映射",
            status=FactorStatus.BACKUP if ic > 0.002 else FactorStatus.OBSERVE,
            ic_mean=ic,
            notes="需验证IC和稳定性",
        ))

    # ============ 新增因子 (2026-04-15 研究发现) ============
    # volume_momentum - 成交量动量因子
    catalog.register(FactorProfile(
        name="volume_momentum",
        family=FactorFamily.SENTIMENT,
        description="成交量趋势强度 (20日成交量移动平均变化率)",
        economic_mechanism="量价配合理论",
        hypothesis="成交量持续放大的股票有动量效应",
        expected_signal=FactorSignal.LONG,
        lookback=40,
        inputs=('volume',),
        computation="_volume_momentum(40)",
        similar_to=["volume_momentum_20d", "relative_volume_20d"],
        dominates=[],
        status=FactorStatus.BACKUP,
        ic_mean=0.015,
        ic_ir=0.08,
        cost_threshold_30bp=True,
        failure_modes=["成交量操纵"],
        regime_sensitivity="趋势市有效",
    ))

    # market_breadth - 市场宽度因子
    catalog.register(FactorProfile(
        name="market_breadth",
        family=FactorFamily.SENTIMENT,
        description="市场上涨股票占比 (20日均线)",
        economic_mechanism="市场情绪同步性",
        hypothesis="市场整体强势时个股表现更好",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=21,
        inputs=('pct_chg',),
        computation="_market_breadth(21)",
        similar_to=["market_breadth_20d"],
        dominates=[],
        status=FactorStatus.BACKUP,
        ic_mean=0.012,
        ic_ir=0.06,
        cost_threshold_30bp=True,
        failure_modes=["市场分化时失效"],
        regime_sensitivity="牛市有效",
    ))

    # ===== Behavioral Finance Factors =====
    catalog.register(FactorProfile(
        name="short_term_reversal",
        family=FactorFamily.BEHAVIORAL,
        description="短期反转效应 (5日收益负向)",
        economic_mechanism="投资者过度反应后的价格回归",
        hypothesis="短期上涨过度的股票会反转",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=10,
        inputs=('pct_chg',),
        computation="_short_term_reversal(10)",
        similar_to=["alpha_001", "alpha_004"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=False,
        failure_modes=["高换手成本"],
        regime_sensitivity="高波动市有效",
    ))

    catalog.register(FactorProfile(
        name="medium_term_reversal",
        family=FactorFamily.BEHAVIORAL,
        description="中期反转效应 (20日收益负向)",
        economic_mechanism="动量衰竭后的价格回归",
        hypothesis="中期动量衰竭后会反转",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=25,
        inputs=('pct_chg',),
        computation="_medium_term_reversal(25)",
        similar_to=["alpha_002"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=False,
        failure_modes=["高换手成本"],
        regime_sensitivity="震荡市有效",
    ))

    catalog.register(FactorProfile(
        name="long_term_reversal",
        family=FactorFamily.BEHAVIORAL,
        description="长期反转效应 (60日收益负向)",
        economic_mechanism="长期价格偏离后的价值回归",
        hypothesis="长期下跌股票有更大反弹概率",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=65,
        inputs=('pct_chg',),
        computation="_long_term_reversal(65)",
        similar_to=["alpha_003"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=False,
        failure_modes=["长期持仓风险"],
        regime_sensitivity="熊市末端有效",
    ))

    catalog.register(FactorProfile(
        name="smart_money_flow",
        family=FactorFamily.BEHAVIORAL,
        description="聪明钱流向指标 (上涨日成交量 - 下跌日成交量)",
        economic_mechanism="机构投资者行为",
        hypothesis="聪明钱流入的股票表现更好",
        expected_signal=FactorSignal.LONG,
        lookback=20,
        inputs=('close', 'volume', 'pct_chg'),
        computation="_smart_money_flow(20)",
        similar_to=["institutional_flow"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["数据延迟"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="disposition_effect",
        family=FactorFamily.BEHAVIORAL,
        description="处置效应 (盈利卖出 vs 亏损持有比例)",
        economic_mechanism="投资者心理学偏差",
        hypothesis="处置效应强的股票有更大下跌风险",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=20,
        inputs=('pct_chg', 'volume'),
        computation="_disposition_effect(20)",
        similar_to=[],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["短期波动干扰"],
        regime_sensitivity="市场敏感",
    ))

    catalog.register(FactorProfile(
        name="herding_indicator",
        family=FactorFamily.BEHAVIORAL,
        description="羊群效应指标 (个股收益与市场收益相关性)",
        economic_mechanism="投资者从众行为",
        hypothesis="高羊群效应的股票波动更大",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=20,
        inputs=('pct_chg',),
        computation="_herding_indicator(20)",
        similar_to=[],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["市场整体影响"],
        regime_sensitivity="高相关性市场",
    ))

    catalog.register(FactorProfile(
        name="momentum_divergence",
        family=FactorFamily.BEHAVIORAL,
        description="动量背离 (60日 - 120日动量差)",
        economic_mechanism="动量周期错位",
        hypothesis="中期动量强于长期时趋势持续",
        expected_signal=FactorSignal.LONG,
        lookback=120,
        inputs=('close',),
        computation="_momentum_60_120_divergence(120)",
        similar_to=["relative_momentum"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势反转风险"],
        regime_sensitivity="趋势市有效",
    ))

    # ===== Pattern Recognition Factors =====
    catalog.register(FactorProfile(
        name="body_ratio",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="K线实体占比 (实体/总 range)",
        economic_mechanism="K线形态强度",
        hypothesis="实体占比适中时趋势信号更可靠",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=1,
        inputs=('open', 'high', 'low', 'close'),
        computation="_body_ratio(1)",
        similar_to=["candle_body_ratio"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["无趋势市场"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="upper_shadow_ratio",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="上影线占比",
        economic_mechanism="上涨压力信号",
        hypothesis="上影线长表明上方卖压重",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=1,
        inputs=('open', 'high', 'low', 'close'),
        computation="_upper_shadow_ratio(1)",
        similar_to=["candle_upper_shadow"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["短期波动"],
        regime_sensitivity="震荡市有效",
    ))

    catalog.register(FactorProfile(
        name="lower_shadow_ratio",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="下影线占比",
        economic_mechanism="下跌支撑信号",
        hypothesis="下影线长表明下方支撑强",
        expected_signal=FactorSignal.POSITIVE,
        lookback=1,
        inputs=('open', 'high', 'low', 'close'),
        computation="_lower_shadow_ratio(1)",
        similar_to=["candle_lower_shadow"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["短期波动"],
        regime_sensitivity="超跌反弹有效",
    ))

    catalog.register(FactorProfile(
        name="price_position_20",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="价格在20日高低点间的位置 (0-1)",
        economic_mechanism="价格相对位置",
        hypothesis="价格在高位时趋势可能反转",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=20,
        inputs=('high', 'low', 'close'),
        computation="_price_position_20d(20)",
        similar_to=["close_position", "close_to_high"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势持续性"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="price_position_60",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="价格在60日高低点间的位置 (0-1)",
        economic_mechanism="价格相对位置",
        hypothesis="价格在长期高位时风险积累",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=60,
        inputs=('high', 'low', 'close'),
        computation="_price_position_60d(60)",
        similar_to=["close_position_60d"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势持续性"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="intraday_range",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="日内振幅",
        economic_mechanism="日内波动程度",
        hypothesis="振幅大时趋势信号更强",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=1,
        inputs=('open', 'high', 'low'),
        computation="_intraday_range(1)",
        similar_to=["high_low_range"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["高波动误导"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="volume_climax",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="成交量高潮 (当前/60日均值)",
        economic_mechanism="极端成交量预示转折",
        hypothesis="成交量高潮后趋势可能反转",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=60,
        inputs=('volume',),
        computation="_volume_climax(60)",
        similar_to=["volume_spike"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=False,
        failure_modes=["趋势持续性"],
        regime_sensitivity="趋势末端有效",
    ))

    catalog.register(FactorProfile(
        name="volume_dry_up_60",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="60日成交量枯竭",
        economic_mechanism="量能萎缩预示变盘",
        hypothesis="成交量枯竭后会有方向选择",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=60,
        inputs=('volume',),
        computation="_volume_dry_up_60(60)",
        similar_to=["volume_dry_up"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["方向不确定"],
        regime_sensitivity="震荡市有效",
    ))

    catalog.register(FactorProfile(
        name="narrow_range_20",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="20日窄幅震荡",
        economic_mechanism="波动率压缩",
        hypothesis="窄幅震荡后会有大幅波动",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=20,
        inputs=('high', 'low'),
        computation="_narrow_range_20(20)",
        similar_to=["volatility_compression"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["方向不确定"],
        regime_sensitivity="震荡市有效",
    ))

    catalog.register(FactorProfile(
        name="narrow_range_60",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="60日窄幅震荡",
        economic_mechanism="长期波动率压缩",
        hypothesis="长期窄幅震荡后有大机会",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=60,
        inputs=('high', 'low'),
        computation="_narrow_range_60(60)",
        similar_to=["volatility_compression"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["方向不确定"],
        regime_sensitivity="震荡市有效",
    ))

    catalog.register(FactorProfile(
        name="close_to_high_5",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="收盘价/5日最高价",
        economic_mechanism="短期强势信号",
        hypothesis="收盘接近最高点表明强势",
        expected_signal=FactorSignal.LONG,
        lookback=5,
        inputs=('close',),
        computation="_close_to_high_5d(5)",
        similar_to=["close_to_high"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势反转"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="close_to_low_5",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="收盘价/5日最低价",
        economic_mechanism="短期弱势信号",
        hypothesis="收盘接近最低点表明弱势",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=5,
        inputs=('close',),
        computation="_close_to_low_5d(5)",
        similar_to=["close_to_low"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["反弹风险"],
        regime_sensitivity="超跌反弹有效",
    ))

    catalog.register(FactorProfile(
        name="three_white_soldiers",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="三白兵形态 (连续3日阳线且递涨)",
        economic_mechanism="K线反转形态",
        hypothesis="三白兵是强势上涨信号",
        expected_signal=FactorSignal.LONG,
        lookback=3,
        inputs=('open', 'close'),
        computation="_three_white_soldiers(3)",
        similar_to=["candle_engulf_bullish"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["形态失败"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="three_black_crows",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="三乌鸦形态 (连续3日阴线且递跌)",
        economic_mechanism="K线反转形态",
        hypothesis="三乌鸦是强势下跌信号",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=3,
        inputs=('open', 'close'),
        computation="_three_black_crows(3)",
        similar_to=["candle_engulf_bearish"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["形态失败"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="inside_bar",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="孕线形态 (当日高低点在昨日范围内)",
        economic_mechanism="波动率收缩",
        hypothesis="孕线后可能有突破",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=2,
        inputs=('high', 'low'),
        computation="_inside_bar(2)",
        similar_to=["volatility_compression"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["方向不确定"],
        regime_sensitivity="突破后有效",
    ))

    catalog.register(FactorProfile(
        name="outside_bar",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="外包线 (当日高低点超出昨日范围)",
        economic_mechanism="波动率扩张",
        hypothesis="外包线是趋势加速信号",
        expected_signal=FactorSignal.LONG,
        lookback=2,
        inputs=('high', 'low'),
        computation="_outside_bar(2)",
        similar_to=["breakout"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["假突破"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="tweezer_top",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="平头顶形态",
        economic_mechanism="K线反转形态",
        hypothesis="平头顶是顶部反转信号",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=2,
        inputs=('high', 'pct_chg'),
        computation="_tweezer_top(2)",
        similar_to=["shooting_star"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["形态失败"],
        regime_sensitivity="顶部反转有效",
    ))

    catalog.register(FactorProfile(
        name="tweezer_bottom",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="平头底形态",
        economic_mechanism="K线反转形态",
        hypothesis="平头底是底部反转信号",
        expected_signal=FactorSignal.POSITIVE,
        lookback=2,
        inputs=('low', 'pct_chg'),
        computation="_tweezer_bottom(2)",
        similar_to=["hammer"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["形态失败"],
        regime_sensitivity="底部反转有效",
    ))

    catalog.register(FactorProfile(
        name="candle_dragonfly",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="蜻蜓线 (下影线长的T字线)",
        economic_mechanism="K线反转形态",
        hypothesis="蜻蜓线是底部支撑信号",
        expected_signal=FactorSignal.POSITIVE,
        lookback=1,
        inputs=('open', 'high', 'low', 'close'),
        computation="_candle_dragonfly(1)",
        similar_to=["hammer", "tweezer_bottom"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["形态失败"],
        regime_sensitivity="底部反转有效",
    ))

    catalog.register(FactorProfile(
        name="candle_gravestone",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="墓碑线 (上影线长的倒T字线)",
        economic_mechanism="K线反转形态",
        hypothesis="墓碑线是顶部压力信号",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=1,
        inputs=('open', 'high', 'low', 'close'),
        computation="_candle_gravestone(1)",
        similar_to=["shooting_star", "tweezer_top"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["形态失败"],
        regime_sensitivity="顶部反转有效",
    ))

    catalog.register(FactorProfile(
        name="candle_stealth_doji",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="隐蔽十字星 (实体很小的K线)",
        economic_mechanism="犹豫信号",
        hypothesis="十字星后可能有趋势延续",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=1,
        inputs=('open', 'high', 'low', 'close'),
        computation="_candle_stealth_doji(1)",
        similar_to=["doji", "candle_doji"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["方向不确定"],
        regime_sensitivity="趋势中继有效",
    ))

    catalog.register(FactorProfile(
        name="consecutive_up_5",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="连续上涨天数 (5日)",
        economic_mechanism="动量累积",
        hypothesis="连续上涨后可能反转",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=5,
        inputs=('pct_chg',),
        computation="_consecutive_up_5(5)",
        similar_to=["consecutive_up"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势持续"],
        regime_sensitivity="趋势反转有效",
    ))

    catalog.register(FactorProfile(
        name="consecutive_down_5",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="连续下跌天数 (5日)",
        economic_mechanism="动量累积",
        hypothesis="连续下跌后可能反弹",
        expected_signal=FactorSignal.POSITIVE,
        lookback=5,
        inputs=('pct_chg',),
        computation="_consecutive_down_5(5)",
        similar_to=["consecutive_down"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势持续"],
        regime_sensitivity="超跌反弹有效",
    ))

    catalog.register(FactorProfile(
        name="consecutive_up_10",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="连续上涨天数 (10日)",
        economic_mechanism="中期动量累积",
        hypothesis="连续上涨后反转概率更高",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=10,
        inputs=('pct_chg',),
        computation="_consecutive_up_10(10)",
        similar_to=["consecutive_up"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势持续"],
        regime_sensitivity="趋势反转有效",
    ))

    catalog.register(FactorProfile(
        name="consecutive_down_10",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="连续下跌天数 (10日)",
        economic_mechanism="中期动量累积",
        hypothesis="连续下跌后反弹概率更高",
        expected_signal=FactorSignal.POSITIVE,
        lookback=10,
        inputs=('pct_chg',),
        computation="_consecutive_down_10(10)",
        similar_to=["consecutive_down"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势持续"],
        regime_sensitivity="超跌反弹有效",
    ))

    catalog.register(FactorProfile(
        name="gap_fill_5d",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="5日内缺口回补概率",
        economic_mechanism="缺口理论",
        hypothesis="缺口回补概率高的股票更稳定",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=10,
        inputs=('high', 'low', 'close'),
        computation="_gap_fill_5d(10)",
        similar_to=["gap_size"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["市场联动"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="volume_ratio_up_day",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="上涨日成交量/下跌日成交量 (20日均值)",
        economic_mechanism="量价配合",
        hypothesis="上涨放量、下跌缩量更健康",
        expected_signal=FactorSignal.LONG,
        lookback=20,
        inputs=('volume', 'pct_chg'),
        computation="_volume_ratio_up_day(20)",
        similar_to=["smart_money_flow"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势反转"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="limit_up_count_20",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="20日内涨停次数",
        economic_mechanism="极端强势信号",
        hypothesis="涨停次数多的股票有更强动能",
        expected_signal=FactorSignal.LONG,
        lookback=20,
        inputs=('pct_chg',),
        computation="_limit_up_count_20d(20)",
        similar_to=["limit_up_count"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=False,
        failure_modes=["涨停板流动性风险"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="limit_down_count_20",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="20日内跌停次数",
        economic_mechanism="极端弱势信号",
        hypothesis="跌停次数多的股票有更大风险",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=20,
        inputs=('pct_chg',),
        computation="_limit_down_count_20d(20)",
        similar_to=["limit_down_count"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=False,
        failure_modes=["跌停板流动性风险"],
        regime_sensitivity="下跌市场有效",
    ))

    catalog.register(FactorProfile(
        name="close_to_high_60",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="收盘价/60日最高价",
        economic_mechanism="长期强势信号",
        hypothesis="收盘接近60日高点表明强势",
        expected_signal=FactorSignal.LONG,
        lookback=60,
        inputs=('close',),
        computation="_close_to_high_60d(60)",
        similar_to=["close_to_high"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势反转"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="close_to_low_60",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="收盘价/60日最低价",
        economic_mechanism="长期弱势信号",
        hypothesis="收盘接近60日低点表明弱势",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=60,
        inputs=('close',),
        computation="_close_to_low_60d(60)",
        similar_to=["close_to_low"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["反弹风险"],
        regime_sensitivity="超跌反弹有效",
    ))

    catalog.register(FactorProfile(
        name="high_close_ratio",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="收盘价/20日最高价",
        economic_mechanism="价格相对位置",
        hypothesis="收盘在高位表明强势",
        expected_signal=FactorSignal.LONG,
        lookback=20,
        inputs=('close',),
        computation="_high_close_ratio(20)",
        similar_to=["close_to_high"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势反转"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="low_close_ratio",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="收盘价/20日最低价",
        economic_mechanism="价格相对位置",
        hypothesis="收盘在低位表明弱势",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=20,
        inputs=('close',),
        computation="_low_close_ratio(20)",
        similar_to=["close_to_low"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["反弹风险"],
        regime_sensitivity="超跌反弹有效",
    ))

    catalog.register(FactorProfile(
        name="volume_price_div",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="量价背离 (价格动量 - 成交量动量)",
        economic_mechanism="量价背离理论",
        hypothesis="价涨量跌可能反转",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=20,
        inputs=('close', 'volume'),
        computation="_volume_price_divergence(20)",
        similar_to=["flow_divergence"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势持续"],
        regime_sensitivity="趋势反转有效",
    ))

    catalog.register(FactorProfile(
        name="volume_sync_market",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="个股成交量与市场同步性",
        economic_mechanism="市场联动效应",
        hypothesis="与市场同步性高的股票更稳定",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=20,
        inputs=('volume',),
        computation="_volume_sync_with_market(20)",
        similar_to=["market_breadth"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["市场整体影响"],
        regime_sensitivity="高相关性市场",
    ))

    catalog.register(FactorProfile(
        name="intraday_range_20",
        family=FactorFamily.TECHNICAL_PATTERN,
        description="20日平均日内振幅",
        economic_mechanism="波动率指标",
        hypothesis="振幅大的股票风险更高",
        expected_signal=FactorSignal.NEGATIVE,
        lookback=20,
        inputs=('open', 'high', 'low'),
        computation="_intraday_range_20d(20)",
        similar_to=["intraday_range"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["高波动误导"],
        regime_sensitivity="高波动市",
    ))

    catalog.register(FactorProfile(
        name="earnings_quality",
        family=FactorFamily.BEHAVIORAL,
        description="盈利质量 (上涨收益占比)",
        economic_mechanism="盈利质量分析",
        hypothesis="盈利质量高的股票更可靠",
        expected_signal=FactorSignal.POSITIVE,
        lookback=20,
        inputs=('pct_chg',),
        computation="_earnings_quality(20)",
        similar_to=["roe_stability"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["财务操纵"],
        regime_sensitivity="基本面驱动市场",
    ))

    catalog.register(FactorProfile(
        name="turnover_rate",
        family=FactorFamily.BEHAVIORAL,
        description="换手率 (当前/250日均值)",
        economic_mechanism="交易活跃度",
        hypothesis="高换手率股票波动大",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=250,
        inputs=('volume', 'close'),
        computation="_turnover_rate(250)",
        similar_to=["turnover_rate_5d"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["高换手成本"],
        regime_sensitivity="高换手市场",
    ))

    catalog.register(FactorProfile(
        name="price_depth_20",
        family=FactorFamily.BEHAVIORAL,
        description="价格深度 (20日成交额/均值)",
        economic_mechanism="市场深度分析",
        hypothesis="深度好的股票更稳定",
        expected_signal=FactorSignal.POSITIVE,
        lookback=20,
        inputs=('amount', 'close'),
        computation="_price_depth_20d(20)",
        similar_to=["liquidity_factors"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["流动性风险"],
        regime_sensitivity="低流动性市场",
    ))

    catalog.register(FactorProfile(
        name="vwap_return_20",
        family=FactorFamily.BEHAVIORAL,
        description="VWAP加权收益 (20日)",
        economic_mechanism="成交量加权价格",
        hypothesis="VWAP收益高表明买入时机好",
        expected_signal=FactorSignal.POSITIVE,
        lookback=20,
        inputs=('pct_chg', 'volume'),
        computation="_vwap_return_20d(20)",
        similar_to=["smart_money_flow"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["高换手成本"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="retail_indicator",
        family=FactorFamily.BEHAVIORAL,
        description="散户指标 (低成交量=机构参与)",
        economic_mechanism="资金流向分析",
        hypothesis="低成交量时段机构在吸筹",
        expected_signal=FactorSignal.POSITIVE,
        lookback=20,
        inputs=('volume',),
        computation="_retail_indicator(20)",
        similar_to=["smart_money_flow"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["数据不精确"],
        regime_sensitivity="机构主导市场",
    ))

    catalog.register(FactorProfile(
        name="institutional_flow_20",
        family=FactorFamily.BEHAVIORAL,
        description="机构资金流向 (20日/60日比例)",
        economic_mechanism="资金流向分析",
        hypothesis="机构流入的股票表现更好",
        expected_signal=FactorSignal.LONG,
        lookback=60,
        inputs=('amount',),
        computation="_institutional_flow_20d(60)",
        similar_to=["smart_money_flow"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["数据延迟"],
        regime_sensitivity="资金驱动市场",
    ))

    catalog.register(FactorProfile(
        name="abnormal_vol_return",
        family=FactorFamily.BEHAVIORAL,
        description="异常量价交互 (成交量比值 × 收益)",
        economic_mechanism="量价异常分析",
        hypothesis="异常放量配合收益有预测力",
        expected_signal=FactorSignal.CONTEXT_DEPENDENT,
        lookback=20,
        inputs=('volume', 'pct_chg'),
        computation="_abnormal_vol_return(20)",
        similar_to=["volume_momentum"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["假信号"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="gain_loss_asymmetry",
        family=FactorFamily.BEHAVIORAL,
        description="盈亏不对称 (上涨日均收益 - 下跌日均损失)",
        economic_mechanism="投资者行为偏差",
        hypothesis="盈亏不对称好的股票更优",
        expected_signal=FactorSignal.POSITIVE,
        lookback=20,
        inputs=('pct_chg',),
        computation="_gain_loss_asymmetry(20)",
        similar_to=["earnings_quality"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["高波动干扰"],
        regime_sensitivity="趋势市有效",
    ))

    catalog.register(FactorProfile(
        name="up_down_volume_ratio",
        family=FactorFamily.BEHAVIORAL,
        description="上涨日成交量/下跌日成交量",
        economic_mechanism="量价配合",
        hypothesis="上涨放量、下跌缩量更健康",
        expected_signal=FactorSignal.LONG,
        lookback=20,
        inputs=('volume', 'pct_chg'),
        computation="_up_down_volume_ratio(20)",
        similar_to=["volume_ratio_up_day", "smart_money_flow"],
        dominates=[],
        status=FactorStatus.POOL,
        ic_mean=None,
        ic_ir=None,
        cost_threshold_30bp=True,
        failure_modes=["趋势反转"],
        regime_sensitivity="趋势市有效",
    ))

    return catalog


def get_research_pool_factors(catalog: FactorCatalog) -> list[str]:
    """从档案获取研究池因子列表"""
    return [p.name for p in catalog.get_research_pool()]


def get_filtered_pool(
    catalog: FactorCatalog,
    min_ic: float = 0.005,
    min_ir: float = 0.0,
    require_cost_threshold: bool = True,
    exclude_families: list[FactorFamily] = None,
) -> list[str]:
    """根据条件筛选因子池"""
    factors = []
    exclude_families = exclude_families or []

    for p in catalog.get_research_pool():
        # 家族过滤
        if p.family in exclude_families:
            continue

        # IC过滤
        if p.ic_mean and p.ic_mean < min_ic:
            continue

        # IR过滤
        if p.ic_ir and p.ic_ir < min_ir:
            continue

        # 成本过滤
        if require_cost_threshold and not p.cost_threshold_30bp:
            continue

        factors.append(p.name)

    return factors


def print_factor_families_summary(catalog: FactorCatalog):
    """打印家族汇总"""
    print("\n" + "=" * 80)
    print("因子家族汇总")
    print("=" * 80)

    for family in FactorFamily:
        profiles = catalog.get_by_family(family)
        if not profiles:
            continue

        # 家族内因子数
        total = len(profiles)
        core = len([p for p in profiles if p.status == FactorStatus.CORE])
        backup = len([p for p in profiles if p.status == FactorStatus.BACKUP])
        observe = len([p for p in profiles if p.status == FactorStatus.OBSERVE])
        reject = len([p for p in profiles if p.status == FactorStatus.REJECT])

        # 平均IC
        valid_ic = [p.ic_mean for p in profiles if p.ic_mean is not None]
        avg_ic = sum(valid_ic) / len(valid_ic) if valid_ic else 0

        print(f"\n【{family.value.upper()}】{total}个因子 (核心:{core} 备用:{backup} 观察:{observe} 拒绝:{reject})")
        print(f"  平均IC: {avg_ic:.4f}")

        # 列出核心和备用
        for p in profiles:
            if p.status in [FactorStatus.CORE, FactorStatus.BACKUP]:
                ic_str = f"IC={p.ic_mean:.3f}" if p.ic_mean else "IC=N/A"
                ir_str = f"IR={p.ic_ir:.2f}" if p.ic_ir else ""
                cost_str = "✅" if p.cost_threshold_30bp else "❌"
                print(f"    {p.status.value}: {p.name:<20} {ic_str:<12} {ir_str:<10} 成本30bp:{cost_str}")

    print("\n" + "=" * 80)
