"""
因子档案系统 - 因子池的"资产管理系统"

目标：
1. 为每个因子建立档案（定义、机制、观测时点、冗余性）
2. 按家族分类因子
3. 识别近似替代品
4. 支持Research Pool筛选
"""
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class FactorFamily(Enum):
    """因子家族分类"""
    MOMENTUM = "momentum"           # 动量
    REVERSAL = "reversal"           # 反转
    VOLATILITY = "volatility"       # 波动率
    LIQUIDITY = "liquidity"         # 流动性
    TECHNICAL_PRICE_LEVEL = "technical_price_level"  # 价格位置
    TECHNICAL_OSCILLATOR = "technical_oscillator"    # 技术震荡
    DISTRIBUTION = "distribution"    # 分布特征
    SENTIMENT = "sentiment"         # 情绪/资金流


class FactorStatus(Enum):
    """因子状态"""
    CORE = "core"                   # 核心池 - IC>0.01, IR>0.1, 稳定3年+
    BACKUP = "backup"              # 备用池 - IC>0.005
    OBSERVE = "observe"             # 观察池 - IC>0
    REJECT = "reject"               # 拒绝池 - IC<=0
    DEPRECATED = "deprecated"       # 已退役


class FactorSignal(Enum):
    """因子信号方向"""
    LONG = "long"                  # 预期未来收益为正
    SHORT = "short"                # 预期未来收益为负
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
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    ic_ir: Optional[float] = None
    ic_positive_ratio: Optional[float] = None
    
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
    
    def get(self, name: str) -> Optional[FactorProfile]:
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
    
    # 短周期动量 - 观察/拒绝
    for w in [3, 5, 10, 20, 30, 60, 90]:
        status = FactorStatus.REJECT if w <= 60 else FactorStatus.OBSERVE
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
            dominates=[f"mom250"] if w < 250 else [],
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
            status=FactorStatus.OBSERVE if w >= 5 else FactorStatus.REJECT,
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
            status=FactorStatus.OBSERVE,
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
            status=FactorStatus.OBSERVE,
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
            status=FactorStatus.OBSERVE,
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
                status=FactorStatus.OBSERVE if w >= 60 else FactorStatus.REJECT,
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
            status=FactorStatus.OBSERVE,
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
    
    for w in [12, 24]:
        catalog.register(FactorProfile(
            name=f"rsi{w}",
            family=FactorFamily.TECHNICAL_OSCILLATOR,
            description=f"{w}日RSI",
            economic_mechanism="超买超卖",
            hypothesis="RSI低预示反弹",
            expected_signal=FactorSignal.LONG,
            lookback=w * 2,
            inputs=('adj_close',),
            computation="_rsi(w)",
            dominates=["rsi6"],
            status=FactorStatus.OBSERVE,
            ic_mean=0.001,
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
            status=FactorStatus.OBSERVE,
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
            status=FactorStatus.OBSERVE,
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
            status=FactorStatus.OBSERVE,
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
            status=FactorStatus.OBSERVE,
            ic_mean=0.001,
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
