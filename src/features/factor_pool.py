"""
统一因子库 - 整合学术、Barra、WorldQuant、行业、形态学因子

使用方式:
from src.features.factor_pool import get_all_factors, get_factors_by_source, print_factor_pool_summary
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class UnifiedFactor:
    """统一因子定义"""
    name: str
    source: str
    category: str
    sub_category: str
    description: str
    economic_interpretation: str
    lookback: int
    data_requirement: list[str]
    formula: Optional[str] = None
    paper_reference: Optional[str] = None


def build_unified_factor_pool() -> list[UnifiedFactor]:
    """构建统一因子池"""
    factors = []
    
    # ===== 学术因子 (Academic Factors) =====
    from src.features.academic_factors import ACADEMIC_FACTORS
    
    for f in ACADEMIC_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="academic",
            category="academic",
            sub_category=f.category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=[],  # 需要根据具体实现确定
            formula=f.computation,
            paper_reference=f.source_paper,
        ))
    
    # ===== Barra因子 =====
    from src.features.barra_factors import BARRA_STYLE_FACTORS
    
    for f in BARRA_STYLE_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="barra",
            category="barra_style",
            sub_category=f.category,
            description=f.description,
            economic_interpretation=f.formula,
            lookback=f.long_lookback,
            data_requirement=f.data_requirement,
            paper_reference="MSCI Barra GEM",
        ))
    
    # ===== WorldQuant Alpha =====
    from src.features.worldquant_alphas import WORLDQUANT_ALPHAS
    
    for a in WORLDQUANT_ALPHAS:
        factors.append(UnifiedFactor(
            name=a.name,
            source="worldquant",
            category="worldquant",
            sub_category=a.category,
            description=a.description,
            economic_interpretation=a.formula,
            lookback=a.lookback,
            data_requirement=a.data_requirement,
            paper_reference="Kakushadze 2016",
        ))
    
    # ===== 行业因子 =====
    from src.features.sector_factors import SECTOR_FACTORS
    
    for f in SECTOR_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="sector",
            category="sector",
            sub_category=f.category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
        ))
    
    # ===== 形态学因子 =====
    from src.features.pattern_factors import PATTERN_FACTORS
    
    for f in PATTERN_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="pattern",
            category="pattern",
            sub_category=f.category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
        ))
    
    # ===== 扩展财务因子 =====
    from src.features.extended_financial_factors import EXTENDED_FINANCIAL_FACTORS
    
    for f in EXTENDED_FINANCIAL_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="extended_financial",
            category="extended_financial",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.computation,
            paper_reference=f.paper_reference,
        ))
    
    # ===== 扩展技术因子 =====
    from src.features.extended_technical_factors import EXTENDED_TECHNICAL_FACTORS
    
    for f in EXTENDED_TECHNICAL_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="extended_technical",
            category="extended_technical",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.formula,
        ))
    
    # ===== 分析师预期因子 =====
    from src.features.analyst_expectation_factors import ANALYST_FACTORS
    
    for f in ANALYST_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="analyst",
            category="analyst",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.formula,
        ))
    
    # ===== 资金流因子 =====
    from src.features.money_flow_factors import MONEY_FLOW_FACTORS
    
    for f in MONEY_FLOW_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="money_flow",
            category="money_flow",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.formula,
        ))
    
    # ===== 情绪因子 =====
    from src.features.sentiment_factors import SENTIMENT_FACTORS
    
    for f in SENTIMENT_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="sentiment",
            category="sentiment",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.formula,
        ))
    
    # ===== 宏观因子 =====
    from src.features.macro_factors import MACRO_FACTORS
    
    for f in MACRO_FACTORS:
        factors.append(UnifiedFactor(
            name=f.name,
            source="macro",
            category="macro",
            sub_category=f.sub_category,
            description=f.description,
            economic_interpretation=f.economic_interpretation,
            lookback=f.lookback,
            data_requirement=f.data_requirement,
            formula=f.formula,
        ))
    
    return factors


def get_all_factors() -> list[UnifiedFactor]:
    """获取所有因子"""
    return build_unified_factor_pool()


def get_factors_by_source(source: str) -> list[UnifiedFactor]:
    """按来源获取因子"""
    return [f for f in build_unified_factor_pool() if f.source == source]


def get_factors_by_category(category: str) -> list[UnifiedFactor]:
    """按类别获取因子"""
    return [f for f in build_unified_factor_pool() if f.category == category]


def inventory() -> pd.DataFrame:
    """导出因子清单"""
    factors = build_unified_factor_pool()
    rows = []
    for f in factors:
        rows.append({
            'name': f.name,
            'source': f.source,
            'category': f.category,
            'sub_category': f.sub_category,
            'description': f.description,
            'lookback': f.lookback,
            'data_requirement': ', '.join(f.data_requirement),
            'formula': f.formula or '',
            'paper': f.paper_reference or '',
        })
    return pd.DataFrame(rows)


def print_factor_pool_summary():
    """打印因子池汇总"""
    factors = build_unified_factor_pool()
    
    print("=" * 100)
    print("统一因子库汇总")
    print("=" * 100)
    
    # 按来源统计
    sources = {}
    for f in factors:
        if f.source not in sources:
            sources[f.source] = []
        sources[f.source].append(f)
    
    print("\n【一、按来源分类】")
    for source, f_list in sorted(sources.items(), key=lambda x: -len(x[1])):
        print(f"  {source:<15}: {len(f_list)}个因子")
    
    # 按大类统计
    categories = {}
    for f in factors:
        if f.category not in categories:
            categories[f.category] = []
        categories[f.category].append(f)
    
    print("\n【二、按大类分类】")
    for cat, f_list in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<25}: {len(f_list)}个因子")
    
    # 详细列表
    print("\n【三、详细因子列表】")
    for source in ['academic', 'barra', 'worldquant', 'sector', 'pattern']:
        if source in sources:
            print(f"\n  === {source.upper()} ===")
            f_list = sources[source]
            
            # 按子类别分组
            sub_cats = {}
            for f in f_list:
                if f.sub_category not in sub_cats:
                    sub_cats[f.sub_category] = []
                sub_cats[f.sub_category].append(f)
            
            for sub, s_list in sorted(sub_cats.items()):
                print(f"    [{sub}] {len(s_list)}个")
                for f in s_list[:3]:  # 只显示前3个
                    print(f"      - {f.name}: {f.description[:50]}")
                if len(s_list) > 3:
                    print(f"      ... 还有{len(s_list)-3}个")
    
    print("\n" + "=" * 100)
    print(f"总计: {len(factors)}个因子")
    print("=" * 100)


def get_data_requirement_summary() -> pd.DataFrame:
    """数据需求汇总"""
    factors = build_unified_factor_pool()
    
    all_requirements = {}
    for f in factors:
        for req in f.data_requirement:
            if req not in all_requirements:
                all_requirements[req] = 0
            all_requirements[req] += 1
    
    df = pd.DataFrame([
        {'data_requirement': k, 'factor_count': v}
        for k, v in sorted(all_requirements.items(), key=lambda x: -x[1])
    ])
    return df


if __name__ == "__main__":
    print_factor_pool_summary()
    
    print("\n")
    df = get_data_requirement_summary()
    print("【数据需求汇总】")
    print(df.to_string(index=False))
