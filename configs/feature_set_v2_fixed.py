"""
特征集定义 - 固定版 (无泄漏)

原则：
1. 特征选择必须在训练窗口内完成，不能用全量数据
2. 但特征集本身可以基于历史经验预先定义
3. 本文件定义的是"候选池"，实际使用需在每个训练窗口内筛选

来源：基于2026-04-12完整IC测试结果
- alpha_004: IC=0.146, IR=0.900 (最强)
- alpha_017: IC=0.099, IR=0.671
- earnings_yield: IC=0.037, IR=0.338
- roe: IC=0.036, IR=0.320
- alpha_006: IC=0.036, IR=0.315
- alpha_002: IC=0.029, IR=0.275
- ocf_per_share: IC=0.031, IR=0.256
- operating_margin: IC=0.022, IR=0.256
- asset_turnover: IC=0.016, IR=0.171
- equity_growth: IC=0.016, IR=0.162

去冗余：剔除alpha_003（与alpha_006相关性0.97）
"""

FEATURE_SET_V2 = {
    'name': 'feature_set_v2_fixed',
    'description': '固定特征集，基于历史IC测试结果，不含未来信息泄漏',
    
    # 主因子 - 按IC IR排序
    'production_factors': [
        {
            'name': 'alpha_004',
            'source': 'worldquant',
            'ic_ir': 0.900,
            'ic_mean': 0.146,
            'category': 'price_volume',
            'description': '(-1 * Ts_Rank(rank(low), 9)) - 简化版'
        },
        {
            'name': 'alpha_017',
            'source': 'worldquant',
            'ic_ir': 0.671,
            'ic_mean': 0.099,
            'category': 'price_volume',
            'description': '复合动量因子'
        },
        {
            'name': 'earnings_yield',
            'source': 'financial',
            'ic_ir': 0.338,
            'ic_mean': 0.037,
            'category': 'value',
            'description': '盈利收益率 = 1/PE_ttm'
        },
        {
            'name': 'roe',
            'source': 'financial',
            'ic_ir': 0.320,
            'ic_mean': 0.036,
            'category': 'profitability',
            'description': '净资产收益率'
        },
        {
            'name': 'alpha_006',
            'source': 'worldquant',
            'ic_ir': 0.315,
            'ic_mean': 0.036,
            'category': 'price_volume',
            'description': '(-1 * correlation(open, volume, 10))'
        },
        {
            'name': 'alpha_002',
            'source': 'worldquant',
            'ic_ir': 0.275,
            'ic_mean': 0.029,
            'category': 'price_volume',
            'description': '(-1 * correlation(rank(delta(log(volume), 2)), rank(ret), 6))'
        },
        {
            'name': 'ocf_per_share',
            'source': 'financial',
            'ic_ir': 0.256,
            'ic_mean': 0.031,
            'category': 'cash_flow',
            'description': '每股经营现金流'
        },
        {
            'name': 'operating_margin',
            'source': 'financial',
            'ic_ir': 0.256,
            'ic_mean': 0.022,
            'category': 'profitability',
            'description': '营业利润率'
        },
        {
            'name': 'asset_turnover',
            'source': 'financial',
            'ic_ir': 0.171,
            'ic_mean': 0.016,
            'category': 'efficiency',
            'description': '资产周转率'
        },
        {
            'name': 'equity_growth',
            'source': 'financial',
            'ic_ir': 0.162,
            'ic_mean': 0.016,
            'category': 'growth',
            'description': '净资产增长率'
        },
    ],
    
    # 预处理规则
    'preprocessing': {
        'winsorize': {
            'financial': [0.01, 0.99],
            'worldquant': [0.05, 0.95],
        },
        'neutralize': {
            'financial': 'industry',
            'worldquant': 'market_cap',
        },
        'add_missing_flag': True,  # 修复：标记缺失值
        'fillna_method': 'median',  # 修复：使用中位数而非0
    },
    
    # 数据要求
    'data_requirements': {
        'min_coverage': 0.5,  # 至少50%的股票有数据
        'min_history': 60,      # 至少60天历史
    },
    
    # 验证配置
    'validation': {
        'ic_threshold': 0.01,     # IC均值需>0.01
        'ir_threshold': 0.05,      # IC IR需>0.05
        'corr_threshold': 0.7,     # 相关性阈值
    },
    
    # 已知风险
    'risks': [
        {
            'factor': 'alpha_004',
            'risk': '简化版定义，非原版WorldQuant公式',
            'mitigation': '需验证简化版与原版IC差异'
        },
        {
            'factor': 'alpha_017',
            'risk': '复合因子，解释性差',
            'mitigation': '保持监控'
        },
        {
            'factor': 'financial_factors',
            'risk': '可能有时点延迟，需严格lag',
            'mitigation': '使用PIT数据，lag一个季度'
        },
    ]
}

# 简化版因子列表（仅名称）
FACTOR_NAMES = [f['name'] for f in FEATURE_SET_V2['production_factors']]

if __name__ == '__main__':
    import json
    print("特征集 V2 (固定版)")
    print("="*50)
    print(f"名称: {FEATURE_SET_V2['name']}")
    print(f"因子数量: {len(FEATURE_SET_V2['production_factors'])}")
    print(f"\n因子列表:")
    for f in FEATURE_SET_V2['production_factors']:
        print(f"  - {f['name']}: IC={f['ic_mean']:.3f}, IR={f['ic_ir']:.3f}")
    
    print(f"\n预处理规则:")
    print(f"  - winsorize: {FEATURE_SET_V2['preprocessing']['winsorize']}")
    print(f"  - add_missing_flag: {FEATURE_SET_V2['preprocessing']['add_missing_flag']}")
    print(f"  - fillna_method: {FEATURE_SET_V2['preprocessing']['fillna_method']}")
