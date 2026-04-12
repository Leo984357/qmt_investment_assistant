"""
特征集定义 - 修复版 (剔除错误的alpha_004)

修复：
- alpha_004简化版与原版IC差异巨大（0.14 vs -0.001）
- 原版alpha_004 IC接近0，无效
- 从候选池中剔除

时间: 2026-04-12
"""

FEATURE_SET_V2 = {
    'name': 'feature_set_v2_corrected',
    'description': '修复版特征集，剔除错误的alpha_004',

    # 主因子 - 按IC IR排序（修复后）
    'production_factors': [
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

    # 剔除的因子
    'removed_factors': [
        {
            'name': 'alpha_004',
            'reason': '简化版IC=0.14 vs 原版IC=-0.001，简化版完全错误',
            'original_ic': 0.146,
            'corrected_ic': -0.001,
        }
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
        'add_missing_flag': True,
        'fillna_method': 'median',
    },

    # 验证配置
    'validation': {
        'ic_threshold': 0.01,
        'ir_threshold': 0.05,
        'corr_threshold': 0.7,
    },

    # 财务因子lag要求
    'financial_lag': {
        'description': '财务数据必须lag一个季度',
        'lag_days': 60,
        'method': 'as_reported_date',
    },
}

# 简化版因子列表（仅名称）
FACTOR_NAMES = [f['name'] for f in FEATURE_SET_V2['production_factors']]

if __name__ == '__main__':
    import json
    print("特征集 V2 (修复版)")
    print("="*50)
    print(f"名称: {FEATURE_SET_V2['name']}")
    print(f"因子数量: {len(FEATURE_SET_V2['production_factors'])}")
    print(f"\n因子列表:")
    for f in FEATURE_SET_V2['production_factors']:
        print(f"  - {f['name']}: IC={f['ic_mean']:.3f}, IR={f['ic_ir']:.3f}")
    
    print(f"\n⚠️ 剔除因子:")
    for f in FEATURE_SET_V2['removed_factors']:
        print(f"  - {f['name']}: {f['reason']}")
