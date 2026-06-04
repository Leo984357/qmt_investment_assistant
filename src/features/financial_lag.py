"""
财务因子lag处理模块

问题：财务数据必须lag，否则可能藏未来函数
- 财报发布日期：4月底(Q1)、8月底(Q2)、10月底(Q3)、次年4月底(Q4)
- 实际可交易日期：发布日期 + N天

修复方案：
1. 对每个财务因子，记录其"报告期"
2. 在使用时lag一个季度（60天）
3. 季度内使用相同值
"""


import pandas as pd

# 财报发布规则
REPORT_SCHEDULE = {
    'Q1': {'month': 4, 'max_day': 30},   # 4月30日前
    'Q2': {'month': 8, 'max_day': 31},   # 8月31日前
    'Q3': {'month': 10, 'max_day': 31},  # 10月31日前
    'Q4': {'month': 4, 'max_day': 30},   # 次年4月30日前（年报）
}


def get_financial_report_period(trade_date: pd.Timestamp) -> str:
    """
    根据交易日期判断当前可用的财报期
    
    Args:
        trade_date: 交易日期
    
    Returns:
        可用财报期，如 '2024Q1'
    """
    year = trade_date.year
    month = trade_date.month
    day = trade_date.day

    # 判断当前处于哪个财报周期
    if month < 4:
        # 1-3月：只能用到去年的Q3（10月发布）
        return f"{year-1}Q3"
    elif month < 8:
        # 4-7月：只能用到去年的Q4（次年4月发布）
        return f"{year-1}Q4"
    elif month < 10:
        # 8-9月：可以用到今年的Q1
        return f"{year}Q1"
    else:
        # 10-12月：可以用到今年的Q2
        return f"{year}Q2"


def lag_financial_data(
    df: pd.DataFrame,
    financial_cols: list[str],
    lag_days: int = 60,
    report_period_col: str = None
) -> pd.DataFrame:
    """
    对财务数据进行lag处理
    
    Args:
        df: 包含trade_date和财务因子的DataFrame
        financial_cols: 财务因子列名
        lag_days: lag天数（默认60天=一个季度）
        report_period_col: 财报期列名（可选）
    
    Returns:
        lag处理后的DataFrame
    """
    df = df.copy()

    # 按日期排序
    df = df.sort_values(['symbol', 'trade_date'])

    # 对每个财务因子进行lag
    for col in financial_cols:
        if col not in df.columns:
            continue

        print(f"  处理 {col}...")

        # lag N天
        df[f'{col}_lagged'] = df.groupby('symbol')[col].shift(lag_days)

        # 季度内填充：同一个季度内使用同一值
        if report_period_col and report_period_col in df.columns:
            df[f'{col}_filled'] = df.groupby(['symbol', report_period_col])[f'{col}_lagged'].fillna(method='ffill')
        else:
            df[f'{col}_filled'] = df[f'{col}_lagged'].fillna(method='ffill')

        # 替换原列
        df[col] = df[f'{col}_filled']

        # 删除临时列
        df.drop([f'{col}_lagged', f'{col}_filled'], axis=1, inplace=True)

    return df


def add_report_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加财报期列
    
    Args:
        df: 包含trade_date的DataFrame
    
    Returns:
        添加了report_period列的DataFrame
    """
    df = df.copy()
    df['report_period'] = df['trade_date'].apply(get_financial_report_period)
    return df


def validate_financial_lag(df: pd.DataFrame, financial_cols: list[str]) -> dict:
    """
    验证财务数据lag处理是否正确
    
    Args:
        df: 处理后的DataFrame
        financial_cols: 财务因子列名
    
    Returns:
        验证报告
    """
    report = {
        'total_rows': len(df),
        'financial_cols': financial_cols,
        'issues': [],
        'warnings': [],
    }

    for col in financial_cols:
        if col not in df.columns:
            report['issues'].append(f"{col} not found in dataframe")
            continue

        # 检查是否有未来值
        lagged = df.groupby('symbol')[col].shift(-1)
        future_diff = (df[col] - lagged).abs().mean()

        if future_diff > 0.01:  # 如果未来值变化超过1%，可能有未来函数
            report['warnings'].append(
                f"{col}: 平均未来变化={future_diff:.4f}，可能存在未来函数"
            )

        # 检查覆盖率
        coverage = df[col].notna().mean()
        if coverage < 0.5:
            report['warnings'].append(
                f"{col}: 覆盖率={coverage:.1%}，过低"
            )

    return report


# ==================== 示例用法 ====================

if __name__ == '__main__':
    # 示例
    print("财务因子Lag处理模块")
    print("="*50)
    print("\n财报发布规则:")
    for q, info in REPORT_SCHEDULE.items():
        print(f"  {q}: {info['month']}月{info['max_day']}日前")

    print("\n使用示例:")
    print("""
    from src.features.financial_lag import lag_financial_data, add_report_period
    
    # 添加财报期
    df = add_report_period(df)
    
    # lag处理
    financial_cols = ['roe', 'earnings_yield', 'operating_margin', ...]
    df = lag_financial_data(df, financial_cols, lag_days=60)
    
    # 验证
    report = validate_financial_lag(df, financial_cols)
    print(report)
    """)
