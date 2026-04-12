"""
数据预处理模块 - 修复版

修复：
1. 添加缺失值标记 (add_missing_flag)
2. 使用中位数填充而非0
3. 记录填充前后对比
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class FeaturePreprocessor:
    """特征预处理器 - 修复缺失值问题"""
    
    def __init__(self, add_missing_flag: bool = True, fillna_method: str = 'median'):
        self.add_missing_flag = add_missing_flag
        self.fillna_method = fillna_method
        self.fill_values: Dict[str, float] = {}  # 存储填充值
        self.missing_flags: List[str] = []  # 缺失标记列名
    
    def fit(self, df: pd.DataFrame, features: List[str]) -> 'FeaturePreprocessor':
        """
        在训练数据上拟合预处理参数
        
        Args:
            df: 训练数据
            features: 特征列名
        """
        self.fill_values = {}
        self.missing_flags = []
        
        for col in features:
            if col not in df.columns:
                continue
            
            # 记录缺失值比例
            missing_ratio = df[col].isna().mean()
            
            # 计算填充值
            if self.fillna_method == 'median':
                fill_val = df[col].median()
            elif self.fillna_method == 'mean':
                fill_val = df[col].mean()
            elif self.fillna_method == 'zero':
                fill_val = 0
            else:
                fill_val = 0
            
            # 处理NaN
            if pd.isna(fill_val):
                fill_val = 0
            
            self.fill_values[col] = fill_val
            
            # 添加缺失标记列
            if self.add_missing_flag:
                missing_col = f'{col}_missing'
                self.missing_flags.append(missing_col)
        
        return self
    
    def transform(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        对数据进行预处理
        
        Args:
            df: 待处理数据
            features: 特征列名
        
        Returns:
            处理后的数据副本
        """
        df = df.copy()
        new_features = list(features)
        
        for col in features:
            if col not in df.columns:
                continue
            
            # 添加缺失标记
            if self.add_missing_flag:
                missing_col = f'{col}_missing'
                df[missing_col] = df[col].isna().astype(int)
                new_features.append(missing_col)
            
            # 填充缺失值
            if col in self.fill_values:
                fill_val = self.fill_values[col]
                df[col] = df[col].fillna(fill_val)
                
                # 替换无穷值
                df[col] = df[col].replace([np.inf, -np.inf], fill_val)
        
        return df, new_features
    
    def fit_transform(self, df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """拟合并转换"""
        self.fit(df, features)
        return self.transform(df, features)
    
    def get_fill_values(self) -> Dict[str, float]:
        """获取填充值"""
        return self.fill_values.copy()
    
    def get_missing_columns(self) -> List[str]:
        """获取缺失标记列名"""
        return self.missing_flags.copy()
    
    def report(self) -> Dict:
        """生成预处理报告"""
        return {
            'fill_values': self.fill_values,
            'missing_flags': self.missing_flags,
            'add_missing_flag': self.add_missing_flag,
            'fillna_method': self.fillna_method,
        }


def preprocess_features(
    df: pd.DataFrame,
    features: List[str],
    add_missing_flag: bool = True,
    fillna_method: str = 'median',
) -> Tuple[pd.DataFrame, List[str], FeaturePreprocessor]:
    """
    预处理特征的便捷函数
    
    Returns:
        (处理后的数据, 处理后的特征列表, 预处理器)
    """
    preprocessor = FeaturePreprocessor(
        add_missing_flag=add_missing_flag,
        fillna_method=fillna_method
    )
    
    return preprocessor.fit_transform(df, features), preprocessor


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    对序列进行缩尾处理
    
    Args:
        s: 输入序列
        lower: 下界百分位
        upper: 上界百分位
    """
    lower_val = s.quantile(lower)
    upper_val = s.quantile(upper)
    return s.clip(lower=lower_val, upper=upper_val)


def neutralize_factor(
    factor: pd.Series,
    industry: pd.Series = None,
    market_cap: pd.Series = None,
) -> pd.Series:
    """
    对因子进行中性化
    
    Args:
        factor: 因子值
        industry: 行业分类（可选）
        market_cap: 市值（可选）
    """
    factor = factor.copy()
    
    if industry is not None:
        # 行业中性化
        for ind in industry.unique():
            mask = industry == ind
            factor.loc[mask] = factor.loc[mask] - factor.loc[mask].mean()
    
    if market_cap is not None:
        # 市值中性化（回归残差）
        valid = ~(factor.isna() | market_cap.isna())
        if valid.sum() > 30:
            from sklearn.linear_model import LinearRegression
            X = market_cap.loc[valid].values.reshape(-1, 1)
            y = factor.loc[valid].values
            lr = LinearRegression()
            lr.fit(X, y)
            factor.loc[valid] = y - lr.predict(X)
    
    return factor


if __name__ == '__main__':
    # 测试
    test_df = pd.DataFrame({
        'factor1': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
        'factor2': [np.nan, 2, 3, 4, np.nan, 6, 7, 8, 9, 10],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })
    
    print("原始数据:")
    print(test_df)
    print()
    
    # 预处理
    features = ['factor1', 'factor2']
    processed, final_features, preprocessor = preprocess_features(
        test_df, features, add_missing_flag=True, fillna_method='median'
    )
    
    print("预处理后:")
    print(processed)
    print()
    
    print("填充值:")
    print(preprocessor.get_fill_values())
    print()
    
    print("缺失标记列:")
    print(preprocessor.get_missing_columns())
