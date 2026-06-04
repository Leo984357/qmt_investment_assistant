"""
IC Decay分析和因子中性化

1. IC Decay: 不同持有期的IC曲线 + 半衰期估计
2. Neutralization: raw / industry-neutral / size-neutral / 双重中性
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class ICDecayResult:
    """IC Decay分析结果"""
    factor_name: str

    # 不同持有期的IC
    ic_by_horizon: dict[int, float]  # {5: 0.02, 10: 0.03, 20: 0.025, 40: 0.015}

    # 半衰期
    half_life: int  # 天

    # Decay速率
    decay_rate: float

    # 推荐调仓周期
    recommended_rebalance: int

    # 置信度
    confidence: float  # 0-1

    def to_dict(self) -> dict:
        return {
            'factor': self.factor_name,
            'ic_decay': self.ic_by_horizon,
            'half_life': self.half_life,
            'decay_rate': self.decay_rate,
            'recommended_rebalance': self.recommended_rebalance,
            'confidence': self.confidence,
        }


class ICDecayAnalyzer:
    """
    IC Decay分析器
    
    用法:
    analyzer = ICDecayAnalyzer()
    
    # 分析因子在不同持有期的IC
    result = analyzer.analyze(
        factor_data,
        factor_name='roe',
        horizons=[1, 5, 10, 20, 40, 60],
    )
    
    print(f"半衰期: {result.half_life}天")
    print(f"推荐调仓周期: {result.recommended_rebalance}天")
    """

    def __init__(self):
        self._cache: dict[str, ICDecayResult] = {}

    def analyze(
        self,
        data: pd.DataFrame,
        factor_name: str,
        horizons: list[int] = [1, 5, 10, 20, 40, 60],
        date_col: str = 'trade_date',
        label_prefix: str = 'fwd_return_',
    ) -> ICDecayResult:
        """
        分析因子在不同持有期的IC
        
        Args:
            data: 包含日期、因子值、标签的DataFrame
            factor_name: 因子名
            horizons: 持有期列表
            date_col: 日期列名
            label_prefix: 标签列名前缀
            
        Returns:
            ICDecayResult
        """
        ic_by_horizon = {}

        for h in horizons:
            label_col = f'{label_prefix}{h}d'

            if label_col in data.columns:
                # 计算IC
                valid = data[[date_col, factor_name, label_col]].dropna()
                if len(valid) > 100:
                    ic_series = valid.groupby(date_col).apply(
                        lambda x: x[factor_name].corr(x[label_col], method='spearman')
                    ).dropna()
                    ic_by_horizon[h] = ic_series.mean()

        # 计算半衰期
        half_life, decay_rate = self._estimate_half_life(ic_by_horizon)

        # 推荐调仓周期
        recommended = self._recommend_rebalance(ic_by_horizon, half_life)

        # 置信度 (基于IC序列的稳定性)
        confidence = self._estimate_confidence(ic_by_horizon)

        result = ICDecayResult(
            factor_name=factor_name,
            ic_by_horizon=ic_by_horizon,
            half_life=half_life,
            decay_rate=decay_rate,
            recommended_rebalance=recommended,
            confidence=confidence,
        )

        self._cache[factor_name] = result
        return result

    def _estimate_half_life(
        self,
        ic_by_horizon: dict[int, float]
    ) -> tuple[int, float]:
        """估计半衰期"""
        if len(ic_by_horizon) < 2:
            return 20, 0.0

        # 按horizon排序
        sorted_items = sorted(ic_by_horizon.items())
        horizons = np.array([h for h, _ in sorted_items])
        ics = np.array([ic for _, ic in sorted_items])

        # 初始IC
        ic0 = ics[0] if horizons[0] == 1 else ics[np.argmax(horizons == horizons.min())]
        if ic0 <= 0:
            return 20, 0.0

        # 指数衰减拟合
        def decay_func(x, k):
            return ic0 * np.exp(-k * x)

        try:
            popt, _ = curve_fit(decay_func, horizons, ics, p0=[0.05], maxfev=1000)
            k = popt[0]
            half_life = int(np.log(2) / k) if k > 0 else 20
            decay_rate = k
        except:
            # Fallback: 简单线性估计
            if len(ics) >= 2:
                # 找到IC减半的点
                half_ic = ic0 / 2
                for i, ic in enumerate(ics):
                    if ic <= half_ic:
                        half_life = int(horizons[i])
                        decay_rate = (ic0 - ic) / horizons[i] / ic0 if horizons[i] > 0 else 0
                        break
                else:
                    half_life = horizons[-1]
                    decay_rate = (ic0 - ics[-1]) / horizons[-1] / ic0 if horizons[-1] > 0 else 0
            else:
                half_life = 20
                decay_rate = 0.0

        return max(5, min(half_life, 120)), decay_rate

    def _recommend_rebalance(
        self,
        ic_by_horizon: dict[int, float],
        half_life: int
    ) -> int:
        """推荐调仓周期"""
        if not ic_by_horizon:
            return 10

        # 找到IC下降到峰值50%的持有期
        max_ic = max(ic_by_horizon.values())
        half_ic = max_ic * 0.5

        for h in sorted(ic_by_horizon.keys()):
            if ic_by_horizon[h] <= half_ic:
                # 建议调仓周期为半衰期的50-80%
                return max(5, int(h * 0.6))

        # 如果没有下降到50%，用半衰期
        return max(5, min(int(half_life * 0.7), 60))

    def _estimate_confidence(self, ic_by_horizon: dict[int, float]) -> float:
        """估计置信度"""
        if len(ic_by_horizon) < 3:
            return 0.3

        # 基于IC是否单调递减
        ics = list(ic_by_horizon.values())
        is_monotonic = all(ics[i] >= ics[i+1] * 0.9 for i in range(len(ics)-1))

        # 基于IC是否都为正
        all_positive = all(ic > 0 for ic in ics)

        # 基于IC绝对值
        avg_ic = np.mean(ics)
        has_signal = avg_ic > 0.01

        confidence = 0.5
        if is_monotonic:
            confidence += 0.2
        if all_positive:
            confidence += 0.2
        if has_signal:
            confidence += 0.1

        return min(1.0, confidence)

    def batch_analyze(
        self,
        data: pd.DataFrame,
        factor_names: list[str],
        horizons: list[int] = [5, 10, 20, 40],
    ) -> pd.DataFrame:
        """批量分析多个因子"""
        results = []
        for f in factor_names:
            if f in data.columns:
                result = self.analyze(data, f, horizons)
                results.append(result.to_dict())
        return pd.DataFrame(results)


class FactorNeutralizer:
    """
    因子中性化器
    
    用法:
    neutralizer = FactorNeutralizer()
    
    # 行业中性化
    industry_neutral = neutralizer.industry_neutral(factor, industry)
    
    # Size中性化
    size_neutral = neutralizer.size_neutral(factor, market_cap)
    
    # 双重中性化
    double_neutral = neutralizer.double_neutral(factor, industry, market_cap)
    """

    def __init__(self):
        pass

    def industry_neutral(
        self,
        factor: pd.Series,
        industry: pd.Series,
        style_factors: pd.DataFrame | None = None,
    ) -> pd.Series:
        """
        行业中性化
        
        对因子值在每个行业内做横截面回归，残差作为中性化后的因子值
        
        Args:
            factor: 原始因子值
            industry: 行业分类 (如申万一级行业代码)
            style_factors: 可选的风格因子 (市值、PB等)
            
        Returns:
            中性化后的因子值
        """
        df = pd.DataFrame({
            'factor': factor,
            'industry': industry,
        })

        if style_factors is not None:
            for col in style_factors.columns:
                df[col] = style_factors[col]

        # 按行业分组回归
        residuals = []
        for ind, group in df.groupby('industry', observed=True):
            if len(group) > 10:
                # 简单版本: 行业均值调整
                group_mean = group['factor'].mean()
                group['factor_neutral'] = group['factor'] - group_mean
            else:
                group['factor_neutral'] = group['factor']
            residuals.append(group[['factor_neutral']])

        result = pd.concat(residuals)
        result = result.reindex(factor.index)
        return result['factor_neutral']

    def size_neutral(
        self,
        factor: pd.Series,
        market_cap: pd.Series,
        n_bins: int = 5,
    ) -> pd.Series:
        """
        Size中性化
        
        将市值分成n组，在每组内做因子值调整
        
        Args:
            factor: 原始因子值
            market_cap: 市值
            n_bins: 分组数
            
        Returns:
            中性化后的因子值
        """
        df = pd.DataFrame({
            'factor': factor,
            'market_cap': market_cap,
        })

        # 按市值分组
        df['size_group'] = pd.qcut(df['market_cap'], n_bins, labels=False, duplicates='drop')

        # 组内均值调整
        group_means = df.groupby('size_group')['factor'].transform('mean')
        df['factor_neutral'] = df['factor'] - group_means

        return df['factor_neutral']

    def double_neutral(
        self,
        factor: pd.Series,
        industry: pd.Series,
        market_cap: pd.Series,
    ) -> pd.Series:
        """
        双重中性化 (行业 + Size)
        
        先做行业中性，再做Size中性
        
        Args:
            factor: 原始因子值
            industry: 行业分类
            market_cap: 市值
            
        Returns:
            中性化后的因子值
        """
        # 先行业中性
        neutral = self.industry_neutral(factor, industry)

        # 再Size中性
        neutral = self.size_neutral(neutral, market_cap)

        return neutral

    def style_neutral(
        self,
        factor: pd.Series,
        style_df: pd.DataFrame,
    ) -> pd.Series:
        """
        风格因子中性化
        
        对市值、PB、PE等风格因子做横截面回归，残差作为中性化后的因子值
        
        Args:
            factor: 原始因子值
            style_df: 风格因子DataFrame (列: market_cap, pb, pe等)
            
        Returns:
            中性化后的因子值
        """
        df = pd.DataFrame({
            'factor': factor,
        })
        for col in style_df.columns:
            df[col] = style_df[col].values

        # 横截面回归
        from sklearn.linear_model import LinearRegression

        X = df[style_df.columns].values
        y = df['factor'].values

        # 处理缺失值
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if valid_mask.sum() < 50:
            return factor

        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        try:
            reg = LinearRegression()
            reg.fit(X_valid, y_valid)
            y_pred = reg.predict(X)
            residuals = y - y_pred
        except:
            residuals = y

        result = pd.Series(residuals, index=factor.index)
        return result


def analyze_factor_neutralization(
    data: pd.DataFrame,
    factor_name: str,
    industry_col: str = 'industry',
    market_cap_col: str = 'market_cap',
) -> dict:
    """
    综合分析因子的中性化效果
    
    Returns:
        dict: 包含raw IC, 行业中性IC, Size中性IC, 双重中性IC
    """
    neutralizer = FactorNeutralizer()

    factor = data[factor_name]
    industry = data.get(industry_col)
    market_cap = data.get(market_cap_col)

    result = {
        'factor': factor_name,
        'raw_ic': factor.corr(data['fwd_return_20d'], method='spearman') if 'fwd_return_20d' in data.columns else None,
    }

    if industry is not None:
        industry_neutral = neutralizer.industry_neutral(factor, industry)
        if 'fwd_return_20d' in data.columns:
            result['industry_neutral_ic'] = industry_neutral.corr(data['fwd_return_20d'], method='spearman')
        result['industry_neutral'] = industry_neutral

    if market_cap is not None:
        size_neutral = neutralizer.size_neutral(factor, market_cap)
        if 'fwd_return_20d' in data.columns:
            result['size_neutral_ic'] = size_neutral.corr(data['fwd_return_20d'], method='spearman')
        result['size_neutral'] = size_neutral

    if industry is not None and market_cap is not None:
        double_neutral = neutralizer.double_neutral(factor, industry, market_cap)
        if 'fwd_return_20d' in data.columns:
            result['double_neutral_ic'] = double_neutral.corr(data['fwd_return_20d'], method='spearman')
        result['double_neutral'] = double_neutral

    return result
