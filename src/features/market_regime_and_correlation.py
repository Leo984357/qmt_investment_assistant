"""
市场状态分类器 & 多维度相关性分析

1. 市场状态分类:
   - 波动率 regime (高/低)
   - 趋势 regime (上行/下行/震荡)
   - 流动性 regime (宽松/收缩)

2. 多维度相关性去重:
   - 因子值截面相关
   - IC时间序列相关
   - 行业暴露相似度
   - 换手结构相似度
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class VolatilityRegime(Enum):
    """波动率状态"""
    HIGH = "high_vol"      # 高波动
    NORMAL = "normal_vol"  # 正常
    LOW = "low_vol"        # 低波动


class TrendRegime(Enum):
    """趋势状态"""
    UPTEND = "uptrend"     # 上涨趋势
    DOWNTREND = "downtrend"  # 下跌趋势
    NEUTRAL = "neutral"    # 震荡


class LiquidityRegime(Enum):
    """流动性状态"""
    EASY = "easy"         # 宽松
    NORMAL = "normal"      # 正常
    TIGHT = "tight"       # 收缩


@dataclass
class MarketRegime:
    """市场状态"""
    date: pd.Timestamp

    # 波动率
    vol_regime: VolatilityRegime
    vol_percentile: float  # 在历史中的波动率分位

    # 趋势
    trend_regime: TrendRegime
    trend_strength: float  # 趋势强度

    # 流动性
    liq_regime: LiquidityRegime
    liq_percentile: float  # 在历史中的流动性分位

    # 合成状态
    composite_regime: str  # 如 "high_vol_uptrend_easy"

    def to_dict(self) -> dict:
        return {
            'date': self.date,
            'vol_regime': self.vol_regime.value if isinstance(self.vol_regime, Enum) else self.vol_regime,
            'vol_percentile': self.vol_percentile,
            'trend_regime': self.trend_regime.value if isinstance(self.trend_regime, Enum) else self.trend_regime,
            'trend_strength': self.trend_strength,
            'liq_regime': self.liq_regime.value if isinstance(self.liq_regime, Enum) else self.liq_regime,
            'liq_percentile': self.liq_percentile,
            'composite_regime': self.composite_regime,
        }


class MarketRegimeClassifier:
    """
    市场状态分类器
    
    用法:
    classifier = MarketRegimeClassifier()
    
    # 分类
    regimes = classifier.classify(market_data)
    
    # 分析因子在不同regime的表现
    result = classifier.analyze_factor_by_regime(
        factor_data,
        factor_name='roe',
        regimes=regimes,
    )
    """

    def __init__(
        self,
        vol_window: int = 20,
        trend_window: int = 60,
        vol_high_threshold: float = 0.7,
        vol_low_threshold: float = 0.3,
    ):
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.vol_high_threshold = vol_high_threshold
        self.vol_low_threshold = vol_low_threshold

        self._vol_history: pd.Series | None = None

    def classify(
        self,
        market_data: pd.DataFrame,
        date_col: str = 'trade_date',
        return_col: str = 'pct_chg',
        volume_col: str = 'volume',
    ) -> pd.DataFrame:
        """
        对市场数据进行状态分类
        
        Args:
            market_data: 包含日期、市场收益、成交量的DataFrame
            date_col: 日期列名
            return_col: 收益列名
            volume_col: 成交量列名
            
        Returns:
            DataFrame with regime classification
        """
        df = market_data.copy()
        df = df.sort_values(date_col)

        # 计算波动率
        df['volatility'] = df[return_col].rolling(self.vol_window).std() * np.sqrt(252)

        # 计算趋势
        df['trend'] = df[return_col].rolling(self.trend_window).mean()
        df['trend'] = df['trend'] * 252  # 年化

        # 计算流动性 (成交量变化率)
        df['volume_ma'] = df[volume_col].rolling(20).mean()
        df['volume_ratio'] = df[volume_col] / df['volume_ma']

        # 初始化历史波动率
        if self._vol_history is None:
            self._vol_history = df['volatility'].dropna()
        else:
            self._vol_history = pd.concat([self._vol_history, df['volatility'].dropna()])

        # 计算分位数
        df['vol_percentile'] = df['volatility'].apply(
            lambda x: (self._vol_history <= x).mean() if pd.notna(x) else np.nan
        )
        df['liq_percentile'] = df['volume_ratio'].rank(pct=True)

        # 分类
        df['vol_regime'] = df['vol_percentile'].apply(self._classify_vol)
        df['trend_regime'] = df['trend'].apply(self._classify_trend)
        df['liq_regime'] = df['volume_ratio'].apply(self._classify_liq)

        # 合成状态
        df['composite_regime'] = (
            df['vol_regime'].astype(str) + '_' +
            df['trend_regime'].astype(str) + '_' +
            df['liq_regime'].astype(str)
        )

        return df

    def _classify_vol(self, percentile: float) -> str:
        """分类波动率状态"""
        if pd.isna(percentile):
            return VolatilityRegime.NORMAL.value
        if percentile >= self.vol_high_threshold:
            return VolatilityRegime.HIGH.value
        elif percentile <= self.vol_low_threshold:
            return VolatilityRegime.LOW.value
        else:
            return VolatilityRegime.NORMAL.value

    def _classify_trend(self, trend_return: float) -> str:
        """分类趋势状态"""
        if pd.isna(trend_return):
            return TrendRegime.NEUTRAL.value
        if trend_return > 0.02:  # 年化收益 > 2%
            return TrendRegime.UPTEND.value
        elif trend_return < -0.02:  # 年化收益 < -2%
            return TrendRegime.DOWNTREND.value
        else:
            return TrendRegime.NEUTRAL.value

    def _classify_liq(self, volume_ratio: float) -> str:
        """分类流动性状态"""
        if pd.isna(volume_ratio):
            return LiquidityRegime.NORMAL.value
        if volume_ratio > 1.2:
            return LiquidityRegime.EASY.value
        elif volume_ratio < 0.8:
            return LiquidityRegime.TIGHT.value
        else:
            return LiquidityRegime.NORMAL.value

    def analyze_factor_by_regime(
        self,
        factor_data: pd.DataFrame,
        factor_name: str,
        regime_data: pd.DataFrame,
        label_col: str = 'fwd_return_20d',
        date_col: str = 'trade_date',
    ) -> dict:
        """
        分析因子在不同市场状态下的表现
        
        Returns:
            dict: 包含各regime下的IC
        """
        df = factor_data.merge(
            regime_data[[date_col, 'vol_regime', 'trend_regime', 'liq_regime', 'composite_regime']],
            on=date_col,
            how='left'
        )

        valid = df.dropna(subset=[factor_name, label_col])

        results = {
            'factor': factor_name,
            'overall_ic': valid[factor_name].corr(valid[label_col], method='spearman'),
            'by_vol_regime': {},
            'by_trend_regime': {},
            'by_liq_regime': {},
            'by_composite_regime': {},
        }

        # 按波动率regime分析
        for regime in valid['vol_regime'].dropna().unique():
            subset = valid[valid['vol_regime'] == regime]
            if len(subset) > 50:
                ic = subset[factor_name].corr(subset[label_col], method='spearman')
                results['by_vol_regime'][regime] = ic

        # 按趋势regime分析
        for regime in valid['trend_regime'].dropna().unique():
            subset = valid[valid['trend_regime'] == regime]
            if len(subset) > 50:
                ic = subset[factor_name].corr(subset[label_col], method='spearman')
                results['by_trend_regime'][regime] = ic

        # 按流动性regime分析
        for regime in valid['liq_regime'].dropna().unique():
            subset = valid[valid['liq_regime'] == regime]
            if len(subset) > 50:
                ic = subset[factor_name].corr(subset[label_col], method='spearman')
                results['by_liq_regime'][regime] = ic

        return results


class MultiDimensionalCorrelationAnalyzer:
    """
    多维度相关性分析器
    
    检测因子冗余不仅看因子值相关，还看:
    1. 因子值截面相关
    2. IC时间序列相关
    3. 行业暴露相似度
    4. 换手结构相似度
    """

    def __init__(self):
        pass

    def analyze_redundancy(
        self,
        factor_data: pd.DataFrame,
        factor_names: list[str],
        date_col: str = 'trade_date',
        label_col: str = 'fwd_return_20d',
        industry_col: str = 'industry',
    ) -> pd.DataFrame:
        """
        多维度冗余分析
        
        Returns:
            DataFrame with redundancy metrics for each factor pair
        """
        results = []

        for i, f1 in enumerate(factor_names):
            for f2 in factor_names[i+1:]:
                if f1 not in factor_data.columns or f2 not in factor_data.columns:
                    continue

                result = {
                    'factor1': f1,
                    'factor2': f2,
                }

                # 1. 因子值截面相关
                cross_corr = factor_data[[f1, f2]].corr().iloc[0, 1]
                result['cross_sectional_corr'] = cross_corr

                # 2. IC时间序列相关
                ic_series1 = self._compute_ic_series(factor_data, f1, date_col, label_col)
                ic_series2 = self._compute_ic_series(factor_data, f2, date_col, label_col)

                if len(ic_series1) > 10 and len(ic_series2) > 10:
                    common_dates = ic_series1.index.intersection(ic_series2.index)
                    if len(common_dates) > 10:
                        ic_corr = ic_series1[common_dates].corr(ic_series2[common_dates])
                        result['ic_time_series_corr'] = ic_corr

                # 3. 行业暴露相似度 (如果可用)
                if industry_col in factor_data.columns:
                    industry_exposure1 = self._compute_industry_exposure(factor_data, f1, industry_col)
                    industry_exposure2 = self._compute_industry_exposure(factor_data, f2, industry_col)

                    if len(industry_exposure1) > 0 and len(industry_exposure2) > 0:
                        # 对齐索引
                        common_industries = industry_exposure1.index.intersection(industry_exposure2.index)
                        if len(common_industries) > 5:
                            exp_corr = industry_exposure1[common_industries].corr(industry_exposure2[common_industries])
                            result['industry_exposure_corr'] = exp_corr

                # 综合冗余得分
                corr_cols = [c for c in ['cross_sectional_corr', 'ic_time_series_corr', 'industry_exposure_corr']
                           if c in result]
                if corr_cols:
                    avg_corr = np.mean([abs(result[c]) for c in corr_cols])
                    result['avg_correlation'] = avg_corr
                    result['is_redundant'] = avg_corr > 0.7

                results.append(result)

        return pd.DataFrame(results)

    def _compute_ic_series(
        self,
        data: pd.DataFrame,
        factor_name: str,
        date_col: str,
        label_col: str,
    ) -> pd.Series:
        """计算IC时间序列"""
        valid = data.dropna(subset=[factor_name, label_col])
        ic_series = valid.groupby(date_col).apply(
            lambda x: x[factor_name].corr(x[label_col], method='spearman')
        )
        return ic_series

    def _compute_industry_exposure(
        self,
        data: pd.DataFrame,
        factor_name: str,
        industry_col: str,
    ) -> pd.Series:
        """计算行业暴露"""
        if industry_col not in data.columns:
            return pd.Series()

        valid = data.dropna(subset=[factor_name, industry_col])

        # 计算每个行业的平均因子值
        industry_exposure = valid.groupby(industry_col)[factor_name].mean()
        return industry_exposure

    def get_redundancy_groups(
        self,
        redundancy_df: pd.DataFrame,
        threshold: float = 0.7,
    ) -> list[list[str]]:
        """识别冗余因子组"""
        if redundancy_df.empty:
            return []

        redundant_pairs = redundancy_df[redundancy_df['is_redundant']]

        # 构建图
        import networkx as nx
        G = nx.Graph()

        for _, row in redundant_pairs.iterrows():
            G.add_edge(row['factor1'], row['factor2'], weight=row['avg_correlation'])

        # 找连通分量
        groups = list(nx.connected_components(G))
        return [list(g) for g in groups]

    def recommend_keep_remove(
        self,
        redundancy_groups: list[list[str]],
        factor_metrics: pd.DataFrame,
    ) -> dict:
        """
        推荐保留/移除的因子
        
        factor_metrics应包含: ic_mean, turnover等指标
        """
        recommendations = {}

        for group in redundancy_groups:
            if len(group) <= 1:
                continue

            # 在每组中，选择:
            # 1. IC最高
            # 2. 换手率最低
            group_metrics = factor_metrics[factor_metrics['factor'].isin(group)]

            if group_metrics.empty:
                continue

            # 综合评分
            group_metrics = group_metrics.copy()
            if 'ic_mean' in group_metrics.columns:
                group_metrics['ic_score'] = group_metrics['ic_mean'].rank(ascending=False)
            if 'turnover' in group_metrics.columns:
                group_metrics['turnover_score'] = group_metrics['turnover'].rank(ascending=True)

            if 'ic_score' in group_metrics.columns and 'turnover_score' in group_metrics.columns:
                group_metrics['composite_score'] = group_metrics['ic_score'] + group_metrics['turnover_score']
                best_factor = group_metrics.loc[group_metrics['composite_score'].idxmin(), 'factor']
            else:
                best_factor = group_metrics.iloc[0]['factor']

            for f in group:
                recommendations[f] = {
                    'action': 'KEEP' if f == best_factor else 'REMOVE',
                    'reason': "保留IC最高/换手最低" if f != best_factor else "最佳候选",
                    'suggested_replacement': best_factor if f != best_factor else None,
                }

        return recommendations


# 导出
__all__ = [
    'MarketRegimeClassifier',
    'MarketRegime',
    'VolatilityRegime',
    'TrendRegime',
    'LiquidityRegime',
    'MultiDimensionalCorrelationAnalyzer',
]
