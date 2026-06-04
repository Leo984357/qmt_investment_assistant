from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class RollingModelResult:
    predictions: pd.DataFrame
    split_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    model_registry: pd.DataFrame


@dataclass
class RegimeWeightConfig:
    """市场状态下的因子权重配置"""
    value_factors: list[str] = field(default_factory=lambda: ['earnings_yield', 'book_to_price', 'pe_ratio', 'pb_ratio'])
    momentum_factors: list[str] = field(default_factory=lambda: ['mom250', 'mom120', 'mom60', 'close_to_high250'])
    quality_factors: list[str] = field(default_factory=lambda: ['roe', 'roa', 'ocf_per_share', 'operating_margin', 'gross_margin'])
    reversal_factors: list[str] = field(default_factory=lambda: ['rev20', 'rev10', 'short_term_reversal'])

    bull_momentum_mult: float = 1.5
    bull_quality_mult: float = 1.2
    bull_value_mult: float = 0.8
    bull_reversal_mult: float = 0.5

    bear_value_mult: float = 1.5
    bear_quality_mult: float = 1.3
    bear_momentum_mult: float = 0.3
    bear_reversal_mult: float = 0.8

    ranging_reversal_mult: float = 1.5
    ranging_quality_mult: float = 1.3
    ranging_value_mult: float = 1.0
    ranging_momentum_mult: float = 0.5


def detect_simple_regime(prices: pd.Series, short_window: int = 20, long_window: int = 60) -> str:
    """简单市场状态检测
    
    Returns: 'bull', 'bear', 'ranging'
    """
    if len(prices) < long_window:
        return 'ranging'

    current = prices.iloc[-1]
    short_ma = prices.tail(short_window).mean()
    long_ma = prices.tail(long_window).mean()

    # 趋势位置
    trend_pos = (current / long_ma - 1) if long_ma > 0 else 0

    # 波动率
    returns = prices.pct_change().dropna()
    vol = returns.tail(short_window).std() * np.sqrt(252)

    if trend_pos > 0.05 and short_ma > long_ma:
        return 'bull'
    elif trend_pos < -0.05 and short_ma < long_ma:
        return 'bear'
    else:
        return 'ranging'


def get_regime_multipliers(factor_name: str, regime: str, config: RegimeWeightConfig) -> float:
    """根据市场状态和因子家族获取权重倍率"""
    if regime == 'bull':
        if factor_name in config.momentum_factors:
            return config.bull_momentum_mult
        elif factor_name in config.quality_factors:
            return config.bull_quality_mult
        elif factor_name in config.value_factors:
            return config.bull_value_mult
        elif factor_name in config.reversal_factors:
            return config.bull_reversal_mult
    elif regime == 'bear':
        if factor_name in config.value_factors:
            return config.bear_value_mult
        elif factor_name in config.quality_factors:
            return config.bear_quality_mult
        elif factor_name in config.momentum_factors:
            return config.bear_momentum_mult
        elif factor_name in config.reversal_factors:
            return config.bear_reversal_mult
    else:  # ranging
        if factor_name in config.reversal_factors:
            return config.ranging_reversal_mult
        elif factor_name in config.quality_factors:
            return config.ranging_quality_mult
        elif factor_name in config.value_factors:
            return config.ranging_value_mult
        elif factor_name in config.momentum_factors:
            return config.ranging_momentum_mult

    return 1.0


@dataclass
class FactorDecayConfig:
    """因子退化监控配置"""
    decay_lookback: int = 60          # 回看窗口 (交易日)
    decay_threshold_ic: float = -0.03  # IC低于此值立即降权
    decay_consecutive_neg: int = 5     # 连续负IC次数阈值
    decay_weight_cut: float = 0.5      # 降权比例
    decay_min_positive_rate: float = 0.35  # 正IC最低比例
    decay_weight_recovery: float = 1.5  # 恢复时乘数


@dataclass
class AdaptiveWeightConfig:
    """自适应权重配置"""
    ic_lookback: int = 120            # IC计算回看窗口
    ic_half_life: int = 20            # IC指数衰减半衰期
    decay_config: FactorDecayConfig = field(default_factory=FactorDecayConfig)
    enable_decay_monitor: bool = True  # 是否启用因子退化监控
    min_factor_weight: float = 0.05    # 因子最低权重
    decay_backup_factors: list[str] | None = None  # 备用因子列表


def _compute_exponential_ic_weights(
    ic_series: pd.Series,
    half_life: int = 20,
) -> np.ndarray:
    """计算指数加权IC权重
    
    越近期的IC权重越高，半衰期控制衰减速度
    """
    if len(ic_series) < 2:
        return np.array([1.0])

    dates = ic_series.index
    weights = np.exp(-np.arange(len(ic_series) - 1, -1, -1) * np.log(2) / half_life)
    return weights / weights.sum()


def _compute_factor_decay_status(
    ic_history: pd.DataFrame,
    factor: str,
    config: FactorDecayConfig,
) -> tuple[str, float]:
    """
    计算因子退化状态
    
    Returns:
        (status, weight_multiplier)
        status: 'active', 'watch', 'reduced', 'offline'
    """
    if factor not in ic_history.columns:
        return 'active', 1.0

    ic_series = ic_history[factor].dropna()
    if len(ic_series) < config.decay_lookback:
        return 'active', 1.0

    recent_ic = ic_series.tail(config.decay_lookback)

    # 检查1: 单次IC低于阈值
    latest_ic = recent_ic.iloc[-1]
    if latest_ic < config.decay_threshold_ic:
        return 'offline', 0.0

    # 检查2: 连续负IC
    consecutive_neg = 0
    consecutive_pos = 0
    for ic in reversed(recent_ic):
        if ic < 0:
            consecutive_neg += 1
            consecutive_pos = 0
        else:
            consecutive_pos += 1
            consecutive_neg = 0
        if consecutive_neg >= config.decay_consecutive_neg:
            break

    if consecutive_neg >= config.decay_consecutive_neg * 3:
        return 'offline', 0.0
    elif consecutive_neg >= config.decay_consecutive_neg * 2:
        return 'reduced', config.decay_weight_cut * config.decay_weight_cut
    elif consecutive_neg >= config.decay_consecutive_neg:
        return 'reduced', config.decay_weight_cut
    elif consecutive_pos >= config.decay_consecutive_neg * 4:
        return 'active', 1.0

    # 检查3: 正IC比例
    positive_rate = (recent_ic > 0).mean()
    if positive_rate < config.decay_min_positive_rate:
        return 'reduced', config.decay_weight_cut

    return 'active', 1.0


class AdaptiveICWeightedModel:
    """
    自适应IC加权模型
    
    改进点：
    1. 指数加权IC：近期IC权重更高
    2. 因子退化监控：自动识别并降权/下线失效因子
    3. 动态权重：根据近期表现调整因子权重
    
    经济逻辑：
    - 因子有效性随市场变化
    - 近期表现比长期平均更能预测未来
    - 因子失效时应减少暴露
    """

    def __init__(
        self,
        feature_names: list[str],
        config: AdaptiveWeightConfig | None = None,
    ):
        self.feature_names = feature_names
        self.config = config or AdaptiveWeightConfig()

        # 因子IC历史记录
        self._factor_ic_history: pd.DataFrame = pd.DataFrame()
        self._factor_weights_history: list[dict] = []

    def fit_walk_forward(
        self,
        dataset: pd.DataFrame,
        feature_names: list[str],
        label_name: str,
        rebalance_dates: list[pd.Timestamp],
        artifact_dir: Path,
        label_horizon: int = 0,
    ) -> RollingModelResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        unique_dates = sorted(pd.to_datetime(dataset['trade_date'].unique()))
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        prediction_frames: list[pd.DataFrame] = []
        metric_rows: list[dict] = []
        importance_frames: list[pd.DataFrame] = []
        registry_rows: list[dict] = []
        total_splits = len(rebalance_dates)
        progress_every = max(1, total_splits // 20) if total_splits else 1
        start_ts = perf_counter()

        for split_idx, signal_date in enumerate(rebalance_dates, start=1):
            signal_date = pd.Timestamp(signal_date)
            if signal_date not in date_to_idx:
                continue

            idx = date_to_idx[signal_date]
            train_end_idx = max(0, idx - int(label_horizon) - 1)
            train_start_idx = max(0, train_end_idx - self.config.ic_lookback)
            train_dates = unique_dates[train_start_idx:train_end_idx]

            train_df = dataset.loc[dataset['trade_date'].isin(train_dates)].dropna(subset=feature_names + [label_name]).copy()

            # 计算每日IC，然后指数加权
            ic_scores = {}
            ic_daily = {}
            for f in feature_names:
                valid = train_df[['trade_date', f, label_name]].dropna()
                if len(valid) > 30:
                    daily_ic = valid.groupby('trade_date').apply(
                        lambda x: x[f].corr(x[label_name]), include_groups=False
                    )
                    ic_daily[f] = daily_ic

                    # 指数加权IC
                    weights = _compute_exponential_ic_weights(daily_ic, self.config.ic_half_life)
                    weighted_ic = (daily_ic.values * weights).sum()
                    ic_scores[f] = max(weighted_ic, 0.001)
                else:
                    ic_scores[f] = 0.001
                    ic_daily[f] = pd.Series(dtype=float)

            # 更新IC历史
            for f, daily_ic in ic_daily.items():
                if len(daily_ic) > 0:
                    # 统一使用trade_date列
                    new_records = pd.DataFrame({
                        'trade_date': daily_ic.index,
                        f: daily_ic.values
                    }).set_index('trade_date')

                    if self._factor_ic_history.empty:
                        self._factor_ic_history = new_records
                    else:
                        # 合并新记录
                        for idx, row in new_records.iterrows():
                            if idx not in self._factor_ic_history.index:
                                self._factor_ic_history.loc[idx] = np.nan
                            self._factor_ic_history.loc[idx, f] = row[f]

            # 因子退化监控
            decay_weights = {}
            decay_status = {}
            if self.config.enable_decay_monitor:
                for f in feature_names:
                    status, multiplier = _compute_factor_decay_status(
                        self._factor_ic_history, f, self.config.decay_config
                    )
                    decay_status[f] = status
                    decay_weights[f] = multiplier

            # 计算最终权重
            base_weights = np.array([ic_scores.get(f, 0.001) for f in feature_names])

            if self.config.enable_decay_monitor:
                decay_array = np.array([decay_weights.get(f, 1.0) for f in feature_names])
                base_weights = base_weights * decay_array

            # 应用最低权重限制
            min_weight = self.config.min_factor_weight
            weights = np.maximum(base_weights, min_weight)
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones(len(feature_names)) / len(feature_names)

            # 记录权重历史
            weight_record = {
                'signal_date': signal_date,
                **{f'weight_{f}': w for f, w in zip(feature_names, weights)},
                **{f'ic_{f}': ic_scores.get(f, 0) for f in feature_names},
                **{f'status_{f}': decay_status.get(f, 'active') for f in feature_names},
            }
            self._factor_weights_history.append(weight_record)

            test_df = dataset.loc[dataset['trade_date'] == signal_date].dropna(subset=feature_names).copy()
            if test_df.empty:
                continue

            factor_matrix = test_df[feature_names].values
            score = np.dot(factor_matrix, weights)
            test_df['score'] = score
            test_df['fallback_used'] = False

            pred = test_df[['trade_date', 'symbol', 'score', 'fallback_used']].copy()
            pred['model_type'] = 'adaptive_ic_weighted'
            pred['signal_date'] = signal_date
            prediction_frames.append(pred)

            importance_frames.append(pd.DataFrame({
                'feature_name': feature_names,
                'importance_gain': weights,
                'signal_date': [signal_date] * len(feature_names),
            }))

            registry_rows.append({
                'signal_date': signal_date,
                'model_type': 'adaptive_ic_weighted',
                'train_samples': len(train_df),
                'valid_samples': 0,
                'train_dates': len(train_dates),
                'valid_dates': 0,
                **{f'ic_{f}': ic_scores.get(f, 0) for f in feature_names},
                **{f'status_{f}': decay_status.get(f, 'active') for f in feature_names},
            })

            if split_idx % progress_every == 0:
                elapsed = perf_counter() - start_ts
                offline_count = sum(1 for s in decay_status.values() if s == 'offline')
                logger.info('AdaptiveICWeighted progress %d/%d splits (%.1fs) offline=%d',
                           split_idx, total_splits, elapsed, offline_count)

        predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        split_metrics = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame()
        feature_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
        model_registry = pd.DataFrame(registry_rows) if registry_rows else pd.DataFrame()

        return RollingModelResult(
            predictions=predictions,
            split_metrics=split_metrics,
            feature_importance=feature_importance,
            model_registry=model_registry,
        )

    def get_factor_ic_history(self) -> pd.DataFrame:
        """获取因子IC历史"""
        return self._factor_ic_history.copy()

    def get_weight_history(self) -> pd.DataFrame:
        """获取权重历史"""
        return pd.DataFrame(self._factor_weights_history)


class RegimeAwareAdaptiveModel:
    """
    市场状态感知自适应模型
    
    在 AdaptiveICWeightedModel 基础上加入市场状态检测:
    - 牛市: 动量/盈利因子权重提高, 反转因子降低
    - 熊市: 价值/质量因子权重提高, 动量因子降低
    - 震荡: 反转/质量因子权重提高, 动量因子降低
    
    经济逻辑:
    - 牛市投资者乐观, 趋势延续性强, 动量有效
    - 熊市投资者悲观, 价值防御性强, 低估值股票抗跌
    - 震荡市方向不明, 超跌反弹机会多, 反转有效
    """

    def __init__(
        self,
        feature_names: list[str],
        config: AdaptiveWeightConfig | None = None,
        regime_config: RegimeWeightConfig | None = None,
        regime_short_window: int = 20,
        regime_long_window: int = 60,
    ):
        self.feature_names = feature_names
        self.config = config or AdaptiveWeightConfig()
        self.regime_config = regime_config or RegimeWeightConfig()
        self.regime_short_window = regime_short_window
        self.regime_long_window = regime_long_window
        self._base_model = AdaptiveICWeightedModel(feature_names, config)
        self._market_prices: pd.Series | None = None

    def fit_walk_forward(
        self,
        dataset: pd.DataFrame,
        feature_names: list[str],
        label_name: str,
        rebalance_dates: list[pd.Timestamp],
        artifact_dir: Path,
        label_horizon: int = 0,
        market_index: pd.Series | None = None,
    ) -> RollingModelResult:
        """walk forward训练, 支持市场状态检测"""

        # 如果提供了市场指数, 构建价格序列
        if market_index is not None:
            self._market_prices = market_index
        elif '000300.SH' in dataset.columns:
            # 尝试从dataset获取沪深300指数
            idx_data = dataset[['trade_date', '000300.SH']].dropna()
            if len(idx_data) > 0:
                self._market_prices = idx_data.set_index('trade_date')['000300.SH'].sort_index()

        # 使用基础模型进行walk-forward
        result = self._base_model.fit_walk_forward(
            dataset=dataset,
            feature_names=feature_names,
            label_name=label_name,
            rebalance_dates=rebalance_dates,
            artifact_dir=artifact_dir,
            label_horizon=label_horizon,
        )

        # 添加市场状态列
        if len(result.predictions) > 0 and self._market_prices is not None:
            result.predictions['regime'] = result.predictions['signal_date'].apply(
                lambda d: detect_simple_regime(
                    self._market_prices.loc[:d] if d in self._market_prices.index else self._market_prices,
                    self.regime_short_window,
                    self.regime_long_window
                )
            )

        return result

    def compute_regime_aware_scores(
        self,
        scores: pd.Series,
        signal_date: pd.Timestamp,
        factor_names: list[str],
    ) -> pd.Series:
        """根据市场状态调整分数"""
        if self._market_prices is None:
            return scores

        # 检测当前市场状态
        prices = self._market_prices.loc[:signal_date]
        regime = detect_simple_regime(prices, self.regime_short_window, self.regime_long_window)

        # 应用状态倍率
        multipliers = {
            f: get_regime_multipliers(f, regime, self.regime_config)
            for f in factor_names
        }

        return scores  # 实际调整在模型层面做


class ICWeightedAverageModel:
    """原始IC加权模型 (保留兼容性)"""

    def __init__(
        self,
        feature_names: list[str],
        ic_weights: dict[str, float] | None = None,
        lookback_days: int = 60,
    ):
        self.feature_names = feature_names
        self.ic_weights = ic_weights or {}
        self.lookback_days = lookback_days

    def fit_walk_forward(
        self,
        dataset: pd.DataFrame,
        feature_names: list[str],
        label_name: str,
        rebalance_dates: list[pd.Timestamp],
        artifact_dir: Path,
        label_horizon: int = 0,
    ) -> RollingModelResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        unique_dates = sorted(pd.to_datetime(dataset['trade_date'].unique()))
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        prediction_frames: list[pd.DataFrame] = []
        metric_rows: list[dict] = []
        importance_frames: list[pd.DataFrame] = []
        registry_rows: list[dict] = []
        total_splits = len(rebalance_dates)
        progress_every = max(1, total_splits // 20) if total_splits else 1
        start_ts = perf_counter()

        for split_idx, signal_date in enumerate(rebalance_dates, start=1):
            signal_date = pd.Timestamp(signal_date)
            if signal_date not in date_to_idx:
                continue

            idx = date_to_idx[signal_date]
            train_end_idx = max(0, idx - int(label_horizon) - 1)
            train_start_idx = max(0, train_end_idx - self.lookback_days)
            train_dates = unique_dates[train_start_idx:train_end_idx]

            train_df = dataset.loc[dataset['trade_date'].isin(train_dates)].dropna(subset=feature_names + [label_name]).copy()

            ic_scores = {}
            for f in feature_names:
                valid = train_df[['trade_date', f, label_name]].dropna()
                if len(valid) > 30:
                    ic = valid.groupby('trade_date').apply(
                        lambda x: x[f].corr(x[label_name]), include_groups=False
                    ).mean()
                    ic_scores[f] = max(ic, 0.001)
                else:
                    ic_scores[f] = 0.001

            weights = np.array([ic_scores.get(f, 0.001) for f in feature_names])
            weights = weights / weights.sum()

            test_df = dataset.loc[dataset['trade_date'] == signal_date].dropna(subset=feature_names).copy()
            if test_df.empty:
                continue

            factor_matrix = test_df[feature_names].values
            score = np.dot(factor_matrix, weights)
            test_df['score'] = score
            test_df['fallback_used'] = False

            pred = test_df[['trade_date', 'symbol', 'score', 'fallback_used']].copy()
            pred['model_type'] = 'ic_weighted_average'
            pred['signal_date'] = signal_date
            prediction_frames.append(pred)

            importance_frames.append(pd.DataFrame({
                'feature_name': feature_names,
                'importance_gain': weights,
                'signal_date': [signal_date] * len(feature_names),
            }))

            registry_rows.append({
                'signal_date': signal_date,
                'model_type': 'ic_weighted_average',
                'train_samples': len(train_df),
                'valid_samples': 0,
                'train_dates': len(train_dates),
                'valid_dates': 0,
                **{f'ic_{f}': ic_scores.get(f, 0) for f in feature_names},
            })

            if split_idx % progress_every == 0:
                elapsed = perf_counter() - start_ts
                logger.info('ICWeightedAverage progress %d/%d splits (%.1fs)', split_idx, total_splits, elapsed)

        predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        split_metrics = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame()
        feature_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
        model_registry = pd.DataFrame(registry_rows) if registry_rows else pd.DataFrame()

        return RollingModelResult(
            predictions=predictions,
            split_metrics=split_metrics,
            feature_importance=feature_importance,
            model_registry=model_registry,
        )


class SimpleAverageModel:
    """简单平均模型 (保留兼容性)"""

    def __init__(
        self,
        feature_names: list[str],
    ):
        self.feature_names = feature_names

    def fit_walk_forward(
        self,
        dataset: pd.DataFrame,
        feature_names: list[str],
        label_name: str,
        rebalance_dates: list[pd.Timestamp],
        artifact_dir: Path,
        label_horizon: int = 0,
    ) -> RollingModelResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        unique_dates = sorted(pd.to_datetime(dataset['trade_date'].unique()))
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        prediction_frames: list[pd.DataFrame] = []
        metric_rows: list[dict] = []
        importance_frames: list[pd.DataFrame] = []
        registry_rows: list[dict] = []
        total_splits = len(rebalance_dates)
        progress_every = max(1, total_splits // 20) if total_splits else 1
        start_ts = perf_counter()

        for split_idx, signal_date in enumerate(rebalance_dates, start=1):
            signal_date = pd.Timestamp(signal_date)
            if signal_date not in date_to_idx:
                continue

            test_df = dataset.loc[dataset['trade_date'] == signal_date].dropna(subset=feature_names).copy()
            if test_df.empty:
                continue

            score = test_df[feature_names].mean(axis=1)
            test_df['score'] = score
            test_df['fallback_used'] = False

            pred = test_df[['trade_date', 'symbol', 'score', 'fallback_used']].copy()
            pred['model_type'] = 'simple_average'
            pred['signal_date'] = signal_date
            prediction_frames.append(pred)

            importance_frames.append(pd.DataFrame({
                'feature_name': feature_names,
                'importance_gain': [1.0] * len(feature_names),
                'signal_date': [signal_date] * len(feature_names),
            }))

            registry_rows.append({
                'signal_date': signal_date,
                'model_type': 'simple_average',
                'train_samples': 0,
                'valid_samples': 0,
                'train_dates': 0,
                'valid_dates': 0,
            })

            if split_idx % progress_every == 0:
                elapsed = perf_counter() - start_ts
                logger.info('SimpleAverage progress %d/%d splits (%.1fs)', split_idx, total_splits, elapsed)

        predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        split_metrics = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame()
        feature_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
        model_registry = pd.DataFrame(registry_rows) if registry_rows else pd.DataFrame()

        return RollingModelResult(
            predictions=predictions,
            split_metrics=split_metrics,
            feature_importance=feature_importance,
            model_registry=model_registry,
        )
