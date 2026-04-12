from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BufferConfig:
    """持仓缓冲区配置"""
    retain_threshold_rank: int = 50  # 老持仓只要在50名内就保留
    min_score_drop: float = 0.1  # 分数下降超过10%才换
    max_retain_ratio: float = 0.6  # 最多保留60%老持仓


@dataclass
class SmootherConfig:
    """权重平滑配置"""
    step_ratio: float = 0.5  # 每次移动50%的差距
    min_change_threshold: float = 0.001  # 变化小于0.1%不调


@dataclass
class CostFilterConfig:
    """成本-边际收益过滤配置"""
    min_alpha_threshold: float = 0.002  # 最小边际收益0.2%
    cost_to_alpha_ratio: float = 0.3  # 成本超过收益30%就跳过
    skip_small_changes: bool = True
    min_weight_change: float = 0.005  # 权重变化小于0.5%跳过


class PositionBuffer:
    """
    持仓缓冲区
    
    作用：减少无意义换手，保留还有价值的老持仓
    
    逻辑：
    - 老持仓只要还在可接受排名内，就保留
    - 新信号必须明显优于老持仓才替换
    - 设置最大保留比例，防止过度保守
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        self.config = config or BufferConfig()
        self._previous_holdings: dict[str, dict] = {}
    
    def apply(
        self,
        new_candidates: pd.DataFrame,
        current_positions: dict[str, float],
        execution_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        应用持仓缓冲区
        
        Args:
            new_candidates: 新候选股票，带score和rank
            current_positions: 当前持仓 {symbol: weight}
            execution_date: 执行日期
        
        Returns:
            过滤后的候选列表
        """
        if new_candidates.empty or not current_positions:
            return new_candidates
        
        config = self.config
        
        # 给候选股票添加当前持仓标记
        new_candidates = new_candidates.copy()
        new_candidates['is_holding'] = new_candidates['symbol'].isin(current_positions.keys())
        new_candidates['current_weight'] = new_candidates['symbol'].map(current_positions).fillna(0)
        
        # 分类处理
        result_list = []
        retained = []
        
        for _, row in new_candidates.iterrows():
            symbol = row['symbol']
            new_score = row['score']
            new_rank = row['rank']
            is_holding = row['is_holding']
            
            if is_holding:
                # 老持仓检查
                if new_rank <= config.retain_threshold_rank:
                    # 仍在可接受排名内，保留
                    retained.append(row)
                else:
                    # 已不在可接受范围，检查是否值得保留
                    prev_score = self._previous_holdings.get(symbol, {}).get('score', new_score)
                    score_retention_ratio = new_score / max(prev_score, 0.001)
                    
                    if score_retention_ratio > (1 - config.min_score_drop):
                        retained.append(row)
                    else:
                        # 分数下降太多，降权处理
                        row['buffer_retained'] = True
                        row['weight_multiplier'] = 0.5
                        retained.append(row)
            else:
                # 新候选
                result_list.append(row)
        
        # 限制保留比例
        max_retain = int(len(new_candidates) * config.max_retain_ratio)
        
        if len(retained) > max_retain:
            # 按分数排序，保留分数高的
            retained = sorted(retained, key=lambda x: x['score'], reverse=True)[:max_retain]
            result_list.extend(retained)
        else:
            result_list.extend(retained)
        
        result = pd.DataFrame(result_list)
        
        # 更新记录
        for _, row in result.iterrows():
            self._previous_holdings[row['symbol']] = {
                'score': row['score'],
                'rank': row['rank'],
                'weight': row.get('target_weight', 0),
            }
        
        return result
    
    def reset(self):
        """重置缓冲区状态"""
        self._previous_holdings = {}
    
    def get_holding_info(self) -> dict:
        """获取当前持仓信息"""
        return self._previous_holdings.copy()


class WeightSmoother:
    """
    权重平滑器
    
    作用：不一次性追到目标权重，而是渐进调整
    
    逻辑：
    - 计算当前权重和目标权重的差距
    - 每次只移动差距的step_ratio比例
    - 变化太小时不调整
    """
    
    def __init__(self, config: Optional[SmootherConfig] = None):
        self.config = config or SmootherConfig()
        self._current_weights: dict[str, float] = {}
    
    def smooth(
        self,
        target_weights: pd.DataFrame,
        execution_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        平滑权重
        
        Args:
            target_weights: 目标权重 {symbol: target_weight}
            execution_date: 执行日期
        
        Returns:
            调整后的权重
        """
        config = self.config
        
        if target_weights.empty:
            return target_weights
        
        result = target_weights.copy()
        
        # 计算每个权重的平滑值
        smoothed_weights = []
        
        for _, row in result.iterrows():
            symbol = row['symbol']
            target = row['target_weight']
            current = self._current_weights.get(symbol, 0)
            
            # 计算差距
            diff = target - current
            
            # 变化太小不调整
            if abs(diff) < config.min_change_threshold:
                smoothed_weight = current
            else:
                # 半步调整
                smoothed_weight = current + diff * config.step_ratio
            
            smoothed_weights.append(smoothed_weight)
            self._current_weights[symbol] = smoothed_weight
        
        result['target_weight'] = smoothed_weights
        result['smoothed'] = True
        
        # 归一化确保总和正确
        total = result['target_weight'].sum()
        if total > 0:
            result['target_weight'] = result['target_weight'] / total * result['gross_exposure'].iloc[0]
        
        return result
    
    def reset(self):
        """重置平滑器状态"""
        self._current_weights = {}
    
    def set_current_weights(self, weights: dict[str, float]):
        """手动设置当前权重（用于初始化）"""
        self._current_weights = weights.copy()
    
    def get_current_weights(self) -> dict:
        """获取当前权重"""
        return self._current_weights.copy()


class CostAlphaFilter:
    """
    成本-边际收益过滤器 (已修复版)
    
    作用：判断调仓是否值得
    
    核心改进：
    1. 边际收益基于历史分组收益校准，不再拍脑袋
    2. 卖出信号单独处理，不因负alpha误判
    3. 区分"成本过高"和"信号本身不够强"
    
    历史分组收益校准逻辑：
    - 按分数rank分bucket (如5组)
    - 计算每个bucket的历史平均收益
    - 用目标股票所在bucket的收益作为预期边际收益
    """
    
    def __init__(self, config: Optional[CostFilterConfig] = None):
        self.config = config or CostFilterConfig()
        
        # 校准后的bucket收益 (rank bucket -> avg return)
        # 默认值，实际使用前需从历史数据计算
        self._bucket_returns: dict[int, float] = {
            5: 0.015,  # top 20% bucket
            4: 0.008,  # 40-60% bucket
            3: 0.002,  # 60-80% bucket
            2: -0.005, # 80-95% bucket
            1: -0.012, # bottom 5% bucket
        }
    
    def calibrate_from_history(
        self,
        historical_data: pd.DataFrame,
        score_col: str,
        return_col: str,
        n_buckets: int = 5,
    ) -> dict:
        """
        从历史数据校准bucket收益
        
        Args:
            historical_data: 历史分数和收益数据
            score_col: 分数列名
            return_col: 收益列名
            n_buckets: 分组数
        
        Returns:
            {bucket: avg_return} 字典
        """
        df = historical_data.dropna(subset=[score_col, return_col]).copy()
        
        # 按分数rank分组
        df['bucket'] = pd.qcut(df[score_col], q=n_buckets, labels=False, duplicates='drop') + 1
        
        # 计算每个bucket的平均收益
        bucket_returns = df.groupby('bucket')[return_col].mean().to_dict()
        
        self._bucket_returns = bucket_returns
        
        return bucket_returns
    
    def set_bucket_returns(self, bucket_returns: dict[int, float]):
        """手动设置bucket收益"""
        self._bucket_returns = bucket_returns
    
    def filter_trades(
        self,
        target_weights: pd.DataFrame,
        current_positions: dict[str, float],
        prices: dict[str, float],
        lot_size: int,
        commission_bps: float = 0.75,
        stamp_duty_bps: float = 10.0,
        slippage_bps: float = 5.0,
        min_trade_value: float = 2000.0,
        total_equity: float = 1000000.0,
    ) -> pd.DataFrame:
        """
        过滤不值得的调仓 (已修复版)
        
        改进：
        1. 卖出时直接交易，不判断estimated_alpha
        2. 买入时用bucket校准的边际收益
        3. 成本-收益比基于真实历史数据
        """
        config = self.config
        
        if target_weights.empty:
            return target_weights
        
        result = target_weights.copy()
        result['trade_direction'] = 'HOLD'
        result['estimated_cost'] = 0.0
        result['estimated_alpha'] = 0.0
        result['filter_reason'] = None
        result['bucket'] = 0
        
        for idx, row in result.iterrows():
            symbol = row['symbol']
            target_weight = row['target_weight']
            current_weight = current_positions.get(symbol, 0)
            
            weight_diff = target_weight - current_weight
            price = prices.get(symbol, 0)
            
            # 变化太小不调整
            if abs(weight_diff) < config.min_weight_change:
                result.at[idx, 'trade_direction'] = 'SKIP_TOO_SMALL'
                result.at[idx, 'filter_reason'] = 'weight_change_too_small'
                result.at[idx, 'target_weight'] = current_weight
                continue
            
            if price <= 0:
                result.at[idx, 'trade_direction'] = 'SKIP_NO_PRICE'
                result.at[idx, 'filter_reason'] = 'no_price_data'
                result.at[idx, 'target_weight'] = current_weight
                continue
            
            if total_equity <= 0 or not np.isfinite(total_equity):
                result.at[idx, 'trade_direction'] = 'SKIP_INVALID_EQUITY'
                result.at[idx, 'filter_reason'] = 'invalid_total_equity'
                result.at[idx, 'target_weight'] = current_weight
                continue
            
            diff_value = abs(total_equity * weight_diff)
            
            if diff_value <= 0 or not np.isfinite(diff_value):
                result.at[idx, 'trade_direction'] = 'SKIP_INVALID_DIFF'
                result.at[idx, 'filter_reason'] = 'invalid_diff_value'
                result.at[idx, 'target_weight'] = current_weight
                continue
            
            # 计算交易股数
            shares = int(diff_value / price / lot_size) * lot_size
            if shares * price < min_trade_value:
                result.at[idx, 'trade_direction'] = 'SKIP_MIN_TRADE'
                result.at[idx, 'filter_reason'] = 'below_min_trade_value'
                result.at[idx, 'target_weight'] = current_weight
                continue
            
            # 计算成本
            is_buy = weight_diff > 0
            commission = shares * price * commission_bps / 10000
            slippage = shares * price * slippage_bps / 10000
            stamp_duty = shares * price * stamp_duty_bps / 10000 if not is_buy else 0
            total_cost = commission + slippage + stamp_duty
            
            # 估算边际收益 (基于bucket校准)
            # 优先使用score_percentile，否则使用score
            score = row.get('score_percentile', row.get('score', 0))
            bucket = self._get_bucket_from_score(score, n_buckets=5)
            bucket_return = self._bucket_returns.get(bucket, 0.005)
            
            # 边际收益 = bucket预期收益 * 目标权重
            estimated_alpha = bucket_return * target_weight * total_equity
            
            result.at[idx, 'estimated_cost'] = total_cost
            result.at[idx, 'estimated_alpha'] = estimated_alpha
            result.at[idx, 'bucket'] = bucket
            
            # 交易决策逻辑
            if is_buy:
                # 买入：检查成本-收益比
                if estimated_alpha <= 0:
                    result.at[idx, 'trade_direction'] = 'SKIP_LOW_ALPHA'
                    result.at[idx, 'filter_reason'] = 'negative_bucket_alpha'
                    result.at[idx, 'target_weight'] = current_weight
                elif estimated_alpha < config.min_alpha_threshold * total_equity:
                    result.at[idx, 'trade_direction'] = 'SKIP_LOW_ALPHA'
                    result.at[idx, 'filter_reason'] = 'alpha_below_threshold'
                    result.at[idx, 'target_weight'] = current_weight
                elif total_cost / max(estimated_alpha, 1) > config.cost_to_alpha_ratio:
                    result.at[idx, 'trade_direction'] = 'SKIP_HIGH_COST'
                    result.at[idx, 'filter_reason'] = 'cost_exceeds_threshold'
                    result.at[idx, 'target_weight'] = current_weight
                else:
                    result.at[idx, 'trade_direction'] = 'BUY'
            else:
                # 卖出：直接允许，除非不满足最小交易额
                # 卖出本身就是alpha，保护资金
                result.at[idx, 'trade_direction'] = 'SELL'
        
        return result
    
    def _get_bucket_from_score(self, score: float, n_buckets: int = 5) -> int:
        """根据分数确定bucket
        
        直接使用[0,1]范围的percentile值，按等分位划分
        """
        # 假设score已经是[0,1]的percentile
        if 0 <= score <= 1:
            if score >= 0.8:
                return 5
            elif score >= 0.6:
                return 4
            elif score >= 0.4:
                return 3
            elif score >= 0.2:
                return 2
            else:
                return 1
        
        # 如果不是percentile，尝试使用rank百分比
        # 这里用线性插值简化处理
        percentile = (score + 1) / 2  # 假设score在[-1, 1]范围
        percentile = max(0, min(1, percentile))
        
        if percentile >= 0.8:
            return 5
        elif percentile >= 0.6:
            return 4
        elif percentile >= 0.4:
            return 3
        elif percentile >= 0.2:
            return 2
        else:
            return 1
    
    def get_trade_summary(self, filtered_weights: pd.DataFrame) -> dict:
        """获取交易摘要"""
        if filtered_weights.empty:
            return {}
        
        summary = {
            'total_trades': len(filtered_weights),
            'buy_trades': len(filtered_weights[filtered_weights['trade_direction'] == 'BUY']),
            'sell_trades': len(filtered_weights[filtered_weights['trade_direction'] == 'SELL']),
            'hold_trades': len(filtered_weights[filtered_weights['trade_direction'] == 'HOLD']),
            'skipped_trades': len(filtered_weights[filtered_weights['trade_direction'].str.startswith('SKIP')]),
            'total_estimated_cost': filtered_weights['estimated_cost'].sum(),
            'total_estimated_alpha': filtered_weights['estimated_alpha'].sum(),
        }
        
        # 统计跳过原因
        skip_reasons = filtered_weights[filtered_weights['trade_direction'].str.startswith('SKIP')]['trade_direction'].value_counts()
        summary['skip_reasons'] = skip_reasons.to_dict()
        
        return summary


class PortfolioEnhancer:
    """
    组合增强器 (已修复版)
    
    整合持仓缓冲区、权重平滑、成本过滤三个组件
    
    修复：
    1. buffer结果正确传递给smoother
    2. chain中每步结果正确流转
    """
    
    def __init__(
        self,
        buffer_config: Optional[BufferConfig] = None,
        smoother_config: Optional[SmootherConfig] = None,
        cost_config: Optional[CostFilterConfig] = None,
    ):
        self.buffer = PositionBuffer(buffer_config)
        self.smoother = WeightSmoother(smoother_config)
        self.cost_filter = CostAlphaFilter(cost_config)
    
    def enhance(
        self,
        candidates: pd.DataFrame,
        current_positions: dict[str, float],
        target_weights: pd.DataFrame,
        prices: dict[str, float],
        execution_date: pd.Timestamp,
        total_equity: float,
        lot_size: int = 100,
        commission_bps: float = 0.75,
        stamp_duty_bps: float = 10.0,
        slippage_bps: float = 5.0,
        min_trade_value: float = 2000.0,
    ) -> tuple[pd.DataFrame, dict]:
        """
        增强组合构建 (已修复版)
        
        流程：
        1. 持仓缓冲区过滤 → buffered_weights
        2. 权重平滑 → 使用buffered结果，不是原始target_weights
        3. 成本-收益过滤
        
        Returns:
            (增强后的目标权重, 交易摘要)
        """
        # Step 1: 持仓缓冲区
        buffered = self.buffer.apply(
            candidates,
            current_positions,
            execution_date
        )
        
        # 记录缓冲保留了哪些股票
        buffered_symbols = set(buffered['symbol'].tolist()) if len(buffered) > 0 else set()
        original_symbols = set(candidates['symbol'].tolist()) if len(candidates) > 0 else set()
        
        # 合并buffer结果到target_weights
        if 'is_holding' in buffered.columns and len(buffered) > 0:
            # 创建symbol -> weight_multiplier的映射
            multiplier_map = {}
            for _, row in buffered.iterrows():
                mult = row.get('weight_multiplier', 1.0)
                if row.get('is_holding', False) and not row.get('buffer_retained', False):
                    mult = 1.0  # 正常保留的不乘
                multiplier_map[row['symbol']] = mult
            
            # 更新target_weights中的权重
            target_weights = target_weights.copy()
            target_weights['weight_multiplier'] = target_weights['symbol'].map(multiplier_map).fillna(1.0)
            target_weights['target_weight'] = target_weights['target_weight'] * target_weights['weight_multiplier']
        else:
            target_weights = target_weights.copy()
        
        # Step 2: 权重平滑 - 使用buffered后的target_weights
        smoothed = self.smoother.smooth(target_weights, execution_date)
        
        # Step 3: 成本-收益过滤
        filtered = self.cost_filter.filter_trades(
            smoothed,
            current_positions,
            prices,
            lot_size,
            commission_bps,
            stamp_duty_bps,
            slippage_bps,
            min_trade_value,
            total_equity,
        )
        
        summary = self.cost_filter.get_trade_summary(filtered)
        summary['buffered_retained'] = len(buffered_symbols)
        summary['buffered_removed'] = len(original_symbols) - len(buffered_symbols)
        
        return filtered, summary
    
    def reset(self):
        """重置所有状态"""
        self.buffer.reset()
        self.smoother.reset()
    
    def set_current_positions(self, positions: dict[str, float], weights: dict[str, float]):
        """设置当前持仓（用于初始化）"""
        self.buffer._previous_holdings = {
            symbol: {'score': 0, 'rank': 0, 'weight': w}
            for symbol, w in weights.items()
        }
        self.smoother.set_current_weights(weights)
