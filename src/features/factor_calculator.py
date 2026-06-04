"""
因子计算引擎 - 批量计算因子

支持:
1. 价格/量基础因子
2. Barra风格因子
3. WorldQuant Alpha因子
4. 行业因子
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorResult:
    """因子计算结果"""
    name: str
    panel: pd.DataFrame
    computation_time_ms: float
    notes: str = ""


class FactorCalculator:
    """因子计算器"""

    def __init__(self, bars: pd.DataFrame, security_master: pd.DataFrame | None = None):
        """
        初始化计算器
        
        Args:
            bars: 日线数据，包含 trade_date, symbol, open, high, low, close, volume, amount, adj_close
            security_master: 股票信息表，包含 symbol, industry, board 等
        """
        self.bars = bars.copy()
        self.security_master = security_master

        # 确保数据排序
        self.bars = self.bars.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

        # 确保日期格式
        self.bars['trade_date'] = pd.to_datetime(self.bars['trade_date'])

        # 预处理
        self._prepare_data()

    def _prepare_data(self):
        """数据预处理"""
        # 计算基础指标
        self.bars['return'] = self.bars.groupby('symbol')['adj_close'].pct_change()

        # 日内range
        self.bars['high_low_range'] = (self.bars['high'] - self.bars['low']) / self.bars['close']

        # 成交量标准化
        self.bars['vol_normalized'] = self.bars.groupby('symbol')['volume'].transform(
            lambda x: x / x.rolling(20, min_periods=1).mean()
        )

        # 复权价收益率
        self.bars['pct_change_1d'] = self.bars.groupby('symbol')['adj_close'].pct_change(1)
        self.bars['pct_change_5d'] = self.bars.groupby('symbol')['adj_close'].pct_change(5)
        self.bars['pct_change_20d'] = self.bars.groupby('symbol')['adj_close'].pct_change(20)

    def calculate_all(
        self,
        factor_names: list[str],
        **kwargs
    ) -> dict[str, pd.DataFrame]:
        """批量计算因子"""
        results = {}
        for name in factor_names:
            try:
                result = self.calculate(name, **kwargs)
                results[name] = result
            except Exception as e:
                print(f"计算 {name} 失败: {e}")
        return results

    def calculate(self, factor_name: str, **kwargs) -> pd.DataFrame:
        """计算单个因子"""
        method_name = f"_calc_{factor_name}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(**kwargs)
        else:
            raise ValueError(f"Unknown factor: {factor_name}")

    # ===== 基础价格因子 =====
    def _calc_mom(self, window: int = 20) -> pd.DataFrame:
        """动量因子"""
        result = self.bars[['trade_date', 'symbol']].copy()
        result['value'] = self.bars.groupby('symbol')['adj_close'].pct_change(window)
        return result

    def _calc_reversal(self, window: int = 5) -> pd.DataFrame:
        """反转因子"""
        result = self.bars[['trade_date', 'symbol']].copy()
        result['value'] = -self.bars.groupby('symbol')['adj_close'].pct_change(window)
        return result

    def _calc_vol(self, window: int = 20) -> pd.DataFrame:
        """波动率因子"""
        result = self.bars[['trade_date', 'symbol']].copy()
        result['value'] = self.bars.groupby('symbol')['return'].transform(
            lambda x: x.rolling(window, min_periods=10).std() * np.sqrt(252)
        )
        return result

    def _calc_vol_rank(self, window: int = 20) -> pd.DataFrame:
        """波动率排名因子"""
        result = self.bars[['trade_date', 'symbol']].copy()
        vol = self.bars.groupby('symbol')['return'].transform(
            lambda x: x.rolling(window, min_periods=10).std()
        )
        result['value'] = vol.groupby(self.bars['trade_date']).rank(pct=True)
        return result

    # ===== 价格位置因子 =====
    def _calc_close_to_high(self, window: int = 20) -> pd.DataFrame:
        """收盘价/窗口最高价"""
        result = self.bars[['trade_date', 'symbol']].copy()
        high = self.bars.groupby('symbol')['adj_close'].transform(
            lambda x: x.rolling(window, min_periods=5).max()
        )
        result['value'] = self.bars['adj_close'] / high.replace(0, np.nan)
        return result

    def _calc_close_to_low(self, window: int = 20) -> pd.DataFrame:
        """收盘价/窗口最低价"""
        result = self.bars[['trade_date', 'symbol']].copy()
        low = self.bars.groupby('symbol')['adj_close'].transform(
            lambda x: x.rolling(window, min_periods=5).min()
        )
        result['value'] = self.bars['adj_close'] / low.replace(0, np.nan)
        return result

    def _calc_high_low_pos(self, window: int = 20) -> pd.DataFrame:
        """(收盘-最低)/(最高-最低)"""
        result = self.bars[['trade_date', 'symbol']].copy()
        high = self.bars.groupby('symbol')['high'].transform(
            lambda x: x.rolling(window, min_periods=5).max()
        )
        low = self.bars.groupby('symbol')['low'].transform(
            lambda x: x.rolling(window, min_periods=5).min()
        )
        denom = high - low
        result['value'] = (self.bars['adj_close'] - low) / denom.replace(0, np.nan)
        return result

    # ===== 技术指标 =====
    def _calc_rsi(self, window: int = 14) -> pd.DataFrame:
        """RSI"""
        result = self.bars[['trade_date', 'symbol']].copy()

        delta = self.bars.groupby('symbol')['adj_close'].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.groupby(self.bars['symbol']).transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )
        avg_loss = loss.groupby(self.bars['symbol']).transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )

        rs = avg_gain / avg_loss.replace(0, np.nan)
        result['value'] = 100 - (100 / (1 + rs))
        return result

    def _calc_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD"""
        result = self.bars[['trade_date', 'symbol']].copy()

        ema_fast = self.bars.groupby('symbol')['adj_close'].transform(
            lambda x: x.ewm(span=fast, adjust=False).mean()
        )
        ema_slow = self.bars.groupby('symbol')['adj_close'].transform(
            lambda x: x.ewm(span=slow, adjust=False).mean()
        )
        macd = ema_fast - ema_slow

        result['value'] = macd.groupby(self.bars['symbol']).transform(
            lambda x: x.ewm(span=signal, adjust=False).mean()
        )
        return result

    # ===== 成交量因子 =====
    def _calc_vol_ratio(self, short: int = 5, long: int = 20) -> pd.DataFrame:
        """成交量比"""
        result = self.bars[['trade_date', 'symbol']].copy()

        vol_short = self.bars.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(short, min_periods=1).mean()
        )
        vol_long = self.bars.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(long, min_periods=1).mean()
        )

        result['value'] = vol_short / vol_long.replace(0, np.nan)
        return result

    def _calc_amount_growth(self, window: int = 20) -> pd.DataFrame:
        """成交额增长"""
        result = self.bars[['trade_date', 'symbol']].copy()
        result['value'] = self.bars.groupby('symbol')['amount'].pct_change(window)
        return result

    # ===== 均线因子 =====
    def _calc_ma_diff(self, short: int = 5, long: int = 20) -> pd.DataFrame:
        """均线差值"""
        result = self.bars[['trade_date', 'symbol']].copy()

        ma_short = self.bars.groupby('symbol')['adj_close'].transform(
            lambda x: x.rolling(short, min_periods=1).mean()
        )
        ma_long = self.bars.groupby('symbol')['adj_close'].transform(
            lambda x: x.rolling(long, min_periods=1).mean()
        )

        result['value'] = ma_short / ma_long.replace(0, np.nan)
        return result

    def _calc_price_to_ma(self, window: int = 20) -> pd.DataFrame:
        """价格/均线"""
        result = self.bars[['trade_date', 'symbol']].copy()

        ma = self.bars.groupby('symbol')['adj_close'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        result['value'] = self.bars['adj_close'] / ma.replace(0, np.nan)
        return result

    # ===== WorldQuant Alpha计算 =====
    def _calc_alpha_001(self) -> pd.DataFrame:
        """Alpha#001: rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 15)) - 0.5"""
        result = self.bars[['trade_date', 'symbol']].copy()

        # SignedPower
        close = self.bars['adj_close']
        returns = self.bars['return']

        signed_power = returns.clip(lower=0).fillna(0) * returns.apply(lambda x: x**2 if x < 0 else 1)

        # Ts_ArgMax over 15 days
        def ts_argmax_15(x):
            return x.rolling(15, min_periods=5).apply(lambda y: y.argmax() if len(y) > 0 else np.nan, raw=True)

        argmax_val = returns.groupby(self.bars['symbol']).transform(ts_argmax_15)

        # Rank
        result['value'] = argmax_val.groupby(self.bars['trade_date']).rank(pct=True) - 0.5
        return result

    def _calc_alpha_002(self) -> pd.DataFrame:
        """Alpha#002: (-1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6))"""
        result = self.bars[['trade_date', 'symbol']].copy()

        log_vol = np.log(self.bars['volume'].replace(0, np.nan))
        delta_log_vol = log_vol.groupby(self.bars['symbol']).diff(2)

        daily_return = (self.bars['close'] - self.bars['open']) / self.bars['open']

        # Rolling correlation over 6 days
        def rolling_corr_6(x_vol, x_ret):
            return x_vol.rolling(6, min_periods=3).corr(x_ret)

        corr = delta_log_vol.groupby(self.bars['symbol']).transform(
            lambda x: rolling_corr_6(x, daily_return.loc[x.index])
        )

        result['value'] = -corr
        return result

    def _calc_alpha_004(self) -> pd.DataFrame:
        """Alpha#004: (-1 * ts_rank(rank(low), 9))"""
        result = self.bars[['trade_date', 'symbol']].copy()

        rank_low = self.bars.groupby('trade_date')['low'].rank(pct=True)

        def ts_rank_9(x):
            return x.rolling(9, min_periods=5).apply(lambda y: (len(y) - y.argsort().argsort()[0]) / len(y), raw=True)

        ts_rank = rank_low.groupby(self.bars['symbol']).transform(ts_rank_9)

        result['value'] = -ts_rank
        return result

    def _calc_alpha_006(self) -> pd.DataFrame:
        """Alpha#006: (-1 * correlation(open, volume, 10))"""
        result = self.bars[['trade_date', 'symbol']].copy()

        def rolling_corr_10(x, y):
            return x.rolling(10, min_periods=5).corr(y)

        corr = self.bars.groupby('symbol').apply(
            lambda g: rolling_corr_10(g['open'], g['volume']),
            include_groups=False
        )

        result['value'] = -corr.values
        return result

    def _calc_alpha_012(self) -> pd.DataFrame:
        """Alpha#012: sign(delta(volume, 1)) * (-1 * delta(close, 1))"""
        result = self.bars[['trade_date', 'symbol']].copy()

        vol_change = self.bars.groupby('symbol')['volume'].diff()
        close_change = self.bars.groupby('symbol')['adj_close'].diff()

        result['value'] = np.sign(vol_change) * (-close_change)
        return result

    def _calc_alpha_013(self) -> pd.DataFrame:
        """Alpha#013: (-1 * rank(covariance(rank(close), rank(volume), 5)))"""
        result = self.bars[['trade_date', 'symbol']].copy()

        rank_close = self.bars.groupby('trade_date')['adj_close'].rank(pct=True)
        rank_vol = self.bars.groupby('trade_date')['volume'].rank(pct=True)

        def rolling_cov_5(x, y):
            return x.rolling(5, min_periods=3).cov(y)

        cov = rank_close.rolling(5, min_periods=3).cov(rank_vol)

        result['value'] = -rank_close.groupby(self.bars['symbol']).transform(
            lambda x: rolling_cov_5(x, rank_vol.loc[x.index])
        )
        return result

    def _calc_alpha_016(self) -> pd.DataFrame:
        """Alpha#016: (-1 * correlation(rank(high), rank(volume), 5))"""
        result = self.bars[['trade_date', 'symbol']].copy()

        rank_high = self.bars.groupby('trade_date')['high'].rank(pct=True)
        rank_vol = self.bars.groupby('trade_date')['volume'].rank(pct=True)

        def rolling_corr_5(x, y):
            return x.rolling(5, min_periods=3).corr(y)

        corr = rank_high.groupby(self.bars['symbol']).transform(
            lambda x: rolling_corr_5(x, rank_vol.loc[x.index])
        )

        result['value'] = -corr
        return result

    def _calc_alpha_031(self) -> pd.DataFrame:
        """Alpha#031: ((ts_rank(volume, 32) * (1 - ts_rank(((close + high) - low), 16))) * (1 - ts_rank(returns, 32)))"""
        result = self.bars[['trade_date', 'symbol']].copy()

        # ts_rank函数
        def ts_rank(x, window):
            return x.rolling(window, min_periods=int(window/2)).apply(
                lambda y: pd.Series(y).rank(pct=True).iloc[-1] if len(y) > 0 else np.nan,
                raw=True
            )

        ts_rank_vol = self.bars.groupby('trade_date')['volume'].rank(pct=True).groupby(
            self.bars['symbol']
        ).transform(lambda x: ts_rank(x, 32))

        hlc = self.bars['high'] + self.bars['low']
        ts_rank_hlc = self.bars.groupby('trade_date')['close'].rank(pct=True).groupby(
            self.bars['symbol']
        ).transform(lambda x: ts_rank(x, 16))

        ts_rank_ret = self.bars.groupby('trade_date')['return'].rank(pct=True).groupby(
            self.bars['symbol']
        ).transform(lambda x: ts_rank(x, 32))

        result['value'] = ts_rank_vol * (1 - ts_rank_hlc) * (1 - ts_rank_ret)
        return result

    def _calc_alpha_041(self) -> pd.DataFrame:
        """Alpha#041: (((sum(close, 7) / 7) - close) + (20 * correlation((close - vwap), delay(close, 5), 230)))"""
        result = self.bars[['trade_date', 'symbol']].copy()

        # 简化版: 使用amount/volume作为VWAP代理
        vwap = self.bars['amount'] / self.bars['volume'].replace(0, np.nan)

        ma_close = self.bars.groupby('symbol')['adj_close'].transform(
            lambda x: x.rolling(7, min_periods=3).mean()
        )

        close_minus_vwap = self.bars['adj_close'] - vwap

        # 简化: 使用过去5日相关性
        def simple_corr(x, y):
            return x.rolling(5, min_periods=3).corr(y)

        corr = close_minus_vwap.groupby(self.bars['symbol']).transform(
            lambda x: simple_corr(x, self.bars.groupby('symbol')['adj_close'].shift(5).loc[x.index])
        )

        result['value'] = (ma_close - self.bars['adj_close']) + 20 * corr
        return result

    # ===== Barra风格因子 =====
    def _calc_beta(self, window: int = 252) -> pd.DataFrame:
        """市场Beta"""
        result = self.bars[['trade_date', 'symbol']].copy()

        # 计算市场收益
        market_ret = self.bars.groupby('trade_date')['adj_close'].last().pct_change()
        market_ret.name = 'market_ret'

        df = self.bars.merge(
            market_ret.reset_index().rename(columns={'index': 'trade_date'}),
            on='trade_date',
            how='left'
        )

        # 滚动Beta
        def rolling_beta(x):
            market = df.loc[x.index, 'market_ret']
            if market.std() == 0:
                return np.nan
            return x.cov(market) / market.var()

        result['value'] = self.bars.groupby('symbol')['return'].transform(rolling_beta)
        return result

    def _calc_size(self) -> pd.DataFrame:
        """对数市值因子 (使用收盘价作为市值代理)"""
        result = self.bars[['trade_date', 'symbol']].copy()

        # 使用收盘价作为市值代理 (实际应使用总市值)
        result['value'] = np.log(self.bars['close'].replace(0, np.nan))
        return result

    def _calc_turnover_rate(self, window: int = 20) -> pd.DataFrame:
        """换手率"""
        result = self.bars[['trade_date', 'symbol']].copy()

        result['value'] = self.bars.groupby('symbol')['volume'].transform(
            lambda x: x / x.rolling(window, min_periods=5).mean()
        )
        return result

    # ===== 行业因子 =====
    def _calc_sector_mom(self, window: int = 20) -> pd.DataFrame:
        """行业动量 (需要security_master)"""
        result = self.bars[['trade_date', 'symbol']].copy()

        if self.security_master is not None:
            # 获取行业
            sector_map = self.security_master.set_index('symbol')['industry'].to_dict()
            self.bars['industry'] = self.bars['symbol'].map(sector_map)

            # 计算行业动量
            sector_ret = self.bars.groupby(['trade_date', 'industry'])['adj_close'].last().pct_change(window)

            # 映射回股票
            self.bars['sector_return'] = self.bars.set_index(['trade_date', 'industry']).index.map(
                sector_ret.to_dict()
            ).fillna(0)

            result['value'] = self.bars['sector_return']
        else:
            result['value'] = 0

        return result

    # ===== 形态学因子 =====
    def _calc_candle_body_ratio(self) -> pd.DataFrame:
        """蜡烛实体占比"""
        result = self.bars[['trade_date', 'symbol']].copy()

        body = (self.bars['close'] - self.bars['open']).abs()
        total_range = self.bars['high'] - self.bars['low']

        result['value'] = body / total_range.replace(0, np.nan)
        return result

    def _calc_upper_shadow(self) -> pd.DataFrame:
        """上影线占比"""
        result = self.bars[['trade_date', 'symbol']].copy()

        upper_shadow = self.bars['high'] - self.bars[['close', 'open']].max(axis=1)
        total_range = self.bars['high'] - self.bars['low']

        result['value'] = upper_shadow / total_range.replace(0, np.nan)
        return result

    def _calc_lower_shadow(self) -> pd.DataFrame:
        """下影线占比"""
        result = self.bars[['trade_date', 'symbol']].copy()

        lower_shadow = self.bars[['close', 'open']].min(axis=1) - self.bars['low']
        total_range = self.bars['high'] - self.bars['low']

        result['value'] = lower_shadow / total_range.replace(0, np.nan)
        return result

    def _calc_volume_price_divergence(self, window: int = 20) -> pd.DataFrame:
        """量价背离"""
        result = self.bars[['trade_date', 'symbol']].copy()

        price_ret = self.bars.groupby('symbol')['adj_close'].pct_change(window)
        vol_change = self.bars.groupby('symbol')['volume'].pct_change(window)

        result['value'] = price_ret - vol_change
        return result

    def _calc_trend_strength(self, window: int = 20) -> pd.DataFrame:
        """趋势强度"""
        result = self.bars[['trade_date', 'symbol']].copy()

        # 使用线性回归斜率作为趋势代理
        def trend_strength(x):
            if len(x) < 5:
                return np.nan
            y = np.arange(len(x))
            return np.polyfit(y, x, 1)[0] / x.mean() if x.mean() != 0 else 0

        result['value'] = self.bars.groupby('symbol')['adj_close'].transform(
            lambda x: x.rolling(window, min_periods=5).apply(trend_strength, raw=True)
        )
        return result


def calculate_factor_panel(
    bars: pd.DataFrame,
    factor_names: list[str],
    security_master: pd.DataFrame | None = None,
    winsorize: bool = True,
    zscore: bool = True,
) -> pd.DataFrame:
    """
    计算因子面板数据的便捷函数
    
    Args:
        bars: 日线数据
        factor_names: 因子名称列表
        security_master: 股票信息表
        winsorize: 是否去极值
        zscore: 是否标准化
    
    Returns:
        因子面板 (trade_date, symbol, factor1, factor2, ...)
    """
    calculator = FactorCalculator(bars, security_master)

    # 计算所有因子
    panels = [bars[['trade_date', 'symbol']].copy()]

    for name in factor_names:
        try:
            df = calculator.calculate(name)
            panels.append(df[['value']].rename(columns={'value': name}))
        except Exception as e:
            print(f"Warning: {name} failed - {e}")

    # 合并
    panel = panels[0]
    for p in panels[1:]:
        panel = panel.merge(p, on=['trade_date', 'symbol'], how='left')

    # 后处理
    if winsorize:
        for col in factor_names:
            if col in panel.columns:
                lower = panel[col].quantile(0.01)
                upper = panel[col].quantile(0.99)
                panel[col] = panel[col].clip(lower, upper)

    if zscore:
        for col in factor_names:
            if col in panel.columns:
                panel[col] = panel.groupby('trade_date')[col].transform(
                    lambda x: (x - x.mean()) / x.std().replace(0, 1)
                )

    return panel


# 常用因子映射
BASIC_FACTORS = {
    'mom20': ('mom', {'window': 20}),
    'mom60': ('mom', {'window': 60}),
    'mom120': ('mom', {'window': 120}),
    'mom250': ('mom', {'window': 250}),
    'rev5': ('reversal', {'window': 5}),
    'rev20': ('reversal', {'window': 20}),
    'vol20': ('vol', {'window': 20}),
    'vol60': ('vol', {'window': 60}),
    'close_to_high20': ('close_to_high', {'window': 20}),
    'close_to_high60': ('close_to_high', {'window': 60}),
    'close_to_high250': ('close_to_high', {'window': 250}),
    'close_to_low20': ('close_to_low', {'window': 20}),
    'high_low_pos20': ('high_low_pos', {'window': 20}),
    'high_low_pos60': ('high_low_pos', {'window': 60}),
    'high_low_pos120': ('high_low_pos', {'window': 120}),
    'rsi6': ('rsi', {'window': 6}),
    'rsi14': ('rsi', {'window': 14}),
    'vol_ratio_5_20': ('vol_ratio', {'short': 5, 'long': 20}),
    'vol_ratio_20_60': ('vol_ratio', {'short': 20, 'long': 60}),
    'amount_growth20': ('amount_growth', {'window': 20}),
    'ma_diff_5_20': ('ma_diff', {'short': 5, 'long': 20}),
    'ma_diff_20_60': ('ma_diff', {'short': 20, 'long': 60}),
    'price_to_ma20': ('price_to_ma', {'window': 20}),
    'price_to_ma60': ('price_to_ma', {'window': 60}),
    'candle_body_ratio': ('candle_body_ratio', {}),
    'upper_shadow': ('upper_shadow', {}),
    'lower_shadow': ('lower_shadow', {}),
    'size': ('size', {}),
    'turnover_rate': ('turnover_rate', {'window': 20}),
}


if __name__ == "__main__":
    # 测试
    bars = pd.read_parquet('data/bronze/daily_bar.parquet')

    print("测试因子计算...")
    calc = FactorCalculator(bars)

    # 计算几个因子
    test_factors = ['mom20', 'mom120', 'close_to_high250', 'rsi14', 'vol_ratio_5_20']

    for f in test_factors:
        result = calc.calculate(f)
        print(f"  {f}: {result['value'].notna().sum()} 个有效值")
