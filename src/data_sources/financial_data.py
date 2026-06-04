"""
财务因子库 - 使用akshare获取财务数据

Barra因子需要的数据:
- Value: book_to_price, earnings_yield, sales_to_price, cashflow_yield
- Growth: earnings_growth, revenue_growth
- Profitability: roe, roa, gross_profitability
- Leverage: debt_to_assets, current_ratio
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logging_utils import get_logger

logger = get_logger(__name__)


class FinancialDataCache:
    """财务数据缓存管理器"""

    def __init__(self, cache_dir: str | None = None):
        from src.ops.paths import RAW_DATA_DIR
        self.cache_dir = RAW_DATA_DIR / 'financial_data' if cache_dir is None else cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, pd.DataFrame] = {}

    def get_financial_indicator(self, symbol: str, start_year: str = '2018') -> pd.DataFrame:
        """获取个股财务指标"""
        cache_path = self.cache_dir / f'{symbol.replace(".", "_")}_indicator.parquet'

        # 如果缓存存在且非空，使用缓存
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            if not df.empty:
                return df

        # 获取新数据
        try:
            import akshare as ak
            # 解析股票代码: sh.600000 -> 600000, sz.000001 -> 000001
            code = symbol.split('.')[1] if '.' in symbol else symbol
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year=start_year)
            if not df.empty:
                df.to_parquet(cache_path, index=False)
            return df
        except Exception as e:
            logger.warning(f'Failed to fetch financial data for {symbol}: {e}')
            return pd.DataFrame()

    def get_market_cap(self, symbol: str) -> pd.DataFrame:
        """获取市值数据"""
        cache_path = self.cache_dir / f'{symbol.replace(".", "_")}_market_cap.parquet'

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
        else:
            try:
                import akshare as ak
                df = ak.stock_a_liquidity(date='20240411')
                df.to_parquet(cache_path, index=False)
            except Exception as e:
                logger.warning(f'Failed to fetch market cap for {symbol}: {e}')
                return pd.DataFrame()

        return df


class FinancialFactorCalculator:
    """财务因子计算器"""

    def __init__(self, financial_df: pd.DataFrame, market_cap_df: pd.DataFrame | None = None):
        self.financial_df = financial_df
        self.market_cap_df = market_cap_df
        self._financial_dict: dict[str, dict] = {}

        if not self.financial_df.empty:
            self._build_financial_dict()

    def _build_financial_dict(self):
        """将财务数据转换为 {报告期: {指标: 值}} 格式"""
        if '日期' not in self.financial_df.columns:
            return

        for _, row in self.financial_df.iterrows():
            period = row['日期']
            if pd.isna(period):
                continue
            self._financial_dict[str(period)] = {}
            for col in self.financial_df.columns:
                if col != '日期':
                    val = row[col]
                    if pd.notna(val):
                        try:
                            self._financial_dict[str(period)][col] = float(val)
                        except (ValueError, TypeError):
                            pass

    def get_latest_value(self, indicator_name: str, periods: int = 4) -> float:
        """获取最近N期的财务指标值"""
        periods_list = list(self._financial_dict.keys())
        if not periods_list:
            return np.nan

        periods_list = sorted(periods_list, reverse=True)
        values = []

        for period in periods_list[:periods]:
            if period in self._financial_dict:
                if indicator_name in self._financial_dict[period]:
                    values.append(self._financial_dict[period][indicator_name])

        if values:
            return np.mean(values)
        return np.nan

    def calculate_value_factors(self) -> dict[str, float]:
        """计算价值因子"""
        factors = {}

        bvps = self.get_latest_value('每股净资产_调整后(元)')
        if pd.notna(bvps):
            factors['book_to_price'] = bvps

        eps = self.get_latest_value('摊薄每股收益(元)')
        if pd.notna(eps):
            factors['earnings_yield'] = eps

        gp_ratio = self.get_latest_value('销售毛利率(%)')
        if pd.notna(gp_ratio):
            factors['gross_margin'] = gp_ratio

        op_profit = self.get_latest_value('营业利润率(%)')
        if pd.notna(op_profit):
            factors['operating_margin'] = op_profit

        net_profit_margin = self.get_latest_value('销售净利率(%)')
        if pd.notna(net_profit_margin):
            factors['net_margin'] = net_profit_margin

        return factors

    def calculate_profitability_factors(self) -> dict[str, float]:
        """计算盈利质量因子"""
        factors = {}

        roe = self.get_latest_value('净资产收益率(%)')
        if pd.notna(roe):
            factors['roe'] = roe

        roe_weighted = self.get_latest_value('加权净资产收益率(%)')
        if pd.notna(roe_weighted):
            factors['roe_weighted'] = roe_weighted

        roa = self.get_latest_value('资产报酬率(%)')
        if pd.notna(roa):
            factors['roa'] = roa

        total_roa = self.get_latest_value('总资产利润率(%)')
        if pd.notna(total_roa):
            factors['total_roa'] = total_roa

        op_cf_per_share = self.get_latest_value('每股经营性现金流(元)')
        if pd.notna(op_cf_per_share):
            factors['ocf_per_share'] = op_cf_per_share

        return factors

    def calculate_growth_factors(self) -> dict[str, float]:
        """计算成长因子"""
        factors = {}

        rev_growth = self.get_latest_value('主营业务收入增长率(%)')
        if pd.notna(rev_growth):
            factors['revenue_growth'] = rev_growth

        profit_growth = self.get_latest_value('净利润增长率(%)')
        if pd.notna(profit_growth):
            factors['profit_growth'] = profit_growth

        equity_growth = self.get_latest_value('净资产增长率(%)')
        if pd.notna(equity_growth):
            factors['equity_growth'] = equity_growth

        asset_growth = self.get_latest_value('总资产增长率(%)')
        if pd.notna(asset_growth):
            factors['asset_growth'] = asset_growth

        return factors

    def calculate_leverage_factors(self) -> dict[str, float]:
        """计算杠杆因子"""
        factors = {}

        debt_ratio = self.get_latest_value('资产负债率(%)')
        if pd.notna(debt_ratio):
            factors['debt_ratio'] = debt_ratio

        current_ratio = self.get_latest_value('流动比率')
        if pd.notna(current_ratio):
            factors['current_ratio'] = current_ratio

        quick_ratio = self.get_latest_value('速动比率')
        if pd.notna(quick_ratio):
            factors['quick_ratio'] = quick_ratio

        cash_ratio = self.get_latest_value('现金比率(%)')
        if pd.notna(cash_ratio):
            factors['cash_ratio'] = cash_ratio

        interest_coverage = self.get_latest_value('利息支付倍数')
        if pd.notna(interest_coverage):
            factors['interest_coverage'] = interest_coverage

        return factors

    def calculate_efficiency_factors(self) -> dict[str, float]:
        """计算效率因子"""
        factors = {}

        asset_turnover = self.get_latest_value('总资产周转率(次)')
        if pd.notna(asset_turnover):
            factors['asset_turnover'] = asset_turnover

        inv_turnover = self.get_latest_value('存货周转率(次)')
        if pd.notna(inv_turnover):
            factors['inv_turnover'] = inv_turnover

        ar_turnover = self.get_latest_value('应收账款周转率(次)')
        if pd.notna(ar_turnover):
            factors['ar_turnover'] = ar_turnover

        return factors

    def calculate_all_factors(self) -> dict[str, float]:
        """计算所有财务因子"""
        factors = {}
        factors.update(self.calculate_value_factors())
        factors.update(self.calculate_profitability_factors())
        factors.update(self.calculate_growth_factors())
        factors.update(self.calculate_leverage_factors())
        factors.update(self.calculate_efficiency_factors())
        return factors


def compute_financial_factors_for_universe(
    symbols: list[str],
    trade_date: str,
    lookback_quarters: int = 4
) -> pd.DataFrame:
    """计算整个股票池的财务因子
    
    Args:
        symbols: 股票代码列表
        trade_date: 交易日期 (用于确定使用哪期财务数据)
        lookback_quarters: 回看多少期财报
    
    Returns:
        DataFrame with columns: [symbol, factor_name, value]
    """
    cache = FinancialDataCache()
    results = []

    for symbol in symbols:
        try:
            financial_df = cache.get_financial_indicator(symbol, start_year='2018')
            if financial_df.empty:
                continue

            calculator = FinancialFactorCalculator(financial_df)
            factors = calculator.calculate_all_factors()

            for factor_name, value in factors.items():
                if pd.notna(value):
                    results.append({
                        'symbol': symbol,
                        'trade_date': trade_date,
                        'factor_name': factor_name,
                        'value': value
                    })
        except Exception as e:
            logger.warning(f'Error computing factors for {symbol}: {e}')
            continue

    return pd.DataFrame(results)


# 财务因子名称映射 (akshare字段 -> Barra因子名)
FINANCIAL_FACTOR_MAPPING = {
    # ===== 盈利/价值因子 =====
    '每股净资产_调整后(元)': 'book_to_price',
    '摊薄每股收益(元)': 'earnings_yield',
    '加权每股收益(元)': 'earnings_yield_weighted',
    '扣除非经常性损益后的每股收益(元)': 'earnings_yield_ex',
    '销售毛利率(%)': 'gross_margin',
    '营业利润率(%)': 'operating_margin',
    '销售净利率(%)': 'net_margin',
    '成本费用利润率(%)': 'cost_profit_margin',
    '主营业务利润率(%)': 'mainbiz_profit_margin',

    # ===== Profitability因子 =====
    '净资产收益率(%)': 'roe',
    '加权净资产收益率(%)': 'roe_weighted',
    '资产报酬率(%)': 'roa',
    '总资产净利润率(%)': 'total_roa_net',
    '净资产报酬率(%)': 'roe_assets',
    '资产报酬率(%)': 'asset_return',
    '主营利润比重(%)': 'mainbiz_profit_ratio',
    '每股经营性现金流(元)': 'ocf_per_share',

    # ===== 每股指标 =====
    '每股未分配利润(元)': 'retained_earnings_per_share',
    '每股资本公积金(元)': 'capital_reserve_per_share',

    # ===== Growth因子 =====
    '主营业务收入增长率(%)': 'revenue_growth',
    '净利润增长率(%)': 'profit_growth',
    '净资产增长率(%)': 'equity_growth',
    '总资产增长率(%)': 'asset_growth',

    # ===== Leverage因子 =====
    '资产负债率(%)': 'debt_ratio',
    '流动比率': 'current_ratio',
    '速动比率': 'quick_ratio',
    '现金比率(%)': 'cash_ratio',
    '利息支付倍数': 'interest_coverage',
    '产权比率(%)': 'equity_ratio',
    '股东权益与固定资产比率(%)': 'equity_fixasset_ratio',
    '长期负债比率(%)': 'longterm_debt_ratio',
    '股东权益比率(%)': 'equity_ratio_total',
    '资本固定化比率(%)': 'capital_fix_ratio',
    '清算价值比率(%)': 'liquidation_ratio',

    # ===== Efficiency因子 =====
    '总资产周转率(次)': 'asset_turnover',
    '存货周转率(次)': 'inv_turnover',
    '应收账款周转率(次)': 'ar_turnover',
    '固定资产周转率(次)': 'fixasset_turnover',
    '存货周转天数(天)': 'inv_turnover_days',
    '应收账款周转天数(天)': 'ar_turnover_days',
    '总资产周转天数(天)': 'asset_turnover_days',

    # ===== 费用因子 =====
    '三项费用比重(%)': 'three_expense_ratio',

    # ===== 股息率 =====
    '股息发放率(%)': 'dividend_payout_ratio',

    # ===== 投资回报 =====
    '投资收益率(%)': 'investment_return_ratio',
}


if __name__ == '__main__':
    # 测试
    cache = FinancialDataCache()
    df = cache.get_financial_indicator('sh.600000', start_year='2020')

    if not df.empty:
        calc = FinancialFactorCalculator(df)
        factors = calc.calculate_all_factors()
        print('=== 600000 财务因子 ===')
        for name, value in factors.items():
            print(f'  {name}: {value:.4f}')
