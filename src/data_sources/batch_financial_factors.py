"""
批量获取财务因子数据

使用方式:
python -m src.data_sources.batch_financial_factors --symbols-file data/symbols.txt --start-year 2018
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.core.logging_utils import get_logger
from src.data_sources.disclosure_date_cache import DisclosureDateCache
from src.data_sources.financial_data import (
    FINANCIAL_FACTOR_MAPPING,
    FinancialDataCache,
    FinancialFactorCalculator,
)
from src.ops.paths import RAW_DATA_DIR

logger = get_logger(__name__)

# 全局披露日期缓存，避免重复请求
_disclosure_cache: DisclosureDateCache | None = None


def _get_disclosure_cache() -> DisclosureDateCache:
    """获取或创建全局披露日期缓存"""
    global _disclosure_cache
    if _disclosure_cache is None:
        _disclosure_cache = DisclosureDateCache()
    return _disclosure_cache


def get_financial_factors_for_symbol(
    symbol: str,
    start_year: str = '2018',
    use_real_disclosure_dates: bool = True,
) -> pd.DataFrame:
    """获取单个股票的财务因子时间序列
    
    Args:
        symbol: 股票代码
        start_year: 开始年份
        use_real_disclosure_dates: 是否使用真实披露日期(从akshare获取)
            若为False，回退到规则化日期
    """
    fin_cache = FinancialDataCache()
    disc_cache = _get_disclosure_cache() if use_real_disclosure_dates else None

    try:
        df = fin_cache.get_financial_indicator(symbol, start_year=start_year)
        if df.empty:
            return pd.DataFrame()

        calc = FinancialFactorCalculator(df)

        # 获取所有可用的财务因子
        results = []
        for _, row in df.iterrows():
            period = row['日期']
            if pd.isna(period):
                continue

            # 获取披露日期
            period_dt = pd.to_datetime(period)
            if disc_cache is not None:
                pub_date = disc_cache.get_report_period_to_disclosure_date(symbol, period_dt)
            else:
                pub_date = _get_pub_date(period)

            record = {
                'symbol': symbol,
                'report_date': period_dt,
                'pub_date': pub_date,  # 财报真实披露日期
            }

            # 提取各指标
            for col, factor_name in FINANCIAL_FACTOR_MAPPING.items():
                if col in row.index and pd.notna(row[col]):
                    try:
                        record[factor_name] = float(row[col])
                    except (ValueError, TypeError):
                        pass

            results.append(record)

        return pd.DataFrame(results)

    except Exception as e:
        logger.warning(f'Error processing {symbol}: {e}')
        return pd.DataFrame()


def _get_pub_date(report_date: str) -> pd.Timestamp:
    """根据报告期推算发布日期
    
    财报发布规则:
    - Q1: 4月底前
    - Q2: 8月底前
    - Q3: 10月底前
    - Q4/年度: 次年4月底前
    """
    date = pd.to_datetime(report_date)

    if date.month == 3:
        return pd.Timestamp(f'{date.year}-04-30')
    elif date.month == 6:
        return pd.Timestamp(f'{date.year}-08-31')
    elif date.month == 9:
        return pd.Timestamp(f'{date.year}-10-31')
    elif date.month == 12:
        return pd.Timestamp(f'{date.year + 1}-04-30')
    else:
        return date


def get_latest_financial_factors(symbols: list[str], as_of_date: str) -> pd.DataFrame:
    """获取指定日期可用的最新财务因子
    
    Args:
        symbols: 股票代码列表
        as_of_date: 参考日期
    
    Returns:
        DataFrame with columns: [symbol, factor_name, value]
    """
    cache = FinancialDataCache()
    as_of = pd.to_datetime(as_of_date)
    results = []

    for symbol in symbols:
        try:
            df = cache.get_financial_indicator(symbol, start_year='2018')
            if df.empty:
                continue

            calc = FinancialFactorCalculator(df)
            factors = calc.calculate_all_factors()

            # 找到最近可用的财务数据
            if '日期' in df.columns:
                df['report_date'] = pd.to_datetime(df['日期'])
                available = df[df['report_date'] <= as_of]

                if not available.empty:
                    latest_date = available['report_date'].max()
                    latest_row = df[df['report_date'] == latest_date].iloc[0]

                    for col, factor_name in FINANCIAL_FACTOR_MAPPING.items():
                        if col in latest_row.index and pd.notna(latest_row[col]):
                            try:
                                results.append({
                                    'symbol': symbol,
                                    'factor_name': factor_name,
                                    'value': float(latest_row[col]),
                                    'report_date': latest_date,
                                })
                            except (ValueError, TypeError):
                                pass

        except Exception:
            continue

    return pd.DataFrame(results)


def build_financial_factor_panel(
    symbols: list[str],
    trade_dates: list[str],
    start_year: str = '2018',
    max_workers: int = 16,
) -> pd.DataFrame:
    """构建财务因子面板
    
    Args:
        symbols: 股票代码列表
        trade_dates: 交易日期列表
        start_year: 开始年份
        max_workers: 并行线程数
    
    Returns:
        DataFrame with columns: [trade_date, symbol, factor_name, value]
    """
    cache = FinancialDataCache()

    # 1. 并行获取所有股票的财务数据
    logger.info(f'Fetching financial data for {len(symbols)} symbols...')

    all_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_financial_factors_for_symbol, sym, start_year): sym
            for sym in symbols
        }

        for i, future in enumerate(as_completed(futures)):
            symbol = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    all_data.append(df)

                if (i + 1) % 50 == 0:
                    logger.info(f'Progress: {i+1}/{len(symbols)}')
            except Exception as e:
                logger.warning(f'Error for {symbol}: {e}')

    if not all_data:
        logger.warning('No financial data retrieved!')
        return pd.DataFrame()

    # 合并所有数据
    financial_panel = pd.concat(all_data, ignore_index=True)
    logger.info(f'Financial data shape: {financial_panel.shape}')

    # 2. 为每个交易日匹配可用的财务数据
    trade_dates = sorted([pd.to_datetime(d) for d in trade_dates])
    results = []

    for trade_date in trade_dates:
        # 找到在交易日前发布的财务数据
        mask = financial_panel['pub_date'] <= trade_date
        available = financial_panel[mask]

        if available.empty:
            continue

        # 对每只股票取最新数据
        latest = available.sort_values('pub_date').groupby(['symbol', 'factor_name']).last().reset_index()

        for _, row in latest.iterrows():
            results.append({
                'trade_date': trade_date,
                'symbol': row['symbol'],
                'factor_name': row['factor_name'],
                'value': row['value'],
            })

    return pd.DataFrame(results)


def save_financial_factor_cache(
    symbols: list[str],
    output_path: Path,
    start_year: str = '2018',
):
    """保存财务因子缓存"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_data = []
    cache = FinancialDataCache()

    for i, symbol in enumerate(symbols):
        try:
            df = cache.get_financial_indicator(symbol, start_year=start_year)
            if not df.empty:
                df['symbol'] = symbol
                all_data.append(df)

            if (i + 1) % 20 == 0:
                logger.info(f'Progress: {i+1}/{len(symbols)}')
                time.sleep(0.5)  # 避免请求过快

        except Exception as e:
            logger.warning(f'Error for {symbol}: {e}')

    if all_data:
        panel = pd.concat(all_data, ignore_index=True)
        panel.to_parquet(output_path, index=False)
        logger.info(f'Saved financial data to {output_path}')
    else:
        logger.warning('No data to save!')


# 注册的财务因子列表
FINANCIAL_FACTORS = [
    # Value
    'book_to_price',
    'earnings_yield',
    'gross_margin',
    'operating_margin',
    'net_margin',
    # Profitability
    'roe',
    'roe_weighted',
    'roa',
    'total_roa',
    'ocf_per_share',
    # Growth
    'revenue_growth',
    'profit_growth',
    'equity_growth',
    'asset_growth',
    # Leverage
    'debt_ratio',
    'current_ratio',
    'quick_ratio',
    'cash_ratio',
    'interest_coverage',
    # Efficiency
    'asset_turnover',
    'inv_turnover',
    'ar_turnover',
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch fetch financial factors')
    parser.add_argument('--symbols-file', type=str, help='File with one symbol per line')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--start-year', type=str, default='2018')
    parser.add_argument('--output', type=str, default=str(RAW_DATA_DIR / 'financial_factors.parquet'))
    parser.add_argument('--workers', type=int, default=8)

    args = parser.parse_args()

    # 获取股票列表
    if args.symbols_file:
        symbols = Path(args.symbols_file).read_text().strip().split('\n')
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = ['sh.600000', 'sh.600036', 'sz.000001']

    logger.info(f'Processing {len(symbols)} symbols...')

    save_financial_factor_cache(
        symbols=symbols,
        output_path=Path(args.output),
        start_year=args.start_year,
    )
