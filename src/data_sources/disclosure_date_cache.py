"""
财报披露日期缓存 - 获取真实披露日期

使用 akshare stock_report_disclosure API 获取真实财报披露日期。
解决了使用规则化 tradeable_date (如"4月底") 导致的精度问题。

披露日期规则:
- 年报: 必须披露，stock_report_disclosure(period='YYYY年报') 提供真实日期
- 半年报: 必须披露，stock_report_disclosure(period='YYYY半年报') 提供真实日期
- 一季报: 非强制披露，通常随年报一起披露，使用年报日期或4月底
- 三季报: 非强制披露，通常随半年报一起披露，使用半年报日期或10月底

使用方式:
    from src.data_sources.disclosure_date_cache import DisclosureDateCache
    
    cache = DisclosureDateCache()
    # 获取单只股票的真实披露日期
    dates = cache.get_disclosure_dates('000001', '2023')
    print(dates)
    # {'Q1': Timestamp('2024-03-15'), 'Q2': Timestamp('2023-08-24'), 
    #  'Q3': Timestamp('2023-10-25'), 'Q4': Timestamp('2024-03-15')}
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.core.logging_utils import get_logger

logger = get_logger(__name__)


class DisclosureDateCache:
    """
    财报披露日期缓存
    
    从 akshare stock_report_disclosure API 获取真实披露日期，
    避免使用规则化日期带来的精度损失。
    """

    def __init__(self, cache_dir: str | None = None):
        from src.ops.paths import RAW_DATA_DIR
        self.cache_dir = RAW_DATA_DIR / 'disclosure_dates' if cache_dir is None else Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, pd.DataFrame] = {}

    def _get_cache_path(self, year: int, period_type: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f'disclosure_{year}_{period_type}.parquet'

    def _fetch_disclosure_dates(self, year: int, period_type: str = '年报') -> pd.DataFrame:
        """
        从 akshare 获取披露日期
        
        Args:
            year: 年份 (如 2023)
            period_type: 期间类型 ('年报' 或 '半年报')
        
        Returns:
            DataFrame with columns: [stock_code, actual_disclosure_date]
        """
        import akshare as ak

        cache_path = self._get_cache_path(year, period_type)

        # 检查磁盘缓存
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if not df.empty:
                    return df
            except Exception:
                pass

        # 获取新数据
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = ak.stock_report_disclosure(market='沪深京', period=f'{year}{period_type}')

                # 标准化列名
                if '股票代码' in df.columns:
                    df = df.rename(columns={'股票代码': 'stock_code', '实际披露': 'actual_date'})

                # 只保留需要的列
                if 'stock_code' in df.columns and 'actual_date' in df.columns:
                    result = df[['stock_code', 'actual_date']].copy()
                    result['actual_date'] = pd.to_datetime(result['actual_date'], errors='coerce')
                    result = result.dropna(subset=['actual_date'])

                    # 保存到磁盘缓存
                    result.to_parquet(cache_path, index=False)
                    logger.info(f'Cached {year}{period_type}: {len(result)} stocks')
                    return result

            except Exception as e:
                logger.warning(f'Attempt {attempt+1}/{max_retries} failed for {year}{period_type}: {e}')
                if attempt < max_retries - 1:
                    time.sleep(2)

        return pd.DataFrame(columns=['stock_code', 'actual_date'])

    def _fetch_annual_and_halfyear(self, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """同时获取年报和半年报日期"""
        annual_df = self._fetch_disclosure_dates(year, '年报')
        halfyear_df = self._fetch_disclosure_dates(year, '半年报')
        return annual_df, halfyear_df

    def get_disclosure_dates(self, symbol: str, year: int) -> dict[str, pd.Timestamp]:
        """
        获取单只股票指定年份的披露日期
        
        Args:
            symbol: 股票代码 (如 '000001' 或 '000001.SZ')
            year: 年份
        
        Returns:
            dict with keys: 'Q1', 'Q2', 'Q3', 'Q4'
            每个值是披露日期的 Timestamp，或 None 如果无法获取
        """
        # 解析股票代码
        code = symbol.split('.')[1] if '.' in symbol else symbol
        # 标准化格式 (去掉前缀0)
        code = code.lstrip('0')

        # 尝试从内存缓存获取
        cache_key = f'{year}'
        if cache_key not in self._memory_cache:
            annual_df, halfyear_df = self._fetch_annual_and_halfyear(year)
            self._memory_cache[cache_key] = (annual_df, halfyear_df)
        else:
            annual_df, halfyear_df = self._memory_cache[cache_key]

        # 查找年报日期 (Q4)
        annual_dates = annual_df[annual_df['stock_code'].str.endswith(code.lstrip('0')) |
                                  annual_df['stock_code'].str.endswith(code)]
        q4_date = None
        if not annual_dates.empty:
            q4_date = annual_dates.iloc[0]['actual_date']

        # 查找半年报日期 (Q2)
        halfyear_dates = halfyear_df[halfyear_df['stock_code'].str.endswith(code.lstrip('0')) |
                                      halfyear_df['stock_code'].str.endswith(code)]
        q2_date = None
        if not halfyear_dates.empty:
            q2_date = halfyear_dates.iloc[0]['actual_date']

        # Q1 和 Q3: 非强制披露
        # Q1 通常随年报一起披露，或在4月底前
        # Q3 通常随半年报一起披露，或在10月底前
        q1_date = q4_date  # 通常年报中包含Q1数据
        if q1_date is None:
            q1_date = pd.Timestamp(f'{year}-04-30')

        q3_date = q2_date  # 通常半年报中包含Q3数据
        if q3_date is None:
            q3_date = pd.Timestamp(f'{year}-10-31')

        return {
            'Q1': q1_date,  # 一季报 (通常包含在年报中)
            'Q2': q2_date,  # 半年报
            'Q3': q3_date,  # 三季报 (通常包含在半年报中)
            'Q4': q4_date,  # 年报
        }

    def get_report_period_to_disclosure_date(
        self,
        symbol: str,
        report_date: pd.Timestamp
    ) -> pd.Timestamp:
        """
        根据报告期获取对应的披露日期
        
        Args:
            symbol: 股票代码
            report_date: 报告期 (如 2023-03-31, 2023-06-30, etc.)
        
        Returns:
            披露日期
        """
        year = report_date.year
        quarter = (report_date.month - 1) // 3 + 1

        dates = self.get_disclosure_dates(symbol, year)

        quarter_key = f'Q{quarter}'
        disclosure_date = dates.get(quarter_key)

        if disclosure_date is None:
            # Fallback to rule-based date
            month = report_date.month
            if month == 3:
                return pd.Timestamp(f'{year}-04-30')
            elif month == 6:
                return pd.Timestamp(f'{year}-08-31')
            elif month == 9:
                return pd.Timestamp(f'{year}-10-31')
            elif month == 12:
                return pd.Timestamp(f'{year + 1}-04-30')

        return disclosure_date

    def warmup_cache(self, years: list[int], n_workers: int = 4):
        """
        预热缓存 - 并行获取多年数据
        
        Args:
            years: 要预热的年份列表
            n_workers: 并行线程数
        """
        def fetch_year(year):
            annual_df, halfyear_df = self._fetch_annual_and_halfyear(year)
            return year, annual_df, halfyear_df

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(fetch_year, year): year for year in years}
            for future in as_completed(futures):
                try:
                    year, annual_df, halfyear_df = future.result()
                    self._memory_cache[str(year)] = (annual_df, halfyear_df)
                    logger.info(f'Warmed cache for {year}: {len(annual_df)} annual, {len(halfyear_df)} halfyear')
                except Exception as e:
                    year = futures[future]
                    logger.warning(f'Failed to warm cache for {year}: {e}')

    def clear_cache(self):
        """清除缓存"""
        self._memory_cache.clear()
        for f in self.cache_dir.glob('*.parquet'):
            f.unlink()
        logger.info(f'Cleared cache at {self.cache_dir}')
