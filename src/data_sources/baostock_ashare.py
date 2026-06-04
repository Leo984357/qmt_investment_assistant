from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass

import akshare as ak
import baostock as bs
import pandas as pd

from src.core.logging_utils import get_logger
from src.ops.paths import RAW_DATA_DIR

from .base import BaseResearchDataSource

DAILY_FIELDS = 'date,code,open,high,low,close,preclose,volume,amount,tradestatus,isST,pctChg'
BAOSTOCK_ADJUST_FLAG = {'': '3', 'none': '3', 'qfq': '2', 'hfq': '1'}
UNIVERSE_ALIASES = {'HS300': 'hs300'}
logger = get_logger(__name__)


@dataclass(frozen=True)
class BaoStockAShareConfig:
    start_date: str
    end_date: str
    universe_name: str = 'HS300'
    formal_start_date: str | None = None
    universe_mode: str = 'point_in_time'
    universe_refresh_frequency_days: int = 1
    price_adjust: str = 'qfq'
    incremental: bool = True


class _BaoStockSession(AbstractContextManager):
    def __enter__(self) -> _BaoStockSession:
        result = bs.login()
        if result.error_code != '0':
            raise RuntimeError(f'baostock 登录失败: {result.error_msg}')
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        try:
            bs.logout()
        except Exception:
            pass
        return None

    @staticmethod
    def query_df(query_result) -> pd.DataFrame:
        if query_result.error_code != '0':
            raise RuntimeError(f'baostock 查询失败: {query_result.error_msg}')
        rows: list[list[str]] = []
        while query_result.next():
            rows.append(query_result.get_row_data())
        return pd.DataFrame(rows, columns=query_result.fields)


class BaoStockAShareDataSource(BaseResearchDataSource):
    supports_incremental = True

    def __init__(self, config: BaoStockAShareConfig):
        if config.universe_name not in UNIVERSE_ALIASES:
            raise NotImplementedError(f'当前真实数据源仅支持 HS300，收到 {config.universe_name}')
        self.config = config
        self.start_date = pd.Timestamp(config.start_date).normalize()
        requested_end_date = _resolve_end_date(config.end_date)
        self.end_date = min(requested_end_date, pd.Timestamp.today().normalize())
        self.formal_start_date = pd.Timestamp(config.formal_start_date).normalize() if config.formal_start_date else self.start_date
        self.refresh_step = max(1, int(config.universe_refresh_frequency_days))
        self.price_adjust = str(config.price_adjust or 'qfq').lower()
        self.daily_cache_dir = RAW_DATA_DIR / 'baostock_daily_raw'
        self.adjusted_daily_cache_dir = self.daily_cache_dir if self.price_adjust in {'', 'none'} else (RAW_DATA_DIR / f'baostock_daily_{self.price_adjust}')
        self.universe_cache_dir = RAW_DATA_DIR / f'baostock_{UNIVERSE_ALIASES[config.universe_name]}_snapshots'
        self.info_cache_dir = RAW_DATA_DIR / 'akshare_symbol_info'
        self.dividend_cache_dir = RAW_DATA_DIR / 'baostock_dividend'
        self.adjust_factor_cache_dir = RAW_DATA_DIR / 'baostock_adjust_factor'
        for path in (self.daily_cache_dir, self.adjusted_daily_cache_dir, self.universe_cache_dir, self.info_cache_dir, self.dividend_cache_dir, self.adjust_factor_cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    def fetch_tables(self, existing_tables: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
        existing_tables = existing_tables or {}
        # Always resolve tables through the session-backed incremental path.
        # The lower-level loaders already reuse parquet caches and only fetch
        # missing ranges, while this avoids silently reusing a too-short cache
        # when the requested start_date moves earlier for warmup training.
        with _BaoStockSession() as session:
            trade_calendar = self._build_trade_calendar(session)
            universe_membership, name_map = self._build_universe_membership(session, trade_calendar)
            daily_bar = self._build_daily_bar(session, universe_membership)
            corporate_actions = self._build_corporate_actions(session, daily_bar)

        security_master = self._build_security_master(
            symbols=sorted(set(universe_membership['symbol'])),
            daily_bar=daily_bar,
            name_map=name_map,
            existing_security=existing_tables.get('security_master'),
        )
        tradability = self._build_tradability(daily_bar, security_master)
        daily_bar = daily_bar.sort_values(['trade_date', 'symbol']).reset_index(drop=True)
        security_master = security_master.sort_values('symbol').reset_index(drop=True)
        tradability = tradability.sort_values(['trade_date', 'symbol']).reset_index(drop=True)
        corporate_actions = corporate_actions.sort_values(['event_date', 'symbol', 'event_type']).reset_index(drop=True)
        return {
            'trade_calendar': trade_calendar,
            'security_master': security_master,
            'daily_bar': daily_bar,
            'universe_membership': universe_membership,
            'tradability': tradability,
            'corporate_actions': corporate_actions,
        }

    def _build_universe_membership_from_cache(self) -> tuple[pd.DataFrame, dict[str, str]]:
        snapshot_paths = sorted(self.universe_cache_dir.glob('*.parquet'))
        if not snapshot_paths:
            return pd.DataFrame(), {}

        snapshots: list[tuple[pd.Timestamp, pd.DataFrame]] = []
        name_map: dict[str, str] = {}
        for path in snapshot_paths:
            refresh_date = pd.Timestamp(path.stem)
            if refresh_date < self.start_date or refresh_date > self.end_date:
                continue
            snapshot = pd.read_parquet(path)
            if snapshot.empty:
                continue
            snapshot = snapshot.copy()
            snapshot['symbol'] = snapshot['code'].map(_normalize_bs_symbol)
            snapshot['security_name'] = snapshot['code_name'].astype(str)
            name_map.update(snapshot.set_index('symbol')['security_name'].to_dict())
            snapshots.append((refresh_date, snapshot[['symbol', 'security_name']].drop_duplicates().reset_index(drop=True)))

        if not snapshots:
            return pd.DataFrame(), {}

        daily_bar = self._build_daily_bar_from_cache(
            pd.DataFrame({'symbol': sorted({symbol for _, frame in snapshots for symbol in frame['symbol'].tolist()})})
        )
        if daily_bar.empty:
            return pd.DataFrame(), {}
        trade_dates = sorted(pd.to_datetime(daily_bar['trade_date']).drop_duplicates().tolist())

        frames: list[pd.DataFrame] = []
        for idx, (refresh_date, snapshot) in enumerate(snapshots):
            period_end = snapshots[idx + 1][0] if idx + 1 < len(snapshots) else trade_dates[-1] + pd.Timedelta(days=1)
            period_dates = [date for date in trade_dates if refresh_date <= date < period_end]
            if not period_dates:
                continue
            weight = 1.0 / max(len(snapshot), 1)
            frames.append(
                pd.DataFrame({'trade_date': period_dates}).merge(
                    snapshot.assign(
                        universe_name=self.config.universe_name,
                        universe_weight=weight,
                        rebalance_id=refresh_date.strftime('%Y-%m-%d'),
                    ),
                    how='cross',
                )[['trade_date', 'universe_name', 'symbol', 'universe_weight', 'rebalance_id']]
            )
        if not frames:
            return pd.DataFrame(), {}
        membership = pd.concat(frames, ignore_index=True).sort_values(['trade_date', 'symbol']).reset_index(drop=True)
        return membership, name_map

    def _build_trade_calendar(self, session: _BaoStockSession) -> pd.DataFrame:
        frame = self._retry(
            lambda: session.query_df(
                bs.query_trade_dates(
                    start_date=self.start_date.strftime('%Y-%m-%d'),
                    end_date=self.end_date.strftime('%Y-%m-%d'),
                )
            )
        )
        frame['trade_date'] = pd.to_datetime(frame['calendar_date'])
        frame['is_trading_day'] = frame['is_trading_day'].astype(str).eq('1')
        frame = frame.loc[frame['is_trading_day'], ['trade_date', 'is_trading_day']]
        return frame.reset_index(drop=True)

    def _build_universe_membership(self, session: _BaoStockSession, trade_calendar: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        trade_dates = sorted(pd.to_datetime(trade_calendar['trade_date']).tolist())
        if not trade_dates:
            raise ValueError('交易日历为空，无法构建股票池。')

        refresh_dates = trade_dates[:: self.refresh_step]
        if refresh_dates[-1] != trade_dates[-1]:
            refresh_dates.append(trade_dates[-1])

        frames: list[pd.DataFrame] = []
        name_map: dict[str, str] = {}
        last_snapshot: pd.DataFrame | None = None
        total_refresh = len(refresh_dates)

        for idx, refresh_date in enumerate(refresh_dates):
            snapshot = self._load_or_fetch_universe_snapshot(session, refresh_date)
            if snapshot.empty:
                if last_snapshot is None:
                    continue
                snapshot = last_snapshot.copy()
            else:
                last_snapshot = snapshot.copy()

            period_end = refresh_dates[idx + 1] if idx + 1 < len(refresh_dates) else trade_dates[-1] + pd.Timedelta(days=1)
            period_dates = [date for date in trade_dates if refresh_date <= date < period_end]
            if not period_dates:
                continue

            snapshot = snapshot.copy()
            snapshot['symbol'] = snapshot['code'].map(_normalize_bs_symbol)
            snapshot['security_name'] = snapshot['code_name'].astype(str)
            update_date = pd.to_datetime(snapshot['updateDate'].iloc[0]) if 'updateDate' in snapshot.columns and not snapshot.empty else refresh_date
            weight = 1.0 / max(len(snapshot), 1)
            name_map.update(snapshot.set_index('symbol')['security_name'].to_dict())
            frames.append(
                pd.DataFrame(
                    {
                        'trade_date': period_dates,
                    }
                ).merge(
                    snapshot[['symbol', 'security_name']].assign(
                        universe_name=self.config.universe_name,
                        universe_weight=weight,
                        rebalance_id=update_date.strftime('%Y-%m-%d'),
                    ),
                    how='cross',
                )[['trade_date', 'universe_name', 'symbol', 'universe_weight', 'rebalance_id']]
            )
            if idx == 0 or idx + 1 == total_refresh or (idx + 1) % max(1, total_refresh // 10) == 0:
                logger.info(
                    'HS300 snapshot progress %s/%s refresh_date=%s members=%s',
                    idx + 1,
                    total_refresh,
                    refresh_date.date(),
                    len(snapshot),
                )

        if not frames:
            raise ValueError('未能构建任何 HS300 股票池成分。')
        membership = pd.concat(frames, ignore_index=True).sort_values(['trade_date', 'symbol']).reset_index(drop=True)
        return membership, name_map

    def _load_or_fetch_universe_snapshot(self, session: _BaoStockSession, refresh_date: pd.Timestamp) -> pd.DataFrame:
        cache_path = self.universe_cache_dir / f'{refresh_date.strftime("%Y%m%d")}.parquet'
        if cache_path.exists() and self.config.incremental:
            return pd.read_parquet(cache_path)

        def _query() -> pd.DataFrame:
            return session.query_df(bs.query_hs300_stocks(date=refresh_date.strftime('%Y-%m-%d')))

        frame = self._retry(_query)
        if not frame.empty:
            frame.to_parquet(cache_path, index=False)
        return frame

    def _build_daily_bar(self, session: _BaoStockSession, universe_membership: pd.DataFrame) -> pd.DataFrame:
        membership = universe_membership.copy()
        membership['trade_date'] = pd.to_datetime(membership['trade_date'])
        symbol_membership = membership.loc[membership['trade_date'] >= self.formal_start_date].copy()
        if symbol_membership.empty:
            symbol_membership = membership
        if self.config.universe_mode == 'current_snapshot':
            latest_trade_date = symbol_membership['trade_date'].max()
            symbols = sorted(set(symbol_membership.loc[symbol_membership['trade_date'] == latest_trade_date, 'symbol']))
        else:
            symbols = sorted(set(symbol_membership['symbol']))
        symbol_end_dates = symbol_membership.groupby('symbol')['trade_date'].max().to_dict() if not symbol_membership.empty else {}
        total_symbols = len(symbols)
        cached_symbols = [
            symbol for symbol in symbols
            if self._local_bar_cache_covers(symbol, required_end_date=symbol_end_dates.get(symbol, self.end_date))
        ]
        missing_symbols = [symbol for symbol in symbols if symbol not in set(cached_symbols)]
        frames: list[pd.DataFrame] = []
        progress = 0

        if cached_symbols:
            with ThreadPoolExecutor(max_workers=min(16, len(cached_symbols))) as pool:
                for symbol, frame in zip(cached_symbols, pool.map(self._load_symbol_bars_from_local_cache, cached_symbols)):
                    progress += 1
                    if frame is not None and not frame.empty:
                        frames.append(frame)
                    if progress == 1 or progress == total_symbols or progress % max(1, total_symbols // 10) == 0:
                        logger.info(
                            'Daily bar progress %s/%s symbol=%s rows=%s source=cache',
                            progress,
                            total_symbols,
                            symbol,
                            0 if frame is None else len(frame),
                        )
        for symbol in missing_symbols:
            frame = self._load_or_fetch_symbol_bars(session, symbol)
            progress += 1
            if frame.empty:
                continue
            frames.append(frame)
            if progress == 1 or progress == total_symbols or progress % max(1, total_symbols // 10) == 0:
                logger.info(
                    'Daily bar progress %s/%s symbol=%s rows=%s source=fetch',
                    progress,
                    total_symbols,
                    symbol,
                    len(frame),
                )
        if not frames:
            raise ValueError('未能抓取任何真实日线数据。')
        return pd.concat(frames, ignore_index=True)

    def _build_daily_bar_from_cache(self, universe_membership: pd.DataFrame) -> pd.DataFrame:
        if 'symbol' in universe_membership.columns:
            symbols = sorted(set(universe_membership['symbol']))
        else:
            symbols = []
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            frame = self._read_cached_symbol_bars(symbol)
            if frame.empty:
                continue
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=self._daily_columns())
        return pd.concat(frames, ignore_index=True).sort_values(['trade_date', 'symbol']).reset_index(drop=True)

    def _build_trade_calendar_from_daily_bar(self, daily_bar: pd.DataFrame) -> pd.DataFrame:
        trade_dates = pd.to_datetime(daily_bar['trade_date']).drop_duplicates().sort_values()
        return pd.DataFrame({'trade_date': trade_dates, 'is_trading_day': True})

    def _read_cached_symbol_bars(self, symbol: str) -> pd.DataFrame:
        cache_path = self.daily_cache_dir / f'{symbol.replace(".", "_")}.parquet'
        if not cache_path.exists():
            return pd.DataFrame(columns=self._daily_columns())
        cached = pd.read_parquet(cache_path)
        if cached.empty:
            return pd.DataFrame(columns=self._daily_columns())
        cached['trade_date'] = pd.to_datetime(cached['trade_date'])
        cached = cached.drop_duplicates(subset=['trade_date', 'symbol']).sort_values('trade_date').reset_index(drop=True)
        return cached.loc[(cached['trade_date'] >= self.start_date) & (cached['trade_date'] <= self.end_date)].reset_index(drop=True)

    def _read_cached_adjusted_bars(self, symbol: str) -> pd.DataFrame:
        if self.price_adjust in {'', 'none'}:
            return pd.DataFrame(columns=['trade_date', 'close'])
        cache_path = self.adjusted_daily_cache_dir / f'{symbol.replace(".", "_")}.parquet'
        if not cache_path.exists():
            return pd.DataFrame(columns=['trade_date', 'close'])
        cached = pd.read_parquet(cache_path)
        if cached.empty:
            return pd.DataFrame(columns=['trade_date', 'close'])
        cached['trade_date'] = pd.to_datetime(cached['trade_date'])
        cached = cached.drop_duplicates(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
        return cached.loc[(cached['trade_date'] >= self.start_date) & (cached['trade_date'] <= self.end_date), ['trade_date', 'close']].reset_index(drop=True)

    def _local_bar_cache_covers(self, symbol: str, required_end_date: pd.Timestamp | None = None) -> bool:
        raw_path = self.daily_cache_dir / f'{symbol.replace(".", "_")}.parquet'
        if not raw_path.exists():
            return False
        required_end = pd.Timestamp(required_end_date) if required_end_date is not None else self.end_date
        raw_dates = pd.to_datetime(pd.read_parquet(raw_path, columns=['trade_date'])['trade_date'])
        if raw_dates.empty or raw_dates.max() < required_end:
            return False
        if self.price_adjust in {'', 'none'}:
            return True
        adjusted_path = self.adjusted_daily_cache_dir / f'{symbol.replace(".", "_")}.parquet'
        if not adjusted_path.exists():
            return False
        adjusted_dates = pd.to_datetime(pd.read_parquet(adjusted_path, columns=['trade_date'])['trade_date'])
        return bool(not adjusted_dates.empty and adjusted_dates.max() >= required_end)

    def _load_symbol_bars_from_local_cache(self, symbol: str) -> pd.DataFrame:
        raw = self._read_cached_symbol_bars(symbol)
        if raw.empty:
            return pd.DataFrame(columns=self._daily_columns())
        adjusted = self._read_cached_adjusted_bars(symbol)
        return self._normalize_daily_bar(raw, symbol, adjusted_frame=adjusted)

    def _load_or_fetch_symbol_bars(self, session: _BaoStockSession, symbol: str) -> pd.DataFrame:
        adjusted = (
            self._load_or_fetch_cached_bar_view(session, symbol, cache_dir=self.adjusted_daily_cache_dir, adjust_mode=self.price_adjust)
            if self.price_adjust not in {'', 'none'}
            else pd.DataFrame()
        )
        raw = self._load_or_fetch_cached_bar_view(session, symbol, cache_dir=self.daily_cache_dir, adjust_mode='none')
        if raw.empty and not adjusted.empty:
            raw = self._load_or_rebuild_raw_from_adjusted(session, symbol, adjusted)
        if raw.empty:
            return pd.DataFrame(columns=self._daily_columns())
        return self._normalize_daily_bar(raw, symbol, adjusted_frame=adjusted)

    def _load_or_rebuild_raw_from_adjusted(self, session: _BaoStockSession, symbol: str, adjusted: pd.DataFrame) -> pd.DataFrame:
        cache_path = self.daily_cache_dir / f'{symbol.replace(".", "_")}.parquet'
        cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
        if not cached.empty:
            cached = self._coerce_bar_cache_types(cached)
            cache_min = pd.to_datetime(cached['trade_date']).min()
            cache_max = pd.to_datetime(cached['trade_date']).max()
            if cache_min <= self.start_date and cache_max >= self.end_date:
                return cached.loc[(cached['trade_date'] >= self.start_date) & (cached['trade_date'] <= self.end_date)].reset_index(drop=True)

        try:
            rebuilt = self._rebuild_raw_from_adjusted(session, symbol, adjusted)
        except Exception:
            rebuilt = self._load_or_fetch_cached_bar_view(session, symbol, cache_dir=self.daily_cache_dir, adjust_mode='none')
            if rebuilt.empty:
                raise
        rebuilt.to_parquet(cache_path, index=False)
        return rebuilt.loc[(rebuilt['trade_date'] >= self.start_date) & (rebuilt['trade_date'] <= self.end_date)].reset_index(drop=True)

    def _rebuild_raw_from_adjusted(self, session: _BaoStockSession, symbol: str, adjusted: pd.DataFrame) -> pd.DataFrame:
        adjusted = self._coerce_bar_cache_types(adjusted)
        factor_frame = self._load_or_fetch_adjust_factor_rows(session, symbol)
        factor_series = self._build_fore_adjust_series(adjusted['trade_date'], factor_frame)
        rebuilt = adjusted.copy()
        price_columns = [column for column in ['open', 'high', 'low', 'close', 'preclose'] if column in rebuilt.columns]
        for column in price_columns:
            rebuilt[column] = pd.to_numeric(rebuilt[column], errors='coerce') / factor_series
        rebuilt['symbol'] = symbol
        return rebuilt

    def _load_or_fetch_cached_bar_view(
        self,
        session: _BaoStockSession,
        symbol: str,
        cache_dir,
        adjust_mode: str,
    ) -> pd.DataFrame:
        cache_path = cache_dir / f'{symbol.replace(".", "_")}.parquet'
        cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
        if not cached.empty:
            cached['trade_date'] = pd.to_datetime(cached['trade_date'])
            cached = cached.sort_values('trade_date').reset_index(drop=True)

        fetch_frames: list[pd.DataFrame] = []
        if cached.empty:
            fetch_frames.append(self._fetch_symbol_bars(session, symbol, self.start_date, self.end_date, adjust_mode=adjust_mode))
        else:
            cache_min = pd.to_datetime(cached['trade_date']).min()
            cache_max = pd.to_datetime(cached['trade_date']).max()
            if self.start_date < cache_min:
                fetch_frames.append(self._fetch_symbol_bars(session, symbol, self.start_date, cache_min - pd.Timedelta(days=1), adjust_mode=adjust_mode))
            if self.end_date > cache_max:
                fetch_frames.append(self._fetch_symbol_bars(session, symbol, cache_max + pd.Timedelta(days=1), self.end_date, adjust_mode=adjust_mode))

        fresh = pd.concat(fetch_frames, ignore_index=True) if fetch_frames else pd.DataFrame()
        if cached.empty and fresh.empty:
            merged = pd.DataFrame()
        elif cached.empty:
            merged = fresh.copy()
        elif fresh.empty:
            merged = cached.copy()
        else:
            merged = pd.concat([cached, fresh], ignore_index=True)
        if merged.empty:
            return merged

        merged = self._coerce_bar_cache_types(merged)
        subset = ['trade_date', 'symbol'] if 'symbol' in merged.columns else ['trade_date']
        merged = merged.drop_duplicates(subset=subset).sort_values('trade_date').reset_index(drop=True)
        merged.to_parquet(cache_path, index=False)
        return merged.loc[(merged['trade_date'] >= self.start_date) & (merged['trade_date'] <= self.end_date)].reset_index(drop=True)

    def _fetch_symbol_bars(self, session: _BaoStockSession, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, adjust_mode: str) -> pd.DataFrame:
        if start_date > end_date:
            return pd.DataFrame()

        def _query_baostock() -> pd.DataFrame:
            return session.query_df(
                bs.query_history_k_data_plus(
                    _to_bs_symbol(symbol),
                    DAILY_FIELDS,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    frequency='d',
                    adjustflag=BAOSTOCK_ADJUST_FLAG.get(adjust_mode, '3'),
                )
            )

        frame = self._retry(_query_baostock, allow_empty=True)
        if frame.empty:
            try:
                frame = self._fetch_symbol_bars_from_akshare(symbol, start_date, end_date, adjust_mode=adjust_mode)
            except Exception:
                frame = pd.DataFrame()
        if frame.empty:
            return pd.DataFrame()

        if 'date' in frame.columns:
            normalized = frame.rename(columns={'date': 'trade_date'}).copy()
        else:
            normalized = frame.copy()
        normalized['trade_date'] = pd.to_datetime(normalized['trade_date'])
        normalized['symbol'] = symbol
        return normalized

    def _fetch_symbol_bars_from_akshare(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, adjust_mode: str) -> pd.DataFrame:
        code = symbol.split('.')[0]
        frame = self._retry(
            lambda: ak.stock_zh_a_hist(
                symbol=code,
                period='daily',
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d'),
                adjust='' if adjust_mode in {'', 'none'} else adjust_mode,
            ),
            allow_empty=True,
        )
        if frame.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                'date': pd.to_datetime(frame['日期']),
                'code': code,
                'open': pd.to_numeric(frame['开盘'], errors='coerce'),
                'high': pd.to_numeric(frame['最高'], errors='coerce'),
                'low': pd.to_numeric(frame['最低'], errors='coerce'),
                'close': pd.to_numeric(frame['收盘'], errors='coerce'),
                'preclose': pd.to_numeric(frame['收盘'], errors='coerce').shift(1),
                'volume': pd.to_numeric(frame['成交量'], errors='coerce'),
                'amount': pd.to_numeric(frame['成交额'], errors='coerce'),
                'tradestatus': '1',
                'isST': '0',
                'pctChg': pd.to_numeric(frame['涨跌幅'], errors='coerce'),
            }
        )

    def _normalize_daily_bar(self, frame: pd.DataFrame, symbol: str, adjusted_frame: pd.DataFrame | None = None) -> pd.DataFrame:
        trade_status = frame['tradestatus'].astype(str) if 'tradestatus' in frame.columns else pd.Series('1', index=frame.index)
        is_st_series = frame['isST'].astype(str) if 'isST' in frame.columns else pd.Series('0', index=frame.index)
        normalized = pd.DataFrame(
            {
                'trade_date': pd.to_datetime(frame['trade_date']),
                'symbol': symbol,
                'open': pd.to_numeric(frame['open'], errors='coerce'),
                'high': pd.to_numeric(frame['high'], errors='coerce'),
                'low': pd.to_numeric(frame['low'], errors='coerce'),
                'close': pd.to_numeric(frame['close'], errors='coerce'),
                'preclose': pd.to_numeric(frame.get('preclose'), errors='coerce'),
                'volume': pd.to_numeric(frame['volume'], errors='coerce'),
                'amount': pd.to_numeric(frame['amount'], errors='coerce'),
                'adj_close': pd.to_numeric(frame['close'], errors='coerce'),
                'status_flag': trade_status.map(lambda value: 'NORMAL' if value == '1' else 'SUSPENDED'),
                'tradestatus': trade_status,
                'is_st_daily': is_st_series.isin({'1', 'True', 'true'}),
                'pct_chg': pd.to_numeric(frame.get('pctChg'), errors='coerce'),
            }
        )
        if adjusted_frame is not None and not adjusted_frame.empty and self.price_adjust not in {'', 'none'}:
            adjusted = adjusted_frame[['trade_date', 'close']].copy()
            adjusted['trade_date'] = pd.to_datetime(adjusted['trade_date'])
            adjusted['adj_close'] = pd.to_numeric(adjusted['close'], errors='coerce')
            normalized = normalized.drop(columns=['adj_close']).merge(adjusted[['trade_date', 'adj_close']], on='trade_date', how='left')
            normalized['adj_close'] = normalized['adj_close'].fillna(normalized['close'])
        if normalized['pct_chg'].isna().all():
            normalized['pct_chg'] = normalized['close'].pct_change().mul(100.0)
        normalized = normalized.dropna(subset=['trade_date', 'open', 'high', 'low', 'close']).reset_index(drop=True)
        normalized['volume'] = normalized['volume'].fillna(0.0).astype(float)
        normalized['amount'] = normalized['amount'].fillna(0.0).astype(float)
        return normalized

    def _build_security_master(
        self,
        symbols: list[str],
        daily_bar: pd.DataFrame,
        name_map: dict[str, str],
        existing_security: pd.DataFrame | None,
    ) -> pd.DataFrame:
        existing_lookup = {}
        if existing_security is not None and not existing_security.empty:
            frame = existing_security.copy()
            frame['listed_date'] = pd.to_datetime(frame['listed_date'], errors='coerce')
            existing_lookup = frame.set_index('symbol').to_dict(orient='index')

        st_flags = daily_bar.groupby('symbol')['is_st_daily'].max().to_dict()
        first_seen = daily_bar.groupby('symbol')['trade_date'].min().to_dict()

        rows: list[dict] = []
        for symbol in symbols:
            cached = existing_lookup.get(symbol, {})
            info = self._load_cached_symbol_info(symbol)
            listed_date = pd.to_datetime(info.get('上市时间'), format='%Y%m%d', errors='coerce')
            if pd.isna(listed_date):
                listed_date = cached.get('listed_date')
            if pd.isna(listed_date):
                listed_date = first_seen.get(symbol)
            if pd.isna(listed_date):
                listed_date = pd.Timestamp('2000-01-01')
            rows.append(
                {
                    'symbol': symbol,
                    'security_name': str(info.get('股票简称') or cached.get('security_name') or name_map.get(symbol) or symbol),
                    'market': symbol.split('.')[-1],
                    'board': _board_from_symbol(symbol),
                    'industry': str(info.get('行业') or cached.get('industry') or 'Unknown'),
                    'listed_date': pd.Timestamp(listed_date),
                    'delisted_date': pd.NaT,
                    'is_st_flag': bool(st_flags.get(symbol, False)),
                }
            )
        return pd.DataFrame(rows)

    def _load_cached_symbol_info(self, symbol: str) -> dict[str, str]:
        cache_path = self.info_cache_dir / f'{symbol.replace(".", "_")}.json'
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding='utf-8'))
        return {}

    def _load_or_fetch_symbol_info(self, symbol: str) -> dict[str, str]:
        cache_path = self.info_cache_dir / f'{symbol.replace(".", "_")}.json'
        if cache_path.exists() and self.config.incremental:
            return json.loads(cache_path.read_text(encoding='utf-8'))

        code = symbol.split('.')[0]
        try:
            frame = self._retry(lambda: ak.stock_individual_info_em(symbol=code), allow_empty=True)
        except Exception as exc:
            logger.warning('AkShare symbol info unavailable symbol=%s error=%s', symbol, exc)
            frame = pd.DataFrame()
        if frame.empty:
            info: dict[str, str] = {}
        else:
            info = {str(row.item): str(row.value) for row in frame.itertuples(index=False)}
        cache_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        return info

    def _build_tradability(self, daily_bar: pd.DataFrame, security_master: pd.DataFrame) -> pd.DataFrame:
        listed_map = security_master.set_index('symbol')['listed_date']
        frame = daily_bar[['trade_date', 'symbol', 'tradestatus', 'is_st_daily', 'pct_chg']].copy()
        frame['listed_date'] = frame['symbol'].map(listed_map)
        frame['days_since_listed'] = (frame['trade_date'] - frame['listed_date']).dt.days
        frame['is_new_listing'] = frame['days_since_listed'].fillna(9999).astype(float) < 60
        frame['is_st'] = frame['is_st_daily'].astype('boolean').fillna(False).astype(bool)
        frame['is_suspended'] = ~frame['tradestatus'].astype(str).eq('1')
        frame['limit_pct'] = frame.apply(
            lambda row: _price_limit_threshold(
                symbol=str(row['symbol']),
                trade_date=pd.Timestamp(row['trade_date']),
                is_st=bool(row['is_st']),
            ),
            axis=1,
        )
        frame['up_limit_hit'] = frame['pct_chg'].fillna(0.0) >= frame['limit_pct']
        frame['down_limit_hit'] = frame['pct_chg'].fillna(0.0) <= -frame['limit_pct']
        frame['can_buy'] = ~(frame['is_suspended'] | frame['up_limit_hit'] | frame['is_st'] | frame['is_new_listing'])
        frame['can_sell'] = ~(frame['is_suspended'] | frame['down_limit_hit'])
        frame['is_tradable'] = frame['can_buy'] | frame['can_sell']
        return frame[
            [
                'trade_date',
                'symbol',
                'is_tradable',
                'can_buy',
                'can_sell',
                'is_suspended',
                'up_limit_hit',
                'down_limit_hit',
                'is_st',
                'is_new_listing',
            ]
        ]

    def _build_corporate_actions(self, session: _BaoStockSession, daily_bar: pd.DataFrame) -> pd.DataFrame:
        if daily_bar.empty:
            return self._empty_corporate_actions()
        action_start = max(self.start_date, self.formal_start_date)
        symbol_ranges = (
            daily_bar.loc[pd.to_datetime(daily_bar['trade_date']) >= action_start, ['symbol', 'trade_date']]
            .assign(trade_date=lambda frame: pd.to_datetime(frame['trade_date']))
            .groupby('symbol')['trade_date']
            .agg(['min', 'max'])
            .reset_index()
        )
        if symbol_ranges.empty:
            return self._empty_corporate_actions()
        rows: list[dict] = []
        total_symbols = len(symbol_ranges)
        for idx, row in enumerate(symbol_ranges.itertuples(index=False), start=1):
            years = list(range(pd.Timestamp(row.min).year, pd.Timestamp(row.max).year + 1))
            for year in years:
                frame = self._load_or_fetch_dividend_rows(session, row.symbol, year)
                if frame.empty:
                    continue
                event_dates = pd.to_datetime(frame['dividOperateDate'], errors='coerce')
                cash_ps = pd.to_numeric(frame['dividCashPsBeforeTax'], errors='coerce').fillna(0.0)
                stock_ps = pd.to_numeric(frame['dividStocksPs'], errors='coerce').fillna(0.0)
                reserve_ps = pd.to_numeric(frame['dividReserveToStockPs'], errors='coerce').fillna(0.0)
                for event_date, cash_value, stock_value, reserve_value in zip(event_dates, cash_ps, stock_ps, reserve_ps):
                    if pd.isna(event_date) or event_date < action_start or event_date > self.end_date:
                        continue
                    if cash_value > 0:
                        rows.append({'event_date': event_date, 'symbol': row.symbol, 'event_type': 'cash_dividend', 'event_value': float(cash_value)})
                    if stock_value > 0:
                        rows.append({'event_date': event_date, 'symbol': row.symbol, 'event_type': 'stock_dividend', 'event_value': float(stock_value)})
                    if reserve_value > 0:
                        rows.append({'event_date': event_date, 'symbol': row.symbol, 'event_type': 'reserve_to_stock', 'event_value': float(reserve_value)})
            if idx == 1 or idx == total_symbols or idx % max(1, total_symbols // 10) == 0:
                logger.info('Corporate action progress %s/%s symbol=%s action_rows=%s', idx, total_symbols, row.symbol, len(rows))
        if not rows:
            return self._empty_corporate_actions()
        return pd.DataFrame(rows).drop_duplicates(subset=['event_date', 'symbol', 'event_type', 'event_value']).reset_index(drop=True)

    def _load_or_fetch_dividend_rows(self, session: _BaoStockSession, symbol: str, year: int) -> pd.DataFrame:
        cache_path = self.dividend_cache_dir / f'{symbol.replace(".", "_")}_{year}.parquet'
        if cache_path.exists() and self.config.incremental:
            return pd.read_parquet(cache_path)

        def _query() -> pd.DataFrame:
            return session.query_df(bs.query_dividend_data(code=_to_bs_symbol(symbol), year=str(year), yearType='operate'))

        frame = self._retry(_query, attempts=5, sleep_seconds=1.2, allow_empty=True)
        if frame.empty:
            frame = pd.DataFrame(
                {
                    'dividOperateDate': pd.Series(dtype='object'),
                    'dividCashPsBeforeTax': pd.Series(dtype='object'),
                    'dividStocksPs': pd.Series(dtype='object'),
                    'dividReserveToStockPs': pd.Series(dtype='object'),
                }
            )
        frame.to_parquet(cache_path, index=False)
        return frame

    def _load_or_fetch_adjust_factor_rows(self, session: _BaoStockSession, symbol: str) -> pd.DataFrame:
        cache_path = self.adjust_factor_cache_dir / f'{symbol.replace(".", "_")}.parquet'
        if cache_path.exists() and self.config.incremental:
            return pd.read_parquet(cache_path)

        def _query() -> pd.DataFrame:
            return session.query_df(
                bs.query_adjust_factor(
                    code=_to_bs_symbol(symbol),
                    start_date='2015-01-01',
                    end_date=self.end_date.strftime('%Y-%m-%d'),
                )
            )

        frame = self._retry(_query, allow_empty=True)
        if frame.empty:
            frame = pd.DataFrame(
                {
                    'dividOperateDate': pd.Series(dtype='datetime64[ns]'),
                    'foreAdjustFactor': pd.Series(dtype='float64'),
                }
            )
        frame.to_parquet(cache_path, index=False)
        return frame

    @staticmethod
    def _empty_corporate_actions() -> pd.DataFrame:
        return pd.DataFrame(
            {
                'event_date': pd.Series(dtype='datetime64[ns]'),
                'symbol': pd.Series(dtype='object'),
                'event_type': pd.Series(dtype='object'),
                'event_value': pd.Series(dtype='float64'),
            }
        )

    @staticmethod
    def _daily_columns() -> list[str]:
        return ['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'adj_close', 'status_flag', 'tradestatus', 'is_st_daily', 'pct_chg']

    @staticmethod
    def _coerce_bar_cache_types(frame: pd.DataFrame) -> pd.DataFrame:
        coerced = frame.copy()
        if 'trade_date' in coerced.columns:
            coerced['trade_date'] = pd.to_datetime(coerced['trade_date'])
        for column in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg', 'pct_chg']:
            if column in coerced.columns:
                coerced[column] = pd.to_numeric(coerced[column], errors='coerce')
        for column in ['code', 'symbol', 'tradestatus', 'isST']:
            if column in coerced.columns:
                coerced[column] = coerced[column].astype(str)
        return coerced

    @staticmethod
    def _build_fore_adjust_series(trade_dates: pd.Series, factor_frame: pd.DataFrame) -> pd.Series:
        date_frame = pd.DataFrame({'trade_date': pd.to_datetime(trade_dates)}).sort_values('trade_date').reset_index(drop=True)
        if factor_frame.empty:
            return pd.Series(1.0, index=date_frame.index, dtype=float)
        factors = factor_frame.copy()
        factors['dividOperateDate'] = pd.to_datetime(factors['dividOperateDate'], errors='coerce')
        factors['foreAdjustFactor'] = pd.to_numeric(factors['foreAdjustFactor'], errors='coerce')
        factors = factors.dropna(subset=['dividOperateDate', 'foreAdjustFactor']).sort_values('dividOperateDate').drop_duplicates(subset=['dividOperateDate'], keep='last')
        if factors.empty:
            return pd.Series(1.0, index=date_frame.index, dtype=float)
        merged = pd.merge_asof(
            date_frame,
            factors[['dividOperateDate', 'foreAdjustFactor']],
            left_on='trade_date',
            right_on='dividOperateDate',
            direction='backward',
        )
        series = merged['foreAdjustFactor'].ffill().fillna(1.0).replace(0.0, 1.0)
        return pd.Series(series.to_numpy(dtype=float), index=date_frame.index, dtype=float)

    @staticmethod
    def _retry(func, attempts: int = 3, sleep_seconds: float = 0.8, allow_empty: bool = False):
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                result = func()
                if isinstance(result, pd.DataFrame) and result.empty and not allow_empty:
                    raise ValueError('返回空数据')
                return result
            except Exception as exc:
                last_error = exc
                if '用户未登录' in str(exc) or 'please login' in str(exc).lower():
                    try:
                        bs.login()
                    except Exception:
                        pass
                if attempt == attempts:
                    break
                time.sleep(sleep_seconds * attempt)
        raise last_error


def _normalize_bs_symbol(code: str) -> str:
    value = str(code).lower()
    if '.' not in value:
        return _normalize_numeric_symbol(value)
    market, numeric = value.split('.', 1)
    return f'{numeric}.{market.upper()}'


def _normalize_numeric_symbol(code: str) -> str:
    numeric = str(code).zfill(6)
    if numeric.startswith(('600', '601', '603', '605', '688', '689', '900')):
        return f'{numeric}.SH'
    if numeric.startswith(('430', '440', '830', '831', '832', '833', '835', '836', '837', '838', '839', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879')):
        return f'{numeric}.BJ'
    return f'{numeric}.SZ'


def _to_bs_symbol(symbol: str) -> str:
    numeric, market = symbol.split('.', 1)
    return f'{market.lower()}.{numeric}'


def _board_from_symbol(symbol: str) -> str:
    numeric = symbol.split('.', 1)[0]
    if numeric.startswith('688'):
        return 'STAR'
    if numeric.startswith('300'):
        return 'ChiNext'
    if numeric.startswith(('4', '8')):
        return 'BSE'
    return 'Main'


def _price_limit_threshold(symbol: str, trade_date: pd.Timestamp, is_st: bool) -> float:
    if is_st:
        return 4.8
    numeric = symbol.split('.', 1)[0]
    if numeric.startswith(('4', '8')):
        return 29.8
    if numeric.startswith('688'):
        return 19.8
    if numeric.startswith('300') and trade_date >= pd.Timestamp('2020-08-24'):
        return 19.8
    return 9.8


def _resolve_end_date(value: str) -> pd.Timestamp:
    normalized = str(value).strip().lower()
    today = pd.Timestamp.today().normalize()
    if normalized in {'today', 'latest'}:
        return today
    if normalized in {'latest_completed', 'latest_complete'}:
        latest = today - pd.Timedelta(days=1)
        while latest.weekday() >= 5:
            latest -= pd.Timedelta(days=1)
        return latest
    return pd.Timestamp(value).normalize()
