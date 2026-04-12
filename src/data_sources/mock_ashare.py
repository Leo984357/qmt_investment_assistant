from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import BaseResearchDataSource

INDUSTRIES = ['Bank', 'Consumer', 'Industrial', 'Healthcare', 'Tech', 'Materials', 'Energy', 'Utility']
BOARDS = ['Main', 'ChiNext', 'STAR']
EVENT_TYPES = ['dividend', 'split', 'rename', 'suspension_notice']


@dataclass(frozen=True)
class MockAShareConfig:
    start_date: str
    end_date: str
    n_symbols_master: int = 320
    n_universe: int = 300
    seed: int = 7
    universe_name: str = 'HS300'


class MockAShareDataSource(BaseResearchDataSource):
    def __init__(self, config: MockAShareConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def fetch_tables(self, existing_tables: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
        dates = pd.bdate_range(self.config.start_date, self.config.end_date)
        security_master = self._build_security_master(dates)
        daily_bar = self._build_daily_bar(dates, security_master)
        universe_membership = self._build_universe_membership(dates, security_master)
        tradability = self._build_tradability(daily_bar, security_master)
        corporate_actions = self._build_corporate_actions(dates, security_master)
        trade_calendar = pd.DataFrame({'trade_date': dates, 'is_trading_day': True})
        return {
            'trade_calendar': trade_calendar,
            'security_master': security_master,
            'daily_bar': daily_bar,
            'universe_membership': universe_membership,
            'tradability': tradability,
            'corporate_actions': corporate_actions,
        }

    def _build_security_master(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        symbols: list[str] = []
        for idx in range(self.config.n_symbols_master):
            if idx % 2 == 0:
                symbols.append(f'{600000 + idx // 2 + 1:06d}.SH')
            else:
                symbols.append(f'{idx // 2 + 1:06d}.SZ')
        listed_offsets = self.rng.integers(0, max(20, min(len(dates) // 5, 120)), size=len(symbols))
        listed_dates = dates[listed_offsets]
        boards = self.rng.choice(BOARDS, size=len(symbols), p=[0.7, 0.15, 0.15])
        industries = self.rng.choice(INDUSTRIES, size=len(symbols))
        is_st = self.rng.random(len(symbols)) < 0.04
        market = ['SH' if symbol.endswith('.SH') else 'SZ' for symbol in symbols]
        return pd.DataFrame(
            {
                'symbol': symbols,
                'security_name': [f'股票{idx + 1:03d}' for idx in range(len(symbols))],
                'market': market,
                'board': boards,
                'industry': industries,
                'listed_date': listed_dates,
                'delisted_date': pd.NaT,
                'is_st_flag': is_st,
            }
        ).sort_values('symbol').reset_index(drop=True)

    def _build_daily_bar(self, dates: pd.DatetimeIndex, security_master: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for idx, row in enumerate(security_master.itertuples(index=False)):
            symbol_dates = dates[dates >= row.listed_date]
            if len(symbol_dates) == 0:
                continue
            local_rng = np.random.default_rng(self.config.seed + idx)
            drift = 0.0002 + (idx % 13) * 0.00003
            vol = 0.012 + (idx % 7) * 0.0015
            returns = local_rng.normal(drift, vol, size=len(symbol_dates))
            close = (18 + idx % 40) * np.exp(np.cumsum(returns))
            open_px = close * (1 + local_rng.normal(0.0, 0.003, size=len(symbol_dates)))
            high = np.maximum(open_px, close) * (1 + np.abs(local_rng.normal(0.002, 0.0015, size=len(symbol_dates))))
            low = np.minimum(open_px, close) * (1 - np.abs(local_rng.normal(0.002, 0.0015, size=len(symbol_dates))))
            volume = local_rng.lognormal(mean=12.2, sigma=0.35, size=len(symbol_dates)).astype(int)
            amount = volume * close * (1 + local_rng.normal(0.0, 0.01, size=len(symbol_dates)))
            frame = pd.DataFrame(
                {
                    'trade_date': symbol_dates,
                    'symbol': row.symbol,
                    'open': open_px.round(4),
                    'high': high.round(4),
                    'low': low.round(4),
                    'close': close.round(4),
                    'volume': volume,
                    'amount': amount.round(2),
                    'adj_close': close.round(4),
                    'status_flag': 'NORMAL',
                }
            )
            frames.append(frame)
        daily_bar = pd.concat(frames, ignore_index=True)
        return daily_bar.sort_values(['trade_date', 'symbol']).reset_index(drop=True)

    def _build_universe_membership(self, dates: pd.DatetimeIndex, security_master: pd.DataFrame) -> pd.DataFrame:
        rebalance_dates = dates[::20]
        frames: list[pd.DataFrame] = []
        for idx, rebalance_date in enumerate(rebalance_dates):
            period_end = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else dates[-1] + pd.Timedelta(days=1)
            period_dates = dates[(dates >= rebalance_date) & (dates < period_end)]
            eligible = security_master.loc[security_master['listed_date'] <= rebalance_date, ['symbol']].copy()
            sample_size = min(self.config.n_universe, len(eligible))
            chosen = eligible.sample(sample_size, random_state=self.config.seed + idx).sort_values('symbol').reset_index(drop=True)
            weight = 1.0 / max(sample_size, 1)
            for period_date in period_dates:
                frames.append(
                    pd.DataFrame(
                        {
                            'trade_date': period_date,
                            'universe_name': self.config.universe_name,
                            'symbol': chosen['symbol'],
                            'universe_weight': weight,
                            'rebalance_id': str(rebalance_date.date()),
                        }
                    )
                )
        membership = pd.concat(frames, ignore_index=True)
        return membership.sort_values(['trade_date', 'symbol']).reset_index(drop=True)

    def _build_tradability(self, daily_bar: pd.DataFrame, security_master: pd.DataFrame) -> pd.DataFrame:
        listed_map = security_master.set_index('symbol')['listed_date']
        st_map = security_master.set_index('symbol')['is_st_flag']
        tradability = daily_bar[['trade_date', 'symbol', 'open', 'close']].copy()
        tradability['return_1d'] = tradability.groupby('symbol')['close'].pct_change().fillna(0.0)
        tradability['listed_date'] = tradability['symbol'].map(listed_map)
        tradability['days_since_listed'] = (tradability['trade_date'] - tradability['listed_date']).dt.days
        tradability['is_new_listing'] = tradability['days_since_listed'] < 60
        tradability['is_st'] = tradability['symbol'].map(st_map).fillna(False).astype(bool)
        local_rng = np.random.default_rng(self.config.seed + 2048)
        tradability['is_suspended'] = local_rng.random(len(tradability)) < 0.01
        tradability['up_limit_hit'] = tradability['return_1d'] > 0.095
        tradability['down_limit_hit'] = tradability['return_1d'] < -0.095
        tradability['can_buy'] = ~(tradability['is_suspended'] | tradability['up_limit_hit'] | tradability['is_st'] | tradability['is_new_listing'])
        tradability['can_sell'] = ~(tradability['is_suspended'] | tradability['down_limit_hit'])
        tradability['is_tradable'] = tradability['can_buy'] | tradability['can_sell']
        return tradability[
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
        ].sort_values(['trade_date', 'symbol']).reset_index(drop=True)

    def _build_corporate_actions(self, dates: pd.DatetimeIndex, security_master: pd.DataFrame) -> pd.DataFrame:
        sample_size = min(60, len(security_master))
        symbols = security_master['symbol'].sample(sample_size, random_state=self.config.seed + 99).tolist()
        event_dates = self.rng.choice(dates[20:-20], size=sample_size, replace=True)
        event_type = self.rng.choice(EVENT_TYPES, size=sample_size, replace=True)
        payload = self.rng.normal(loc=1.0, scale=0.08, size=sample_size).round(4)
        return pd.DataFrame(
            {
                'event_date': pd.to_datetime(event_dates),
                'symbol': symbols,
                'event_type': event_type,
                'event_value': payload,
            }
        ).sort_values(['event_date', 'symbol']).reset_index(drop=True)
