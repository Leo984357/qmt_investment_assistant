from __future__ import annotations

import json
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd


def compute_drawdown(nav: pd.DataFrame) -> pd.DataFrame:
    series = nav[['trade_date', 'nav']].copy()
    series['rolling_peak'] = series['nav'].cummax()
    series['drawdown'] = series['nav'] / series['rolling_peak'] - 1.0
    return series[['trade_date', 'nav', 'rolling_peak', 'drawdown']]


def compute_monthly_returns(nav: pd.DataFrame) -> pd.DataFrame:
    series = nav[['trade_date', 'nav']].copy()
    series['trade_date'] = pd.to_datetime(series['trade_date'])
    monthly = series.set_index('trade_date')['nav'].resample('ME').last().pct_change().fillna(0.0)
    return monthly.rename('monthly_return').reset_index()


def compute_rank_ic(predictions: pd.DataFrame, labels: pd.DataFrame, label_name: str) -> pd.DataFrame:
    merged = predictions.merge(labels[['trade_date', 'symbol', label_name]], on=['trade_date', 'symbol'], how='left')
    rows = []
    for trade_date, frame in merged.groupby('trade_date'):
        if frame[label_name].notna().sum() < 2:
            continue
        if frame['score'].nunique(dropna=True) < 2 or frame[label_name].nunique(dropna=True) < 2:
            corr = 0.0
        else:
            corr = frame['score'].corr(frame[label_name], method='spearman')
        rows.append(
            {
                'trade_date': trade_date,
                'rank_ic': float(0.0 if pd.isna(corr) else corr),
            }
        )
    return pd.DataFrame(rows)


def compute_benchmark_nav(
    strategy_nav: pd.DataFrame,
    daily_bar: pd.DataFrame,
    universe_membership: pd.DataFrame,
    data_source: str,
    universe_name: str,
) -> pd.DataFrame:
    if strategy_nav.empty:
        return pd.DataFrame(columns=['trade_date', 'benchmark_nav', 'benchmark_return', 'benchmark_name'])

    nav_dates = pd.DataFrame({'trade_date': pd.to_datetime(strategy_nav['trade_date']).sort_values().drop_duplicates()})
    benchmark_name = f'{universe_name}等权基准'
    benchmark = pd.DataFrame()

    if data_source == 'baostock_ashare' and universe_name.upper() == 'HS300':
        try:
            benchmark = _fetch_csindex_benchmark('000300', nav_dates['trade_date'].min(), nav_dates['trade_date'].max())
            if not benchmark.empty:
                benchmark_name = '沪深300'
        except Exception:
            benchmark = pd.DataFrame()

    if benchmark.empty:
        benchmark = _compute_universe_equal_weight_benchmark(nav_dates, daily_bar, universe_membership, universe_name)
    else:
        benchmark = nav_dates.merge(benchmark, on='trade_date', how='left').sort_values('trade_date').reset_index(drop=True)
        benchmark['benchmark_close'] = benchmark['benchmark_close'].ffill().bfill()
        if benchmark['benchmark_close'].notna().sum() == 0:
            benchmark = _compute_universe_equal_weight_benchmark(nav_dates, daily_bar, universe_membership, universe_name)
            benchmark_name = f'{universe_name}等权基准'
        else:
            start_close = float(benchmark['benchmark_close'].dropna().iloc[0])
            benchmark['benchmark_nav'] = benchmark['benchmark_close'] / max(start_close, 1e-9)
            benchmark['benchmark_return'] = benchmark['benchmark_nav'].pct_change().fillna(0.0)

    if benchmark.empty:
        return pd.DataFrame(columns=['trade_date', 'benchmark_nav', 'benchmark_return', 'benchmark_name'])
    benchmark['benchmark_name'] = benchmark_name
    return benchmark[['trade_date', 'benchmark_nav', 'benchmark_return', 'benchmark_name']]


def performance_summary(nav: pd.DataFrame, trades: pd.DataFrame, rank_ic: pd.DataFrame, benchmark_nav: pd.DataFrame | None = None) -> dict:
    daily_return = nav['daily_return'].fillna(0.0)
    total_return = float(nav['nav'].iloc[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (252 / max(len(nav), 1)) - 1.0)
    annual_vol = float(daily_return.std(ddof=0) * np.sqrt(252))
    sharpe = annual_return / annual_vol if annual_vol > 1e-12 else 0.0
    drawdown = compute_drawdown(nav)
    avg_turnover = float(nav['turnover'].mean()) if not nav.empty else 0.0
    total_cost = float(nav['trade_cost'].sum()) if 'trade_cost' in nav.columns else 0.0
    avg_rank_ic = float(rank_ic['rank_ic'].mean()) if not rank_ic.empty else 0.0
    summary = {
        'total_return': round(total_return, 4),
        'annual_return': round(annual_return, 4),
        'annual_vol': round(annual_vol, 4),
        'sharpe_like': round(sharpe, 4),
        'max_drawdown': round(float(drawdown['drawdown'].min()), 4),
        'trade_count': int(len(trades)),
        'avg_turnover': round(avg_turnover, 4),
        'total_cost': round(total_cost, 2),
        'avg_rank_ic': round(avg_rank_ic, 4),
    }
    if benchmark_nav is not None and not benchmark_nav.empty:
        benchmark_total_return = float(benchmark_nav['benchmark_nav'].iloc[-1] - 1.0)
        merged = nav[['trade_date', 'nav']].merge(benchmark_nav[['trade_date', 'benchmark_nav']], on='trade_date', how='left')
        merged['benchmark_nav'] = merged['benchmark_nav'].ffill().bfill()
        merged = merged.dropna(subset=['benchmark_nav']).copy()
        if not merged.empty:
            excess_total_return = float(merged['nav'].iloc[-1] / max(merged['benchmark_nav'].iloc[-1], 1e-9) - 1.0)
            summary['benchmark_total_return'] = round(benchmark_total_return, 4)
            summary['excess_total_return'] = round(excess_total_return, 4)
    return summary


def trim_backtest_window(
    nav: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    rank_ic: pd.DataFrame | None = None,
    benchmark_nav: pd.DataFrame | None = None,
    active_start: str | pd.Timestamp | None = None,
    rank_ic_start: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if nav.empty:
        empty = pd.DataFrame()
        return empty, trades if trades is not None else empty, rank_ic if rank_ic is not None else empty, benchmark_nav if benchmark_nav is not None else empty
    if active_start is None:
        active_start = pd.to_datetime(nav['trade_date']).min()
    start_ts = pd.Timestamp(active_start)
    rank_ic_start_ts = pd.Timestamp(rank_ic_start) if rank_ic_start is not None else start_ts
    trimmed_nav = nav.loc[pd.to_datetime(nav['trade_date']) >= start_ts].reset_index(drop=True)
    trimmed_trades = trades.loc[pd.to_datetime(trades['trade_date']) >= start_ts].reset_index(drop=True) if trades is not None and not trades.empty else pd.DataFrame()
    trimmed_rank_ic = rank_ic.loc[pd.to_datetime(rank_ic['trade_date']) >= rank_ic_start_ts].reset_index(drop=True) if rank_ic is not None and not rank_ic.empty else pd.DataFrame()
    trimmed_benchmark = benchmark_nav.loc[pd.to_datetime(benchmark_nav['trade_date']) >= start_ts].reset_index(drop=True) if benchmark_nav is not None and not benchmark_nav.empty else pd.DataFrame()
    return trimmed_nav, trimmed_trades, trimmed_rank_ic, trimmed_benchmark


def latest_signal_report(target_weights: pd.DataFrame, security_master: pd.DataFrame) -> pd.DataFrame:
    if target_weights.empty:
        return pd.DataFrame(columns=['signal_date', 'execution_date', 'symbol', 'security_name', 'industry', 'rank', 'score', 'target_weight'])
    latest_date = target_weights['signal_date'].max()
    latest = target_weights.loc[target_weights['signal_date'] == latest_date].copy()
    master = security_master[['symbol', 'security_name', 'industry']]
    latest = latest.merge(master, on='symbol', how='left')
    return latest[['signal_date', 'execution_date', 'symbol', 'security_name', 'industry', 'rank', 'score', 'target_weight']].sort_values('rank')


def write_markdown_report(
    output_path: Path,
    summary: dict,
    latest_signal: pd.DataFrame,
    split_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latest_signal_md = latest_signal.to_markdown(index=False) if not latest_signal.empty else '_无最新信号_'
    split_metrics_md = split_metrics.head(20).to_markdown(index=False) if not split_metrics.empty else '_无切分指标_'
    feature_importance_md = feature_importance.head(20).to_markdown(index=False) if not feature_importance.empty else '_无特征重要性_'
    content = (
        '# Research Run Report\n\n'
        '## Summary\n'
        f'```json\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n```\n\n'
        '## Latest Signal\n'
        f'{latest_signal_md}\n\n'
        '## Split Metrics\n'
        f'{split_metrics_md}\n\n'
        '## Feature Importance\n'
        f'{feature_importance_md}\n'
    )
    output_path.write_text(content, encoding='utf-8')


def _fetch_csindex_benchmark(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    frame = ak.stock_zh_index_hist_csindex(
        symbol=symbol,
        start_date=pd.Timestamp(start_date).strftime('%Y%m%d'),
        end_date=pd.Timestamp(end_date).strftime('%Y%m%d'),
    )
    if frame.empty:
        return pd.DataFrame()
    benchmark = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(frame['日期']),
            'benchmark_close': pd.to_numeric(frame['收盘'], errors='coerce'),
        }
    )
    benchmark = benchmark.dropna(subset=['trade_date', 'benchmark_close']).drop_duplicates(subset=['trade_date'], keep='last')
    return benchmark.sort_values('trade_date').reset_index(drop=True)


def _compute_universe_equal_weight_benchmark(
    nav_dates: pd.DataFrame,
    daily_bar: pd.DataFrame,
    universe_membership: pd.DataFrame,
    universe_name: str,
) -> pd.DataFrame:
    bars = daily_bar[['trade_date', 'symbol', 'adj_close']].copy()
    bars['trade_date'] = pd.to_datetime(bars['trade_date'])
    bars = bars.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    bars['asset_return'] = bars.groupby('symbol')['adj_close'].pct_change().fillna(0.0)

    membership = universe_membership.loc[universe_membership['universe_name'] == universe_name, ['trade_date', 'symbol', 'universe_weight']].copy()
    membership['trade_date'] = pd.to_datetime(membership['trade_date'])
    frame = membership.merge(bars[['trade_date', 'symbol', 'asset_return']], on=['trade_date', 'symbol'], how='left')
    frame['weighted_return'] = frame['universe_weight'].fillna(0.0) * frame['asset_return'].fillna(0.0)
    benchmark = frame.groupby('trade_date', as_index=False)['weighted_return'].sum().rename(columns={'weighted_return': 'benchmark_return'})
    benchmark = nav_dates.merge(benchmark, on='trade_date', how='left').sort_values('trade_date').reset_index(drop=True)
    benchmark['benchmark_return'] = benchmark['benchmark_return'].fillna(0.0)
    benchmark['benchmark_nav'] = (1.0 + benchmark['benchmark_return']).cumprod()
    return benchmark[['trade_date', 'benchmark_nav', 'benchmark_return']]
