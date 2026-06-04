from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.paths import DATA_DIR
from src.core.schemas import StrategyOutput

TICKERS = ['000001.SZ', '000333.SZ', '000651.SZ', '600036.SH', '600519.SH', '601318.SH', '300750.SZ', '688981.SH']


def _ensure_sample_prices(path: Path, days: int = 260):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    data = {}
    for i, t in enumerate(TICKERS):
        drift = 0.0004 + i * 0.00003
        vol = 0.015 + (i % 3) * 0.002
        rets = rng.normal(drift, vol, size=len(dates))
        prices = 30 * np.exp(np.cumsum(rets))
        data[t] = prices.round(2)
    pd.DataFrame(data, index=dates).to_csv(path, encoding='utf-8-sig')


def _calc_metrics(equity: pd.Series) -> dict:
    ret = equity.pct_change().fillna(0.0)
    cum = float(equity.iloc[-1] / equity.iloc[0] - 1)
    ann = float((1 + cum) ** (252 / max(len(equity), 1)) - 1)
    vol = float(ret.std(ddof=0) * np.sqrt(252))
    sharpe = ann / vol if vol > 1e-12 else 0.0
    max_dd = float((equity / equity.cummax() - 1).min())
    return {
        'cum_return': round(cum, 4),
        'annual_return': round(ann, 4),
        'annual_vol': round(vol, 4),
        'sharpe_like': round(sharpe, 4),
        'max_drawdown': round(max_dd, 4),
    }


def run_demo_research(config) -> StrategyOutput:
    price_path = DATA_DIR / 'research' / 'sample_prices.csv'
    history_days = max(config.common.latest_n_days, config.strategy.long_lookback + 40)
    _ensure_sample_prices(price_path, days=max(history_days, 260))
    prices = pd.read_csv(price_path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().tail(history_days)

    mom_lb = config.strategy.momentum_lookback
    long_lb = config.strategy.long_lookback
    top_n = config.strategy.top_n
    max_single = config.strategy.max_single_weight
    min_price = config.strategy.min_price
    target_gross = min(config.strategy.target_gross, 1.0 - config.strategy.rebalance_cash_buffer)

    rets = prices.pct_change().fillna(0.0)
    mom_short = prices.iloc[-1] / prices.iloc[-mom_lb] - 1.0
    mom_long = prices.iloc[-1] / prices.iloc[-long_lb] - 1.0
    vol_20 = rets.tail(20).std(ddof=0)
    score = 0.65 * mom_short + 0.35 * mom_long - 0.15 * vol_20
    score = score.sort_values(ascending=False)

    signals = pd.DataFrame({
        'ticker': score.index,
        'score': score.values,
        'mom_short': mom_short.reindex(score.index).values,
        'mom_long': mom_long.reindex(score.index).values,
        'vol20': vol_20.reindex(score.index).values,
        'latest_price': prices.iloc[-1].reindex(score.index).values,
    })
    signals = signals[signals['latest_price'] >= min_price].reset_index(drop=True)

    selected = signals.head(top_n).copy()
    if selected.empty:
        target_portfolio = pd.DataFrame(columns=['ticker', 'reference_price', 'target_weight'])
    else:
        per_name = min(max_single, target_gross / max(len(selected), 1))
        selected['target_weight'] = per_name
        target_portfolio = selected[['ticker', 'latest_price', 'target_weight']].rename(columns={'latest_price': 'reference_price'})

    daily_target = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    rebal_dates = rets.index[::20]
    for dt in rebal_dates:
        end_loc = rets.index.get_loc(dt)
        if end_loc < max(mom_lb, long_lb):
            continue
        sub_prices = prices.iloc[:end_loc + 1]
        sub_rets = sub_prices.pct_change().fillna(0.0)
        s = 0.65 * (sub_prices.iloc[-1] / sub_prices.iloc[-mom_lb] - 1.0) + 0.35 * (sub_prices.iloc[-1] / sub_prices.iloc[-long_lb] - 1.0) - 0.15 * sub_rets.tail(20).std(ddof=0)
        eligible = sub_prices.iloc[-1] >= min_price
        names = s[eligible].sort_values(ascending=False).head(top_n).index.tolist()
        daily_target.loc[dt, names] = min(max_single, target_gross / max(top_n, 1))
    daily_target = daily_target.replace(0, np.nan).ffill().fillna(0.0)
    strat_ret = (daily_target.shift(1).fillna(0.0) * rets).sum(axis=1)
    equity = (1 + strat_ret).cumprod()
    metrics = _calc_metrics(equity)
    diagnostics = {
        'selected_count': int((target_portfolio['target_weight'] > 0).sum()),
        'latest_rebalance_universe': selected['ticker'].tolist(),
        'target_gross': float(target_portfolio['target_weight'].sum()),
        'history_days': int(len(prices)),
        'min_price_filter': float(min_price),
    }
    return StrategyOutput(
        as_of_date=str(prices.index[-1].date()),
        signals=signals,
        target_portfolio=target_portfolio,
        diagnostics=diagnostics,
        equity_curve=equity.to_frame('equity'),
        metrics=metrics,
    )
