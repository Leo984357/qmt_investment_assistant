from __future__ import annotations

from src.core.module_registry import ModuleMetadata

from .construction import build_target_weights
from .registry import PortfolioContext, PortfolioRegistry, register_portfolio


def _qmt_topn_equal_weight(context: PortfolioContext):
    cfg = context.portfolio_config
    backtest_cfg = context.backtest_config
    return build_target_weights(
        predictions=context.signal_scores,
        universe_membership=context.universe_membership,
        tradability=context.tradability,
        market_reference=context.market_reference,
        trade_calendar=context.trade_calendar,
        universe_name=context.universe_name,
        top_n=cfg.top_n,
        gross_exposure=cfg.gross_exposure,
        defensive_gross=cfg.defensive_gross,
        max_single_weight=cfg.max_single_weight,
        market_filter_lookback=cfg.market_filter_lookback,
        market_filter_threshold=cfg.market_filter_threshold,
        trade_delay_days=backtest_cfg.trade_delay_days,
        risk_model=cfg.risk_model,
        risk_ma_short_window=cfg.risk_ma_short_window,
        risk_ma_long_window=cfg.risk_ma_long_window,
        risk_momentum_window=cfg.risk_momentum_window,
        risk_mid_exposure=cfg.risk_mid_exposure,
        risk_low_exposure=cfg.risk_low_exposure,
        risk_crash_exposure=cfg.risk_crash_exposure,
        candidate_filter_mode=cfg.candidate_filter_mode,
    )


def default_portfolio_registry() -> PortfolioRegistry:
    registry = PortfolioRegistry()
    register_portfolio(
        registry,
        ModuleMetadata(
            name='qmt_topn_equal_weight',
            version='v1',
            category='portfolio_constructor',
            description='Top-N equal-weight constructor with QMT-style market-risk ladder.',
            owner='platform',
            inputs=('signal_scores', 'universe_membership', 'tradability', 'market_reference'),
            outputs=('target_weights', 'filtered_candidates'),
            tags=('top_n', 'equal_weight', 'qmt_style'),
        ),
        builder=_qmt_topn_equal_weight,
    )
    return registry
