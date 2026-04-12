from pathlib import Path

import pandas as pd

from src.backtest.engine import run_backtest
from src.data_sources.baostock_ashare import BaoStockAShareConfig, BaoStockAShareDataSource
from src.evaluation.reporting import trim_backtest_window
from src.experiment.runner import _build_market_reference, _build_signal_dates, _build_source_data_spec_with_warmup
from src.experiment.spec import BacktestSpec, DataSpec, EvaluationSpec, ExperimentSpec, FeatureSetSpec, LabelSpec, ModelSpec, PortfolioSpec, SignalSpec
from src.features.simple_definitions import simple_factor_registry as default_feature_registry
from src.models.lightgbm_cross_sectional import CrossSectionalLightGBMModel
from src.portfolio.construction import _active_gross_exposure, build_target_weights
from src.universe.membership import build_current_snapshot_universe


def test_anchor_based_signal_dates_match_qmt_semantics():
    trade_calendar = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(
                ['2022-09-01', '2022-09-02', '2022-09-05', '2022-09-06', '2022-09-07', '2022-09-08', '2022-09-09']
            )
        }
    )
    available_dates = sorted(pd.to_datetime(trade_calendar['trade_date']).tolist())

    signal_dates = _build_signal_dates(
        available_dates=available_dates,
        trade_calendar=trade_calendar,
        rebalance_frequency_days=5,
        trade_delay_days=1,
        anchor_mode='fixed',
        anchor_date='2022-09-01',
        backtest_start_date='2022-09-01',
    )

    assert signal_dates == [pd.Timestamp('2022-09-07')]


def test_backtest_start_anchor_rephases_rebalance_calendar():
    trade_calendar = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(
                ['2022-09-01', '2022-09-02', '2022-09-05', '2022-09-06', '2022-09-07', '2022-09-08', '2022-09-09']
            )
        }
    )
    available_dates = sorted(pd.to_datetime(trade_calendar['trade_date']).tolist())

    signal_dates = _build_signal_dates(
        available_dates=available_dates,
        trade_calendar=trade_calendar,
        rebalance_frequency_days=5,
        trade_delay_days=1,
        anchor_mode='follow_backtest_start',
        anchor_date=None,
        backtest_start_date='2022-09-02',
    )

    assert signal_dates == [pd.Timestamp('2022-09-08')]


def test_source_data_spec_adds_warmup_window_before_research_start():
    spec = ExperimentSpec(
        name='warmup_case',
        description='',
        data=DataSpec(
            source='baostock_ashare',
            snapshot_id='warmup_case',
            start_date='2020-01-02',
            end_date='2020-12-31',
            universe_name='HS300',
        ),
        features=FeatureSetSpec(
            set_name='f',
            version='v1',
            names=['mom20', 'mom60', 'mom120', 'rev5', 'vol20', 'amihud_illiq_20d'],
            winsorize_limits=(0.0, 1.0),
            zscore=False,
            fill_missing=False,
        ),
        label=LabelSpec(name='fwd_return_20d', horizon=20),
        model=ModelSpec(
            family='lightgbm_regression',
            version='v1',
            registry_stage='research',
            fallback_model='mom60_zscore',
            train_window_days=380,
            valid_window_days=0,
            min_train_samples=600,
        ),
        signal=SignalSpec(),
        portfolio=PortfolioSpec(
            top_n=25,
            weighting='equal',
            gross_exposure=1.0,
            defensive_gross=0.45,
            max_single_weight=0.05,
            cash_buffer=0.0,
            min_trade_value=5000,
            market_filter_lookback=20,
            market_filter_threshold=0.0,
        ),
        backtest=BacktestSpec(
            initial_cash=250000,
            lot_size=100,
            commission_bps=0.75,
            stamp_duty_bps=10.0,
            slippage_bps=20.0,
            rebalance_frequency_days=5,
            trade_delay_days=1,
            anchor_mode='follow_backtest_start',
            anchor_date=None,
            execution_constraint_mode='strict_ashare',
        ),
        evaluation=EvaluationSpec(),
    )

    source_spec = _build_source_data_spec_with_warmup(spec)

    assert pd.Timestamp(source_spec.start_date) < pd.Timestamp(spec.data.start_date)


def test_feature_definitions_match_script_formulas():
    registry = default_feature_registry()
    dates = pd.date_range('2024-01-01', periods=70, freq='B')
    close = pd.Series(range(100, 170), dtype=float)
    volume = pd.Series([100.0] * 50 + [200.0] * 20)
    bars = pd.DataFrame(
        {
            'trade_date': dates,
            'symbol': '000001.SZ',
            'adj_close': close,
            'close': close,
            'volume': volume,
            'amount': volume * close,
        }
    )

    vol20 = registry.get('vol20').compute(bars).iloc[-1]
    amihud = registry.get('amihud_illiq_20d').compute(bars).iloc[-1]

    manual_vol20 = bars['adj_close'].pct_change().rolling(20).std().iloc[-1]

    assert vol20 > 0
    assert round(vol20, 10) == round(manual_vol20, 10)
    assert amihud >= 0  # Amihud illiquidity should be non-negative


def test_feature_panel_preserves_structural_nans_without_fill():
    registry = default_feature_registry()
    dates = pd.date_range('2024-01-01', periods=30, freq='B')
    bars = pd.DataFrame(
        {
            'trade_date': dates,
            'symbol': '000001.SZ',
            'adj_close': pd.Series(range(100, 130), dtype=float),
            'volume': pd.Series([1000.0] * len(dates)),
            'amount': pd.Series(range(100, 130), dtype=float) * 1000.0,
        }
    )
    panel, _, _ = registry.compute_panel(
        daily_bar=bars,
        feature_names=['mom20', 'mom60'],
        winsorize_limits=(0.0, 1.0),
        zscore=False,
        fill_missing=False,
    )

    early = panel.loc[panel['trade_date'] == pd.Timestamp('2024-01-08')].iloc[0]
    later = panel.loc[panel['trade_date'] == pd.Timestamp('2024-01-30')].iloc[0]

    assert pd.isna(early['mom20'])
    assert pd.isna(early['mom60'])
    assert not pd.isna(later['mom20'])


def test_daily_bar_keeps_raw_execution_prices_and_adjusted_signal_price():
    source = BaoStockAShareDataSource(
        BaoStockAShareConfig(
            start_date='2024-01-01',
            end_date='2024-01-31',
            universe_name='HS300',
            price_adjust='qfq',
        )
    )
    raw_frame = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03']),
            'open': [10.0, 10.5],
            'high': [10.2, 10.8],
            'low': [9.9, 10.4],
            'close': [10.1, 10.6],
            'preclose': [9.8, 10.1],
            'volume': [1000, 1100],
            'amount': [10100, 11660],
            'tradestatus': ['1', '1'],
            'isST': ['0', '0'],
            'pctChg': [3.06, 4.95],
        }
    )
    adjusted_frame = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03']),
            'close': [8.1, 8.4],
        }
    )

    normalized = source._normalize_daily_bar(raw_frame, '000001.SZ', adjusted_frame=adjusted_frame)

    assert list(normalized['close']) == [10.1, 10.6]
    assert list(normalized['adj_close']) == [8.1, 8.4]


def test_bar_cache_type_coercion_handles_mixed_object_columns():
    frame = pd.DataFrame(
        {
            'trade_date': ['2024-01-02', '2024-01-03'],
            'code': ['000001', '000001'],
            'open': ['10.0', 10.5],
            'high': ['10.2', 10.8],
            'low': ['9.9', 10.4],
            'close': ['10.1', 10.6],
            'preclose': ['9.8', 10.1],
            'volume': ['1000', 1100.0],
            'amount': ['10100', 11660.0],
            'tradestatus': [1, '1'],
            'isST': [0, '0'],
            'pctChg': ['3.06', 4.95],
        }
    )

    coerced = BaoStockAShareDataSource._coerce_bar_cache_types(frame)

    assert str(coerced['trade_date'].dtype).startswith('datetime64')
    assert str(coerced['open'].dtype).startswith('float')
    assert str(coerced['volume'].dtype).startswith('float')
    assert coerced['tradestatus'].tolist() == ['1', '1']


def test_fore_adjust_series_forward_fills_latest_factor():
    trade_dates = pd.Series(pd.to_datetime(['2024-06-13', '2024-06-14', '2024-10-09', '2024-10-10', '2024-10-11']))
    factor_frame = pd.DataFrame(
        {
            'dividOperateDate': ['2024-06-14', '2024-10-10'],
            'foreAdjustFactor': ['0.90', '0.95'],
        }
    )

    series = BaoStockAShareDataSource._build_fore_adjust_series(trade_dates, factor_frame)

    assert series.round(2).tolist() == [1.00, 0.90, 0.90, 0.95, 0.95]


def test_walk_forward_excludes_recent_unrealized_labels_from_training(tmp_path):
    dates = pd.date_range('2024-01-01', periods=40, freq='B')
    dataset = pd.DataFrame(
        {
            'trade_date': dates,
            'symbol': ['000001.SZ'] * len(dates),
            'mom20': range(len(dates)),
            'target': range(len(dates)),
        }
    )
    model = CrossSectionalLightGBMModel(
        params={},
        train_window_days=10,
        valid_window_days=0,
        min_train_samples=999,
        training_embargo_days=0,
        registry_stage='research',
        fallback_model='mom20_zscore',
        fallback_feature='mom20',
    )
    result = model.fit_walk_forward(
        dataset=dataset,
        feature_names=['mom20'],
        label_name='target',
        rebalance_dates=[pd.Timestamp('2024-02-23')],
        artifact_dir=tmp_path,
        label_horizon=5,
    )

    split = result.split_metrics.iloc[0]
    assert int(split['train_rows']) == 11
    assert pd.Timestamp(split['train_end_date']) == pd.Timestamp('2024-02-16')


def test_walk_forward_respects_qmt_training_embargo():
    dates = pd.date_range('2024-01-01', periods=60, freq='B')
    dataset = pd.DataFrame(
        {
            'trade_date': dates,
            'symbol': ['000001.SZ'] * len(dates),
            'mom20': range(len(dates)),
            'target': range(len(dates)),
        }
    )
    model = CrossSectionalLightGBMModel(
        params={},
        train_window_days=10,
        valid_window_days=0,
        min_train_samples=999,
        training_embargo_days=10,
        registry_stage='research',
        fallback_model='mom20_zscore',
        fallback_feature='mom20',
    )
    result = model.fit_walk_forward(
        dataset=dataset,
        feature_names=['mom20'],
        label_name='target',
        rebalance_dates=[pd.Timestamp('2024-03-22')],
        artifact_dir=Path('.'),
        label_horizon=5,
    )

    split = result.split_metrics.iloc[0]
    assert int(split['train_rows']) == 11
    assert pd.Timestamp(split['train_end_date']) == pd.Timestamp('2024-03-01')


def test_backtest_applies_cash_dividend_and_stock_bonus():
    daily_bar = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03']),
            'symbol': ['000001.SZ', '000001.SZ'],
            'open': [10.0, 10.0],
            'high': [10.0, 10.0],
            'low': [10.0, 10.0],
            'close': [10.0, 10.0],
            'volume': [10000.0, 10000.0],
            'amount': [100000.0, 100000.0],
            'adj_close': [10.0, 10.0],
            'status_flag': ['NORMAL', 'NORMAL'],
        }
    )
    target_weights = pd.DataFrame(
        {
            'signal_date': pd.to_datetime(['2024-01-02']),
            'execution_date': pd.to_datetime(['2024-01-02']),
            'symbol': ['000001.SZ'],
            'rank': [1],
            'score': [1.0],
            'target_weight': [1.0],
            'gross_exposure': [1.0],
            'market_momentum': [0.0],
            'model_type': ['fallback_mom60_zscore'],
            'fallback_used': [True],
        }
    )
    tradability = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03']),
            'symbol': ['000001.SZ', '000001.SZ'],
            'is_tradable': [True, True],
            'can_buy': [True, True],
            'can_sell': [True, True],
            'is_suspended': [False, False],
            'up_limit_hit': [False, False],
            'down_limit_hit': [False, False],
            'is_st': [False, False],
            'is_new_listing': [False, False],
        }
    )
    corporate_actions = pd.DataFrame(
        {
            'event_date': pd.to_datetime(['2024-01-03', '2024-01-03']),
            'symbol': ['000001.SZ', '000001.SZ'],
            'event_type': ['cash_dividend', 'stock_dividend'],
            'event_value': [0.1, 0.1],
        }
    )

    result = run_backtest(
        daily_bar=daily_bar,
        target_weights=target_weights,
        tradability=tradability,
        corporate_actions=corporate_actions,
        initial_cash=1000.0,
        lot_size=100,
        commission_bps=0.0,
        stamp_duty_bps=0.0,
        slippage_bps=0.0,
        min_trade_value=0.0,
        execution_constraint_mode='strict_ashare',
    )

    last_nav = result.nav.iloc[-1]
    last_position = result.positions.iloc[-1]
    assert round(float(last_nav['cash']), 2) == 10.0
    assert int(last_position['shares']) == 110
    assert round(float(last_nav['equity']), 2) == 1110.0


def test_qmt_risk_ladder_matches_four_bucket_logic():
    dates = pd.date_range('2024-01-01', periods=6, freq='B')
    gross = _active_gross_exposure(
        signal_date=dates[-1],
        market_reference=pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 2.0], index=dates),
        gross_exposure=1.0,
        defensive_gross=0.35,
        market_filter_lookback=20,
        market_filter_threshold=0.0,
        risk_model='qmt_style_ladder',
        risk_ma_short_window=3,
        risk_ma_long_window=5,
        risk_momentum_window=2,
        risk_mid_exposure=0.85,
        risk_low_exposure=0.65,
        risk_crash_exposure=0.45,
    )
    mid = _active_gross_exposure(
        signal_date=dates[-1],
        market_reference=pd.Series([1.0, 1.0, 1.0, 1.2, 1.2, 1.1], index=dates),
        gross_exposure=1.0,
        defensive_gross=0.35,
        market_filter_lookback=20,
        market_filter_threshold=0.0,
        risk_model='qmt_style_ladder',
        risk_ma_short_window=3,
        risk_ma_long_window=5,
        risk_momentum_window=2,
        risk_mid_exposure=0.85,
        risk_low_exposure=0.65,
        risk_crash_exposure=0.45,
    )
    low = _active_gross_exposure(
        signal_date=dates[-1],
        market_reference=pd.Series([1.4, 1.4, 1.0, 1.0, 1.2, 1.05], index=dates),
        gross_exposure=1.0,
        defensive_gross=0.35,
        market_filter_lookback=20,
        market_filter_threshold=0.0,
        risk_model='qmt_style_ladder',
        risk_ma_short_window=3,
        risk_ma_long_window=5,
        risk_momentum_window=2,
        risk_mid_exposure=0.85,
        risk_low_exposure=0.65,
        risk_crash_exposure=0.45,
    )
    crash = _active_gross_exposure(
        signal_date=dates[-1],
        market_reference=pd.Series([1.2, 1.2, 1.2, 1.1, 1.0, 0.9], index=dates),
        gross_exposure=1.0,
        defensive_gross=0.35,
        market_filter_lookback=20,
        market_filter_threshold=0.0,
        risk_model='qmt_style_ladder',
        risk_ma_short_window=3,
        risk_ma_long_window=5,
        risk_momentum_window=2,
        risk_mid_exposure=0.85,
        risk_low_exposure=0.65,
        risk_crash_exposure=0.45,
    )

    assert gross == 1.0
    assert mid == 0.85
    assert low == 0.65
    assert crash == 0.45


def test_current_snapshot_universe_uses_latest_membership_for_all_dates():
    universe_membership = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03']),
            'universe_name': ['HS300', 'HS300', 'HS300', 'HS300'],
            'symbol': ['000001.SZ', '000002.SZ', '000001.SZ', '000003.SZ'],
            'universe_weight': [0.5, 0.5, 0.5, 0.5],
            'rebalance_id': ['2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03'],
        }
    )
    trade_calendar = pd.DataFrame({'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04'])})

    current_snapshot = build_current_snapshot_universe(universe_membership, 'HS300', trade_calendar)

    assert sorted(current_snapshot.loc[current_snapshot['trade_date'] == pd.Timestamp('2024-01-02'), 'symbol'].tolist()) == ['000001.SZ', '000003.SZ']
    assert current_snapshot['rebalance_id'].nunique() == 1


def test_market_reference_uses_point_in_time_universe_only():
    universe = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03']),
            'universe_name': ['HS300'] * 4,
            'symbol': ['000001.SZ', '000002.SZ', '000002.SZ', '000003.SZ'],
            'universe_weight': [0.5, 0.5, 0.5, 0.5],
            'rebalance_id': ['2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03'],
        }
    )
    daily_bar = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03', '2024-01-03']),
            'symbol': ['000001.SZ', '000002.SZ', '000002.SZ', '000003.SZ', '999999.SZ'],
            'adj_close': [10.0, 20.0, 21.0, 39.0, 1000.0],
        }
    )

    market_reference = _build_market_reference(universe, daily_bar)

    assert float(market_reference.loc[pd.Timestamp('2024-01-02')]) == 15.0
    assert float(market_reference.loc[pd.Timestamp('2024-01-03')]) == 30.0


def test_trim_backtest_window_keeps_rank_ic_from_signal_start():
    nav = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04']),
            'nav': [1.0, 1.01, 1.02],
            'daily_return': [0.0, 0.01, 0.0099],
            'turnover': [0.0, 0.2, 0.0],
            'trade_cost': [0.0, 2.0, 0.0],
        }
    )
    trades = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-03']),
            'symbol': ['000001.SZ'],
        }
    )
    rank_ic = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03']),
            'rank_ic': [0.1, 0.2],
        }
    )
    benchmark = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04']),
            'benchmark_nav': [1.0, 1.0, 1.01],
            'benchmark_return': [0.0, 0.0, 0.01],
            'benchmark_name': ['沪深300'] * 3,
        }
    )

    trimmed_nav, trimmed_trades, trimmed_rank_ic, trimmed_benchmark = trim_backtest_window(
        nav=nav,
        trades=trades,
        rank_ic=rank_ic,
        benchmark_nav=benchmark,
        active_start='2024-01-03',
        rank_ic_start='2024-01-02',
    )

    assert trimmed_nav['trade_date'].tolist() == [pd.Timestamp('2024-01-03'), pd.Timestamp('2024-01-04')]
    assert trimmed_trades['trade_date'].tolist() == [pd.Timestamp('2024-01-03')]
    assert trimmed_rank_ic['trade_date'].tolist() == [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')]
    assert trimmed_benchmark['trade_date'].tolist() == [pd.Timestamp('2024-01-03'), pd.Timestamp('2024-01-04')]


def test_strict_ashare_candidate_filter_uses_execution_day_tradability():
    predictions = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02']),
            'symbol': ['000001.SZ'],
            'score': [1.0],
            'model_type': ['lightgbm_mom60_blend'],
            'fallback_used': [False],
        }
    )
    universe_membership = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02']),
            'universe_name': ['HS300'],
            'symbol': ['000001.SZ'],
            'universe_weight': [1.0],
            'rebalance_id': ['2024-01-02'],
        }
    )
    tradability = pd.DataFrame(
        {
            'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03']),
            'symbol': ['000001.SZ', '000001.SZ'],
            'is_st': [False, False],
            'is_new_listing': [False, False],
            'is_tradable': [True, False],
            'can_buy': [True, False],
            'can_sell': [True, True],
        }
    )
    trade_calendar = pd.DataFrame({'trade_date': pd.to_datetime(['2024-01-02', '2024-01-03'])})

    result = build_target_weights(
        predictions=predictions,
        universe_membership=universe_membership,
        tradability=tradability,
        market_reference=pd.Series([1.0, 1.0], index=pd.to_datetime(['2024-01-02', '2024-01-03'])),
        trade_calendar=trade_calendar,
        universe_name='HS300',
        top_n=1,
        gross_exposure=1.0,
        defensive_gross=0.35,
        max_single_weight=1.0,
        market_filter_lookback=20,
        market_filter_threshold=0.0,
        trade_delay_days=1,
        risk_model='qmt_style_ladder',
        candidate_filter_mode='strict_ashare',
    )

    assert result.target_weights.empty
    assert result.filtered_candidates['reason'].tolist() == ['temporarily_restricted']
