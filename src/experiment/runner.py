from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.backtest.engine import run_backtest
from src.core.logging_utils import get_logger
from src.data_sources.factory import build_data_source
from src.data_store.catalog import LocalResearchCatalog
from src.data_store.schemas import SCHEMAS
from src.evaluation.registry import EvaluationContext
from src.evaluation.reporting import compute_benchmark_nav, compute_drawdown, compute_monthly_returns, compute_rank_ic, latest_signal_report, performance_summary, trim_backtest_window, write_markdown_report
from src.evaluation.suites import default_evaluation_registry
from src.features.simple_definitions import simple_factor_registry as default_feature_registry
from src.labels.definitions import default_label_registry
from src.models.definitions import default_model_registry
from src.ops.paths import ARTIFACT_RUNS_DIR, ensure_platform_dirs
from src.portfolio.definitions import default_portfolio_registry
from src.portfolio.enhancer import PortfolioEnhancer, BufferConfig, SmootherConfig, CostFilterConfig
from src.portfolio.registry import PortfolioContext
from src.signals.definitions import default_signal_registry
from src.signals.registry import SignalContext
from src.universe.membership import build_current_snapshot_universe, build_point_in_time_universe

from .manifest import build_artifact_inventory, build_data_contract_report, build_experiment_manifest
from .spec import ExperimentSpec, load_experiment_spec
from .tracker import MLflowTracker

logger = get_logger(__name__)


def run_experiment(config_path: str | Path) -> dict:
    overall_start = perf_counter()
    ensure_platform_dirs()
    spec = load_experiment_spec(config_path)
    run_id = f"{spec.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{spec.config_hash[:8]}"
    run_dir = ARTIFACT_RUNS_DIR / run_id
    for subdir in ['config', 'metadata', 'data', 'features', 'labels', 'datasets', 'models', 'signals', 'backtest', 'evaluation', 'reports']:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    logger.info('Experiment started run_id=%s config=%s data_range=%s..%s universe=%s', run_id, config_path, spec.data.start_date, spec.data.end_date, spec.data.universe_name)
    stage_timings: dict[str, float] = {}

    spec.save(run_dir / 'config' / 'resolved_experiment.yaml')

    catalog = LocalResearchCatalog()
    source_spec = build_source_data_spec_with_warmup(spec)
    source = build_data_source(source_spec)
    if spec.data.bootstrap_if_missing and not _catalog_is_ready_for_spec(catalog, source):
        logger.info('Catalog bootstrap started source=%s incremental=%s', spec.data.source, spec.data.incremental)
        bootstrap_start = perf_counter()
        catalog.bootstrap(source)
        stage_timings['bootstrap'] = perf_counter() - bootstrap_start
        logger.info('Catalog bootstrap complete elapsed=%.1fs', stage_timings['bootstrap'])
    else:
        logger.info('Catalog already ready; using cached silver tables.')
        stage_timings['bootstrap'] = 0.0

    load_start = perf_counter()
    trade_calendar = catalog.read_table('trade_calendar')
    security_master = catalog.read_table('security_master')
    daily_bar = catalog.read_table('daily_bar')
    universe_membership = catalog.read_table('universe_membership')
    tradability = catalog.read_table('tradability')
    corporate_actions = catalog.read_table('corporate_actions')
    snapshot_manifest = catalog.snapshot_manifest(zone='silver')
    (run_dir / 'metadata' / 'data_snapshot.json').write_text(json.dumps(snapshot_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    stage_timings['load_data'] = perf_counter() - load_start
    logger.info('Data loaded trade_dates=%s bars=%s universe_rows=%s elapsed=%.1fs', trade_calendar['trade_date'].nunique(), len(daily_bar), len(universe_membership), stage_timings['load_data'])

    feature_start = perf_counter()
    universe = _resolve_universe_frame(spec, universe_membership, trade_calendar)
    feature_registry = default_feature_registry()
    feature_panel, feature_long, feature_inventory = feature_registry.compute_panel(
        daily_bar=daily_bar,
        feature_names=spec.features.names,
        winsorize_limits=tuple(spec.features.winsorize_limits),
        zscore=spec.features.zscore,
        fill_missing=spec.features.fill_missing,
    )
    label_registry = default_label_registry()
    label_panel, label_inventory = label_registry.compute_panel(daily_bar, spec.label.name)
    stage_timings['feature_label'] = perf_counter() - feature_start
    logger.info('Feature/label build complete universe_rows=%s feature_rows=%s label_rows=%s elapsed=%.1fs', len(universe), len(feature_panel), len(label_panel), stage_timings['feature_label'])

    dataset_start = perf_counter()
    model_dataset = (
        universe[['trade_date', 'symbol', 'universe_weight']]
        .merge(feature_panel, on=['trade_date', 'symbol'], how='left')
        .merge(label_panel, on=['trade_date', 'symbol'], how='left')
        .merge(tradability, on=['trade_date', 'symbol'], how='left')
    )
    model_dataset = model_dataset.dropna(subset=spec.features.names).sort_values(['trade_date', 'symbol']).reset_index(drop=True)
    stage_timings['dataset'] = perf_counter() - dataset_start
    logger.info('Model dataset ready rows=%s dates=%s symbols=%s elapsed=%.1fs', len(model_dataset), model_dataset['trade_date'].nunique(), model_dataset['symbol'].nunique(), stage_timings['dataset'])

    feature_panel.to_parquet(run_dir / 'features' / 'feature_panel.parquet', index=False)
    feature_long.to_parquet(run_dir / 'features' / 'feature_long.parquet', index=False)
    feature_inventory.to_csv(run_dir / 'features' / 'feature_inventory.csv', index=False)
    feature_registry.inventory().to_csv(run_dir / 'features' / 'feature_registry.csv', index=False)
    label_panel.to_parquet(run_dir / 'labels' / 'label_panel.parquet', index=False)
    label_inventory.to_csv(run_dir / 'labels' / 'label_inventory.csv', index=False)
    label_registry.inventory().to_csv(run_dir / 'labels' / 'label_registry.csv', index=False)
    model_dataset.to_parquet(run_dir / 'datasets' / 'model_dataset.parquet', index=False)

    signal_dates = _build_signal_dates(
        available_dates=sorted(pd.to_datetime(model_dataset['trade_date'].unique())),
        trade_calendar=trade_calendar,
        rebalance_frequency_days=spec.backtest.rebalance_frequency_days,
        trade_delay_days=spec.backtest.trade_delay_days,
        anchor_mode=spec.backtest.anchor_mode,
        anchor_date=spec.backtest.anchor_date,
        backtest_start_date=spec.data.start_date,
    )
    logger.info('Signal schedule built signal_dates=%s anchor_mode=%s first_signal=%s last_signal=%s', len(signal_dates), spec.backtest.anchor_mode, signal_dates[0].date() if signal_dates else None, signal_dates[-1].date() if signal_dates else None)
    model_registry = default_model_registry()
    signal_registry = default_signal_registry()
    portfolio_registry = default_portfolio_registry()
    evaluation_registry = default_evaluation_registry()
    model_runner = model_registry.build(spec.model.family, spec.model)
    model_start = perf_counter()
    model_result = model_runner.fit_walk_forward(
        dataset=model_dataset,
        feature_names=spec.features.names,
        label_name=spec.label.name,
        rebalance_dates=signal_dates,
        artifact_dir=run_dir / 'models',
        label_horizon=spec.label.horizon,
    )
    stage_timings['model'] = perf_counter() - model_start
    logger.info('Model stage complete predictions=%s fallback_rate=%.2f%% elapsed=%.1fs', len(model_result.predictions), 100.0 * float(model_result.predictions['fallback_used'].mean()) if not model_result.predictions.empty else 0.0, stage_timings['model'])
    model_result.predictions.to_parquet(run_dir / 'signals' / 'predictions.parquet', index=False)
    model_result.split_metrics.to_csv(run_dir / 'models' / 'split_metrics.csv', index=False)
    model_result.feature_importance.to_csv(run_dir / 'models' / 'feature_importance.csv', index=False)
    model_result.model_registry.to_csv(run_dir / 'models' / 'model_registry.csv', index=False)

    signal_start = perf_counter()
    signal_result = signal_registry.run(
        spec.signal.name,
        SignalContext(
            predictions=model_result.predictions,
            model_dataset=model_dataset,
            feature_names=spec.features.names,
            label_name=spec.label.name,
        ),
        params=spec.signal.params,
    )
    stage_timings['signal'] = perf_counter() - signal_start
    signal_scores = signal_result.signal_scores
    signal_scores.to_parquet(run_dir / 'signals' / 'signal_scores.parquet', index=False)
    (run_dir / 'signals' / 'signal_summary.json').write_text(json.dumps(signal_result.summary, ensure_ascii=False, indent=2), encoding='utf-8')
    for table_name, table in signal_result.tables.items():
        table.to_parquet(run_dir / 'signals' / f'{table_name}.parquet', index=False)
    logger.info('Signal stage complete signal_dates=%s scored_rows=%s elapsed=%.1fs', signal_scores['trade_date'].nunique() if not signal_scores.empty else 0, len(signal_scores), stage_timings['signal'])

    market_reference = _build_market_reference(universe, daily_bar)
    portfolio_start = perf_counter()
    portfolio_result = portfolio_registry.build(
        spec.portfolio.constructor,
        PortfolioContext(
            signal_scores=signal_scores,
            universe_membership=universe,
            tradability=tradability,
            market_reference=market_reference,
            trade_calendar=trade_calendar,
            universe_name=spec.data.universe_name,
            portfolio_config=spec.portfolio,
            backtest_config=spec.backtest,
        ),
    )
    stage_timings['portfolio'] = perf_counter() - portfolio_start
    portfolio_result.target_weights.to_parquet(run_dir / 'signals' / 'target_weights.parquet', index=False)
    portfolio_result.filtered_candidates.to_parquet(run_dir / 'signals' / 'filtered_candidates.parquet', index=False)
    logger.info('Portfolio construction complete target_rows=%s selected_dates=%s elapsed=%.1fs', len(portfolio_result.target_weights), portfolio_result.target_weights['signal_date'].nunique() if not portfolio_result.target_weights.empty else 0, stage_timings['portfolio'])

    # ========== Phase 6: 组合增强 (持仓缓冲区 + 权重平滑 + 成本过滤) ==========
    logger.info("=" * 60)
    logger.info("Phase 6: Portfolio Enhancement (Buffer + Smooth + CostFilter)")
    logger.info("=" * 60)
    enhancer_start = perf_counter()
    enhancer = PortfolioEnhancer(
        buffer_config=BufferConfig(
            retain_threshold_rank=spec.enhancer.buffer_retain_threshold_rank,
            max_retain_ratio=spec.enhancer.buffer_max_retain_ratio,
        ),
        smoother_config=SmootherConfig(
            step_ratio=spec.enhancer.smoother_step_ratio,
            min_change_threshold=spec.enhancer.smoother_min_change_threshold,
        ),
        cost_config=CostFilterConfig(
            min_alpha_threshold=spec.enhancer.cost_min_alpha_threshold,
            cost_to_alpha_ratio=spec.enhancer.cost_cost_to_alpha_ratio,
        ),
    )
    
    logger.info("Enhancer config: Buffer(retain_rank=50,max=60%), Smooth(step=50%), CostFilter(alpha=0.2%)")
    
    # 准备价格数据
    close_matrix = daily_bar.pivot(index='trade_date', columns='symbol', values='close').sort_index()
    
    # 获取调仓日期
    tw_dates = sorted(portfolio_result.target_weights['execution_date'].unique())
    current_positions: dict[str, float] = {}  # weights (e.g., 0.05 for 5%)
    total_equity: float = spec.backtest.initial_cash  # start with initial cash
    prev_prices: dict[str, float] = {}  # 上一期价格，用于估算 equity 变化
    enhanced_weights_list = []
    enhancement_summary = {
        'total_rebalances': len(tw_dates),
        'buffered_retained': 0,
        'buffered_removed': 0,
        'filtered_trades': 0,
    }
    
    for exec_date in tw_dates:
        exec_date_ts = pd.Timestamp(exec_date)
        day_tw = portfolio_result.target_weights[portfolio_result.target_weights['execution_date'] == exec_date].copy()
        day_candidates = portfolio_result.filtered_candidates[portfolio_result.filtered_candidates['execution_date'] == exec_date_ts].copy()
        
        # 添加 rank 列 (基于 score)
        if 'rank' not in day_candidates.columns and 'score' in day_candidates.columns:
            day_candidates['rank'] = day_candidates['score'].rank(ascending=False, method='min')
        
        # 获取当日价格
        if exec_date_ts in close_matrix.index:
            prices = close_matrix.loc[exec_date_ts].dropna().to_dict()
        else:
            prices = {}
        
        # 估算 equity 变化 (基于持仓股票的价格变动)
        if current_positions and prev_prices:
            equity_change = 0.0
            for sym, weight in current_positions.items():
                curr_price = prices.get(sym, prev_prices.get(sym, 0))
                prev_price = prev_prices.get(sym, curr_price)
                if prev_price > 0 and curr_price > 0:
                    price_ret = (curr_price - prev_price) / prev_price
                    position_value = weight * total_equity
                    equity_change += position_value * price_ret
            total_equity += equity_change
            total_equity = max(total_equity, spec.backtest.initial_cash * 0.5)  # 防止为负
        
        # 应用增强器
        enhanced_tw, summary = enhancer.enhance(
            candidates=day_candidates,
            current_positions=current_positions,
            target_weights=day_tw,
            prices=prices,
            execution_date=exec_date_ts,
            total_equity=total_equity,
            lot_size=spec.backtest.lot_size,
            commission_bps=spec.backtest.commission_bps,
            stamp_duty_bps=spec.backtest.stamp_duty_bps,
            slippage_bps=spec.backtest.slippage_bps,
            min_trade_value=spec.portfolio.min_trade_value,
        )
        
        # 更新 prev_prices
        prev_prices = prices.copy()
        
        enhancement_summary['buffered_retained'] += summary.get('buffered_retained', 0)
        enhancement_summary['buffered_removed'] += summary.get('buffered_removed', 0)
        enhancement_summary['filtered_trades'] += summary.get('filtered_trades', 0)
        
        enhanced_weights_list.append(enhanced_tw)
        
        # 更新当前持仓 (权重)
        current_positions = {
            row['symbol']: row['target_weight']
            for _, row in enhanced_tw.iterrows()
            if row['target_weight'] > 0
        }
    
    # 合并增强后的权重
    if enhanced_weights_list:
        enhanced_target_weights = pd.concat(enhanced_weights_list, ignore_index=True)
    else:
        enhanced_target_weights = portfolio_result.target_weights.copy()
    
    enhanced_target_weights.to_parquet(run_dir / 'signals' / 'target_weights_enhanced.parquet', index=False)
    stage_timings['enhancement'] = perf_counter() - enhancer_start
    logger.info("Portfolio enhancement complete rebalances=%s buffered_retained=%s buffered_removed=%s filtered_trades=%s elapsed=%.1fs",
                enhancement_summary['total_rebalances'],
                enhancement_summary['buffered_retained'],
                enhancement_summary['buffered_removed'],
                enhancement_summary['filtered_trades'],
                stage_timings['enhancement'])
    logger.info("=" * 60)

    backtest_start = perf_counter()
    backtest_result = run_backtest(
        daily_bar=daily_bar,
        target_weights=enhanced_target_weights,
        tradability=tradability,
        corporate_actions=corporate_actions,
        initial_cash=spec.backtest.initial_cash,
        lot_size=spec.backtest.lot_size,
        commission_bps=spec.backtest.commission_bps,
        stamp_duty_bps=spec.backtest.stamp_duty_bps,
        slippage_bps=spec.backtest.slippage_bps,
        min_trade_value=spec.portfolio.min_trade_value,
        execution_constraint_mode=spec.backtest.execution_constraint_mode,
    )
    stage_timings['backtest'] = perf_counter() - backtest_start
    logger.info('Backtest complete nav_rows=%s trades=%s elapsed=%.1fs', len(backtest_result.nav), len(backtest_result.trades), stage_timings['backtest'])
    benchmark_nav = compute_benchmark_nav(
        strategy_nav=backtest_result.nav,
        daily_bar=daily_bar,
        universe_membership=universe,
        data_source=spec.data.source,
        universe_name=spec.data.universe_name,
    )
    drawdown = compute_drawdown(backtest_result.nav)
    monthly_returns = compute_monthly_returns(backtest_result.nav)
    rank_ic = compute_rank_ic(signal_scores, label_panel, spec.label.name)
    latest_signal = latest_signal_report(portfolio_result.target_weights, security_master)
    active_backtest_start = (
        pd.to_datetime(portfolio_result.target_weights['execution_date']).min()
        if not portfolio_result.target_weights.empty
        else pd.to_datetime(backtest_result.nav['trade_date']).min()
    )
    active_signal_start = (
        pd.to_datetime(signal_scores['trade_date']).min()
        if not signal_scores.empty
        else active_backtest_start
    )
    active_nav, active_trades, active_rank_ic, active_benchmark = trim_backtest_window(
        nav=backtest_result.nav,
        trades=backtest_result.trades,
        rank_ic=rank_ic,
        benchmark_nav=benchmark_nav,
        active_start=active_backtest_start,
        rank_ic_start=active_signal_start,
    )
    active_positions = (
        backtest_result.positions.loc[pd.to_datetime(backtest_result.positions['trade_date']) >= pd.Timestamp(active_backtest_start)].reset_index(drop=True)
        if not backtest_result.positions.empty
        else backtest_result.positions.copy()
    )
    active_drawdown = compute_drawdown(active_nav)
    active_monthly_returns = compute_monthly_returns(active_nav)
    summary = performance_summary(active_nav, active_trades, active_rank_ic, active_benchmark)
    evaluation_start = perf_counter()
    evaluation_result = evaluation_registry.run(
        spec.evaluation.suite,
        EvaluationContext(
            signal_scores=signal_scores,
            label_panel=label_panel,
            label_name=spec.label.name,
            model_dataset=model_dataset,
            target_weights=portfolio_result.target_weights,
            rank_ic=active_rank_ic,
            nav=active_nav,
            trades=active_trades,
        ),
        params=spec.evaluation.params,
    )
    stage_timings['evaluation'] = perf_counter() - evaluation_start
    summary.update({key: value for key, value in evaluation_result.metrics.items() if isinstance(value, (int, float))})
    summary.update({
        'enhancement_rebalances': enhancement_summary['total_rebalances'],
        'enhancement_buffered_retained': enhancement_summary['buffered_retained'],
        'enhancement_buffered_removed': enhancement_summary['buffered_removed'],
        'enhancement_filtered_trades': enhancement_summary['filtered_trades'],
    })
    summary.update(
        {
            'run_id': run_id,
            'strategy_name': spec.name,
            'experiment_name': spec.name,
            'description': spec.description,
            'config_hash': spec.config_hash,
            'artifact_dir': str(run_dir),
            'prediction_dates': int(signal_scores['trade_date'].nunique()),
            'feature_set': spec.features.set_name,
            'label_name': spec.label.name,
            'model_family': spec.model.family,
            'signal_name': spec.signal.name,
            'portfolio_constructor': spec.portfolio.constructor,
            'evaluation_suite': spec.evaluation.suite,
            'universe_name': spec.data.universe_name,
            'data_start': str(pd.to_datetime(backtest_result.nav['trade_date']).min().date()),
            'data_end': str(pd.to_datetime(backtest_result.nav['trade_date']).max().date()),
            'research_start': spec.data.start_date,
            'warmup_data_start': source_spec.start_date,
            'signal_start': str(pd.Timestamp(active_signal_start).date()),
            'signal_end': str(pd.to_datetime(signal_scores['trade_date']).max().date()) if not signal_scores.empty else None,
            'backtest_start': str(pd.Timestamp(active_backtest_start).date()),
            'backtest_end': str(pd.to_datetime(backtest_result.nav['trade_date']).max().date()),
        }
    )

    active_nav.to_parquet(run_dir / 'backtest' / 'nav.parquet', index=False)
    active_benchmark.to_parquet(run_dir / 'backtest' / 'benchmark_nav.parquet', index=False)
    active_trades.to_parquet(run_dir / 'backtest' / 'trades.parquet', index=False)
    active_positions.to_parquet(run_dir / 'backtest' / 'positions.parquet', index=False)
    active_drawdown.to_parquet(run_dir / 'backtest' / 'drawdown.parquet', index=False)
    active_monthly_returns.to_parquet(run_dir / 'backtest' / 'monthly_returns.parquet', index=False)
    active_rank_ic.to_parquet(run_dir / 'backtest' / 'rank_ic.parquet', index=False)
    backtest_result.nav.to_parquet(run_dir / 'backtest' / 'nav_full.parquet', index=False)
    benchmark_nav.to_parquet(run_dir / 'backtest' / 'benchmark_nav_full.parquet', index=False)
    backtest_result.trades.to_parquet(run_dir / 'backtest' / 'trades_full.parquet', index=False)
    backtest_result.positions.to_parquet(run_dir / 'backtest' / 'positions_full.parquet', index=False)
    drawdown.to_parquet(run_dir / 'backtest' / 'drawdown_full.parquet', index=False)
    monthly_returns.to_parquet(run_dir / 'backtest' / 'monthly_returns_full.parquet', index=False)
    rank_ic.to_parquet(run_dir / 'backtest' / 'rank_ic_full.parquet', index=False)
    latest_signal.to_parquet(run_dir / 'signals' / 'latest_signal.parquet', index=False)
    for table_name, table in evaluation_result.tables.items():
        table.to_parquet(run_dir / 'evaluation' / f'{table_name}.parquet', index=False)

    dataset_summary = {
        'dataset_rows': int(len(model_dataset)),
        'dataset_dates': int(model_dataset['trade_date'].nunique()),
        'feature_names': spec.features.names,
        'signal_rows': int(len(signal_scores)),
        'trade_calendar_rows': int(len(trade_calendar)),
        'universe_rows': int(len(universe)),
    }
    (run_dir / 'metadata' / 'dataset_summary.json').write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_markdown_report(
        output_path=run_dir / 'reports' / 'run_report.md',
        summary=summary,
        latest_signal=latest_signal,
        split_metrics=model_result.split_metrics,
        feature_importance=model_result.feature_importance,
    )
    (run_dir / 'reports' / 'factor_diagnostics.md').write_text(evaluation_result.markdown, encoding='utf-8')

    selected_modules = {
        'model': {'name': spec.model.family, 'version': spec.model.version},
        'signal': {'name': spec.signal.name, 'version': spec.signal.version},
        'portfolio': {'name': spec.portfolio.constructor, 'weighting': spec.portfolio.weighting},
        'evaluation': {'name': spec.evaluation.suite, 'version': spec.evaluation.version},
    }
    registry_catalog = {
        'features': feature_registry.inventory().to_dict(orient='records'),
        'labels': label_registry.inventory().to_dict(orient='records'),
        'models': model_registry.inventory().to_dict(orient='records'),
        'signals': signal_registry.inventory().to_dict(orient='records'),
        'portfolio_constructors': portfolio_registry.inventory().to_dict(orient='records'),
        'evaluation_suites': evaluation_registry.inventory().to_dict(orient='records'),
    }
    data_contract = build_data_contract_report(
        {
            'trade_calendar': trade_calendar,
            'security_master': security_master,
            'daily_bar': daily_bar,
            'universe_membership': universe_membership,
            'tradability': tradability,
            'corporate_actions': corporate_actions,
        }
    )
    stage_timings['total'] = perf_counter() - overall_start
    (run_dir / 'metadata' / 'stage_timings.json').write_text(json.dumps(stage_timings, ensure_ascii=False, indent=2), encoding='utf-8')
    (run_dir / 'metadata' / 'data_contract.json').write_text(json.dumps(data_contract, ensure_ascii=False, indent=2), encoding='utf-8')
    (run_dir / 'metadata' / 'registry_catalog.json').write_text(json.dumps(registry_catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    (run_dir / 'metadata' / 'selected_modules.json').write_text(json.dumps(selected_modules, ensure_ascii=False, indent=2), encoding='utf-8')
    (run_dir / 'metadata' / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    artifact_inventory = build_artifact_inventory(run_dir)
    artifact_inventory.to_parquet(run_dir / 'metadata' / 'artifact_inventory.parquet', index=False)
    (run_dir / 'metadata' / 'artifact_inventory.json').write_text(json.dumps(artifact_inventory.to_dict(orient='records'), ensure_ascii=False, indent=2), encoding='utf-8')
    experiment_manifest = build_experiment_manifest(
        spec=spec,
        run_id=run_id,
        summary=summary,
        dataset_summary=dataset_summary,
        data_snapshot=snapshot_manifest,
        data_contract=data_contract,
        stage_timings=stage_timings,
        registry_catalog=registry_catalog,
        selected_modules=selected_modules,
        artifact_inventory=artifact_inventory,
    )
    (run_dir / 'metadata' / 'experiment_manifest.json').write_text(json.dumps(experiment_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('Evaluation and manifest complete diagnostics_tables=%s artifact_count=%s', len(evaluation_result.tables), len(artifact_inventory))

    tracker = MLflowTracker(spec.name)
    mlflow_run_id = tracker.log_run(
        run_name=run_id,
        params=spec.flattened_params(),
        metrics={key: value for key, value in summary.items() if isinstance(value, (int, float))},
        tags={'snapshot_id': spec.data.snapshot_id, 'registry_stage': spec.model.registry_stage, 'universe_name': spec.data.universe_name},
        artifact_dir=run_dir,
    )
    summary['mlflow_run_id'] = mlflow_run_id
    (run_dir / 'metadata' / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    experiment_manifest['summary'] = summary
    (run_dir / 'metadata' / 'experiment_manifest.json').write_text(json.dumps(experiment_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('Experiment complete run_id=%s total_elapsed=%.1fs annual_return=%s max_drawdown=%s', run_id, stage_timings['total'], summary.get('annual_return'), summary.get('max_drawdown'))
    return summary


def _resolve_universe_frame(spec: ExperimentSpec, universe_membership: pd.DataFrame, trade_calendar: pd.DataFrame) -> pd.DataFrame:
    if spec.data.universe_mode == 'current_snapshot':
        return build_current_snapshot_universe(universe_membership, spec.data.universe_name, trade_calendar)
    return build_point_in_time_universe(universe_membership, spec.data.universe_name)


def build_source_data_spec_with_warmup(spec: ExperimentSpec):
    warmup_trade_days = _required_warmup_trade_days(spec)
    warmup_start = _warmup_start_date(spec.data.start_date, warmup_trade_days)
    return replace(spec.data, start_date=warmup_start, formal_start_date=spec.data.start_date)


def _build_source_data_spec_with_warmup(spec: ExperimentSpec):
    return build_source_data_spec_with_warmup(spec)


def _required_warmup_trade_days(spec: ExperimentSpec) -> int:
    feature_registry = default_feature_registry()
    max_feature_lookback = max((feature_registry.get(name).lookback for name in spec.features.names), default=0)
    return int(
        max_feature_lookback
        + spec.model.train_window_days
        + 10
        + spec.model.valid_window_days
        + spec.label.horizon
        + spec.backtest.rebalance_frequency_days
        + spec.backtest.trade_delay_days
    )


def _warmup_start_date(start_date: str, warmup_trade_days: int) -> str:
    start_ts = pd.Timestamp(start_date)
    ref_calendar_path = Path('data/raw/baostock_daily_raw/000001_SZ.parquet')
    if ref_calendar_path.exists():
        ref_dates = sorted(pd.to_datetime(pd.read_parquet(ref_calendar_path, columns=['trade_date'])['trade_date']).drop_duplicates())
        if ref_dates:
            anchor_idx = bisect_left(ref_dates, start_ts)
            warmup_idx = max(0, anchor_idx - int(warmup_trade_days))
            return str(pd.Timestamp(ref_dates[warmup_idx]).date())
    calendar_buffer_days = max(90, int(round(warmup_trade_days * 1.45)) + 20)
    return str((start_ts - pd.Timedelta(days=calendar_buffer_days)).date())


def _build_market_reference(universe: pd.DataFrame, daily_bar: pd.DataFrame) -> pd.Series:
    if universe.empty or daily_bar.empty:
        return pd.Series(dtype=float)
    frame = (
        universe[['trade_date', 'symbol']]
        .merge(daily_bar[['trade_date', 'symbol', 'adj_close']], on=['trade_date', 'symbol'], how='left')
        .sort_values(['trade_date', 'symbol'])
    )
    market_reference = frame.groupby('trade_date')['adj_close'].mean().sort_index()
    market_reference.index = pd.to_datetime(market_reference.index)
    return market_reference


def _catalog_has_required_tables(catalog: LocalResearchCatalog) -> bool:
    return all(catalog.table_exists(table_name, 'silver') for table_name in SCHEMAS)


def _catalog_is_ready_for_spec(catalog: LocalResearchCatalog, source) -> bool:
    if not _catalog_has_required_tables(catalog):
        return False
    try:
        trade_calendar = catalog.read_table('trade_calendar', zone='silver')
        daily_bar = catalog.read_table('daily_bar', zone='silver')
    except FileNotFoundError:
        return False
    if trade_calendar.empty or daily_bar.empty:
        return False
    available_start = min(pd.to_datetime(trade_calendar['trade_date']).min(), pd.to_datetime(daily_bar['trade_date']).min())
    available_end = max(pd.to_datetime(trade_calendar['trade_date']).max(), pd.to_datetime(daily_bar['trade_date']).max())
    requested_start = pd.Timestamp(getattr(source, 'start_date', available_start))
    requested_end = pd.Timestamp(getattr(source, 'end_date', available_end))
    return available_start <= requested_start and available_end >= requested_end


def _build_signal_dates(
    available_dates: list[pd.Timestamp],
    trade_calendar: pd.DataFrame,
    rebalance_frequency_days: int,
    trade_delay_days: int,
    anchor_mode: str,
    anchor_date: str | None,
    backtest_start_date: str | None,
) -> list[pd.Timestamp]:
    available_set = {pd.Timestamp(date) for date in available_dates}
    calendar_dates = sorted(pd.to_datetime(trade_calendar['trade_date']).tolist())
    if not calendar_dates:
        return []
    if anchor_mode == 'follow_backtest_start':
        anchor_ts = pd.Timestamp(backtest_start_date or calendar_dates[0])
    elif anchor_date:
        anchor_ts = pd.Timestamp(anchor_date)
    else:
        return [pd.Timestamp(date) for date in available_dates[::rebalance_frequency_days]]
    anchored_calendar = [date for date in calendar_dates if date >= anchor_ts]
    signal_dates: list[pd.Timestamp] = []
    for idx, execution_date in enumerate(anchored_calendar):
        if idx % rebalance_frequency_days != 0:
            continue
        signal_idx = idx - trade_delay_days
        if signal_idx < 0:
            continue
        signal_date = anchored_calendar[signal_idx]
        if signal_date in available_set and execution_date in available_set:
            signal_dates.append(signal_date)
    return signal_dates
