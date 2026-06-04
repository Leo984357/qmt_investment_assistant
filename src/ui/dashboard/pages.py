from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src.evaluation.reporting import compute_drawdown, compute_monthly_returns
from src.experiment.runner import run_experiment

from .charts import *
from .helpers import *


def _render_quick_actions() -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button('看今日组合', use_container_width=True, on_click=_jump_to_page, args=('signal', 'daily'))
    with col2:
        st.button('看回测复盘', use_container_width=True, on_click=_jump_to_page, args=('backtest', 'review'))
    with col3:
        st.button('看因子池', use_container_width=True, on_click=_jump_to_page, args=('factorlab', 'develop'))
    with col4:
        st.button('进实验开发', use_container_width=True, on_click=_jump_to_page, args=('studio', 'develop'))


def _render_overview(bundle: dict) -> None:
    summary = bundle['summary']
    status = _status_snapshot(bundle)
    latest_signal = bundle['latest_signal']
    data_snapshot = bundle['data_snapshot']
    benchmark_nav = bundle.get('benchmark_nav', pd.DataFrame())
    split_metrics = bundle['split_metrics']
    rebalance_delta = _build_rebalance_delta(bundle['target_weights'])

    _section_header('日常入口', '决策总台', '把成熟平台常见的“先结论、后下钻”路径放到最前面，先看今天该做什么，再决定是否进入研究细节。')
    st.markdown('\n'.join([f'- {line}' for line in _decision_brief(bundle)]))
    _render_quick_actions()

    cards = [
        {'label': '总收益', 'value': _format_pct(summary.get('total_return')), 'sub': '完整回测结果', 'tone': _tone_from_value(summary.get('total_return', 0.0))},
        {'label': '年化收益', 'value': _format_pct(summary.get('annual_return')), 'sub': '按交易日折算后的年化表现', 'tone': _tone_from_value(summary.get('annual_return', 0.0))},
        {'label': '最大回撤', 'value': _format_pct(summary.get('max_drawdown')), 'sub': '从峰值到谷值的最大跌幅', 'tone': _tone_from_value(summary.get('max_drawdown', 0.0), invert=True)},
        {'label': '平均 Rank IC', 'value': _format_number(summary.get('avg_rank_ic'), 3), 'sub': '横截面信号质量', 'tone': _tone_from_value(summary.get('avg_rank_ic', 0.0))},
        {'label': '成交笔数', 'value': str(int(summary.get('trade_count', 0))), 'sub': '回测中实际执行的订单数', 'tone': 'neutral'},
        {'label': '总成本', 'value': _format_currency(summary.get('total_cost')), 'sub': '佣金、税费和滑点', 'tone': 'warn'},
    ]
    if 'benchmark_total_return' in summary:
        benchmark_name = str(benchmark_nav['benchmark_name'].iloc[0]) if not benchmark_nav.empty and 'benchmark_name' in benchmark_nav.columns else '基准'
        cards.insert(2, {'label': '基准收益', 'value': _format_pct(summary.get('benchmark_total_return')), 'sub': benchmark_name, 'tone': 'neutral'})
    if 'excess_total_return' in summary:
        cards.insert(3, {'label': '总超额收益', 'value': _format_pct(summary.get('excess_total_return')), 'sub': '策略相对基准', 'tone': _tone_from_value(summary.get('excess_total_return', 0.0))})
    _render_metric_grid(cards)

    summary_backtest_start = summary.get('backtest_start')
    summary_backtest_end = summary.get('backtest_end')
    summary_signal_start = summary.get('signal_start')
    summary_signal_end = summary.get('signal_end')
    summary_data_start = summary.get('data_start')
    summary_data_end = summary.get('data_end')
    summary_research_start = summary.get('research_start', summary_backtest_start)
    summary_warmup_start = summary.get('warmup_data_start', summary_data_start)
    _render_status_grid(
        [
            {
                'label': '风险状态',
                'value': status['risk_flag'],
                'note': f"当前总暴露 {_format_pct(status['gross_exposure'])}，样本内共生成 {summary.get('prediction_dates', 0)} 个信号日。",
            },
            {
                'label': '模型运行',
                'value': f"Fallback {_format_pct(_fallback_rate(bundle))}",
                'note': f"当前模型 {summary.get('model_family', 'N/A')}，特征集 {summary.get('feature_set', 'N/A')}。",
            },
            {
                'label': '数据快照',
                'value': f"{len(data_snapshot)} 张 silver 表",
                'note': f"预热数据期 {summary_warmup_start or summary_data_start or 'N/A'} 至 {summary_data_end or _latest_data_date(bundle)}；正式研究起点 {summary_research_start or 'N/A'}。",
            },
            {
                'label': '回测区间',
                'value': f"{summary_backtest_start or 'N/A'} -> {summary_backtest_end or 'N/A'}",
                'note': f"信号期 {summary_signal_start or 'N/A'} 至 {summary_signal_end or 'N/A'}；回测按首个执行日开始，不含前面的空仓等待期。",
            },
            {'label': '组合姿态', 'value': status['market_posture'], 'note': f"最新目标组合共 {len(latest_signal)} 只股票，采用 Top-N 目标持仓。"},
        ]
    )

    left, right = st.columns([1.5, 1])
    with left:
        st.altair_chart(_nav_chart(bundle['nav'], bundle.get('benchmark_nav', pd.DataFrame())), use_container_width=True)
        st.altair_chart(_drawdown_chart(bundle['drawdown']), use_container_width=True)
    with right:
        st.altair_chart(_feature_importance_chart(bundle['feature_importance']), use_container_width=True)
        st.dataframe(
            _styled_table(
                split_metrics.tail(10).sort_values('signal_date', ascending=False),
                {'valid_rank_ic': '{:.3f}', 'valid_rmse': '{:.4f}'},
            ),
            use_container_width=True,
            height=290,
        )

    _section_header('今日动作', '调仓摘要', '相对上一期目标组合，当前这期新增了哪些票、删掉了哪些票、哪些票需要显著加减仓。')
    if rebalance_delta.empty:
        st.info('当前还没有足够的目标组合历史来生成调仓摘要。')
    else:
        action_counts = rebalance_delta['action'].value_counts()
        _render_metric_grid(
            [
                {'label': '新增买入', 'value': str(int(action_counts.get('买入', 0))), 'sub': '相对上一期权重上升或新进'},
                {'label': '减仓/移除', 'value': str(int(action_counts.get('卖出', 0))), 'sub': '相对上一期权重下降或移出'},
                {'label': '维持', 'value': str(int(action_counts.get('维持', 0))), 'sub': '相对上一期目标权重不变'},
                {'label': '最大调仓幅度', 'value': _format_pct(float(rebalance_delta['delta_weight'].abs().max())), 'sub': '单票权重变化最大值'},
            ]
        )
        display_delta = _to_display_frame(
            rebalance_delta[['symbol', 'action', 'prev_weight', 'target_weight', 'delta_weight', 'prev_rank', 'rank', 'score']],
            {
                'symbol': '代码',
                'action': '动作',
                'prev_weight': '上期权重',
                'target_weight': '本期权重',
                'delta_weight': '权重变化',
                'prev_rank': '上期排名',
                'rank': '本期排名',
                'score': '分数',
            },
        )
        st.dataframe(
            _styled_table(display_delta.head(20), {'上期权重': '{:.2%}', '本期权重': '{:.2%}', '权重变化': '{:.2%}', '分数': '{:.4f}'}),
            use_container_width=True,
            height=320,
        )

    tabs = st.tabs(['最新信号', '数据快照', '运行说明'])
    with tabs[0]:
        signal_cols = st.columns([1.2, 1])
        with signal_cols[0]:
            st.altair_chart(_signal_score_chart(latest_signal), use_container_width=True)
        with signal_cols[1]:
            st.altair_chart(_industry_weight_chart(latest_signal), use_container_width=True)
        st.dataframe(
            _styled_table(latest_signal, {'score': '{:.4f}', 'target_weight': '{:.2%}'}),
            use_container_width=True,
            height=340,
        )
    with tabs[1]:
        snapshot_table = data_snapshot.copy()
        if 'sha256' in snapshot_table.columns:
            snapshot_table['sha256'] = snapshot_table['sha256'].astype(str).str.slice(0, 12) + '...'
        snapshot_table = _to_display_frame(
            snapshot_table,
            {'table_name': '表名', 'zone': '分层', 'path': '路径', 'row_count': '行数', 'sha256': '校验值', 'created_at': '生成时间'},
        )
        st.dataframe(snapshot_table, use_container_width=True, height=320)
    with tabs[2]:
        st.markdown(
            f"""
            <div class='artifact-shell'>
            <p><strong>运行目录</strong> <span class='code-chip'>{bundle['run_dir']}</span></p>
            <p><strong>配置哈希</strong> <span class='code-chip'>{summary.get('config_hash', 'N/A')[:12]}</span></p>
            <p><strong>MLflow 运行</strong> <span class='code-chip'>{summary.get('mlflow_run_id', 'N/A')}</span></p>
            <p><strong>运行命令</strong> <span class='code-chip'>python -m src.cli experiment --config configs/experiments/hs300_lightgbm.yaml</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if bundle['report_markdown']:
            st.markdown(bundle['report_markdown'])


def _render_ledger(ledger: pd.DataFrame, selected_run_name: str) -> None:
    _section_header('研究记忆', '实验台账', '不用逐个打开工件，也能直接比较不同运行的收益、回撤、信号质量、成本和样本规模。')
    if ledger.empty:
        st.info('当前还没有可用的实验台账。')
        return

    run_options = ledger['run_name'].tolist()
    default_compare = [selected_run_name] if selected_run_name in run_options else run_options[:1]
    compare_runs = st.multiselect('选择对比运行', run_options, default=default_compare[:2], max_selections=min(3, len(run_options)))

    cards = [
        {'label': '已跟踪运行', 'value': str(len(ledger)), 'sub': '已有运行摘要的实验次数'},
        {'label': '中位收益', 'value': _format_pct(float(ledger['total_return'].median())), 'sub': '台账中位数水平'},
        {'label': '最佳夏普近似', 'value': _format_number(float(ledger['sharpe_like'].max()), 2), 'sub': ledger.loc[ledger['sharpe_like'].idxmax(), 'run_name']},
        {'label': '最浅回撤', 'value': _format_pct(float(ledger['max_drawdown'].max())), 'sub': ledger.loc[ledger['max_drawdown'].idxmax(), 'run_name']},
    ]
    _render_metric_grid(cards)

    st.altair_chart(_ledger_scatter_chart(ledger), use_container_width=True)

    compare_df = ledger.loc[ledger['run_name'].isin(compare_runs)].copy()
    if not compare_df.empty:
        compare_view = compare_df[
            ['run_name', 'experiment_name', 'total_return', 'annual_return', 'sharpe_like', 'max_drawdown', 'avg_rank_ic', 'trade_count', 'dataset_rows']
        ].sort_values('created_at', ascending=False)
        st.dataframe(
            _styled_table(
                compare_view,
                {
                    'total_return': '{:.2%}',
                    'annual_return': '{:.2%}',
                    'sharpe_like': '{:.2f}',
                    'max_drawdown': '{:.2%}',
                    'avg_rank_ic': '{:.3f}',
                    'dataset_rows': '{:,.0f}',
                },
            ),
            use_container_width=True,
            height=220,
        )

    ledger_view = ledger[
        ['created_at', 'run_name', 'experiment_name', 'universe_name', 'model_family', 'feature_set', 'total_return', 'sharpe_like', 'max_drawdown', 'avg_rank_ic', 'trade_count']
    ].copy()
    st.dataframe(
        _styled_table(
            ledger_view,
            {'created_at': lambda v: pd.Timestamp(v).strftime('%Y-%m-%d %H:%M'), 'total_return': '{:.2%}', 'sharpe_like': '{:.2f}', 'max_drawdown': '{:.2%}', 'avg_rank_ic': '{:.3f}'},
        ),
        use_container_width=True,
        height=420,
    )


def _render_data_status(bundle: dict) -> None:
    _section_header('输入治理', '数据状态', '查看当前研究运行的数据覆盖、schema 清单、特征注册表和标签定义。')
    dataset_summary = bundle['dataset_summary']
    data_snapshot = bundle['data_snapshot']
    feature_inventory = bundle['feature_inventory']
    feature_registry = bundle.get('feature_registry', pd.DataFrame())
    label_inventory = bundle['label_inventory']
    filtered_candidates = bundle['filtered_candidates']
    data_contract = bundle.get('data_contract', pd.DataFrame())
    stage_timings = bundle.get('stage_timings', {})
    selected_modules = bundle.get('selected_modules', {})
    data_cfg = bundle['config'].get('data', {})
    latest_snapshot_ts = pd.to_datetime(data_snapshot['created_at']).max().strftime('%Y-%m-%d %H:%M') if not data_snapshot.empty and 'created_at' in data_snapshot.columns else 'N/A'

    cards = [
        {'label': '样本行数', 'value': f"{dataset_summary.get('dataset_rows', 0):,}", 'sub': '完成拼接后的训练样本'},
        {'label': '样本日期数', 'value': f"{dataset_summary.get('dataset_dates', 0):,}", 'sub': '时点化日频样本'},
        {'label': '特征数量', 'value': str(len(dataset_summary.get('feature_names', []))), 'sub': '已注册的特征组件'},
        {'label': '数据模式', 'value': _source_label(data_cfg.get('source', 'mock_ashare')), 'sub': f"快照 {data_cfg.get('snapshot_id', 'N/A')}"},
        {'label': 'Silver 表数', 'value': str(len(data_snapshot)), 'sub': f'最新快照时间 {latest_snapshot_ts}'},
    ]
    _render_metric_grid(cards)

    left, right = st.columns([1.1, 1])
    with left:
        snapshot_view = data_snapshot.copy()
        if 'sha256' in snapshot_view.columns:
            snapshot_view['sha256'] = snapshot_view['sha256'].astype(str).str.slice(0, 16) + '...'
        snapshot_view = _to_display_frame(
            snapshot_view,
            {'table_name': '表名', 'zone': '分层', 'path': '路径', 'row_count': '行数', 'sha256': '校验值', 'created_at': '生成时间'},
        )
        st.dataframe(snapshot_view, use_container_width=True, height=320)
    with right:
        if not filtered_candidates.empty:
            st.altair_chart(_reason_chart(filtered_candidates), use_container_width=True)
        else:
            st.info('当前没有可展示的候选过滤诊断。')

    tabs = st.tabs(['特征注册表', '标签注册表', '数据契约', '运行阶段', '解析后配置'])
    with tabs[0]:
        registry_view = feature_registry if not feature_registry.empty else feature_inventory
        st.dataframe(registry_view, use_container_width=True, height=300)
    with tabs[1]:
        st.dataframe(label_inventory, use_container_width=True, height=180)
    with tabs[2]:
        st.dataframe(data_contract, use_container_width=True, height=240)
    with tabs[3]:
        stage_frame = pd.DataFrame(
            [{'stage': key, 'seconds': value} for key, value in stage_timings.items()]
        ).sort_values('seconds', ascending=False) if stage_timings else pd.DataFrame()
        if not stage_frame.empty:
            st.dataframe(_styled_table(stage_frame, {'seconds': '{:.2f}'}), use_container_width=True, height=220)
        if selected_modules:
            st.code(json.dumps(selected_modules, ensure_ascii=False, indent=2), language='json')
    with tabs[4]:
        st.code(yaml.safe_dump(bundle['config'], allow_unicode=True, sort_keys=False), language='yaml')


def _render_backtest(bundle: dict) -> None:
    _section_header('绩效实验室', '回测分析', '查看净值路径、回撤区间、换手、月度分布、信号 IC 和成交结构。')
    nav = bundle['nav']
    benchmark_nav = bundle.get('benchmark_nav', pd.DataFrame())
    rank_ic = bundle['rank_ic']
    trades = bundle['trades']
    if nav.empty:
        st.info('当前没有可展示的回测结果。')
        return

    nav_min_date = pd.to_datetime(nav['trade_date']).min().date()
    signal_min_date = pd.to_datetime(rank_ic['trade_date']).min().date() if not rank_ic.empty else nav_min_date
    min_date = min(nav_min_date, signal_min_date)
    max_date = pd.to_datetime(nav['trade_date']).max().date()
    default_start = pd.to_datetime(bundle['summary'].get('signal_start', bundle['summary'].get('backtest_start', min_date))).date()
    default_end = pd.to_datetime(bundle['summary'].get('backtest_end', max_date)).date()
    selected_range = st.slider(
        '分析时间范围',
        min_value=min_date,
        max_value=max_date,
        value=(max(min_date, default_start), min(max_date, default_end)),
        format='YYYY-MM-DD',
    )
    start_date, end_date = selected_range
    st.caption(
        f"预热数据期：{bundle['summary'].get('warmup_data_start', bundle['summary'].get('data_start', str(min_date)))} 至 {bundle['summary'].get('data_end', str(max_date))}；"
        f"正式研究起点：{bundle['summary'].get('research_start', bundle['summary'].get('backtest_start', str(nav_min_date)))}；"
        f"信号期：{bundle['summary'].get('signal_start', str(signal_min_date))} 至 {bundle['summary'].get('signal_end', str(max_date))}；"
        f"回测期：{bundle['summary'].get('backtest_start', str(nav_min_date))} 至 {bundle['summary'].get('backtest_end', str(max_date))}。"
    )

    nav_window = _filter_date_window(nav, start_date, end_date)
    benchmark_window = _filter_date_window(benchmark_nav, start_date, end_date)
    drawdown_window = compute_drawdown(nav_window[['trade_date', 'nav']]) if not nav_window.empty else pd.DataFrame()
    monthly_returns_window = compute_monthly_returns(nav_window[['trade_date', 'nav']]) if not nav_window.empty else pd.DataFrame()
    rank_ic_window = _filter_date_window(rank_ic, start_date, end_date)
    trades_window = _filter_date_window(trades, start_date, end_date)

    strategy_return = float(nav_window['nav'].iloc[-1] / max(nav_window['nav'].iloc[0], 1e-9) - 1.0) if len(nav_window) >= 2 else 0.0
    cards = [
        {'label': '策略区间收益', 'value': _format_pct(strategy_return), 'sub': '当前时间窗口内'},
        {'label': '窗口最大回撤', 'value': _format_pct(float(drawdown_window['drawdown'].min()) if not drawdown_window.empty else 0.0), 'sub': '按当前窗口重算'},
        {'label': '窗口交易笔数', 'value': f"{len(trades_window):,}", 'sub': '当前时间范围内成交'},
        {'label': '窗口平均IC', 'value': _format_number(float(rank_ic_window['rank_ic'].mean()) if not rank_ic_window.empty else 0.0, 3), 'sub': '当前时间范围内信号质量'},
    ]
    if not benchmark_window.empty and 'benchmark_nav' in benchmark_window.columns:
        benchmark_return = float(benchmark_window['benchmark_nav'].iloc[-1] / max(benchmark_window['benchmark_nav'].iloc[0], 1e-9) - 1.0) if len(benchmark_window) >= 2 else 0.0
        strategy_growth = float(nav_window['nav'].iloc[-1] / max(nav_window['nav'].iloc[0], 1e-9)) if len(nav_window) >= 2 else 1.0
        benchmark_growth = float(benchmark_window['benchmark_nav'].iloc[-1] / max(benchmark_window['benchmark_nav'].iloc[0], 1e-9)) if len(benchmark_window) >= 2 else 1.0
        excess_return = strategy_growth / max(benchmark_growth, 1e-9) - 1.0 if len(nav_window) >= 2 and len(benchmark_window) >= 2 else 0.0
        cards.insert(1, {'label': '基准区间收益', 'value': _format_pct(benchmark_return), 'sub': str(benchmark_window['benchmark_name'].iloc[0]) if 'benchmark_name' in benchmark_window.columns else '基准'})
        cards.insert(2, {'label': '区间超额收益', 'value': _format_pct(excess_return), 'sub': '策略相对基准'})
    _render_metric_grid(cards)

    top, bottom = st.tabs(['绩效表现', '执行细节'])
    with top:
        col1, col2 = st.columns([1.35, 1])
        with col1:
            st.altair_chart(_nav_chart(nav_window, benchmark_window), use_container_width=True)
            excess_chart = _excess_nav_chart(nav_window, benchmark_window)
            if excess_chart is not None:
                st.altair_chart(excess_chart, use_container_width=True)
            st.altair_chart(_drawdown_chart(drawdown_window), use_container_width=True)
        with col2:
            st.altair_chart(_monthly_heatmap(monthly_returns_window), use_container_width=True)
            st.altair_chart(_rank_ic_chart(rank_ic_window), use_container_width=True)

    with bottom:
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.altair_chart(_turnover_chart(nav_window), use_container_width=True)
        with col2:
            if not trades_window.empty:
                side_frame = trades_window.groupby('side', as_index=False).agg(trade_count=('symbol', 'size'), notional=('notional', 'sum'))
                chart = (
                    alt.Chart(side_frame)
                    .mark_bar(cornerRadiusEnd=6)
                    .encode(
                        x=alt.X('notional:Q', title='成交额'),
                        y=alt.Y('side:N', title=None),
                        color=alt.Color('side:N', scale=alt.Scale(domain=['BUY', 'SELL'], range=['#0f766e', '#b45309']), legend=None),
                        tooltip=[alt.Tooltip('side:N', title='方向'), alt.Tooltip('trade_count:Q', title='笔数'), alt.Tooltip('notional:Q', title='成交额', format=',.0f')],
                    )
                    .properties(height=220)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info('当前时间范围内没有成交明细。')
        top_trades = trades_window.groupby('symbol', as_index=False).agg(
            trade_count=('symbol', 'size'),
            total_notional=('notional', 'sum'),
            total_fee=('fee', 'sum'),
        ).sort_values('total_notional', ascending=False).head(20) if not trades_window.empty else pd.DataFrame()
        top_trades = _to_display_frame(
            top_trades,
            {'symbol': '代码', 'trade_count': '成交笔数', 'total_notional': '累计成交额', 'total_fee': '累计费用'},
        )
        st.dataframe(
            _styled_table(top_trades, {'累计成交额': '{:,.0f}', '累计费用': '{:,.2f}'}),
            use_container_width=True,
            height=300,
        )


def _render_signal_desk(bundle: dict) -> None:
    _section_header('决策台', '最新信号与组合构建', '查看最新候选排序、入选组合、过滤原因，以及目标权重的历史变化。')
    latest_signal = bundle['latest_signal']
    filtered_candidates = bundle['filtered_candidates']
    target_weights = bundle['target_weights']
    rebalance_delta = _build_rebalance_delta(target_weights)

    if latest_signal.empty:
        st.warning('这个运行当前没有最新信号。')
        return

    signal_date = latest_signal['signal_date'].max()
    execution_date = latest_signal['execution_date'].max()
    cards = [
        {'label': '信号日', 'value': pd.Timestamp(signal_date).strftime('%Y-%m-%d'), 'sub': '研究决策生成日期'},
        {'label': '执行日', 'value': pd.Timestamp(execution_date).strftime('%Y-%m-%d'), 'sub': '已应用交易延迟'},
        {'label': '持仓数', 'value': str(len(latest_signal)), 'sub': '最新目标组合入选股票数'},
        {'label': '总暴露', 'value': _format_pct(float(latest_signal['target_weight'].sum())), 'sub': '目标组合总仓位'},
    ]
    _render_metric_grid(cards)

    top_left, top_right = st.columns([1.2, 1])
    with top_left:
        st.altair_chart(_signal_score_chart(latest_signal), use_container_width=True)
    with top_right:
        st.altair_chart(_industry_weight_chart(latest_signal), use_container_width=True)
        if not filtered_candidates.empty:
            st.altair_chart(_reason_chart(filtered_candidates), use_container_width=True)

    tabs = st.tabs(['目标组合', '调仓差额', '过滤候选', '权重历史'])
    with tabs[0]:
        display_signal = _to_display_frame(
            latest_signal,
            {
                'signal_date': '信号日',
                'execution_date': '执行日',
                'symbol': '代码',
                'security_name': '名称',
                'industry': '行业',
                'rank': '排名',
                'score': '分数',
                'target_weight': '目标权重',
            },
        )
        st.dataframe(
            _styled_table(display_signal, {'分数': '{:.4f}', '目标权重': '{:.2%}'}),
            use_container_width=True,
            height=360,
        )
    with tabs[1]:
        if rebalance_delta.empty:
            st.info('当前没有足够的历史信号来计算调仓差额。')
        else:
            display_delta = _to_display_frame(
                rebalance_delta[['symbol', 'action', 'prev_weight', 'target_weight', 'delta_weight', 'prev_rank', 'rank', 'score']],
                {
                    'symbol': '代码',
                    'action': '动作',
                    'prev_weight': '上期权重',
                    'target_weight': '本期权重',
                    'delta_weight': '权重变化',
                    'prev_rank': '上期排名',
                    'rank': '本期排名',
                    'score': '分数',
                },
            )
            st.dataframe(
                _styled_table(display_delta, {'上期权重': '{:.2%}', '本期权重': '{:.2%}', '权重变化': '{:.2%}', '分数': '{:.4f}'}),
                use_container_width=True,
                height=360,
            )
    with tabs[2]:
        filtered_view = filtered_candidates.sort_values(['signal_date', 'score'], ascending=[False, False]).copy()
        filtered_view = _to_display_frame(
            filtered_view,
            {
                'trade_date': '交易日',
                'symbol': '代码',
                'score': '分数',
                'model_type': '模型类型',
                'fallback_used': '是否回退',
                'in_universe': '在股票池内',
                'is_st': 'ST',
                'is_new_listing': '次新',
                'is_tradable': '可交易',
                'reason': '原因',
                'signal_date': '信号日',
                'execution_date': '执行日',
            },
        )
        st.dataframe(
            _styled_table(filtered_view, {'分数': '{:.4f}'}),
            use_container_width=True,
            height=360,
        )
    with tabs[3]:
        symbol_options = latest_signal['symbol'].tolist()
        selected_symbol = st.selectbox('查看权重历史', symbol_options)
        history = target_weights.loc[target_weights['symbol'] == selected_symbol, ['signal_date', 'target_weight', 'score']].sort_values('signal_date')
        if not history.empty:
            chart = (
                alt.Chart(history)
                .mark_line(point=True, color='#0f766e', strokeWidth=3)
                .encode(
                    x=alt.X('signal_date:T', title='信号日'),
                    y=alt.Y('target_weight:Q', title='目标权重', axis=alt.Axis(format='%')),
                    tooltip=[
                        alt.Tooltip('signal_date:T', title='信号日'),
                        alt.Tooltip('target_weight:Q', title='目标权重', format='.2%'),
                        alt.Tooltip('score:Q', title='分数', format='.4f'),
                    ],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
            display_history = _to_display_frame(history, {'signal_date': '信号日', 'target_weight': '目标权重', 'score': '分数'})
            st.dataframe(_styled_table(display_history, {'目标权重': '{:.2%}', '分数': '{:.4f}'}), use_container_width=True)


def _render_factor_lab(bundle: dict) -> None:
    _section_header('研究工厂', '因子池与特征池', '先研究单因子有效性、稳定性、相关性和经济逻辑，再决定哪些特征进入下一轮模型实验。')
    summary, daily_ic, corr_frame = _factor_research_bundle(bundle)
    if summary.empty:
        st.info('当前运行缺少可分析的因子数据。')
        return

    cards = [
        {'label': '因子数量', 'value': str(len(summary)), 'sub': '当前实验已注册因子数'},
        {'label': '平均 IC', 'value': _format_number(float(summary['ic_mean'].mean()), 3), 'sub': '单因子 Rank IC 均值'},
        {'label': '转弱/失效', 'value': str(int(summary['state'].isin(['转弱', '失效']).sum())), 'sub': '近20期状态较弱的因子数'},
        {'label': '平均覆盖率', 'value': _format_pct(float(summary['coverage'].mean())), 'sub': '因子可用样本覆盖'},
    ]
    _render_metric_grid(cards)

    left, right = st.columns([1.05, 1])
    with left:
        st.altair_chart(_factor_ic_bar_chart(summary), use_container_width=True)
    with right:
        st.altair_chart(_factor_corr_heatmap(corr_frame), use_container_width=True)

    tabs = st.tabs(['因子画像', 'IC 诊断', '特征编排'])
    with tabs[0]:
        quantile_summary = bundle.get('evaluation_quantile_summary', pd.DataFrame())
        quantile_returns = bundle.get('evaluation_quantile_returns', pd.DataFrame())
        profile_col, spread_col = st.columns(2)
        with profile_col:
            st.altair_chart(_quantile_profile_chart(quantile_summary), use_container_width=True)
        with spread_col:
            st.altair_chart(_quantile_spread_chart(quantile_returns), use_container_width=True)

        display_summary = _to_display_frame(
            summary[
                [
                    'factor_name',
                    'state',
                    'ic_mean',
                    'recent_20d_ic',
                    'ic_ir',
                    'positive_rate',
                    'economic_meaning',
                    'logic',
                    'failure_modes',
                ]
            ],
            {
                'factor_name': '因子',
                'state': '状态',
                'ic_mean': '平均IC',
                'recent_20d_ic': '近20期IC',
                'ic_ir': 'IC_IR',
                'positive_rate': '正IC占比',
                'economic_meaning': '经济含义',
                'logic': '逻辑',
                'failure_modes': '失效场景',
            },
        )
        st.dataframe(
            _styled_table(display_summary, {'平均IC': '{:.3f}', '近20期IC': '{:.3f}', 'IC_IR': '{:.2f}', '正IC占比': '{:.2%}'}),
            use_container_width=True,
            height=420,
        )
        if not corr_frame.empty:
            corr_pairs = corr_frame.melt(id_vars=['factor_name'], var_name='peer_factor', value_name='corr')
            corr_pairs = corr_pairs.loc[corr_pairs['factor_name'] < corr_pairs['peer_factor']].copy()
            corr_pairs['abs_corr'] = corr_pairs['corr'].abs()
            corr_pairs = corr_pairs.sort_values('abs_corr', ascending=False).head(12)
            corr_pairs = _to_display_frame(
                corr_pairs[['factor_name', 'peer_factor', 'corr', 'abs_corr']],
                {'factor_name': '因子A', 'peer_factor': '因子B', 'corr': '相关系数', 'abs_corr': '绝对相关'},
            )
            st.markdown('#### 高相关因子对')
            st.dataframe(_styled_table(corr_pairs, {'相关系数': '{:.3f}', '绝对相关': '{:.3f}'}), use_container_width=True, height=220)
        if not quantile_summary.empty:
            st.markdown('#### 当前整套信号的分组收益摘要')
            quantile_display = _to_display_frame(
                quantile_summary,
                {
                    'bucket': '分组',
                    'mean_forward_return': '平均远期收益',
                    'std_forward_return': '收益波动',
                    'observations': '样本期数',
                },
            )
            st.dataframe(
                _styled_table(quantile_display, {'平均远期收益': '{:.4f}', '收益波动': '{:.4f}'}),
                use_container_width=True,
                height=220,
            )

    with tabs[1]:
        factor_options = summary['factor_name'].tolist()
        selected_factors = st.multiselect('查看因子 IC 曲线', factor_options, default=factor_options[: min(4, len(factor_options))])
        st.altair_chart(_factor_ic_timeseries_chart(daily_ic, selected_factors), use_container_width=True)
        turnover = bundle.get('evaluation_selection_turnover', pd.DataFrame())
        coverage = bundle.get('evaluation_signal_coverage', pd.DataFrame())
        col1, col2 = st.columns(2)
        with col1:
            if not turnover.empty:
                turnover_display = _to_display_frame(
                    turnover,
                    {
                        'signal_date': '信号日',
                        'prev_signal_date': '上期信号日',
                        'selected_count': '入选数',
                        'overlap_count': '重合数',
                        'selection_turnover': '选择换手',
                    },
                )
                st.dataframe(_styled_table(turnover_display, {'选择换手': '{:.2%}'}), use_container_width=True, height=260)
        with col2:
            if not coverage.empty:
                coverage_display = _to_display_frame(
                    coverage,
                    {
                        'trade_date': '日期',
                        'candidate_count': '候选数',
                        'labeled_count': '有标签数',
                        'scored_count': '打分数',
                        'coverage_ratio': '覆盖率',
                        'labeled_ratio': '标签覆盖率',
                    },
                )
                st.dataframe(_styled_table(coverage_display, {'覆盖率': '{:.2%}', '标签覆盖率': '{:.2%}'}), use_container_width=True, height=260)

    with tabs[2]:
        st.caption('拖拽排序这版先用“优先级”字段替代；你可以先勾选、再按优先级编排，然后直接发起一轮新实验。')
        current_features = bundle.get('config', {}).get('features', {}).get('names', [])
        editor_frame = summary[
            ['factor_name', 'state', 'recent_20d_ic', 'ic_mean', 'economic_meaning', 'logic']
        ].copy()
        editor_frame['纳入下一轮'] = editor_frame['factor_name'].isin(current_features)
        editor_frame['优先级'] = editor_frame['factor_name'].map({name: idx + 1 for idx, name in enumerate(current_features)}).fillna(len(current_features) + 1).astype(int)
        edited = st.data_editor(
            editor_frame,
            use_container_width=True,
            hide_index=True,
            column_config={
                'factor_name': st.column_config.TextColumn('因子', disabled=True),
                'state': st.column_config.TextColumn('状态', disabled=True),
                'recent_20d_ic': st.column_config.NumberColumn('近20期IC', format='%.3f', disabled=True),
                'ic_mean': st.column_config.NumberColumn('平均IC', format='%.3f', disabled=True),
                'economic_meaning': st.column_config.TextColumn('经济含义', disabled=True),
                'logic': st.column_config.TextColumn('逻辑', disabled=True),
                '纳入下一轮': st.column_config.CheckboxColumn('纳入下一轮'),
                '优先级': st.column_config.NumberColumn('优先级', min_value=1, max_value=99, step=1),
            },
            height=380,
            key='factor_pool_editor',
        )
        selected_features = (
            edited.loc[edited['纳入下一轮'], ['factor_name', '优先级']]
            .sort_values(['优先级', 'factor_name'])
            ['factor_name']
            .tolist()
        )
        st.code(' -> '.join(selected_features) if selected_features else '当前未选择任何因子')
        col1, col2 = st.columns([1.1, 1.2])
        with col1:
            factor_experiment_name = st.text_input('新实验名称', value=f"{bundle.get('config', {}).get('name', 'factorlab')}_factorlab")
        with col2:
            factor_experiment_note = st.text_input('实验说明补充', value='来自因子池页面的特征编排实验')
        if st.button('用所选特征生成并运行新实验', use_container_width=True, disabled=not selected_features):
            ui_config = copy.deepcopy(bundle['config'])
            ui_config['name'] = factor_experiment_name
            ui_config['description'] = factor_experiment_note
            ui_config.setdefault('features', {})
            ui_config['features']['names'] = selected_features
            ui_config['features']['set_name'] = f"{ui_config['features'].get('set_name', 'feature_set')}_pool"
            config_path = _save_ui_config(ui_config, factor_experiment_name)
            try:
                with st.spinner('正在根据新的因子编排运行实验...'):
                    summary_result = run_experiment(config_path)
                st.session_state['preferred_run_name'] = summary_result['run_id']
                st.cache_data.clear()
                st.success(f"新实验已完成：{summary_result['run_id']}")
                st.code(f'python -m src.cli experiment --config {config_path}')
                st.rerun()
            except Exception as exc:
                st.error(f'运行失败：{exc}')


def _render_studio(bundle: dict) -> None:
    _section_header('实验工作台', '调参与调试', '在界面里修改回测假设、组合规则和模型参数，并基于当前配置直接生成一轮新的研究运行。')
    config = bundle['config']
    registry_catalog = bundle.get('registry_catalog', {})
    data_cfg = config.get('data', {})
    feature_cfg = config.get('features', {})
    label_cfg = config.get('label', {})
    model_cfg = config.get('model', {})
    signal_cfg = config.get('signal', {})
    portfolio_cfg = config.get('portfolio', {})
    backtest_cfg = config.get('backtest', {})
    evaluation_cfg = config.get('evaluation', {})
    feature_options = feature_cfg.get('names', AVAILABLE_FEATURES)
    if registry_catalog.get('features'):
        feature_options = [row.get('feature_name') for row in registry_catalog['features'] if row.get('feature_name')]
    model_options = [row.get('name') for row in registry_catalog.get('models', []) if row.get('name')] or [model_cfg.get('family', 'lightgbm_regression')]
    signal_options = [row.get('name') for row in registry_catalog.get('signals', []) if row.get('name')] or [signal_cfg.get('name', 'cross_sectional_score')]
    portfolio_options = [row.get('name') for row in registry_catalog.get('portfolio_constructors', []) if row.get('name')] or [portfolio_cfg.get('constructor', 'qmt_topn_equal_weight')]
    evaluation_options = [row.get('name') for row in registry_catalog.get('evaluation_suites', []) if row.get('name')] or [evaluation_cfg.get('suite', 'basic_factor_diagnostics')]

    tabs = st.tabs(['参数实验', '批量试验', '模型调试', '使用原则'])
    with tabs[0]:
        st.caption('每次点击运行都会保存一份独立 YAML 到 `artifacts/ui_configs/`，不会覆盖原实验。')
        with st.form('studio_experiment_form'):
            st.markdown('#### 实验与数据')
            col1, col2, col3 = st.columns(3)
            experiment_name = col1.text_input('实验名称', value=f"{config.get('name', 'ui_experiment')}_ui")
            start_date_value = col2.date_input('开始日期', value=_safe_date_input_value(data_cfg.get('start_date', '2024-01-02'), '2024-01-02'), format='YYYY-MM-DD')
            end_date_raw = str(data_cfg.get('end_date', 'latest_completed'))
            end_date_mode = '最新完成日' if end_date_raw == 'latest_completed' else '指定日期'
            end_date_mode = col3.selectbox('结束日期模式', ['最新完成日', '指定日期'], index=['最新完成日', '指定日期'].index(end_date_mode))
            end_date_value = None
            if end_date_mode == '指定日期':
                end_date_value = st.date_input('结束日期', value=_safe_date_input_value(end_date_raw if end_date_raw != 'latest_completed' else datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d')), format='YYYY-MM-DD')
            description = st.text_area('实验说明', value=config.get('description', ''), height=80)
            current_source = data_cfg.get('source', 'baostock_ashare')
            source_labels = list(DATA_SOURCE_OPTIONS.keys())
            selected_source_label = next((label for label, value in DATA_SOURCE_OPTIONS.items() if value == current_source), source_labels[0])
            col1, col2, col3 = st.columns(3)
            data_mode_label = col1.selectbox('数据模式', source_labels, index=source_labels.index(selected_source_label))
            selected_source = DATA_SOURCE_OPTIONS[data_mode_label]
            universe_name = col2.text_input('股票池名称', value=str(data_cfg.get('universe_name', 'HS300')))
            snapshot_id = col3.text_input('数据快照编号', value=str(data_cfg.get('snapshot_id', f'{experiment_name}_snapshot')))

            model_random_state_default = int(model_cfg.get('params', {}).get('random_state') or data_cfg.get('seed') or 7)
            n_symbols_master = None
            n_universe = None
            data_seed = None
            incremental = bool(data_cfg.get('incremental', True))
            universe_refresh_frequency_days = int(data_cfg.get('universe_refresh_frequency_days', 1))
            price_adjust = str(data_cfg.get('price_adjust', 'qfq'))
            universe_mode = str(data_cfg.get('universe_mode', 'point_in_time'))

            if _is_real_source(selected_source):
                col1, col2, col3, col4 = st.columns(4)
                incremental = col1.checkbox('启用增量抓取', value=incremental)
                universe_refresh_frequency_days = col2.number_input('股票池刷新频率(天)', min_value=1, value=universe_refresh_frequency_days, step=1)
                price_adjust = col3.selectbox('价格口径', ['qfq', 'hfq', 'none'], index=['qfq', 'hfq', 'none'].index(price_adjust if price_adjust in {'qfq', 'hfq', 'none'} else 'qfq'))
                universe_mode = col4.selectbox('股票池口径', ['point_in_time', 'current_snapshot'], index=['point_in_time', 'current_snapshot'].index(universe_mode if universe_mode in {'point_in_time', 'current_snapshot'} else 'point_in_time'), format_func=lambda value: '点时成分' if value == 'point_in_time' else '当前快照')
                if universe_mode == 'current_snapshot':
                    st.caption('当前快照口径会把最新一期 HS300 成分扩展到整段历史，贴近原 QMT 脚本。')
                else:
                    st.caption('点时成分口径更严格，适合研究回测。')
            else:
                col1, col2, col3 = st.columns(3)
                n_symbols_master = col1.number_input('主股票池规模', min_value=20, value=int(data_cfg.get('n_symbols_master', 320) or 320), step=10)
                n_universe = col2.number_input('研究股票池规模', min_value=10, value=int(data_cfg.get('n_universe', 300) or 300), step=10)
                data_seed = col3.number_input('模拟数据种子', min_value=1, value=int(data_cfg.get('seed', 7) or 7), step=1)
                st.caption('模拟模式只用于测试平台链路，不代表真实市场，也不会替代真实数据研究。')

            st.markdown('#### 特征与标签')
            feature_names = st.multiselect('特征集合', feature_options, default=feature_cfg.get('names', feature_options))
            col1, col2, col3, col4 = st.columns(4)
            winsor_low = col1.number_input('截尾下限', min_value=0.0, max_value=0.2, value=float(feature_cfg.get('winsorize_limits', [0.01, 0.99])[0]), step=0.01, format='%.2f')
            winsor_high = col2.number_input('截尾上限', min_value=0.8, max_value=1.0, value=float(feature_cfg.get('winsorize_limits', [0.01, 0.99])[1]), step=0.01, format='%.2f')
            zscore = col3.checkbox('横截面标准化', value=bool(feature_cfg.get('zscore', True)))
            fill_missing = col4.checkbox('缺失值中位数填充', value=bool(feature_cfg.get('fill_missing', False)))
            label_name = st.text_input('标签名称', value=str(label_cfg.get('name', 'fwd_return_20d')))

            st.markdown('#### 模型参数')
            col1, col2, col3, col4 = st.columns(4)
            model_family = col1.selectbox('模型模块', model_options, index=model_options.index(model_cfg.get('family', model_options[0])) if model_cfg.get('family', model_options[0]) in model_options else 0)
            signal_name = col2.selectbox('信号模块', signal_options, index=signal_options.index(signal_cfg.get('name', signal_options[0])) if signal_cfg.get('name', signal_options[0]) in signal_options else 0)
            portfolio_constructor = col3.selectbox('组合器', portfolio_options, index=portfolio_options.index(portfolio_cfg.get('constructor', portfolio_options[0])) if portfolio_cfg.get('constructor', portfolio_options[0]) in portfolio_options else 0)
            evaluation_suite = col4.selectbox('评估套件', evaluation_options, index=evaluation_options.index(evaluation_cfg.get('suite', evaluation_options[0])) if evaluation_cfg.get('suite', evaluation_options[0]) in evaluation_options else 0)
            col1, col2, col3 = st.columns(3)
            train_window_days = col1.number_input('训练窗口天数', min_value=20, value=int(model_cfg.get('train_window_days', 380)), step=10)
            valid_window_days = col2.number_input('验证窗口天数', min_value=0, value=int(model_cfg.get('valid_window_days', 0)), step=5)
            min_train_samples = col3.number_input('最少训练样本数', min_value=100, value=int(model_cfg.get('min_train_samples', 600)), step=100)
            col1, col2 = st.columns(2)
            training_embargo_days = col1.number_input('训练保护间隔(天)', min_value=0, value=int(model_cfg.get('training_embargo_days', 0)), step=1)
            col2.caption('默认按原脚本主逻辑为 0；只有你想额外拉开训练样本与信号日距离时才需要增加。')
            col1, col2, col3, col4 = st.columns(4)
            n_estimators = col1.number_input('树数量', min_value=20, value=int(model_cfg.get('params', {}).get('n_estimators', 220)), step=10)
            learning_rate = col2.number_input('学习率', min_value=0.001, max_value=0.5, value=float(model_cfg.get('params', {}).get('learning_rate', 0.04)), step=0.01, format='%.3f')
            num_leaves = col3.number_input('叶子数', min_value=4, value=int(model_cfg.get('params', {}).get('num_leaves', 95)), step=1)
            min_child_samples = col4.number_input('最小叶子样本数', min_value=5, value=int(model_cfg.get('params', {}).get('min_child_samples', 35)), step=5)
            col1, col2, col3, col4 = st.columns(4)
            subsample = col1.number_input('样本采样比', min_value=0.3, max_value=1.0, value=float(model_cfg.get('params', {}).get('subsample', 0.85)), step=0.05, format='%.2f')
            colsample_bytree = col2.number_input('特征采样比', min_value=0.3, max_value=1.0, value=float(model_cfg.get('params', {}).get('colsample_bytree', 0.9)), step=0.05, format='%.2f')
            model_random_state = col3.number_input('模型随机状态', min_value=1, value=model_random_state_default, step=1)
            reg_lambda = col4.number_input('L2 正则', min_value=0.0, value=float(model_cfg.get('params', {}).get('reg_lambda', model_cfg.get('params', {}).get('lambda_l2', 2.0))), step=0.5, format='%.2f')
            col1, col2, col3, col4 = st.columns(4)
            max_depth = col1.number_input('最大深度', min_value=-1, value=int(model_cfg.get('params', {}).get('max_depth', 6)), step=1)
            subsample_freq = col2.number_input('采样频率', min_value=0, value=int(model_cfg.get('params', {}).get('subsample_freq', 1)), step=1)
            fallback_feature = col3.text_input('回退特征', value=str(model_cfg.get('fallback_feature', 'mom60') or ''))
            score_blend_feature = col4.text_input('融合特征', value=str(model_cfg.get('score_blend_feature', 'mom60') or ''))
            col1, col2, col3, col4 = st.columns(4)
            score_blend_weight_model = col1.number_input('模型分数权重', min_value=0.0, max_value=1.0, value=float(model_cfg.get('score_blend_weight_model', 1.0)), step=0.05, format='%.2f')
            score_blend_weight_feature = col2.number_input('特征分数权重', min_value=0.0, max_value=1.0, value=float(model_cfg.get('score_blend_weight_feature', 0.0)), step=0.05, format='%.2f')
            label_clip_default = model_cfg.get('label_clip', [-0.3, 0.3])
            label_clip_low = col3.number_input('标签截尾下限', min_value=-1.0, max_value=0.0, value=float(label_clip_default[0] if isinstance(label_clip_default, (list, tuple)) else -0.3), step=0.05, format='%.2f')
            label_clip_high = col4.number_input('标签截尾上限', min_value=0.0, max_value=1.0, value=float(label_clip_default[1] if isinstance(label_clip_default, (list, tuple)) else 0.3), step=0.05, format='%.2f')

            st.markdown('#### 组合与回测')
            col1, col2, col3, col4 = st.columns(4)
            top_n = col1.number_input('持仓数 Top N', min_value=1, value=int(portfolio_cfg.get('top_n', 25)), step=1)
            gross_exposure = col2.number_input('正常总暴露', min_value=0.1, max_value=1.0, value=float(portfolio_cfg.get('gross_exposure', 0.95)), step=0.05, format='%.2f')
            defensive_gross = col3.number_input('防守总暴露', min_value=0.0, max_value=1.0, value=float(portfolio_cfg.get('defensive_gross', 0.35)), step=0.05, format='%.2f')
            max_single_weight = col4.number_input('单票权重上限', min_value=0.01, max_value=0.5, value=float(portfolio_cfg.get('max_single_weight', 0.05)), step=0.01, format='%.2f')
            col1, col2, col3, col4 = st.columns(4)
            min_trade_value = col1.number_input('最小成交额', min_value=0.0, value=float(portfolio_cfg.get('min_trade_value', 0)), step=500.0)
            market_filter_lookback = col2.number_input('市场过滤窗口', min_value=5, value=int(portfolio_cfg.get('market_filter_lookback', 60)), step=5)
            market_filter_threshold = col3.number_input('市场过滤阈值', min_value=-1.0, max_value=1.0, value=float(portfolio_cfg.get('market_filter_threshold', 0.0)), step=0.01, format='%.2f')
            initial_cash = col4.number_input('初始资金', min_value=10000.0, value=float(backtest_cfg.get('initial_cash', 250000)), step=10000.0)
            col1, col2 = st.columns(2)
            risk_model = col1.selectbox('风险模型', ['two_tier_momentum', 'qmt_style_ladder'], index=['two_tier_momentum', 'qmt_style_ladder'].index(str(portfolio_cfg.get('risk_model', 'two_tier_momentum')) if str(portfolio_cfg.get('risk_model', 'two_tier_momentum')) in {'two_tier_momentum', 'qmt_style_ladder'} else 'two_tier_momentum'), format_func=lambda value: '二档动量过滤' if value == 'two_tier_momentum' else '原脚本四档仓位')
            candidate_filter_mode = 'strict_ashare'
            col2.text_input('选股过滤', value='严格A股约束', disabled=True)
            col1, col2, col3, col4 = st.columns(4)
            risk_ma_short_window = col1.number_input('短均线窗口', min_value=5, value=int(portfolio_cfg.get('risk_ma_short_window', 60)), step=5)
            risk_ma_long_window = col2.number_input('长均线窗口', min_value=20, value=int(portfolio_cfg.get('risk_ma_long_window', 120)), step=5)
            risk_momentum_window = col3.number_input('风险动量窗口', min_value=5, value=int(portfolio_cfg.get('risk_momentum_window', 20)), step=5)
            risk_mid_exposure = col4.number_input('中档暴露', min_value=0.0, max_value=1.0, value=float(portfolio_cfg.get('risk_mid_exposure', 0.85)), step=0.05, format='%.2f')
            col1, col2 = st.columns(2)
            risk_low_exposure = col1.number_input('低档暴露', min_value=0.0, max_value=1.0, value=float(portfolio_cfg.get('risk_low_exposure', 0.65)), step=0.05, format='%.2f')
            risk_crash_exposure = col2.number_input('极弱市暴露', min_value=0.0, max_value=1.0, value=float(portfolio_cfg.get('risk_crash_exposure', 0.45)), step=0.05, format='%.2f')
            col1, col2, col3, col4, col5 = st.columns(5)
            rebalance_frequency_days = col1.number_input('调仓频率(天)', min_value=1, value=int(backtest_cfg.get('rebalance_frequency_days', 5)), step=1)
            trade_delay_days = col2.number_input('交易延迟(天)', min_value=0, value=int(backtest_cfg.get('trade_delay_days', 1)), step=1)
            lot_size = col3.number_input('最小交易股数', min_value=1, value=int(backtest_cfg.get('lot_size', 100)), step=100)
            commission_bps = col4.number_input('佣金(bps)', min_value=0.0, value=float(backtest_cfg.get('commission_bps', 0.75)), step=0.05)
            slippage_bps = col5.number_input('滑点(bps)', min_value=0.0, value=float(backtest_cfg.get('slippage_bps', 5)), step=0.5)
            col1, col2, col3, col4 = st.columns(4)
            stamp_duty_bps = col1.number_input('印花税(bps)', min_value=0.0, value=float(backtest_cfg.get('stamp_duty_bps', 10)), step=0.5)
            anchor_mode = col2.selectbox(
                '调仓起算方式',
                ['follow_backtest_start', 'fixed'],
                index=['follow_backtest_start', 'fixed'].index(str(backtest_cfg.get('anchor_mode', 'follow_backtest_start')) if str(backtest_cfg.get('anchor_mode', 'follow_backtest_start')) in {'follow_backtest_start', 'fixed'} else 'follow_backtest_start'),
                format_func=lambda value: '跟随回测起点' if value == 'follow_backtest_start' else '固定锚点',
            )
            anchor_date_default = _safe_date_input_value(backtest_cfg.get('anchor_date', '2022-09-01') or '2022-09-01', '2022-09-01')
            anchor_date_value = col3.date_input('固定锚点日期', value=anchor_date_default, format='YYYY-MM-DD', disabled=anchor_mode == 'follow_backtest_start')
            execution_constraint_mode = 'strict_ashare'
            col4.text_input('执行限制', value='严格A股约束', disabled=True)

            submitted = st.form_submit_button('保存配置并运行新实验', use_container_width=True)

        if submitted:
            start_date = _normalize_ui_date_string(start_date_value.isoformat(), '开始日期')
            end_date = 'latest_completed' if end_date_mode == '最新完成日' else _normalize_ui_date_string(end_date_value.isoformat(), '结束日期')
            anchor_date = None if anchor_mode == 'follow_backtest_start' else _normalize_ui_date_string(anchor_date_value.isoformat(), '固定锚点日期')
            ui_config = copy.deepcopy(config)
            ui_config['name'] = experiment_name
            ui_config['description'] = description

            ui_config['data'] = copy.deepcopy(data_cfg)
            ui_config['data'].update(
                {
                    'source': selected_source,
                    'snapshot_id': snapshot_id,
                    'start_date': start_date,
                    'end_date': end_date,
                    'universe_name': universe_name,
                    'universe_mode': universe_mode,
                    'bootstrap_if_missing': True,
                    'incremental': bool(incremental),
                    'universe_refresh_frequency_days': int(universe_refresh_frequency_days),
                    'price_adjust': price_adjust,
                    'n_symbols_master': int(n_symbols_master) if n_symbols_master is not None else None,
                    'n_universe': int(n_universe) if n_universe is not None else None,
                    'seed': int(data_seed) if data_seed is not None else None,
                }
            )

            ui_config['features'] = copy.deepcopy(feature_cfg)
            ui_config['features'].update(
                {
                    'set_name': feature_cfg.get('set_name', 'ui_feature_set'),
                    'version': feature_cfg.get('version', 'ui'),
                    'names': list(feature_names),
                    'winsorize_limits': [float(winsor_low), float(winsor_high)],
                    'zscore': bool(zscore),
                    'fill_missing': bool(fill_missing),
                }
            )

            ui_config['label'] = copy.deepcopy(label_cfg)
            ui_config['label'].update(
                {
                    'name': label_name,
                    'horizon': int(label_cfg.get('horizon', 20)),
                }
            )

            model_params = copy.deepcopy(model_cfg.get('params', {}))
            model_params.update(
                {
                    'n_estimators': int(n_estimators),
                    'learning_rate': float(learning_rate),
                    'num_leaves': int(num_leaves),
                    'max_depth': int(max_depth),
                    'subsample': float(subsample),
                    'subsample_freq': int(subsample_freq),
                    'colsample_bytree': float(colsample_bytree),
                    'min_child_samples': int(min_child_samples),
                    'reg_lambda': float(reg_lambda),
                    'random_state': int(model_random_state),
                    'verbosity': -1,
                }
            )
            ui_config['model'] = copy.deepcopy(model_cfg)
            ui_config['model'].update(
                {
                    'family': model_family,
                    'version': model_cfg.get('version', 'ui'),
                    'registry_stage': model_cfg.get('registry_stage', 'research'),
                    'fallback_model': model_cfg.get('fallback_model', 'mom60_zscore'),
                    'fallback_feature': fallback_feature.strip() or None,
                    'score_blend_feature': score_blend_feature.strip() or None,
                    'score_blend_weight_model': float(score_blend_weight_model),
                    'score_blend_weight_feature': float(score_blend_weight_feature),
                    'label_clip': [float(label_clip_low), float(label_clip_high)],
                    'train_window_days': int(train_window_days),
                    'valid_window_days': int(valid_window_days),
                    'min_train_samples': int(min_train_samples),
                    'training_embargo_days': int(training_embargo_days),
                    'params': model_params,
                }
            )

            ui_config['signal'] = copy.deepcopy(signal_cfg)
            ui_config['signal'].update(
                {
                    'name': signal_name,
                    'version': signal_cfg.get('version', 'ui'),
                    'params': copy.deepcopy(signal_cfg.get('params', {})),
                }
            )

            ui_config['portfolio'] = copy.deepcopy(portfolio_cfg)
            ui_config['portfolio'].update(
                {
                    'top_n': int(top_n),
                    'weighting': portfolio_cfg.get('weighting', 'equal'),
                    'gross_exposure': float(gross_exposure),
                    'defensive_gross': float(defensive_gross),
                    'max_single_weight': float(max_single_weight),
                    'cash_buffer': float(portfolio_cfg.get('cash_buffer', 0.05)),
                    'min_trade_value': float(min_trade_value),
                    'market_filter_lookback': int(market_filter_lookback),
                    'market_filter_threshold': float(market_filter_threshold),
                    'risk_model': risk_model,
                    'risk_ma_short_window': int(risk_ma_short_window),
                    'risk_ma_long_window': int(risk_ma_long_window),
                    'risk_momentum_window': int(risk_momentum_window),
                    'risk_mid_exposure': float(risk_mid_exposure),
                    'risk_low_exposure': float(risk_low_exposure),
                    'risk_crash_exposure': float(risk_crash_exposure),
                    'candidate_filter_mode': candidate_filter_mode,
                    'constructor': portfolio_constructor,
                }
            )

            ui_config['backtest'] = copy.deepcopy(backtest_cfg)
            ui_config['backtest'].update(
                {
                    'initial_cash': float(initial_cash),
                    'lot_size': int(lot_size),
                    'commission_bps': float(commission_bps),
                    'stamp_duty_bps': float(stamp_duty_bps),
                    'slippage_bps': float(slippage_bps),
                    'rebalance_frequency_days': int(rebalance_frequency_days),
                    'trade_delay_days': int(trade_delay_days),
                    'anchor_mode': anchor_mode,
                    'anchor_date': anchor_date.strip() or None,
                    'execution_constraint_mode': execution_constraint_mode,
                }
            )

            ui_config['evaluation'] = copy.deepcopy(evaluation_cfg)
            ui_config['evaluation'].update(
                {
                    'suite': evaluation_suite,
                    'version': evaluation_cfg.get('version', 'ui'),
                    'params': copy.deepcopy(evaluation_cfg.get('params', {})),
                }
            )
            config_path = _save_ui_config(ui_config, experiment_name)
            try:
                with st.spinner('正在运行新实验，请稍候...'):
                    summary = run_experiment(config_path)
                st.session_state['preferred_run_name'] = summary['run_id']
                st.session_state['last_ui_config_path'] = str(config_path)
                st.cache_data.clear()
                st.success(f"新实验已完成：{summary['run_id']}")
                st.code(f'python -m src.cli experiment --config {config_path}')
                st.rerun()
            except Exception as exc:
                st.error(f'运行失败：{exc}')
                st.code(yaml.safe_dump(ui_config, allow_unicode=True, sort_keys=False), language='yaml')

    with tabs[1]:
        st.caption('适合做一维参数扫描。每组参数都会保存成独立 YAML，并顺序跑出独立 run，便于回到实验台账里比较。')
        sweep_field_label = st.selectbox('扫描参数', list(SWEEP_FIELD_OPTIONS.keys()))
        sweep_path, sweep_caster = SWEEP_FIELD_OPTIONS[sweep_field_label]
        default_sweep_values = {
            '调仓频率(天)': '3,5,10',
            '持仓数 Top N': '15,25,35',
            '训练窗口天数': '180,240,320',
            '验证窗口天数': '20,60,120',
            '树数量': '80,120,180',
            '学习率': '0.03,0.05,0.08',
            '叶子数': '31,63,95',
            '正常总暴露': '0.85,0.95,1.00',
            '防守总暴露': '0.25,0.35,0.45',
        }
        sweep_values_raw = st.text_area('参数取值', value=default_sweep_values.get(sweep_field_label, ''), height=96, help='用逗号分隔，例如 3,5,10')
        sweep_prefix = st.text_input('批量实验名前缀', value=f"{config.get('name', 'ui_experiment')}_grid")
        if st.button('生成并顺序运行批量实验', use_container_width=True):
            try:
                sweep_values = _parse_sweep_values(sweep_values_raw, sweep_caster)
            except Exception as exc:
                st.error(f'参数解析失败：{exc}')
                sweep_values = []
            if not sweep_values:
                st.warning('至少提供一个有效参数值。')
            elif len(sweep_values) > 8:
                st.warning('单次最多运行 8 组，避免界面阻塞过久。')
            else:
                results = []
                progress = st.progress(0.0)
                status_box = st.empty()
                for idx, sweep_value in enumerate(sweep_values, start=1):
                    ui_config = copy.deepcopy(config)
                    ui_config['name'] = f"{sweep_prefix}_{idx:02d}"
                    ui_config['description'] = f"批量试验: {sweep_field_label}={sweep_value}"
                    _set_nested_config_value(ui_config, sweep_path, sweep_value)
                    config_path = _save_ui_config(ui_config, ui_config['name'])
                    status_box.info(f'正在运行 {idx}/{len(sweep_values)}: {sweep_field_label}={sweep_value}')
                    try:
                        summary = run_experiment(config_path)
                        results.append(
                            {
                                'run_id': summary['run_id'],
                                '参数': sweep_value,
                                '总收益': summary.get('total_return', 0.0),
                                '年化收益': summary.get('annual_return', 0.0),
                                '最大回撤': summary.get('max_drawdown', 0.0),
                                '平均RankIC': summary.get('avg_rank_ic', 0.0),
                            }
                        )
                    except Exception as exc:
                        results.append({'run_id': f'FAILED_{idx:02d}', '参数': sweep_value, '错误': str(exc)})
                    progress.progress(idx / len(sweep_values))
                st.cache_data.clear()
                status_box.success('批量实验执行完成。')
                results_frame = pd.DataFrame(results)
                if not results_frame.empty:
                    st.dataframe(
                        _styled_table(
                            results_frame,
                            {'总收益': '{:.2%}', '年化收益': '{:.2%}', '最大回撤': '{:.2%}', '平均RankIC': '{:.3f}'},
                        ),
                        use_container_width=True,
                        height=280,
                    )

    with tabs[2]:
        split_metrics = bundle['split_metrics']
        model_registry = bundle['model_registry']
        feature_importance = bundle['feature_importance']
        predictions = bundle['predictions']
        label_name = config.get('label', {}).get('name', 'fwd_return_20d')

        cards = [
            {'label': '切分次数', 'value': str(len(split_metrics)), 'sub': '已完成的 walk-forward 轮数'},
            {'label': '回退比例', 'value': _format_pct(_fallback_rate(bundle)), 'sub': '使用 fallback 的比例'},
            {'label': '平均验证 IC', 'value': _format_number(float(split_metrics['valid_rank_ic'].mean()) if not split_metrics.empty else 0.0, 3), 'sub': 'split_metrics 的均值'},
            {'label': '平均验证 RMSE', 'value': _format_number(float(split_metrics['valid_rmse'].mean()) if not split_metrics.empty else 0.0, 4), 'sub': '验证期预测误差'},
        ]
        _render_metric_grid(cards)

        left, right = st.columns([1, 1.2])
        with left:
            st.altair_chart(_feature_importance_chart(feature_importance), use_container_width=True)
        with right:
            display_registry = _to_display_frame(
                model_registry,
                {'signal_date': '信号日', 'model_type': '模型类型', 'registry_stage': '阶段', 'model_path': '模型路径', 'fallback_model': '回退模型'},
            )
            st.dataframe(display_registry, use_container_width=True, height=320)

        if not split_metrics.empty:
            signal_options = sorted(pd.to_datetime(split_metrics['signal_date']).dt.strftime('%Y-%m-%d').tolist())
            selected_signal_str = st.selectbox('选择一个信号日做模型调试', signal_options)
            selected_signal_date = pd.Timestamp(selected_signal_str)
            split_row = split_metrics.loc[pd.to_datetime(split_metrics['signal_date']) == selected_signal_date].head(1)
            debug_frame = _prediction_debug_frame(bundle, selected_signal_date)

            metric_cols = st.columns(5)
            metric_cols[0].metric('训练样本', int(split_row['train_rows'].iloc[0]))
            metric_cols[1].metric('验证样本', int(split_row['valid_rows'].iloc[0]))
            metric_cols[2].metric('测试样本', int(split_row['test_rows'].iloc[0]))
            metric_cols[3].metric('验证 IC', _format_number(float(split_row['valid_rank_ic'].iloc[0]), 3))
            metric_cols[4].metric('验证 RMSE', _format_number(float(split_row['valid_rmse'].iloc[0]), 4))

            top_debug = debug_frame.head(30).copy()
            display_debug = _to_display_frame(
                top_debug,
                {
                    'trade_date': '信号日',
                    'symbol': '代码',
                    'score': '预测分数',
                    'model_type': '模型类型',
                    'fallback_used': '是否回退',
                    label_name: '实际标签',
                },
            )
            st.dataframe(
                _styled_table(display_debug, {'预测分数': '{:.4f}', '实际标签': '{:.4f}'}),
                use_container_width=True,
                height=360,
            )
        else:
            st.info('当前没有可调试的切分记录。')

    with tabs[3]:
        st.markdown(
            '\n'.join(
                [
                    '- 这里推荐把“一个想法”拆成多个独立运行，不要反复覆盖同一份实验配置。',
                    '- 批量试验适合扫单一参数，跑完后直接去“策略比较”或“实验台账”看结果。',
                    '- 改回测假设时，优先动交易延迟、调仓频率、成本、最小成交额和持仓数。',
                    '- 调模型时，优先看验证 IC、回退比例、特征重要性，而不是只看总收益。',
                    '- 如果界面结果和你预期不一致，先去“工件浏览”里看生成的 YAML、split_metrics 和 latest_signal。',
                ]
            )
        )


def _render_artifacts(bundle: dict) -> None:
    _section_header('审计留痕', '工件浏览', '浏览、预览和下载当前运行的配置、报告、数据集、模型和回测产物。')
    artifact_files = bundle['artifact_files']
    run_dir = bundle['run_dir']
    validation = bundle.get('validation', {})
    if artifact_files.empty:
        st.info('当前运行没有找到工件文件。')
        return

    if validation:
        passed = bool(validation.get('passed', False))
        passed_count = int(validation.get('passed_count', 0))
        failed_count = int(validation.get('failed_count', 0))
        cards = [
            {'label': '有效性检查', 'value': '通过' if passed else '未通过', 'sub': '自动一致性校验'},
            {'label': '通过项', 'value': str(passed_count), 'sub': '规则项数'},
            {'label': '失败项', 'value': str(failed_count), 'sub': '需要回看'},
        ]
        _render_metric_grid(cards)
        validation_frame = pd.DataFrame(validation.get('checks', []))
        if not validation_frame.empty:
            validation_frame['status'] = validation_frame['passed'].map(lambda value: '通过' if value else '失败')
            display_validation = validation_frame[['status', 'name', 'detail']].rename(columns={'status': '状态', 'name': '检查项', 'detail': '细节'})
            st.dataframe(display_validation, use_container_width=True, height=220)

    left, right = st.columns([1, 1.35])
    with left:
        artifact_display = _to_display_frame(artifact_files, {'artifact_path': '工件路径', 'size_kb': '大小(KB)', 'modified_at': '修改时间'})
        st.dataframe(artifact_display, use_container_width=True, height=420)
        preferred_artifacts = [
            'metadata/experiment_manifest.json',
            'reports/factor_diagnostics.md',
            'reports/run_report.md',
            'metadata/run_summary.json',
        ]
        artifact_options = artifact_files['artifact_path'].tolist()
        default_artifact = next((path for path in preferred_artifacts if path in artifact_options), artifact_options[0])
        selected_artifact = st.selectbox('选择要预览的工件', artifact_options, index=artifact_options.index(default_artifact))
        artifact_path = run_dir / selected_artifact
        st.download_button('下载当前工件', data=artifact_path.read_bytes(), file_name=artifact_path.name)
    with right:
        suffix = artifact_path.suffix.lower()
        if suffix == '.parquet':
            frame = _read_parquet(str(artifact_path))
            st.dataframe(frame.head(500), use_container_width=True, height=420)
        elif suffix == '.csv':
            frame = _read_csv(str(artifact_path))
            st.dataframe(frame.head(500), use_container_width=True, height=420)
        elif suffix == '.json':
            st.json(_read_json(str(artifact_path)))
        elif suffix in {'.yaml', '.yml'}:
            st.code(yaml.safe_dump(_read_yaml(str(artifact_path)), allow_unicode=True, sort_keys=False), language='yaml')
        elif suffix == '.md':
            text = _read_text(str(artifact_path), limit=50000)
            preview_tab, source_tab = st.tabs(['渲染预览', '源码'])
            with preview_tab:
                st.markdown(text)
            with source_tab:
                st.code(text, language='markdown')
        else:
            st.code(_read_text(str(artifact_path), limit=30000))



__all__ = [
    '_render_quick_actions', '_render_overview', '_render_ledger',
    '_render_data_status', '_render_backtest', '_render_signal_desk',
    '_render_factor_lab', '_render_studio', '_render_artifacts',
]
