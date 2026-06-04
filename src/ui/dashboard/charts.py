from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from .helpers import *

def _metric_card(label: str, value: str, sub: str, tone: str = 'neutral') -> str:
    return (
        f"<div class='metric-card {tone}'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-sub'>{sub}</div>"
        "</div>"
    )


def _render_metric_grid(cards: list[dict]) -> None:
    html = ''.join(_metric_card(card['label'], card['value'], card['sub'], card.get('tone', 'neutral')) for card in cards)
    st.markdown(f"<div class='metric-grid'>{html}</div>", unsafe_allow_html=True)


def _render_status_grid(items: list[dict]) -> None:
    html = ''.join(
        (
            "<div class='status-card'>"
            f"<div class='status-label'>{item['label']}</div>"
            f"<div class='status-value'>{item['value']}</div>"
            f"<div class='status-note'>{item['note']}</div>"
            "</div>"
        )
        for item in items
    )
    st.markdown(f"<div class='status-grid'>{html}</div>", unsafe_allow_html=True)


def _section_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        (
            f"<div class='section-kicker'>{kicker}</div>"
            f"<div class='section-title'>{title}</div>"
            f"<div class='section-subtitle'>{subtitle}</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_hero(bundle: dict, selected_run_name: str) -> None:
    summary = bundle['summary']
    config = bundle['config']
    status = _status_snapshot(bundle)
    data_cfg = config.get('data', {})
    badges = [
        config.get('name', selected_run_name),
        _source_label(data_cfg.get('source', 'mock_ashare')),
        summary.get('universe_name', 'N/A'),
        summary.get('model_family', 'N/A'),
        summary.get('feature_set', 'N/A'),
        status['market_posture'],
        f"MLflow {summary.get('mlflow_run_id', 'N/A')[:8]}",
    ]
    badge_html = ''.join(f"<span class='badge'>{badge}</span>" for badge in badges)
    st.markdown(
        (
            "<div class='hero-panel'>"
            f"<div class='hero-title'>{config.get('name', selected_run_name)}</div>"
            f"<div class='hero-subtitle'>{config.get('description', '本地优先、可审计、以组合研究为核心的投研工作台。')}</div>"
            f"<div class='badge-row'>{badge_html}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _filter_date_window(frame: pd.DataFrame, start_date, end_date, date_col: str = 'trade_date') -> pd.DataFrame:
    if frame.empty or date_col not in frame.columns:
        return frame
    data = frame.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    return data.loc[(data[date_col] >= pd.Timestamp(start_date)) & (data[date_col] <= pd.Timestamp(end_date))].reset_index(drop=True)


def _factor_research_bundle(bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_panel = bundle.get('feature_panel', pd.DataFrame())
    label_panel = bundle.get('label_panel', pd.DataFrame())
    feature_inventory = bundle.get('feature_inventory', pd.DataFrame())
    label_name = bundle.get('config', {}).get('label', {}).get('name', 'fwd_return_20d')
    feature_names = [name for name in bundle.get('config', {}).get('features', {}).get('names', []) if name in feature_panel.columns]
    if feature_panel.empty or label_panel.empty or not feature_names:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    merged = feature_panel[['trade_date', 'symbol'] + feature_names].merge(label_panel[['trade_date', 'symbol', label_name]], on=['trade_date', 'symbol'], how='left')
    summary_rows = []
    ic_frames = []
    for feature_name in feature_names:
        feature_frame = merged[['trade_date', feature_name, label_name]].dropna().copy()
        if feature_frame.empty:
            continue
        daily_ic = (
            feature_frame.groupby('trade_date')
            .apply(lambda frame: frame[feature_name].corr(frame[label_name], method='spearman') if frame[feature_name].nunique() > 1 and frame[label_name].nunique() > 1 else 0.0)
            .rename('rank_ic')
            .reset_index()
        )
        daily_ic['factor_name'] = feature_name
        daily_ic = daily_ic[['trade_date', 'factor_name', 'rank_ic']].sort_values('trade_date').reset_index(drop=True)
        rolling_ic = daily_ic['rank_ic'].rolling(20, min_periods=5).mean()
        recent_ic = float(rolling_ic.iloc[-1]) if not rolling_ic.empty and not pd.isna(rolling_ic.iloc[-1]) else float(daily_ic['rank_ic'].tail(20).mean())
        failure_date = None
        if not rolling_ic.empty and not pd.isna(rolling_ic.iloc[-1]) and rolling_ic.iloc[-1] <= 0:
            negative_dates = daily_ic.loc[rolling_ic <= 0, 'trade_date']
            failure_date = str(pd.to_datetime(negative_dates.iloc[-1]).date()) if not negative_dates.empty else None
        state = '有效'
        if recent_ic <= 0:
            state = '失效'
        elif recent_ic < 0.01:
            state = '转弱'
        summary_rows.append(
            {
                'factor_name': feature_name,
                'ic_mean': float(daily_ic['rank_ic'].mean()),
                'ic_std': float(daily_ic['rank_ic'].std(ddof=0)),
                'ic_ir': float(daily_ic['rank_ic'].mean() / max(daily_ic['rank_ic'].std(ddof=0), 1e-9)),
                'positive_rate': float((daily_ic['rank_ic'] > 0).mean()),
                'recent_20d_ic': recent_ic,
                'coverage': float(feature_frame[feature_name].notna().mean()),
                'observations': int(len(feature_frame)),
                'state': state,
                'failure_since': failure_date,
            }
        )
        ic_frames.append(daily_ic)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty and not feature_inventory.empty:
        inventory = feature_inventory.rename(columns={'feature_name': 'factor_name'})
        summary = summary.merge(inventory, on='factor_name', how='left')
    corr_frame = merged[feature_names].corr(method='spearman').reset_index().rename(columns={'index': 'factor_name'}) if feature_names else pd.DataFrame()
    daily_ic_frame = pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame(columns=['trade_date', 'factor_name', 'rank_ic'])
    if not summary.empty:
        summary = summary.sort_values(['recent_20d_ic', 'ic_mean'], ascending=False).reset_index(drop=True)
    return summary, daily_ic_frame, corr_frame


def _factor_ic_bar_chart(summary: pd.DataFrame) -> alt.Chart:
    if summary.empty:
        return alt.Chart(pd.DataFrame({'factor_name': [], 'ic_mean': []})).mark_bar().encode(x='factor_name:N', y='ic_mean:Q').properties(height=260)
    frame = summary[['factor_name', 'ic_mean', 'recent_20d_ic', 'state']].copy()
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X('factor_name:N', title='因子'),
            y=alt.Y('ic_mean:Q', title='平均 Rank IC'),
            color=alt.Color('recent_20d_ic:Q', title='近20期 IC', scale=alt.Scale(scheme='tealblues')),
            tooltip=[
                alt.Tooltip('factor_name:N', title='因子'),
                alt.Tooltip('ic_mean:Q', title='平均 IC', format='.3f'),
                alt.Tooltip('recent_20d_ic:Q', title='近20期 IC', format='.3f'),
                alt.Tooltip('state:N', title='状态'),
            ],
        )
        .properties(height=280)
    )


def _factor_ic_timeseries_chart(daily_ic: pd.DataFrame, selected_factors: list[str]) -> alt.Chart:
    if daily_ic.empty:
        return alt.Chart(pd.DataFrame({'trade_date': [], 'rank_ic': [], 'factor_name': []})).mark_line().encode(x='trade_date:T', y='rank_ic:Q').properties(height=280)
    frame = daily_ic.loc[daily_ic['factor_name'].isin(selected_factors)].copy() if selected_factors else daily_ic.copy()
    return (
        alt.Chart(frame)
        .mark_line(point=False, strokeWidth=2.4)
        .encode(
            x=alt.X('trade_date:T', title='日期'),
            y=alt.Y('rank_ic:Q', title='因子 Rank IC', scale=alt.Scale(zero=False)),
            color=alt.Color('factor_name:N', title='因子'),
            tooltip=[
                alt.Tooltip('trade_date:T', title='日期'),
                alt.Tooltip('factor_name:N', title='因子'),
                alt.Tooltip('rank_ic:Q', title='Rank IC', format='.3f'),
            ],
        )
        .properties(height=300)
    )


def _factor_corr_heatmap(corr_frame: pd.DataFrame) -> alt.Chart:
    if corr_frame.empty:
        return alt.Chart(pd.DataFrame({'factor_name': [], 'peer': [], 'corr': []})).mark_rect().encode(x='factor_name:N', y='peer:N', color='corr:Q').properties(height=320)
    melted = corr_frame.melt(id_vars=['factor_name'], var_name='peer', value_name='corr')
    return (
        alt.Chart(melted)
        .mark_rect()
        .encode(
            x=alt.X('factor_name:N', title='因子'),
            y=alt.Y('peer:N', title='对比因子'),
            color=alt.Color('corr:Q', title='相关系数', scale=alt.Scale(scheme='tealblues', domain=[-1, 1])),
            tooltip=[
                alt.Tooltip('factor_name:N', title='因子'),
                alt.Tooltip('peer:N', title='对比因子'),
                alt.Tooltip('corr:Q', title='相关系数', format='.3f'),
            ],
        )
        .properties(height=320)
    )


def _quantile_profile_chart(quantile_summary: pd.DataFrame) -> alt.Chart:
    if quantile_summary.empty:
        return alt.Chart(pd.DataFrame({'bucket': [], 'mean_forward_return': []})).mark_bar().encode(x='bucket:N', y='mean_forward_return:Q').properties(height=280)
    frame = quantile_summary.loc[quantile_summary['bucket'].astype(str) != 'long_short'].copy()
    if frame.empty:
        return alt.Chart(pd.DataFrame({'bucket': [], 'mean_forward_return': []})).mark_bar().encode(x='bucket:N', y='mean_forward_return:Q').properties(height=280)
    frame['bucket'] = frame['bucket'].astype(str)
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color='#0f766e')
        .encode(
            x=alt.X('bucket:N', title='分位组'),
            y=alt.Y('mean_forward_return:Q', title='平均远期收益'),
            tooltip=[
                alt.Tooltip('bucket:N', title='分组'),
                alt.Tooltip('mean_forward_return:Q', title='平均远期收益', format='.4f'),
                alt.Tooltip('std_forward_return:Q', title='波动', format='.4f'),
                alt.Tooltip('observations:Q', title='样本期数'),
            ],
        )
        .properties(height=280)
    )


def _quantile_spread_chart(quantile_returns: pd.DataFrame) -> alt.Chart:
    if quantile_returns.empty:
        return alt.Chart(pd.DataFrame({'trade_date': [], 'long_short_spread': []})).mark_line().encode(x='trade_date:T', y='long_short_spread:Q').properties(height=300)
    frame = quantile_returns.copy()
    if frame.empty or 'quantile' not in frame.columns:
        return alt.Chart(pd.DataFrame({'trade_date': [], 'long_short_spread': []})).mark_line().encode(x='trade_date:T', y='long_short_spread:Q').properties(height=300)
    top_bucket = int(frame['quantile'].max())
    bottom_bucket = int(frame['quantile'].min())
    spread = (
        frame.loc[frame['quantile'] == top_bucket, ['trade_date', 'avg_forward_return']]
        .rename(columns={'avg_forward_return': 'top_bucket_return'})
        .merge(
            frame.loc[frame['quantile'] == bottom_bucket, ['trade_date', 'avg_forward_return']].rename(columns={'avg_forward_return': 'bottom_bucket_return'}),
            on='trade_date',
            how='inner',
        )
        .sort_values('trade_date')
        .reset_index(drop=True)
    )
    if spread.empty:
        return alt.Chart(pd.DataFrame({'trade_date': [], 'long_short_spread': []})).mark_line().encode(x='trade_date:T', y='long_short_spread:Q').properties(height=300)
    spread['long_short_spread'] = spread['top_bucket_return'] - spread['bottom_bucket_return']
    spread['rolling_spread_20'] = spread['long_short_spread'].rolling(20, min_periods=5).mean()
    base = alt.Chart(spread).encode(x=alt.X('trade_date:T', title='信号日'))
    bars = base.mark_bar(color='rgba(15, 118, 110, 0.25)').encode(
        y=alt.Y('long_short_spread:Q', title='顶底分位远期收益差'),
        tooltip=[
            alt.Tooltip('trade_date:T', title='信号日'),
            alt.Tooltip('long_short_spread:Q', title='顶底分位差', format='.4f'),
            alt.Tooltip('rolling_spread_20:Q', title='20期滚动均值', format='.4f'),
        ],
    )
    line = base.mark_line(color='#d97706', strokeWidth=2.8).encode(y='rolling_spread_20:Q')
    return (bars + line).properties(height=300)


def _nav_comparison_frame(nav: pd.DataFrame, benchmark_nav: pd.DataFrame) -> pd.DataFrame:
    strategy = nav[['trade_date', 'nav']].copy()
    strategy['series'] = '策略'
    strategy = strategy.rename(columns={'nav': 'value'})

    frames = [strategy]
    if not benchmark_nav.empty and {'trade_date', 'benchmark_nav'}.issubset(benchmark_nav.columns):
        benchmark = benchmark_nav[['trade_date', 'benchmark_nav']].copy()
        benchmark['series'] = benchmark_nav['benchmark_name'].iloc[0] if 'benchmark_name' in benchmark_nav.columns and not benchmark_nav.empty else '基准'
        benchmark = benchmark.rename(columns={'benchmark_nav': 'value'})
        frames.append(benchmark)
    return pd.concat(frames, ignore_index=True)


def _nav_chart(nav: pd.DataFrame, benchmark_nav: pd.DataFrame) -> alt.Chart:
    frame = _nav_comparison_frame(nav, benchmark_nav)
    overview = nav[['trade_date', 'nav']].copy()
    brush = alt.selection_interval(encodings=['x'], name='时间窗')
    series_order = frame['series'].drop_duplicates().tolist()
    palette = ['#0f766e', '#d97706', '#60a5fa', '#f97316']
    color_scale = alt.Scale(domain=series_order, range=palette[: len(series_order)])
    top = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2.8)
        .encode(
            x=alt.X('trade_date:T', title='交易日', scale=alt.Scale(domain=brush)),
            y=alt.Y('value:Q', title='净值', scale=alt.Scale(zero=False)),
            color=alt.Color('series:N', title=None, scale=color_scale),
            tooltip=[
                alt.Tooltip('trade_date:T', title='日期'),
                alt.Tooltip('series:N', title='序列'),
                alt.Tooltip('value:Q', format='.3f', title='净值'),
            ],
        )
        .properties(height=320)
    )
    bottom = (
        alt.Chart(overview)
        .mark_area(color='rgba(18, 111, 143, 0.25)', line={'color': '#126f8f'})
        .encode(
            x=alt.X('trade_date:T', title='拖动下方区域缩放时间'),
            y=alt.Y('nav:Q', title=None),
            tooltip=[alt.Tooltip('trade_date:T', title='日期'), alt.Tooltip('nav:Q', format='.3f', title='策略净值')],
        )
        .add_params(brush)
        .properties(height=72)
    )
    return alt.vconcat(top, bottom, spacing=8)


def _excess_nav_chart(nav: pd.DataFrame, benchmark_nav: pd.DataFrame) -> alt.Chart | None:
    if benchmark_nav.empty or 'benchmark_nav' not in benchmark_nav.columns:
        return None
    frame = nav[['trade_date', 'nav']].merge(benchmark_nav[['trade_date', 'benchmark_nav']], on='trade_date', how='left')
    frame['benchmark_nav'] = frame['benchmark_nav'].ffill().bfill()
    frame = frame.dropna(subset=['benchmark_nav']).copy()
    if frame.empty:
        return None
    frame['excess_nav'] = frame['nav'] / frame['benchmark_nav'] - 1.0
    return (
        alt.Chart(frame)
        .mark_area(line={'color': '#7c3aed'}, color='rgba(124, 58, 237, 0.18)')
        .encode(
            x=alt.X('trade_date:T', title='交易日'),
            y=alt.Y('excess_nav:Q', title='超额收益', axis=alt.Axis(format='%')),
            tooltip=[
                alt.Tooltip('trade_date:T', title='日期'),
                alt.Tooltip('excess_nav:Q', title='超额收益', format='.2%'),
            ],
        )
        .properties(height=180)
    )


def _drawdown_chart(drawdown: pd.DataFrame) -> alt.Chart:
    if drawdown.empty:
        return alt.Chart(pd.DataFrame({'trade_date': [], 'drawdown': []})).mark_area().encode(x='trade_date:T', y='drawdown:Q').properties(height=220)
    frame = drawdown[['trade_date', 'drawdown']].copy()
    return (
        alt.Chart(frame)
        .mark_area(line={'color': '#b45309'}, color='rgba(180, 83, 9, 0.22)')
        .encode(
            x=alt.X('trade_date:T', title='交易日'),
            y=alt.Y('drawdown:Q', title='回撤', scale=alt.Scale(domain=[min(float(frame['drawdown'].min()), -0.01), 0])),
            tooltip=[alt.Tooltip('trade_date:T', title='日期'), alt.Tooltip('drawdown:Q', format='.2%', title='回撤')],
        )
        .properties(height=220)
    )


def _turnover_chart(nav: pd.DataFrame) -> alt.Chart:
    if nav.empty:
        return alt.Chart(pd.DataFrame({'trade_date': [], 'turnover': [], 'trade_cost': []})).mark_bar().encode(x='trade_date:T', y='turnover:Q').properties(height=220)
    frame = nav[['trade_date', 'turnover', 'trade_cost']].copy()
    return (
        alt.Chart(frame)
        .mark_bar(color='#126f8f', opacity=0.75)
        .encode(
            x=alt.X('trade_date:T', title='交易日'),
            y=alt.Y('turnover:Q', title='换手率', axis=alt.Axis(format='%')),
            tooltip=[
                alt.Tooltip('trade_date:T', title='日期'),
                alt.Tooltip('turnover:Q', title='换手率', format='.2%'),
                alt.Tooltip('trade_cost:Q', title='交易成本', format=',.2f'),
            ],
        )
        .properties(height=220)
    )


def _monthly_heatmap(monthly_returns: pd.DataFrame) -> alt.Chart:
    if monthly_returns.empty:
        return alt.Chart(pd.DataFrame({'month': [], 'year': [], 'monthly_return': []})).mark_rect().encode(x='month:N', y='year:O').properties(height=280)
    frame = monthly_returns.copy()
    frame['trade_date'] = pd.to_datetime(frame['trade_date'])
    frame['year'] = frame['trade_date'].dt.year.astype(str)
    frame['month'] = frame['trade_date'].dt.month.astype(str) + '月'
    base = alt.Chart(frame).encode(
        x=alt.X('month:N', sort=MONTH_ORDER, title=None),
        y=alt.Y('year:O', title=None),
    )
    heatmap = base.mark_rect(cornerRadius=6).encode(
            color=alt.Color(
                'monthly_return:Q',
                scale=alt.Scale(domainMid=0, range=['#b42318', '#f7f0e2', '#0f766e']),
                legend=alt.Legend(format='.1%'),
                title='月收益',
            ),
            tooltip=[
                alt.Tooltip('year:N', title='年份'),
                alt.Tooltip('month:N', title='月份'),
                alt.Tooltip('monthly_return:Q', format='.2%', title='收益'),
            ],
        )
    text = base.mark_text(fontSize=11, color='#142321').encode(text=alt.Text('monthly_return:Q', format='.1%'))
    return (heatmap + text).properties(height=280)


def _rank_ic_chart(rank_ic: pd.DataFrame) -> alt.Chart:
    if rank_ic.empty:
        return alt.Chart(pd.DataFrame({'trade_date': [], 'rank_ic': []})).mark_line().encode(x='trade_date:T', y='rank_ic:Q').properties(height=260)
    frame = rank_ic.copy()
    mean_ic = float(frame['rank_ic'].mean()) if not frame.empty else 0.0
    line = (
        alt.Chart(frame)
        .mark_line(color='#0f766e', strokeWidth=2.5)
        .encode(
            x=alt.X('trade_date:T', title='信号日'),
            y=alt.Y('rank_ic:Q', title='Rank IC', scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('trade_date:T', title='日期'), alt.Tooltip('rank_ic:Q', format='.3f', title='Rank IC')],
        )
    )
    rule = alt.Chart(pd.DataFrame({'mean_rank_ic': [mean_ic]})).mark_rule(color='#b45309', strokeDash=[8, 4]).encode(y='mean_rank_ic:Q')
    return (line + rule).properties(height=260)


def _feature_importance_chart(feature_importance: pd.DataFrame) -> alt.Chart:
    frame = feature_importance.sort_values('importance_gain', ascending=True).copy()
    return (
        alt.Chart(frame)
        .mark_bar(color='#126f8f', cornerRadiusEnd=6)
        .encode(
            x=alt.X('importance_gain:Q', title='平均增益'),
            y=alt.Y('feature_name:N', sort=None, title=None),
            tooltip=[alt.Tooltip('feature_name:N', title='特征'), alt.Tooltip('importance_gain:Q', format='.3f', title='增益')],
        )
        .properties(height=max(220, len(frame) * 36))
    )


def _reason_chart(filtered_candidates: pd.DataFrame) -> alt.Chart:
    frame = filtered_candidates['reason'].value_counts().rename_axis('reason').reset_index(name='count')
    return (
        alt.Chart(frame)
        .mark_bar(color='#0f766e', cornerRadiusEnd=6)
        .encode(
            x=alt.X('count:Q', title='数量'),
            y=alt.Y('reason:N', sort='-x', title=None),
            tooltip=[alt.Tooltip('reason:N', title='原因'), alt.Tooltip('count:Q', title='数量')],
        )
        .properties(height=max(220, len(frame) * 38))
    )


def _signal_score_chart(latest_signal: pd.DataFrame) -> alt.Chart:
    frame = latest_signal.sort_values('rank').copy()
    frame['label'] = frame['symbol'] + ' · ' + frame['security_name'].fillna('')
    return (
        alt.Chart(frame)
        .mark_bar(color='#0f766e', cornerRadiusEnd=6)
        .encode(
            x=alt.X('score:Q', title='模型分数'),
            y=alt.Y('label:N', sort='-x', title=None),
            color=alt.Color('industry:N', legend=None, scale=alt.Scale(scheme='tableau20')),
            tooltip=[
                alt.Tooltip('symbol:N', title='代码'),
                alt.Tooltip('security_name:N', title='名称'),
                alt.Tooltip('industry:N', title='行业'),
                alt.Tooltip('score:Q', title='分数', format='.4f'),
                alt.Tooltip('target_weight:Q', title='目标权重', format='.2%'),
            ],
        )
        .properties(height=max(280, len(frame) * 28))
    )


def _industry_weight_chart(latest_signal: pd.DataFrame) -> alt.Chart:
    frame = latest_signal.groupby('industry', as_index=False)['target_weight'].sum().sort_values('target_weight', ascending=False)
    return (
        alt.Chart(frame)
        .mark_arc(innerRadius=42, outerRadius=90)
        .encode(
            theta=alt.Theta('target_weight:Q'),
            color=alt.Color('industry:N', scale=alt.Scale(scheme='tableau20')),
            tooltip=[alt.Tooltip('industry:N', title='行业'), alt.Tooltip('target_weight:Q', title='权重', format='.2%')],
        )
        .properties(height=260)
    )


def _ledger_scatter_chart(ledger: pd.DataFrame) -> alt.Chart:
    frame = ledger.copy()
    return (
        alt.Chart(frame)
        .mark_circle(size=180, opacity=0.86)
        .encode(
            x=alt.X('sharpe_like:Q', title='夏普近似'),
            y=alt.Y('total_return:Q', title='总收益', axis=alt.Axis(format='%')),
            color=alt.Color('avg_rank_ic:Q', title='平均 Rank IC', scale=alt.Scale(scheme='teals')),
            tooltip=[
                alt.Tooltip('run_name:N', title='运行'),
                alt.Tooltip('experiment_name:N', title='实验'),
                alt.Tooltip('total_return:Q', title='总收益', format='.2%'),
                alt.Tooltip('sharpe_like:Q', title='夏普近似', format='.2f'),
                alt.Tooltip('max_drawdown:Q', title='最大回撤', format='.2%'),
            ],
        )
        .properties(height=320)
    )


def _styled_table(df: pd.DataFrame, formats: dict[str, str] | None = None) -> pd.io.formats.style.Styler:
    style = df.style.hide(axis='index')
    if formats:
        available_formats = {column: fmt for column, fmt in formats.items() if column in df.columns}
        style = style.format(available_formats)
    return style


def _fallback_rate(bundle: dict) -> float:
    split_metrics = bundle['split_metrics']
    if split_metrics.empty or 'fallback_used' not in split_metrics.columns:
        return 0.0
    return float(split_metrics['fallback_used'].mean())


def _jump_to_page(page_key: str, mode_key: str | None = None) -> None:
    if mode_key:
        st.session_state['workspace_mode'] = mode_key
    st.session_state['page_key'] = page_key
    st.rerun()


def _decision_brief(bundle: dict) -> list[str]:
    summary = bundle['summary']
    latest_signal = bundle['latest_signal']
    benchmark_nav = bundle.get('benchmark_nav', pd.DataFrame())
    risk_flag = _status_snapshot(bundle)['risk_flag']

    lines = [f"风险状态是 {risk_flag}。"]
    if not latest_signal.empty:
        top_names = latest_signal['security_name'].fillna(latest_signal['symbol']).head(3).tolist()
        lines.append(f"最新组合共 {len(latest_signal)} 只，前排候选是 {' / '.join(top_names)}。")
    if 'benchmark_total_return' in summary and 'excess_total_return' in summary:
        benchmark_name = str(benchmark_nav['benchmark_name'].iloc[0]) if not benchmark_nav.empty and 'benchmark_name' in benchmark_nav.columns else '基准'
        lines.append(f"本轮回测相对 {benchmark_name} 的总超额是 {_format_pct(summary.get('excess_total_return'))}。")
    else:
        lines.append(f"本轮总收益 {_format_pct(summary.get('total_return'))}，最大回撤 {_format_pct(summary.get('max_drawdown'))}。")
    if risk_flag in {'重点关注', '需要复盘'}:
        lines.append('建议先看回测复盘，确认回撤、超额和风险状态。')
    else:
        lines.append('建议先看今日组合或因子池，再决定是否继续下钻到回测或调参。')
    return lines



__all__ = [
    '_metric_card', '_render_metric_grid', '_render_status_grid',
    '_section_header', '_render_hero', '_filter_date_window',
    '_factor_research_bundle', '_factor_ic_bar_chart',
    '_factor_ic_timeseries_chart', '_factor_corr_heatmap',
    '_quantile_profile_chart', '_quantile_spread_chart',
    '_nav_comparison_frame', '_nav_chart', '_excess_nav_chart',
    '_drawdown_chart', '_turnover_chart', '_monthly_heatmap',
    '_rank_ic_chart', '_feature_importance_chart', '_reason_chart',
    '_signal_score_chart', '_industry_weight_chart', '_ledger_scatter_chart',
    '_styled_table', '_fallback_rate', '_jump_to_page', '_decision_brief',
]
