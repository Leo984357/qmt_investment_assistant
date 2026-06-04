from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.evaluation.reporting import compute_drawdown, compute_monthly_returns
from src.experiment.runner import run_experiment
from src.experiment.validation import validate_run
from src.ops.paths import ARTIFACT_RUNS_DIR, ARTIFACTS_DIR

PAGE_OPTIONS = {
    'overview': '决策总台',
    'signal': '今日组合',
    'backtest': '回测复盘',
    'factorlab': '因子池',
    'ledger': '策略比较',
    'studio': '实验开发',
    'data': '数据底座',
    'artifacts': '审计工件',
}
WORKSPACE_MODES = {
    'daily': {
        'label': '日常决策',
        'description': '把今天该看什么、该不该调、组合长什么样放在最前面。',
        'pages': ['overview', 'signal', 'backtest'],
        'checklist': ['1. 先看决策总台。', '2. 再看今日组合。', '3. 最后看回测复盘。'],
    },
    'review': {
        'label': '研究复盘',
        'description': '重点看策略相对基准表现、回撤、IC 和跨实验比较。',
        'pages': ['overview', 'backtest', 'factorlab', 'ledger'],
        'checklist': ['1. 先看决策总台。', '2. 再看回测复盘。', '3. 再看因子池。', '4. 最后看策略比较。'],
    },
    'develop': {
        'label': '实验开发',
        'description': '重点看调参与调试、数据底座和审计工件。',
        'pages': ['overview', 'factorlab', 'studio', 'data', 'artifacts'],
        'checklist': ['1. 先看决策总台。', '2. 再看因子池。', '3. 再看实验开发。', '4. 需要时追到数据底座和审计工件。'],
    },
}

MONTH_ORDER = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
AVAILABLE_FEATURES = ['mom20', 'mom60', 'mom120', 'rev5', 'vol20', 'vol60', 'liq20']
DATA_SOURCE_LABELS = {
    'baostock_ashare': '真实A股',
    'mock_ashare': '模拟A股',
}
DATA_SOURCE_OPTIONS = {
    '真实A股 (Baostock + AKShare)': 'baostock_ashare',
    '模拟A股 (Mock)': 'mock_ashare',
}
SWEEP_FIELD_OPTIONS = {
    '调仓频率(天)': ('backtest.rebalance_frequency_days', int),
    '持仓数 Top N': ('portfolio.top_n', int),
    '训练窗口天数': ('model.train_window_days', int),
    '验证窗口天数': ('model.valid_window_days', int),
    '树数量': ('model.params.n_estimators', int),
    '学习率': ('model.params.learning_rate', float),
    '叶子数': ('model.params.num_leaves', int),
    '正常总暴露': ('portfolio.gross_exposure', float),
    '防守总暴露': ('portfolio.defensive_gross', float),
}


def _source_label(source: str) -> str:
    return DATA_SOURCE_LABELS.get(str(source), str(source))


def _is_real_source(source: str) -> bool:
    return str(source) != 'mock_ashare'


def _safe_date_input_value(value, fallback: str) -> datetime.date:
    try:
        return pd.Timestamp(str(value)).date()
    except Exception:
        return pd.Timestamp(fallback).date()


def _normalize_ui_date_string(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
        raise ValueError(f'{field_name} 必须是 YYYY-MM-DD，收到 {text!r}')
    ts = pd.Timestamp(text)
    if ts.year < 1990 or ts.year > 2100:
        raise ValueError(f'{field_name} 年份超出允许范围，收到 {text!r}')
    return ts.strftime('%Y-%m-%d')


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #232629;
          --panel: rgba(34, 38, 41, 0.92);
          --panel-strong: #2b3035;
          --ink: #f5f7fa;
          --muted: #b4bcc6;
          --line: rgba(255, 255, 255, 0.08);
          --accent: #34c3b6;
          --accent-soft: rgba(52, 195, 182, 0.14);
          --alert: #e7a23d;
          --critical: #ff6b6b;
          --good: #3ddc97;
          --hero-a: rgba(52, 195, 182, 0.10);
          --hero-b: rgba(84, 128, 255, 0.08);
          --hero-c: rgba(231, 162, 61, 0.08);
        }
        .stApp {
          background:
            radial-gradient(circle at top left, var(--hero-a), transparent 30%),
            radial-gradient(circle at top right, var(--hero-b), transparent 26%),
            linear-gradient(180deg, #232629 0%, #1d2023 100%);
          color: var(--ink);
          font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
        }
        .block-container {
          padding-top: 2rem;
          padding-bottom: 2rem;
          max-width: 1440px;
        }
        h1, h2, h3, h4 {
          font-family: "Space Grotesk", "IBM Plex Sans", "Avenir Next", sans-serif;
          color: var(--ink);
          letter-spacing: -0.02em;
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(38, 42, 46, 0.98), rgba(27, 30, 34, 0.98));
          border-right: 1px solid var(--line);
        }
        .hero-panel {
          padding: 1.5rem 1.6rem;
          border: 1px solid var(--line);
          background: linear-gradient(135deg, rgba(42, 47, 51, 0.98), rgba(31, 35, 39, 0.94));
          border-radius: 28px;
          box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
          margin-bottom: 1rem;
        }
        .hero-title {
          font-size: 2.15rem;
          line-height: 1.05;
          font-weight: 700;
          margin-bottom: 0.4rem;
        }
        .hero-subtitle {
          color: var(--muted);
          font-size: 1rem;
          line-height: 1.5;
          max-width: 72rem;
        }
        .badge-row {
          display: flex;
          gap: 0.45rem;
          flex-wrap: wrap;
          margin-top: 1rem;
        }
        .badge {
          display: inline-flex;
          align-items: center;
          gap: 0.25rem;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.1);
          background: rgba(255,255,255,0.06);
          color: var(--ink);
          padding: 0.3rem 0.7rem;
          font-size: 0.82rem;
          font-weight: 600;
        }
        .section-kicker {
          color: var(--muted);
          font-size: 0.82rem;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          margin-bottom: 0.25rem;
        }
        .section-title {
          font-size: 1.35rem;
          font-weight: 700;
          margin: 0 0 0.2rem 0;
        }
        .section-subtitle {
          color: var(--muted);
          margin: 0 0 0.8rem 0;
        }
        .metric-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 0.8rem;
          margin: 0.6rem 0 1.1rem 0;
        }
        .metric-card {
          padding: 1rem 1rem 0.95rem 1rem;
          border-radius: 22px;
          background: linear-gradient(180deg, rgba(43, 48, 53, 0.96), rgba(35, 39, 43, 0.94));
          border: 1px solid var(--line);
          box-shadow: 0 10px 20px rgba(0, 0, 0, 0.18);
        }
        .metric-card.good {
          background: linear-gradient(180deg, rgba(25, 74, 58, 0.96), rgba(32, 55, 48, 0.94));
        }
        .metric-card.warn {
          background: linear-gradient(180deg, rgba(83, 58, 24, 0.96), rgba(49, 39, 26, 0.94));
        }
        .metric-card.critical {
          background: linear-gradient(180deg, rgba(88, 34, 34, 0.96), rgba(52, 31, 31, 0.94));
        }
        .metric-label {
          color: var(--muted);
          font-size: 0.82rem;
          margin-bottom: 0.35rem;
        }
        .metric-value {
          font-size: 1.65rem;
          font-weight: 700;
          line-height: 1;
        }
        .metric-sub {
          color: var(--muted);
          font-size: 0.82rem;
          margin-top: 0.38rem;
        }
        .status-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
          gap: 0.8rem;
          margin-bottom: 1rem;
        }
        .status-card {
          padding: 1rem 1rem 0.95rem 1rem;
          border-radius: 24px;
          background: rgba(43, 48, 53, 0.82);
          border: 1px solid var(--line);
        }
        .status-label {
          color: var(--muted);
          font-size: 0.78rem;
          text-transform: uppercase;
          letter-spacing: 0.11em;
        }
        .status-value {
          font-size: 1.25rem;
          font-weight: 700;
          margin-top: 0.3rem;
        }
        .status-note {
          color: var(--muted);
          margin-top: 0.3rem;
          font-size: 0.84rem;
          line-height: 1.45;
        }
        .code-chip {
          font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
          font-size: 0.84rem;
          background: rgba(52, 195, 182, 0.14);
          border-radius: 10px;
          padding: 0.18rem 0.42rem;
        }
        div[data-testid="stMetricValue"] { color: var(--ink); }
        div[data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMarkdownContainer"], label, .stSelectbox label, .stMultiSelect label, .stRadio label {
          color: var(--ink) !important;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="base-input"] > div,
        [data-baseweb="input"] > div {
          background: #2b3035 !important;
          color: var(--ink) !important;
          border-color: var(--line) !important;
        }
        [data-baseweb="select"] input,
        [data-baseweb="base-input"] input,
        [data-baseweb="input"] input,
        textarea {
          color: var(--ink) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
          gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
          background: rgba(255,255,255,0.04);
          border-radius: 14px 14px 0 0;
          color: var(--muted);
          border: 1px solid var(--line);
          border-bottom: none;
        }
        .stTabs [aria-selected="true"] {
          color: var(--ink) !important;
          background: rgba(52, 195, 182, 0.10) !important;
        }
        div[data-testid="stDataFrame"] *,
        div[data-testid="stTable"] * {
          color: var(--ink) !important;
        }
        div[data-testid="stDataFrame"] {
          background: rgba(43, 48, 53, 0.88);
        }
        pre, code {
          color: #f7fafc !important;
        }
        .stAlert {
          background: rgba(43, 48, 53, 0.92);
          color: var(--ink);
        }
        div[data-testid="stDataFrame"] {
          border-radius: 18px;
          overflow: hidden;
          border: 1px solid var(--line);
        }
        .artifact-shell {
          border-radius: 22px;
          border: 1px solid var(--line);
          background: rgba(43, 48, 53, 0.88);
          padding: 1rem 1rem 0.2rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _list_run_dirs() -> list[str]:
    if not ARTIFACT_RUNS_DIR.exists():
        return []
    return [str(path) for path in sorted([path for path in ARTIFACT_RUNS_DIR.iterdir() if path.is_dir()], reverse=True)]


@st.cache_data(show_spinner=False)
def _read_json(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


@st.cache_data(show_spinner=False)
def _read_yaml(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


@st.cache_data(show_spinner=False)
def _read_parquet(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_text(path_str: str, limit: int = 20000) -> str:
    path = Path(path_str)
    if not path.exists():
        return ''
    content = path.read_text(encoding='utf-8', errors='ignore')
    return content[:limit]


@st.cache_data(show_spinner=False)
def _load_run_bundle(run_dir_str: str) -> dict:
    run_dir = Path(run_dir_str)
    summary = _read_json(str(run_dir / 'metadata' / 'run_summary.json'))
    dataset_summary = _read_json(str(run_dir / 'metadata' / 'dataset_summary.json'))
    data_snapshot = pd.DataFrame(_read_json(str(run_dir / 'metadata' / 'data_snapshot.json')))
    data_contract = pd.DataFrame(_read_json(str(run_dir / 'metadata' / 'data_contract.json')))
    stage_timings = _read_json(str(run_dir / 'metadata' / 'stage_timings.json'))
    registry_catalog = _read_json(str(run_dir / 'metadata' / 'registry_catalog.json'))
    selected_modules = _read_json(str(run_dir / 'metadata' / 'selected_modules.json'))
    experiment_manifest = _read_json(str(run_dir / 'metadata' / 'experiment_manifest.json'))
    resolved_config = _read_yaml(str(run_dir / 'config' / 'resolved_experiment.yaml'))
    latest_signal = _read_parquet(str(run_dir / 'signals' / 'latest_signal.parquet'))
    filtered_candidates = _read_parquet(str(run_dir / 'signals' / 'filtered_candidates.parquet'))
    target_weights = _read_parquet(str(run_dir / 'signals' / 'target_weights.parquet'))
    predictions = _read_parquet(str(run_dir / 'signals' / 'predictions.parquet'))
    signal_scores = _read_parquet(str(run_dir / 'signals' / 'signal_scores.parquet'))
    daily_signal_stats = _read_parquet(str(run_dir / 'signals' / 'daily_signal_stats.parquet'))
    nav = _read_parquet(str(run_dir / 'backtest' / 'nav.parquet'))
    benchmark_nav = _read_parquet(str(run_dir / 'backtest' / 'benchmark_nav.parquet'))
    drawdown = _read_parquet(str(run_dir / 'backtest' / 'drawdown.parquet'))
    monthly_returns = _read_parquet(str(run_dir / 'backtest' / 'monthly_returns.parquet'))
    rank_ic = _read_parquet(str(run_dir / 'backtest' / 'rank_ic.parquet'))
    trades = _read_parquet(str(run_dir / 'backtest' / 'trades.parquet'))
    positions = _read_parquet(str(run_dir / 'backtest' / 'positions.parquet'))
    split_metrics = _read_csv(str(run_dir / 'models' / 'split_metrics.csv'))
    feature_importance = _read_csv(str(run_dir / 'models' / 'feature_importance.csv'))
    feature_panel = _read_parquet(str(run_dir / 'features' / 'feature_panel.parquet'))
    feature_inventory = _read_csv(str(run_dir / 'features' / 'feature_inventory.csv'))
    feature_registry = _read_csv(str(run_dir / 'features' / 'feature_registry.csv'))
    label_inventory = _read_csv(str(run_dir / 'labels' / 'label_inventory.csv'))
    label_panel = _read_parquet(str(run_dir / 'labels' / 'label_panel.parquet'))
    model_registry = _read_csv(str(run_dir / 'models' / 'model_registry.csv'))
    report_markdown = _read_text(str(run_dir / 'reports' / 'run_report.md'))
    factor_report_markdown = _read_text(str(run_dir / 'reports' / 'factor_diagnostics.md'))
    evaluation_ic_summary = _read_parquet(str(run_dir / 'evaluation' / 'ic_summary.parquet'))
    evaluation_quantile_returns = _read_parquet(str(run_dir / 'evaluation' / 'quantile_returns.parquet'))
    evaluation_quantile_summary = _read_parquet(str(run_dir / 'evaluation' / 'quantile_summary.parquet'))
    evaluation_selection_turnover = _read_parquet(str(run_dir / 'evaluation' / 'selection_turnover.parquet'))
    evaluation_signal_coverage = _read_parquet(str(run_dir / 'evaluation' / 'signal_coverage.parquet'))
    try:
        validation = validate_run(run_path=run_dir)
    except Exception as exc:
        validation = {'passed': False, 'failed_count': 1, 'checks': [{'name': 'validation_runtime_error', 'passed': False, 'detail': str(exc)}]}

    for frame in [
        latest_signal,
        filtered_candidates,
        target_weights,
        predictions,
        signal_scores,
        daily_signal_stats,
        nav,
        benchmark_nav,
        drawdown,
        monthly_returns,
        rank_ic,
        trades,
        positions,
        split_metrics,
        model_registry,
        label_panel,
        feature_panel,
        evaluation_quantile_returns,
        evaluation_selection_turnover,
        evaluation_signal_coverage,
    ]:
        for column in ['trade_date', 'signal_date', 'execution_date', 'created_at']:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column])

    artifact_files = []
    for path in sorted(run_dir.rglob('*'), key=lambda item: _artifact_priority(item.relative_to(run_dir).as_posix()) if item.is_file() else (999, item.as_posix())):
        if path.is_file():
            artifact_files.append(
                {
                    'artifact_path': path.relative_to(run_dir).as_posix(),
                    'size_kb': round(path.stat().st_size / 1024, 2),
                    'modified_at': pd.Timestamp(path.stat().st_mtime, unit='s'),
                }
            )

    return {
        'run_dir': run_dir,
        'summary': summary,
        'dataset_summary': dataset_summary,
        'data_snapshot': data_snapshot,
        'data_contract': data_contract,
        'stage_timings': stage_timings,
        'registry_catalog': registry_catalog,
        'selected_modules': selected_modules,
        'experiment_manifest': experiment_manifest,
        'config': resolved_config,
        'latest_signal': latest_signal,
        'filtered_candidates': filtered_candidates,
        'target_weights': target_weights,
        'predictions': predictions,
        'signal_scores': signal_scores,
        'daily_signal_stats': daily_signal_stats,
        'nav': nav,
        'benchmark_nav': benchmark_nav,
        'drawdown': drawdown,
        'monthly_returns': monthly_returns,
        'rank_ic': rank_ic,
        'trades': trades,
        'positions': positions,
        'split_metrics': split_metrics,
        'feature_importance': feature_importance,
        'feature_panel': feature_panel,
        'feature_inventory': feature_inventory,
        'feature_registry': feature_registry,
        'label_inventory': label_inventory,
        'label_panel': label_panel,
        'model_registry': model_registry,
        'report_markdown': report_markdown,
        'factor_report_markdown': factor_report_markdown,
        'evaluation_ic_summary': evaluation_ic_summary,
        'evaluation_quantile_returns': evaluation_quantile_returns,
        'evaluation_quantile_summary': evaluation_quantile_summary,
        'evaluation_selection_turnover': evaluation_selection_turnover,
        'evaluation_signal_coverage': evaluation_signal_coverage,
        'artifact_files': pd.DataFrame(artifact_files),
        'validation': validation,
    }


@st.cache_data(show_spinner=False)
def _build_ledger(run_dir_values: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir_str in run_dir_values:
        run_dir = Path(run_dir_str)
        summary = _read_json(str(run_dir / 'metadata' / 'run_summary.json'))
        if not summary:
            continue
        dataset_summary = _read_json(str(run_dir / 'metadata' / 'dataset_summary.json'))
        config = _read_yaml(str(run_dir / 'config' / 'resolved_experiment.yaml'))
        latest_signal = _read_parquet(str(run_dir / 'signals' / 'latest_signal.parquet'))
        rows.append(
            {
                'run_name': run_dir.name,
                'run_dir': run_dir_str,
                'created_at': pd.Timestamp(run_dir.stat().st_mtime, unit='s'),
                'experiment_name': config.get('name', run_dir.name),
                'description': config.get('description', ''),
                'universe_name': summary.get('universe_name', config.get('data', {}).get('universe_name', 'N/A')),
                'model_family': summary.get('model_family', config.get('model', {}).get('family', 'N/A')),
                'feature_set': summary.get('feature_set', config.get('features', {}).get('set_name', 'N/A')),
                'label_name': summary.get('label_name', config.get('label', {}).get('name', 'N/A')),
                'top_n': config.get('portfolio', {}).get('top_n'),
                'total_return': summary.get('total_return', 0.0),
                'annual_return': summary.get('annual_return', 0.0),
                'sharpe_like': summary.get('sharpe_like', 0.0),
                'max_drawdown': summary.get('max_drawdown', 0.0),
                'avg_rank_ic': summary.get('avg_rank_ic', 0.0),
                'trade_count': summary.get('trade_count', 0),
                'total_cost': summary.get('total_cost', 0.0),
                'prediction_dates': summary.get('prediction_dates', 0),
                'dataset_rows': dataset_summary.get('dataset_rows', 0),
                'signal_count': int(len(latest_signal)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('created_at', ascending=False).reset_index(drop=True)


def _format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return 'N/A'
    return f'{float(value) * 100:.2f}%'


def _format_currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return 'N/A'
    return f'¥{float(value):,.0f}'


def _format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return 'N/A'
    return f'{float(value):,.{digits}f}'


def _tone_from_value(value: float, invert: bool = False) -> str:
    if pd.isna(value):
        return 'warn'
    score = -value if invert else value
    if score > 0:
        return 'good'
    if score < 0:
        return 'critical'
    return 'warn'


def _status_snapshot(bundle: dict) -> dict:
    summary = bundle['summary']
    split_metrics = bundle['split_metrics']
    latest_signal = bundle['latest_signal']
    fallback_rate = float(split_metrics['fallback_used'].mean()) if not split_metrics.empty and 'fallback_used' in split_metrics.columns else 0.0
    gross = float(latest_signal['target_weight'].sum()) if not latest_signal.empty else 0.0
    market_posture = '进攻' if gross >= 0.85 else ('平衡' if gross >= 0.55 else '防守')
    if summary.get('max_drawdown', 0.0) <= -0.15:
        risk_flag = '重点关注'
    elif summary.get('total_return', 0.0) < 0:
        risk_flag = '需要复盘'
    else:
        risk_flag = '健康'
    return {
        'fallback_rate': fallback_rate,
        'market_posture': market_posture,
        'risk_flag': risk_flag,
        'gross_exposure': gross,
    }


def _build_rebalance_delta(target_weights: pd.DataFrame) -> pd.DataFrame:
    if target_weights.empty:
        return pd.DataFrame(columns=['symbol', 'prev_weight', 'target_weight', 'delta_weight', 'action', 'prev_rank', 'rank', 'score'])
    signal_dates = sorted(pd.to_datetime(target_weights['signal_date'].dropna().unique()))
    latest_date = signal_dates[-1]
    previous_date = signal_dates[-2] if len(signal_dates) > 1 else None
    latest = target_weights.loc[target_weights['signal_date'] == latest_date, ['symbol', 'target_weight', 'rank', 'score']].copy()
    latest = latest.rename(columns={'target_weight': 'latest_weight', 'rank': 'latest_rank', 'score': 'latest_score'})
    if previous_date is not None:
        previous = target_weights.loc[target_weights['signal_date'] == previous_date, ['symbol', 'target_weight', 'rank']].copy()
        previous = previous.rename(columns={'target_weight': 'prev_weight', 'rank': 'prev_rank'})
    else:
        previous = pd.DataFrame(columns=['symbol', 'prev_weight', 'prev_rank'])
    merged = previous.merge(latest, on='symbol', how='outer').fillna({'prev_weight': 0.0, 'latest_weight': 0.0})
    merged['delta_weight'] = merged['latest_weight'] - merged['prev_weight']
    merged['action'] = merged['delta_weight'].apply(lambda value: '买入' if value > 1e-9 else ('卖出' if value < -1e-9 else '维持'))
    merged['signal_date'] = latest_date
    merged['previous_signal_date'] = previous_date
    merged = merged.rename(columns={'latest_weight': 'target_weight', 'latest_rank': 'rank', 'latest_score': 'score'})
    return merged.sort_values(['action', 'delta_weight'], ascending=[True, False]).reset_index(drop=True)


def _latest_data_date(bundle: dict) -> str:
    for frame_name in ['nav', 'target_weights', 'predictions']:
        frame = bundle.get(frame_name, pd.DataFrame())
        for column in ['trade_date', 'signal_date', 'execution_date']:
            if column in frame.columns and not frame.empty:
                return pd.Timestamp(frame[column].max()).strftime('%Y-%m-%d')
    return 'N/A'


def _to_display_frame(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    available = {key: value for key, value in mapping.items() if key in df.columns}
    return df.rename(columns=available)


def _save_ui_config(config: dict, experiment_name: str) -> Path:
    ui_dir = ARTIFACTS_DIR / 'ui_configs'
    ui_dir.mkdir(parents=True, exist_ok=True)
    safe_name = ''.join(char if char.isalnum() or char in {'_', '-'} else '_' for char in experiment_name).strip('_') or 'ui_experiment'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = ui_dir / f'{safe_name}_{timestamp}.yaml'
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return path


def _artifact_priority(path_str: str) -> tuple[int, str]:
    priority_map = {
        'metadata/experiment_manifest.json': 0,
        'metadata/run_summary.json': 1,
        'reports/run_report.md': 2,
        'reports/factor_diagnostics.md': 3,
        'metadata/artifact_inventory.parquet': 4,
        'metadata/data_contract.json': 5,
    }
    return priority_map.get(path_str, 99), path_str


def _set_nested_config_value(config: dict, dotted_path: str, value) -> None:
    current = config
    parts = dotted_path.split('.')
    for key in parts[:-1]:
        current = current.setdefault(key, {})
    current[parts[-1]] = value


def _parse_sweep_values(raw_values: str, caster) -> list:
    values = []
    for token in raw_values.replace('\n', ',').split(','):
        token = token.strip()
        if not token:
            continue
        values.append(caster(token))
    return values


def _prediction_debug_frame(bundle: dict, signal_date: pd.Timestamp) -> pd.DataFrame:
    predictions = bundle['predictions'].copy()
    if predictions.empty:
        return pd.DataFrame()
    label_panel = bundle.get('label_panel', pd.DataFrame()).copy()
    label_name = bundle['config'].get('label', {}).get('name', 'fwd_return_20d')
    frame = predictions.loc[predictions['trade_date'] == signal_date].copy()
    if not label_panel.empty and label_name in label_panel.columns:
        frame = frame.merge(label_panel[['trade_date', 'symbol', label_name]], on=['trade_date', 'symbol'], how='left')
    return frame.sort_values('score', ascending=False).reset_index(drop=True)


__all__ = [
    'PAGE_OPTIONS', 'WORKSPACE_MODES', 'MONTH_ORDER', 'AVAILABLE_FEATURES',
    'DATA_SOURCE_LABELS', 'DATA_SOURCE_OPTIONS', 'SWEEP_FIELD_OPTIONS',
    '_source_label', '_is_real_source', '_safe_date_input_value',
    '_normalize_ui_date_string', '_inject_theme', '_list_run_dirs',
    '_read_json', '_read_yaml', '_read_parquet', '_read_csv', '_read_text',
    '_load_run_bundle', '_build_ledger', '_format_pct', '_format_currency',
    '_format_number', '_tone_from_value', '_status_snapshot',
    '_build_rebalance_delta', '_latest_data_date', '_to_display_frame',
    '_save_ui_config', '_artifact_priority', '_set_nested_config_value',
    '_parse_sweep_values', '_prediction_debug_frame',
]
