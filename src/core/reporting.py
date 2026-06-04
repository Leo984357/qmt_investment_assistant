from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _table_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return '<p><em>无数据</em></p>'
    return df.head(max_rows).to_html(index=False, border=0)


def _table_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return '_无数据_'
    sample = df.head(max_rows)
    try:
        return sample.to_markdown(index=False)
    except Exception:
        # Fallback that avoids optional tabulate dependency.
        return '```\n' + sample.to_string(index=False) + '\n```'


def write_research_report(report_dir: Path, as_of_date: str, metrics: dict, signals: pd.DataFrame, target_portfolio: pd.DataFrame, diagnostics: dict) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {'as_of_date': as_of_date, 'metrics': metrics, 'diagnostics': diagnostics}
    md = report_dir / f'research_report_{as_of_date}.md'
    js = report_dir / f'research_report_{as_of_date}.json'
    html = report_dir / f'research_report_{as_of_date}.html'
    md.write_text(
        f"# 研究报告 {as_of_date}\n\n## 指标\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n\n## 诊断\n{json.dumps(diagnostics, ensure_ascii=False, indent=2)}\n\n## Signals\n{_table_markdown(signals, max_rows=15)}\n\n## Target Portfolio\n{_table_markdown(target_portfolio, max_rows=50)}\n",
        encoding='utf-8'
    )
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    html.write_text(
        f"<html><head><meta charset='utf-8'><title>Research {as_of_date}</title></head><body><h1>研究报告 {as_of_date}</h1><h2>指标</h2><pre>{json.dumps(metrics, ensure_ascii=False, indent=2)}</pre><h2>诊断</h2><pre>{json.dumps(diagnostics, ensure_ascii=False, indent=2)}</pre><h2>Signals</h2>{_table_html(signals)}<h2>Target Portfolio</h2>{_table_html(target_portfolio)}</body></html>",
        encoding='utf-8'
    )
    return {'markdown': str(md), 'json': str(js), 'html': str(html)}


def write_decision_report(report_dir: Path, as_of_date: str, packet) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    md = report_dir / f'decision_report_{as_of_date}.md'
    html = report_dir / f'decision_report_{as_of_date}.html'
    md.write_text(
        f"# 决策报告 {as_of_date}\n\n动作：**{packet.action}**\n\n## 理由\n" + '\n'.join([f'- {x}' for x in packet.rationale]) + "\n\n## 风险\n" + json.dumps(packet.risk_summary, ensure_ascii=False, indent=2) + "\n\n## 调仓差异\n" + _table_markdown(packet.rebalance_delta, max_rows=50),
        encoding='utf-8'
    )
    html.write_text(
        f"<html><head><meta charset='utf-8'><title>Decision {as_of_date}</title></head><body><h1>决策报告 {as_of_date}</h1><p>动作：<strong>{packet.action}</strong></p><h2>理由</h2><ul>{''.join(f'<li>{x}</li>' for x in packet.rationale)}</ul><h2>风险</h2><pre>{json.dumps(packet.risk_summary, ensure_ascii=False, indent=2)}</pre><h2>调仓差异</h2>{_table_html(packet.rebalance_delta, max_rows=50)}</body></html>",
        encoding='utf-8'
    )
    return {'markdown': str(md), 'html': str(html)}


def write_review_report(report_dir: Path, as_of_date: str, review) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    md = report_dir / f'review_report_{as_of_date}.md'
    html = report_dir / f'review_report_{as_of_date}.html'
    md.write_text(
        f"# 复盘报告 {as_of_date}\n\n## 执行摘要\n{json.dumps(review.execution_summary, ensure_ascii=False, indent=2)}\n\n## 偏差原因\n" + '\n'.join([f'- {x}' for x in review.deviation_reasons]) + "\n\n## 后续动作\n" + '\n'.join([f'- {x}' for x in review.next_actions]) + "\n\n## Target vs Actual\n" + _table_markdown(review.target_vs_actual, max_rows=50),
        encoding='utf-8'
    )
    html.write_text(
        f"<html><head><meta charset='utf-8'><title>Review {as_of_date}</title></head><body><h1>复盘报告 {as_of_date}</h1><h2>执行摘要</h2><pre>{json.dumps(review.execution_summary, ensure_ascii=False, indent=2)}</pre><h2>偏差原因</h2><ul>{''.join(f'<li>{x}</li>' for x in review.deviation_reasons)}</ul><h2>后续动作</h2><ul>{''.join(f'<li>{x}</li>' for x in review.next_actions)}</ul><h2>Target vs Actual</h2>{_table_html(review.target_vs_actual, max_rows=50)}</body></html>",
        encoding='utf-8'
    )
    return {'markdown': str(md), 'html': str(html)}
