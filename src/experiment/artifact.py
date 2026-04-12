"""
实验结果工件 (Experiment Artifact)

记录实验的可复现信息，包括:
- run_id: 唯一运行标识
- config_hash: 实验配置哈希
- data_snapshot_hash: 数据快照哈希
- key_metrics: 关键指标
- gate_result: 策略门控结果
- timestamp: 运行时间

使用方式:
    from src.experiment.artifact import ExperimentArtifact, save_artifact
    
    artifact = ExperimentArtifact(
        experiment_name='hs300_ridge_final',
        spec=experiment_spec,
        data_snapshot_hash='abc123...',
        metrics={'sharpe': 1.066, 'total_return': 0.645},
        gate_result=gate_result,
    )
    save_artifact(artifact, output_dir)
"""
from __future__ import annotations
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml


@dataclass
class ExperimentMetrics:
    """实验关键指标"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    annual_return: float
    annual_volatility: float
    ic_mean: float
    ic_ir: float
    avg_turnover: float
    total_cost: float
    num_trades: int
    win_rate: float
    excess_return: float
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> ExperimentMetrics:
        return cls(**data)


@dataclass
class ExperimentArtifact:
    """实验结果工件"""
    run_id: str
    experiment_name: str
    config_hash: str
    data_snapshot_hash: str
    research_contract_version: str
    timestamp: str
    metrics: ExperimentMetrics
    gate_passed: bool
    gate_score: float
    gate_details: dict
    feature_names: list[str]
    model_family: str
    backtest_start: str
    backtest_end: str
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    parent_run_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'config_hash': self.config_hash,
            'data_snapshot_hash': self.data_snapshot_hash,
            'research_contract_version': self.research_contract_version,
            'timestamp': self.timestamp,
            'metrics': self.metrics.to_dict(),
            'gate_passed': self.gate_passed,
            'gate_score': self.gate_score,
            'gate_details': self.gate_details,
            'feature_names': self.feature_names,
            'model_family': self.model_family,
            'backtest_start': self.backtest_start,
            'backtest_end': self.backtest_end,
            'notes': self.notes,
            'tags': self.tags,
            'parent_run_id': self.parent_run_id,
        }
    
    def to_markdown(self) -> str:
        lines = [
            f"# Experiment Artifact: {self.experiment_name}",
            f"",
            f"**Run ID:** `{self.run_id}`",
            f"**Timestamp:** {self.timestamp}",
            f"**Config Hash:** `{self.config_hash[:16]}...`",
            f"",
            f"## Gate Status",
            f"",
            f"{'✅ PASSED' if self.gate_passed else '❌ FAILED'} (Score: {self.gate_score:.1f}/100)",
            f"",
            f"## Key Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Return | {self.metrics.total_return:.2%} |",
            f"| Sharpe Ratio | {self.metrics.sharpe_ratio:.3f} |",
            f"| Max Drawdown | {self.metrics.max_drawdown:.2%} |",
            f"| Annual Return | {self.metrics.annual_return:.2%} |",
            f"| Annual Volatility | {self.metrics.annual_volatility:.2%} |",
            f"| IC Mean | {self.metrics.ic_mean:.4f} |",
            f"| IC IR | {self.metrics.ic_ir:.4f} |",
            f"| Avg Turnover | {self.metrics.avg_turnover:.2%} |",
            f"| Total Cost | ¥{self.metrics.total_cost:,.0f} |",
            f"| Win Rate | {self.metrics.win_rate:.1%} |",
            f"| Excess Return | {self.metrics.excess_return:.2%} |",
            f"",
            f"## Configuration",
            f"",
            f"- **Features:** {', '.join(self.feature_names)}",
            f"- **Model:** {self.model_family}",
            f"- **Period:** {self.backtest_start} to {self.backtest_end}",
            f"- **Research Contract:** {self.research_contract_version}",
            f"",
        ]
        
        if self.tags:
            lines.append(f"## Tags")
            lines.append(f"")
            for tag in self.tags:
                lines.append(f"- {tag}")
            lines.append("")
        
        if self.notes:
            lines.append(f"## Notes")
            lines.append(f"")
            lines.append(self.notes)
            lines.append("")
        
        return "\n".join(lines)
    
    @classmethod
    def from_dict(cls, data: dict) -> ExperimentArtifact:
        data = dict(data)
        data['metrics'] = ExperimentMetrics.from_dict(data['metrics'])
        return cls(**data)


def compute_config_hash(spec_dict: dict) -> str:
    """计算配置哈希"""
    # 排除随机种子等非确定性字段
    stable_dict = {k: v for k, v in spec_dict.items() if k not in ['seed', 'random_state']}
    payload = json.dumps(stable_dict, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def compute_data_hash(snapshot_id: str, universe_name: str, start_date: str, end_date: str) -> str:
    """计算数据快照哈希"""
    payload = f"{snapshot_id}:{universe_name}:{start_date}:{end_date}"
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def create_artifact_from_experiment(
    experiment_name: str,
    spec: Any,
    nav: pd.DataFrame,
    trades: pd.DataFrame,
    rank_ic: pd.DataFrame,
    benchmark_nav: Optional[pd.DataFrame],
    gate_result: Any,
    notes: str = "",
    tags: Optional[list[str]] = None,
    parent_run_id: Optional[str] = None,
) -> ExperimentArtifact:
    """从实验结果创建工件"""
    # 计算哈希
    config_hash = compute_config_hash(spec.flattened_params())
    data_hash = compute_data_hash(
        spec.data.snapshot_id,
        spec.data.universe_name,
        spec.data.start_date,
        spec.data.end_date,
    )
    
    # 计算指标
    metrics = _compute_metrics(nav, trades, rank_ic, benchmark_nav)
    
    # 获取回测时间范围
    backtest_start = nav['trade_date'].min() if not nav.empty else ""
    backtest_end = nav['trade_date'].max() if not nav.empty else ""
    
    return ExperimentArtifact(
        run_id=str(uuid.uuid4())[:8],
        experiment_name=experiment_name,
        config_hash=config_hash,
        data_snapshot_hash=data_hash,
        research_contract_version='v1',
        timestamp=datetime.now().isoformat(),
        metrics=metrics,
        gate_passed=gate_result.passed if gate_result else False,
        gate_score=gate_result.overall_score if gate_result else 0,
        gate_details=gate_result.to_dict() if gate_result else {},
        feature_names=spec.features.names,
        model_family=spec.model.family,
        backtest_start=str(backtest_start)[:10] if backtest_start else "",
        backtest_end=str(backtest_end)[:10] if backtest_end else "",
        notes=notes,
        tags=tags or [],
        parent_run_id=parent_run_id,
    )


def _compute_metrics(
    nav: pd.DataFrame,
    trades: pd.DataFrame,
    rank_ic: pd.DataFrame,
    benchmark_nav: Optional[pd.DataFrame],
) -> ExperimentMetrics:
    """计算实验关键指标"""
    # 净值指标
    if nav.empty:
        return ExperimentMetrics(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            annual_return=0.0,
            annual_volatility=0.0,
            ic_mean=0.0,
            ic_ir=0.0,
            avg_turnover=0.0,
            total_cost=0.0,
            num_trades=0,
            win_rate=0.0,
            excess_return=0.0,
        )
    
    nav_series = nav.set_index('trade_date')['nav']
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1 if len(nav_series) > 1 else 0.0
    
    # 最大回撤
    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax
    max_drawdown = abs(drawdown.min())
    
    # 日收益统计
    daily_returns = nav_series.pct_change().dropna()
    annual_return = daily_returns.mean() * 252
    annual_volatility = daily_returns.std() * (252 ** 0.5)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    
    # IC指标
    ic_mean = rank_ic['rank_ic'].mean() if not rank_ic.empty else 0.0
    ic_std = rank_ic['rank_ic'].std() if not rank_ic.empty else 1.0
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    
    # 交易指标
    total_cost = trades['cost'].sum() if not trades.empty and 'cost' in trades.columns else 0.0
    num_trades = len(trades) if not trades.empty else 0
    
    # 估算平均换手率
    if not trades.empty and 'trade_value' in trades.columns and 'execution_date' in trades.columns:
        avg_turnover = trades.groupby('execution_date')['trade_value'].sum().mean()
    else:
        avg_turnover = 0.0
    
    # 胜率
    if len(daily_returns) > 0:
        win_rate = (daily_returns > 0).sum() / len(daily_returns)
    else:
        win_rate = 0.0
    
    # 超额收益
    if benchmark_nav is not None and not benchmark_nav.empty:
        strategy_ret = total_return
        # Handle different column names: 'nav' or 'benchmark_nav'
        if 'nav' in benchmark_nav.columns:
            benchmark_col = 'nav'
        elif 'benchmark_nav' in benchmark_nav.columns:
            benchmark_col = 'benchmark_nav'
        else:
            benchmark_ret = 0.0
            excess_return = 0.0
            benchmark_nav = None
        if benchmark_nav is not None:
            benchmark_ret = benchmark_nav[benchmark_col].iloc[-1] / benchmark_nav[benchmark_col].iloc[0] - 1
            excess_return = strategy_ret - benchmark_ret
    else:
        excess_return = 0.0
    
    return ExperimentMetrics(
        total_return=float(total_return),
        sharpe_ratio=float(sharpe_ratio),
        max_drawdown=float(max_drawdown),
        annual_return=float(annual_return),
        annual_volatility=float(annual_volatility),
        ic_mean=float(ic_mean),
        ic_ir=float(ic_ir),
        avg_turnover=float(avg_turnover),
        total_cost=float(total_cost),
        num_trades=int(num_trades),
        win_rate=float(win_rate),
        excess_return=float(excess_return),
    )


def save_artifact(artifact: ExperimentArtifact, output_dir: Path) -> Path:
    """保存工件到文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON
    json_path = output_dir / f"{artifact.run_id}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(artifact.to_dict(), f, ensure_ascii=False, indent=2)
    
    # 保存Markdown
    md_path = output_dir / f"{artifact.run_id}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(artifact.to_markdown())
    
    # 保存YAML
    yaml_path = output_dir / f"{artifact.run_id}.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(artifact.to_dict(), f, allow_unicode=True, sort_keys=False)
    
    return json_path


def load_artifact(path: Path) -> ExperimentArtifact:
    """加载工件"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return ExperimentArtifact.from_dict(data)


def load_artifacts_from_dir(dir_path: Path) -> list[ExperimentArtifact]:
    """从目录加载所有工件"""
    artifacts = []
    for json_file in Path(dir_path).glob('*.json'):
        try:
            artifacts.append(load_artifact(json_file))
        except Exception:
            pass
    return sorted(artifacts, key=lambda a: a.timestamp, reverse=True)


def find_recommended_artifacts(dir_path: Path) -> list[ExperimentArtifact]:
    """找出所有通过门控的工件"""
    artifacts = load_artifacts_from_dir(dir_path)
    return [a for a in artifacts if a.gate_passed]
