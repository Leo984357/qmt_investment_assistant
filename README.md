# QMT Investment Assistant

> HS300 多因子选股系统 · LightGBM / Ridge 模型 · QMT 实盘对接

本地优先、可复现、可审计的 A 股量化研究系统。支持从因子研究、模型训练、回测验证到实盘下单的完整流程。

---

## Quick Start

```bash
git clone https://github.com/Leo984357/qmt_investment_assistant.git
cd qmt_investment_assistant
pip install -e ".[dev]"

# 审计配置
python -m src.cli audit-config --config configs/experiments/hs300_single_close_to_high250.yaml

# 运行实验
python -m src.cli experiment --config configs/experiments/hs300_single_close_to_high250.yaml
```

---

## Architecture

```
CLI (src/cli.py)
├── experiment/      实验运行器 — 因子计算 → 模型训练 → 信号生成 → 回测
├── models/          模型族 (Ridge, LightGBM, Ensemble)
├── features/        因子库 — 285+ 因子 (WorldQuant Alpha, Barra, 自定义)
├── portfolio/       组合构建 + 风险增强器
├── backtest/        回测引擎 (佣金/滑点/印花税)
├── evaluation/      评估门控 (IC/IR, 回撤, 夏普, Gate 评审)
├── data_sources/    数据接入 (Baostock, AKShare, 财务数据)
└── risk/            风控引擎 (市场状态感知仓位管理)
```

---

## Key Results

| 模型 | 累计收益 | 基准 (HS300) | 夏普 | 最大回撤 | IC |
|------|---------|-------------|------|---------|----|
| Ridge (9因子) | +42.3% | +11.87% | 1.52 | -8.2% | 0.048 |
| LightGBM (自适应) | +81.87% | +11.87% | 1.88 | -12.5% | 0.035 |

*(2022-09 ~ 2026-03, HS300 成分股, 10天调仓周期)*

### Core Factors

| 因子 | IC | IC IR | 类别 |
|------|-----|-------|------|
| earnings_yield | 0.048 | 0.430 | 价值 |
| roe | 0.041 | 0.363 | 质量 |
| mom250 (close_to_high250) | 0.035 | 0.160 | 动量 |
| ocf_per_share | 0.032 | 0.266 | 质量 |
| operating_margin | 0.027 | 0.307 | 质量 |
| close_to_high250 | 0.023 | 0.110 | 动量 |

---

## Tech Stack

| 领域 | 技术 |
|------|------|
| 语言 | Python 3.12+ |
| 数据 | Baostock / AKShare / Tushare |
| 存储 | Parquet / SQLite / DuckDB |
| ML | LightGBM / Ridge / scikit-learn |
| 回测 | 自研 (全成本模拟) |
| 交易 | QMT (xtquant) |
| 实验 | MLflow / 自研 Artifact 系统 |
| 监控 | Streamlit 仪表盘 |

---

## 项目结构

```
src/
├── cli.py                    CLI 入口
├── experiment/               实验运行器 + 规范 + 验证
├── models/                   Ridge / LightGBM / 集成模型
├── features/                 285+ 因子 (WorldQuant Alpha + Barra + 自定义)
├── portfolio/                组合构建 + 风险暴露增强
├── backtest/                 回测引擎
├── evaluation/               评估指标 + IC 监控 + Gate 评审
├── risk/                     风控引擎
├── data_sources/             多数据源接入
├── data_store/               数据持久化层
├── services/                 决策/执行/研究工作流
└── ui/                       Streamlit 仪表盘
```

---

## 研究流程

```
研究问题 → 因子治理 → 候选生成 → 冻结验证 → Holdout → 晋级 → 监控
```

系统强制 `research_protocol.stage` 阶段管理，防止数据挖掘偏差。

---

## Documentation

- [`METHODOLOGY.md`](METHODOLOGY.md) — 完整方法论（研究哲学、因子系统、评估门控、Walk-Forward）
- [`configs/experiments/README.md`](configs/experiments/README.md) — 实验配置说明
