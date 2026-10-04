# QMT Investment Assistant

**A 股量化研究工作台：把因子想法整理成可检查的实验、回测与模拟执行记录。**

[![CI](https://img.shields.io/github/actions/workflow/status/Leo984357/qmt_investment_assistant/ci.yml?branch=main&label=CI&logo=github&style=flat-square)](https://github.com/Leo984357/qmt_investment_assistant/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white&style=flat-square)](pyproject.toml)

[快速开始](#快速开始) · [实验交付物](#一次实验会留下什么) · [方法论](METHODOLOGY.md) · [配置与研究阶段](configs/experiments/README.md)

围绕沪深 300 多因子研究，统一数据接入、因子计算、模型训练、组合构建和成本后评估。每次实验保存配置、数据快照、中间表和评审报告，便于比较候选方案，也便于追溯一个结果来自哪里。

**当前状态：研究与模拟执行。** 已实现 mock 账户与模拟成交；[QMT 适配层](src/adapters/qmt/README.md)仍是待接入接口。

## 项目能做什么

| 研究任务 | 实现与输出 |
|---|---|
| 比较因子与模型 | 配置化因子选择，simple average、Ridge、LightGBM 与自适应 IC 加权 |
| 管理数据与实验 | DuckDB / Parquet 本地数据目录，实验配置与数据快照记录 |
| 评估组合表现 | 佣金、印花税、滑点、调仓延迟与 A 股交易约束；输出净值、交易和持仓 |
| 检查研究证据 | 因子诊断、策略 Gate、运行产物校验，区分发现、验证与 holdout 阶段 |
| 查看与复盘 | Streamlit 研究面板；研究 → 决策 → 模拟执行 → 复盘工作流 |

```text
实验配置 → 数据与因子 → Walk-Forward 模型训练 → 信号与组合
                                                ↓
运行档案 ← 诊断报告与策略 Gate ← 成本后回测与评估
```

## 快速开始

需要 **Python 3.12+**。在项目根目录执行；真实行情实验需要联网，并会在本地建立数据缓存。

```bash
git clone https://github.com/Leo984357/qmt_investment_assistant.git
cd qmt_investment_assistant
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell：.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

**1. 先检查配置。** 此命令检查研究阶段和因子登记情况，不启动行情实验。

```bash
python -m src.cli audit-config \
  --config configs/experiments/hs300_single_close_to_high250.yaml
```

这个示例是 `discovery` 阶段的价格位置 / 长动量候选。审计中的研究阶段提示是预期输出；详细阶段规则见[实验配置说明](configs/experiments/README.md)。

**2. 获取数据并运行实验。** 示例配置的日期范围为 `2022-09-01` 至 `2026-03-28`，研究因子为 `close_to_high250`。首次运行包含行情获取与历史窗口准备，耗时取决于网络和本地缓存。

```bash
python -m src.cli experiment \
  --config configs/experiments/hs300_single_close_to_high250.yaml
```

**3. 检查产物并打开研究面板。**

```bash
python -m src.cli validate-run
python -m src.cli runs --limit 5
python -m src.cli dashboard
```

`validate-run` 默认检查最近一次实验，也可以用 `--run-id` 指定运行记录。它校验运行产物；研究结论仍需结合数据口径、独立验证窗口和策略 Gate 阅读。

## 一次实验会留下什么

产物写入 `artifacts/runs/<run_id>/`。以下是[实验运行器](src/experiment/runner.py)定义的输出结构，实际内容随配置与运行阶段变化。

```text
artifacts/runs/<run_id>/
├── config/resolved_experiment.yaml       # 实际使用的配置
├── features/                            # 因子面板与清单
├── models/                              # 分段评估、重要性与模型记录
├── signals/                             # 预测、信号和目标权重
├── backtest/                            # 净值、基准、成交、持仓与回撤
├── reports/
│   ├── run_report.md                    # 实验总览
│   ├── factor_diagnostics.md            # 因子诊断
│   └── strategy_gate.md                 # 策略评审
└── metadata/                            # 快照、数据合同、运行摘要与产物清单
```

阅读结果时，先看 `run_report.md` 和 `strategy_gate.md`，再沿配置、数据快照和交易记录追溯指标。数据与运行档案默认保存在本地，公开仓库主要提供代码、配置和研究文档。

## 研究状态与结果口径

研究流程使用 `diagnostic → discovery → validation → holdout → production` 阶段标记。候选晋级需要冻结逻辑、独立窗口、成本后评估和完整运行档案；具体要求见[研究阶段与晋级要求](configs/experiments/README.md)。

公开成果以代码、配置和方法文档为主。绩效展示需要同时提供配置、数据时间范围、回测区间、成本、基准和对应 `run_id`；当前仓库未附可独立复核的绩效档案，详见[结果口径说明](docs/presentation-evidence.md)。

## 代码导航

| 入口 | 内容 |
|---|---|
| [`src/cli.py`](src/cli.py) | 实验、审计、产物检查与界面命令 |
| [`src/experiment/`](src/experiment/) | 实验规范、运行器与校验 |
| [`src/features/`](src/features/) / [`src/models/`](src/models/) | 因子与模型 |
| [`src/portfolio/`](src/portfolio/) / [`src/backtest/`](src/backtest/) | 组合与回测 |
| [`src/evaluation/`](src/evaluation/) | 诊断与评估 |
| [`src/services/`](src/services/) / [`src/adapters/`](src/adapters/) | 决策、模拟执行与适配层 |
| [`tests/`](tests/) | 配置、数据、模型与工作流测试 |

开发检查：安装开发依赖后执行 `python -m pytest`。方法与设计背景见 [METHODOLOGY.md](METHODOLOGY.md)。
