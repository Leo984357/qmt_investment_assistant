# QMT Investment Assistant - 量化投资研究系统

一个本地优先、可复现、可审计的 A 股组合研究操作系统。

---

## 唯一正式入口 (Agent必读)

```
python -m src.cli experiment --config configs/experiments/<name>.yaml
```

**禁止用于正式结论的入口**：
- ❌ `src/research_pipeline.py` - 废弃
- ❌ `src/research_orchestrator.py` - 废弃
- ❌ `scripts/*_ic_test.py` - 旧标签口径
- ❌ `scripts/archive/*` - 不可复现

**详细规范**: [docs/AGENT_WORKFLOW.md](docs/AGENT_WORKFLOW.md)

---

## 快速开始

### 1. 安装依赖

```bash
cd /Users/leolee/Desktop/qmt_investment_assistant
pip install -r requirements.txt
```

### 2. 审计配置

```bash
python -m src.cli audit-config --config configs/experiments/hs300_single_close_to_high250.yaml
```

### 3. 运行实验

```bash
# discovery候选，不是正式策略
python -m src.cli experiment --config configs/experiments/hs300_single_close_to_high250.yaml

# 查看实验结果
python -m src.cli runs --limit 5
```

### 4. 配置阶段

所有实验必须区分 `research_protocol.stage`：

| stage | 含义 | 允许结论 |
|------|------|----------|
| diagnostic | 诊断 | 不产出策略结论 |
| discovery | 候选发现 | 只能称为 candidate |
| validation | 冻结验证 | 只能称为 validated candidate |
| holdout | 锁定样本 | 只能作为晋级证据 |
| production | 已晋级 | 当前研究合同下的正式策略 |

---

## 项目结构

```
qmt_investment_assistant/
├── README.md                    # 本文件
├── AGENTS.md                   # Agent指令 (详细研究流程)
├── requirements.txt            # 依赖
│
├── configs/                    # 配置文件
│   ├── README.md               # 配置说明
│   ├── experiments/            # 实验配置
│   │   ├── README.md          # 实验配置说明
│   │   ├── hs300_single_close_to_high250.yaml  # discovery候选
│   │   ├── hs300_mom_volume.yaml               # discovery候选
│   │   ├── diagnostic/                         # 诊断配置
│   │   └── ARCHIVE/           # 归档配置
│   └── ...
│
├── scripts/                    # 脚本
│   ├── README.md               # 脚本说明
│   ├── README.md               # 脚本说明
│   └── archive/                # 归档脚本
│
├── src/                       # 源代码
│   ├── cli.py                 # CLI入口
│   ├── experiment/             # 实验运行器
│   ├── models/                 # 模型 (Ridge, LightGBM, SimpleAverage)
│   ├── portfolio/              # 组合构建 + 增强器
│   ├── features/               # 因子定义
│   └── backtest/               # 回测引擎
│
├── data/                      # 数据目录
│   └── silver/                 # 清洗后数据
│
├── artifacts/                 # 实验产物
│   ├── runs/                  # 实验运行结果
│   └── reports/               # 实验报告
│
└── mlruns/                   # MLflow追踪
```

---

## 当前策略状态

当前没有可以直接称为 production 的策略。已有好结果必须先视为 discovery candidate，再经过冻结验证和 locked holdout。

| 配置 | 阶段 | 说明 |
|------|------|------|
| `hs300_single_close_to_high250.yaml` | discovery | 单因子价格位置/长动量候选 |
| `hs300_mom_volume.yaml` | discovery | 动量 + 成交量确认候选，含未入档因子 |

### 运行命令

```bash
# 先审计，再运行
python -m src.cli audit-config --config configs/experiments/hs300_single_close_to_high250.yaml
python -m src.cli experiment --config configs/experiments/hs300_single_close_to_high250.yaml

# 查看实验列表
python -m src.cli runs --limit 10
```

---

## 研究流程

```
研究问题 → 假设 → 因子治理 → 候选生成 → 冻结协议 → 验证 → holdout → 晋级 → 监控
```

### 完整流程

1. **研究问题**: 明确市场、股票池、频率、标签、成本、基准和成功标准。
2. **假设登记**: 用经济机制解释候选信号，不能用回测收益倒推故事。
3. **因子治理**: 因子必须可计算、时点干净，并在 catalog 中登记状态。
4. **候选生成**: discovery 阶段允许探索，但必须标记 `data_mined: true`。
5. **冻结验证**: validation 前冻结因子、模型、参数、组合和成本假设。
6. **Locked Holdout**: holdout 只作为最终证据，不允许边看边改。
7. **多重检验控制**: 记录试过多少因子、参数、模型和候选策略，并用随机基准、permutation 或白噪声基线惩罚数据挖掘。
8. **风险归因**: 检查收益是否来自行业、size、beta、流动性或主题暴露，而不是 alpha。
9. **Overlay 分离**: 缓冲、平滑、成本过滤是执行增强；市场择时和仓位开关是 overlay strategy，必须单独验证。
10. **晋级生产**: production 必须显式 `data_mined: false` 并通过 Gate/Artifact/审计。
11. **退化监控**: 用滞后兑现 IC 跟踪因子健康。

---

## 核心因子 (6个)

**因子档案**: `src/features/factor_catalog.py` (144因子)

| 因子 | IC | IC IR | 状态 |
|------|-----|-------|------|
| earnings_yield | 0.048 | 0.430 | CORE |
| roe | 0.041 | 0.363 | CORE |
| operating_margin | 0.027 | 0.307 | CORE |
| ocf_per_share | 0.032 | 0.266 | CORE |
| mom250 | 0.035 | 0.160 | CORE |
| close_to_high250 | 0.023 | 0.110 | CORE |

**完整因子档案**: `python -c "from src.features.factor_catalog import build_default_catalog; c=build_default_catalog(); print(c.inventory())"`

---

## 关键配置参数

### Ridge回归基线

```yaml
# configs/experiments/hs300_ridge_baseline.yaml
model:
  family: ridge_regression
  params:
    alpha: 1.0

features:
  names: [roe, earnings_yield, operating_margin, equity_growth,
          ocf_per_share, revenue_growth, asset_turnover, gross_margin, cash_ratio]

portfolio:
  top_n: 15
  rebalance_days: 10

backtest:
  initial_cash: 250000
  commission_bps: 0.75
  stamp_duty_bps: 10
  slippage_bps: 5
```

### 成本计算

- 买入: 佣金 0.075% + 滑点 0.05% = 0.125%
- 卖出: 佣金 0.075% + 印花税 0.1% + 滑点 0.05% = 0.225%

---

## 实验产物

运行后结果保存在 `artifacts/runs/<run_id>/`:

```
<run_id>/
├── config/                     # 实验配置
├── metadata/                   # 元数据
├── features/                   # 因子数据
├── labels/                     # 标签数据
├── datasets/                   # 模型数据集
├── models/                     # 模型文件
├── signals/                    # 信号
│   ├── predictions.parquet
│   ├── target_weights.parquet
│   └── target_weights_enhanced.parquet  # 增强后权重
├── backtest/                  # 回测结果
│   ├── nav.parquet            # 净值曲线
│   ├── trades.parquet         # 交易明细
│   └── monthly_returns.parquet
└── reports/                   # 报告
    └── run_report.md
```

---

## 常见问题

### Q: 如何增加新的因子?
编辑 `src/features/simple_definitions.py` 添加因子定义。

### Q: 如何修改风控参数?
编辑实验配置文件中的 `portfolio` 部分:
```yaml
portfolio:
  risk_model: qmt_style_ladder    # 四档仓位风控
  risk_mid_exposure: 0.80        # 中等市场仓位
  risk_low_exposure: 0.55        # 低市场仓位
```

### Q: 如何切换模型?
```yaml
model:
  family: ridge_regression        # 可选: simple_average, lightgbm
  params:
    alpha: 1.0                    # Ridge正则化参数
```

---

## 研究合同 (冻结)

| 项目 | 值 |
|------|-----|
| 股票池 | HS300, point-in-time |
| 时间范围 | 2022-09-01 ~ 2026-03-28 |
| 调仓周期 | 10天 |
| 标签定义 | fwd_return_20d |
| 成本 | 佣金0.75bp + 印花税10bp + 滑点5bp |

---

## 相关文档

- `AGENTS.md` - Agent指令 (详细研究流程、因子健康检查、Walk-Forward验证)
- `configs/experiments/README.md` - 实验配置详解
- `scripts/README.md` - 脚本使用说明
