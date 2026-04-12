# QMT Investment Assistant - 量化投资研究系统

一个本地优先、可复现、可审计的 A 股组合研究操作系统。

---

## 快速开始

### 1. 安装依赖

```bash
cd /Users/leolee/Desktop/qmt_investment_assistant
pip install -r requirements.txt
```

### 2. 初始化数据 (首次运行)

```bash
python scripts/bootstrap_data.py
```

### 3. 运行实验 (推荐配置)

```bash
# Ridge回归基线 (最高收益)
python -m src.cli experiment --config configs/experiments/hs300_ridge_baseline.yaml

# Ridge+Support因子 (最高Sharpe)
python -m src.cli experiment --config configs/experiments/hs300_ridge_with_support.yaml
```

### 4. 查看结果

```bash
python -m src.cli runs --limit 5
```

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
│   │   ├── hs300_ridge_baseline.yaml           # ⭐ 推荐配置
│   │   ├── hs300_ridge_with_support.yaml       # 最高Sharpe
│   │   ├── hs300_ridge_full_risk.yaml          # 保守风控
│   │   └── ARCHIVE/           # 归档配置
│   └── ...
│
├── scripts/                    # 脚本
│   ├── README.md               # 脚本说明
│   ├── run_experiment.py       # 运行实验 (CLI入口)
│   ├── run_best_strategy.py    # 最佳策略回测
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

## 推荐实验配置

| 配置 | 特点 | 总收益 | Sharpe | 最大回撤 |
|------|------|--------|--------|----------|
| `hs300_ridge_baseline.yaml` ⭐ | 平衡型 | 34.7% | 0.715 | -14.0% |
| `hs300_baseline_rank_sum.yaml` | 简单基线 | 5.3% | 0.333 | -20.5% |

> **修复说明**: 已修复财务因子 point-in-time（避免未来函数）、Ridge embargo（标签隔离）、组合增强器日期错配和单位混用问题。修复后 IC 更可信但略有下降。

### 运行命令

```bash
# 推荐: Ridge回归基线
python -m src.cli experiment --config configs/experiments/hs300_ridge_baseline.yaml

# 最高Sharpe配置
python -m src.cli experiment --config configs/experiments/hs300_ridge_with_support.yaml

# 查看实验列表
python -m src.cli runs --limit 10
```

---

## 研究流程

```
数据 → 因子 → 模型 → 信号 → 组合 → [增强器] → 回测 → 评估
                                    ↑
                        PositionBuffer (减少换手)
                        WeightSmoother (平滑过渡)
                        CostFilter (成本过滤)
```

### 完整流程

1. **数据准备**: `scripts/bootstrap_data.py`
2. **特征构建**: 因子计算 + Z-score标准化
3. **模型训练**: Ridge回归 / LightGBM / 简单平均
4. **信号生成**: 横截面分数排序
5. **组合构建**: Top-N等权 + 风控
6. **组合增强**: PositionBuffer + WeightSmoother + CostFilter
7. **回测验证**: 真实价格 + 成本模拟
8. **评估报告**: IC、收益、回撤、月报

---

## 核心因子 (9个)

| 因子 | IC | IC IR | 类型 |
|------|-----|-------|------|
| alpha_017 | 0.094 | 0.588 | WorldQuant |
| earnings_yield | 0.041 | 0.368 | 财务 |
| roe | 0.036 | 0.320 | 财务 |
| alpha_006 | 0.036 | 0.315 | WorldQuant |
| alpha_002 | 0.030 | 0.227 | WorldQuant |
| ocf_per_share | 0.032 | 0.266 | 财务 |
| operating_margin | 0.022 | 0.256 | 财务 |
| asset_turnover | 0.016 | 0.171 | 财务 |
| equity_growth | 0.016 | 0.162 | 财务 |

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
