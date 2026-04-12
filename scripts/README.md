# Scripts 目录说明

## 快速使用

### 运行实验 (最常用)
```bash
# Ridge回归基线 (推荐)
python -m src.cli experiment --config configs/experiments/hs300_ridge_baseline.yaml

# Ridge+Support因子 (最高Sharpe)
python -m src.cli experiment --config configs/experiments/hs300_ridge_with_support.yaml

# 简单平均基线 (对比用)
python -m src.cli experiment --config configs/experiments/hs300_baseline_rank_sum.yaml
```

### 查看实验结果
```bash
python -m src.cli runs --limit 10
```

---

## 脚本分类

### 核心脚本 (Core)

| 脚本 | 说明 |
|------|------|
| `run_experiment.py` | 通过CLI运行实验 (推荐方式) |
| `run_best_strategy.py` | 运行最佳策略回测 (独立脚本) |

**使用方式**:
```bash
python -m src.cli experiment --config configs/experiments/hs300_ridge_baseline.yaml
```

### 数据获取 (Data)

| 脚本 | 说明 | 使用频率 |
|------|------|----------|
| `bootstrap_data.py` | 初始化DuckDB数据目录 | 首次 |
| `fetch_all_data.py` | 获取全部数据 (价格/财务) | 首次/更新 |
| `fetch_analyst_data.py` | 获取分析师预期数据 | 更新 |

### 研究脚本 (Research)

| 脚本 | 说明 | 状态 |
|------|------|------|
| `full_factor_ic_test.py` | 完整因子IC测试 (100因子) | 一次性 |
| `strategy_discovery_v2.py` | 策略发现 v2 | 一次性 |
| `walk_forward_ridge_support.py` | Ridge+Support OOS验证 | 一次性 |
| `cost_sensitivity_analysis.py` | 成本敏感性分析 | 一次性 |
| `multi_model_cost_comparison.py` | 多模型成本对比 | 一次性 |

### 验证工具 (Verification)

| 脚本 | 说明 |
|------|------|
| `verify_alpha_004.py` | 验证alpha_004因子定义 |
| `verify_enhancer_real.py` | 验证组合增强器 |
| `verify_portfolio_enhancer.py` | 验证投资组合增强器 |
| `verify_new_strategy.py` | 验证新策略 |
| `calibrate_cost_filter.py` | 校准CostAlphaFilter |
| `true_walk_forward_oos_v2.py` | 真实Walk-Forward OOS验证 |

### 可视化 (UI)

| 脚本 | 说明 |
|------|------|
| `launch_dashboard.py` | 启动Streamlit研究看板 |

```bash
python scripts/launch_dashboard.py
```

---

## 目录结构

```
scripts/
├── README.md                    # 本文件
├── archive/                     # 归档脚本 (废弃/重复)
│   ├── fetch_data_v2.py         # → 使用 fetch_all_data.py
│   ├── true_walk_forward_oos.py  # → 使用 true_walk_forward_oos_v2.py
│   └── strategy_discovery_pipeline.py
│
├── run_experiment.py            # 运行CLI实验 ⭐
├── run_best_strategy.py         # 最佳策略回测 ⭐
├── bootstrap_data.py            # 初始化数据
├── fetch_all_data.py            # 获取数据
├── launch_dashboard.py          # 启动看板
│
└── [研究/验证脚本]               # 一次性使用
```

---

## 运行示例

### 1. 初始化数据
```bash
python scripts/bootstrap_data.py
```

### 2. 运行实验
```bash
python -m src.cli experiment --config configs/experiments/hs300_ridge_baseline.yaml
```

### 3. 查看结果
```bash
python -m src.cli runs --limit 5
```

### 4. 启动看板
```bash
python scripts/launch_dashboard.py
```

---

## 依赖关系

```
bootstrap_data.py
    ↓
fetch_all_data.py
    ↓
run_experiment.py → configs/experiments/*.yaml
    ↓
launch_dashboard.py → artifacts/runs/<run_id>/
```
