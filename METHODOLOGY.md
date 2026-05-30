# QMT Investment Assistant — 方法论

> 沪深300多因子选股：从研究到可复现评估的完整框架

---

## 1. 概览

QMT Investment Assistant 是一个**本地优先、可复现、可审计**的 A 股量化研究系统。核心任务是验证多因子选股策略在严格的成本假设和点-in-time 约束下是否仍有统计显著的超额收益。

### 设计哲学

- **先基线，再 ML**：任何机器学习模型必须优于 simple average 基线，否则不采用
- **可复现高于一切**：每次实验记录完整的配置、数据快照、中间产物和评估结果
- **诚实面对成本**：佣金 + 印花税 + 滑点全部计入，不做"成本前"的虚假收益
- **研究阶段分离**：diagnostic → discovery → validation → holdout → production，每阶段有独立的结论限制

---

## 2. 统一研究合同

所有实验必须遵守冻结的研究合同 `configs/research_contract_v1.yaml`：

| 项目 | 值 |
|------|-----|
| **股票池** | HS300, point-in-time（使用历史成分股，避免未来函数） |
| **研究周期** | 2022-09-01 ~ 2026-03-28 |
| **调仓频率** | 每 10 个交易日 |
| **标签** | `fwd_return_20d` = `price[T+20] / price[T+1] - 1` |
| **成本假设** | 买入 0.75bp 佣金 + 5bp 滑点；卖出 +10bp 印花税 |
| **最小交易单位** | 100 股（lot_size），最低 2000 元 |
| **可交易过滤** | 剔除 ST、新股、停牌、涨跌停 |

### 为什么要 Point-in-Time 股票池

实盘中，你只能知道**历史这一刻**的沪深300成分股。如果用现在的成分股去算历史收益，就是生存者偏差（survivorship bias）。本项目使用 baostock 的历史成分股 membership 数据，在每期调仓时只使用当时真实的成分股。

### 标签定义细节

```
信号日 T → T+1 可执行 → 持有至 T+20 → 计算收益
label = adj_close[T+20] / adj_close[T+1] - 1
```

关键约束：
- T 日收盘产生信号后，**T+1** 才是理论可执行起点
- 禁止使用 `pct_change(20).shift(-20)` 等忽视执行延迟的旧标签口径
- 标签使用后复权价格（adj_close），考虑分红送转

---

## 3. 研究流水线架构

每一次实验的执行入口是唯一的：

```bash
python -m src.cli experiment --config configs/experiments/<name>.yaml
```

内部流程分为 10 个阶段：

```
数据加载 → 因子计算 → 标签计算 → 数据集组装
    → Walk-Forward 模型训练 → 信号生成 → 组合构建
    → 组合增强 → 回测 → 评估 + 策略门控
```

### 3.1 数据层

数据通过 `LocalResearchCatalog` 统一管理：

```
source (baostock) → bootstrap → silver layer (parquet) → experiment
```

自动 warmup：系统根据最大因子 lookback + 训练窗口长度 + 标签 horizon，向前推算需要的 warmup 交易日，保证所有数据计算时点正确。

核心数据表：

| 表 | 内容 |
|----|------|
| `trade_calendar` | A 股交易日历 |
| `security_master` | 证券 master 表 |
| `daily_bar` | 日线行情（开高低收、成交量、复权因子） |
| `universe_membership` | 历史成分股 membership（point-in-time） |
| `tradability` | 可交易状态（ST、新股、停牌标记） |
| `corporate_actions` | 公司事件（分红、送转） |

### 3.2 因子系统

因子系统是项目中**最大最完善的模块**，包含 ~150 个因子和完整的生命周期管理。

#### 因子档案（Factor Catalog）

每个因子在 `src/features/factor_catalog.py` 中有完整档案：

```python
FactorProfile(
    name='close_to_high250',
    family=FactorFamily.TECHNICAL_PRICE_LEVEL,
    description='收盘价距离250日最高点的比例',
    economic_mechanism='价格位置效应：接近高点的股票有趋势延续倾向',
    expected_signal=FactorSignal.LONG,
    lookback=250,
    inputs=('close',),
    status=FactorStatus.CORE,
)
```

#### 因子状态管理

因子有明确的 6 级状态机，基于 IC 表现自动升降级：

```
CORE → 核心池（IC>0.01, IR>0.1, 稳定3年+）
BACKUP → 备用池
OBSERVE → 观察池
REJECT → 拒绝池（IC<=0）
DEPRECATED → 已退役
POOL → 研究池（未评估）
```

自动拒绝的因子示例（主要基于 A 股实证）：

| 因子 | 拒绝原因 |
|------|---------|
| `vol20/60/120` | A 股波动率溢价不成立 |
| `mom20/60` | 短期动量在 A 股反转效应更强 |
| `rev5/10` | 短期反转 IC 不稳定 |
| `high_low_pos20/60` | IC 显著为负 |
| `alpha_004` 等 | WorldQuant Alpha 在 A 股表现不佳 |

#### 因子 IC 监控与退化检测

每个实验运行时自动启动 `RealizedICMonitor`，根据已兑现的 forward return 计算因子健康度：

| 条件 | 动作 |
|------|------|
| 连续 5 次调仓 IC < 0 | 进入观察 |
| 连续 10 次调仓 IC < 0 | 降权 50% |
| 连续 15 次调仓 IC < 0 | 下线 |
| 单次 IC < -0.03 | 立即下线 |
| 连续 20 次调仓 IC > 0 | 逐步恢复 |

注意：IC 只能在 horizon + delay 已兑现后计算，不能使用"实时 IC"。

#### 核心因子档案（当前）

| 因子 | 方向 | IC | IC IR | 状态 |
|------|------|----|-------|------|
| `earnings_yield` | 价值 | 0.048 | 0.430 | CORE |
| `roe` | 质量 | 0.041 | 0.363 | CORE |
| `operating_margin` | 盈利 | 0.027 | 0.307 | CORE |
| `ocf_per_share` | 现金流 | 0.032 | 0.266 | CORE |
| `mom250` | 动量 | 0.035 | 0.160 | CORE |
| `close_to_high250` | 价格位置 | 0.023 | 0.110 | CORE |

### 3.3 模型系统

支持四种模型，所有模型使用同一套训练窗口协议和标签：

| 模型 | 适用场景 |
|------|---------|
| `simple_average` | 基线：因子得分等权平均 |
| `ic_weighted_average` | 基线：历史 IC 加权 |
| `ridge_regression` | 线性模型：因子权重优化 |
| `lightgbm_regression` | 非线性模型：交互效应捕捉 |

所有模型使用 **Walk-Forward** 训练：

```
每期调仓日:
  1. 用前 500 个交易日的数据训练
  2. 对当前期做预测
  3. 滚动到下一期
```

Walk-Forward 避免了未来信息的泄漏，比全样本训练+回测更接近真实表现。

### 3.4 组合构建

组合构建流程：

```
signal_scores → 过滤候选（universe + tradability）→ 选 Top-N → 等权/加权 → 目标权重
```

风险模型（可配置）：
- `qmt_style_ladder`：四档风控，根据市场动量调整暴露
- 牛市暴露 1.0 → 中等 0.80 → 低 0.55 → 危机 0.40

### 3.5 组合增强

组合增强是一个独立的执行层，与信号预测分离：

```
目标权重 → 持仓缓冲区 → 权重平滑 → 成本过滤 → 增强后权重
```

- **缓冲区**：排名在阈值内的持仓保留，减少无意义换手
- **平滑器**：限制单期权重调整幅度，降低冲击成本
- **成本过滤器**：预期 alpha 不足以覆盖交易成本的调仓被跳过

**重要**：组合增强是执行层优化，不能创造新的 alpha。市场择时、仓位开关等"overlay strategy"必须单独验证和归因。

### 3.6 回测引擎

回测引擎 `src/backtest/engine.py` 模拟真实交易约束：

- 执行价使用当日开盘价（若缺失则 fallback 到收盘价）
- 买入计佣金 + 滑点，卖出再加印花税
- 支持 lot_size（100 股）和 min_trade_value（2000 元）
- 使用 point-in-time tradability 限制买卖
- 处理分红送转等公司事件

成本明细：

| 方向 | 费用 | 费率 |
|------|------|------|
| 买入 | 佣金 | 0.75 bp |
| 买入 | 滑点 | 5 bp |
| **买入总计** | | **1.25 bp** |
| 卖出 | 佣金 | 0.75 bp |
| 卖出 | 印花税 | 10 bp |
| 卖出 | 滑点 | 5 bp |
| **卖出总计** | | **15.75 bp** |

### 3.7 评估与策略门控

每次实验产出完整的评估报告，核心是 **Strategy Gate**（策略准入门控）：

| 门控 | 阈值 | 说明 |
|------|------|------|
| IC 均值 | > 0.02 | 因子预测力 |
| IC IR | > 0.15 | 因子稳定性 |
| 分组单调性 | > 0.60 | Q5 > Q4 > ... > Q1 |
| 超额收益 | > 0 | 成本后跑赢基准 |
| 最大回撤 | < 30% | 风险控制 |
| 平均换手 | < 50% | 可执行性 |
| 年度胜率 | > 60% | 年份稳定性 |
| 对比基线 | Sharpe > 基线 | 相对于 frozen baseline 的提升 |

所有 8 个门控必须全部通过，策略才能标记为通过。

---

## 4. 研究阶段与结论分级

这是本系统**最重要的方法论原则**：策略结论必须明确其所在研究阶段，不同阶段对应不同的结论约束。

```
diagnostic → discovery → validation → holdout → production
```

| 阶段 | 含义 | 允许说什么 |
|------|------|-----------|
| **diagnostic** | 数据/因子诊断 | 不产出策略结论，"发现候选现象" |
| **discovery** | 候选策略发现 | "这是一个 candidate" |
| **validation** | 冻结验证 | "这是一个 validated candidate" |
| **holdout** | 锁定样本 | "这是 holdout 证据" |
| **production** | 已晋级策略 | "在当前研究合同下通过准入" |

### 多重检验控制

策略研究不能只看"最好的一次结果"。本项目要求正式结论中记录：

- 尝试了多少个候选策略（candidate_count）
- 尝试了多少个因子（factor_trial_count）
- 尝试了多少组参数（parameter_trial_count）
- 使用的修正方法（Bonferroni / FDR / 锁定的 holdout）
- 是否通过随机基线、permutation test、白噪声基线验证

### 风险归因

即使 Gate 通过，收益也可能来自风险暴露而非 alpha。必须检查：

- 行业暴露是否解释了收益
- Size / Beta / 流动性暴露
- 因子中性化后是否仍有效
- 主动风险是否集中在少数行业或个股

### Overlay 分离

以下操作属于 **Overlay Strategy**，不能混在"组合增强"里：

- 市场择时仓位控制
- 风险开关
- 行业偏离约束

如果启用 overlay 策略，必须单独验证、单独归因，并登记 hypothesis_id。

---

## 5. 实验产物与可复现

每次正式实验保留完整的 artifact 到 `artifacts/runs/<run_id>/`：

```
├── config/resolved_experiment.yaml    # 完全展开的配置
├── metadata/
│   ├── data_snapshot.json             # 数据快照信息
│   ├── experiment_manifest.json       # 完整实验清单
│   ├── run_summary.json              # 关键指标汇总
│   └── stage_timings.json            # 各阶段耗时
├── features/
│   ├── feature_panel.parquet         # 因子面板数据
│   └── factor_ic_history.parquet     # IC 历史
├── labels/label_panel.parquet
├── datasets/model_dataset.parquet
├── models/
│   ├── split_metrics.csv             # 各期训练/测试指标
│   └── feature_importance.csv        # 特征重要性
├── signals/
│   ├── predictions.parquet           # 模型预测
│   ├── signal_scores.parquet         # 信号得分
│   ├── target_weights.parquet        # 目标权重
│   └── target_weights_enhanced.parquet  # 增强后权重
├── backtest/
│   ├── nav.parquet                   # 策略净值
│   ├── trades.parquet               # 交易明细
│   └── rank_ic.parquet              # Rank IC 序列
└── reports/
    ├── run_report.md                 # 完整实验报告
    ├── strategy_gate.md              # 门控结果
    └── factor_decay_report.md        # 因子退化报告
```

实验中全部中间产物都可追溯，任何结论必须能指向对应的 run_id 和 config_hash。

---

## 6. 关键结果摘要

### Ridge 回归基线（推荐起点）

```yaml
features: [roe, earnings_yield, operating_margin, equity_growth,
           ocf_per_share, revenue_growth, asset_turnover, gross_margin, cash_ratio]
model: ridge_regression (alpha=1.0)
portfolio: top-15, 10日调仓
```

| 指标 | 值 |
|------|-----|
| 总收益 | 64.5% |
| Sharpe | 1.066 |
| 最大回撤 | -13.4% |

### Ridge + Support 因子（最高 Sharpe）

额外加入 `mom120` + `vol20`，Sharpe 提升至 1.277，回撤降至 -9.0%。

### Ridge + 四档风控（最保守）

启用 `qmt_style_ladder` 后，Sharpe 0.70，回撤 -19.1%，收益 35.1%。

---

## 7. 重要限制与风险披露

1. **回测不等于实盘**：当前滑点模型未考虑冲击成本、盘口深度和容量限制
2. **成本估算为近似**：实际佣金、印花税因券商和账户规模而异
3. **没有 Live Trading 验证**：策略实盘表现可能与回测有显著差异
4. **样本有限**：研究周期覆盖约 3.5 年，未经历完整牛熊周期
5. **多重检验**：已导出多个组合配置，当前最优结果存在过拟合风险

---

*生成于 2026-05-30 · 对应研究合同 research_contract_v1*
