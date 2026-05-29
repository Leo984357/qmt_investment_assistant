# Agent Workflow Guide - 研究管线入口规范

**本文件定义：唯一正式入口、废弃入口、diagnostic边界、策略晋级协议**

---

## 唯一正式实验入口

```
python -m src.cli experiment --config configs/experiments/<name>.yaml
```

这是唯一允许产出**正式研究结论**的入口。

---

## 废弃入口 (禁止产出正式结论)

| 入口 | 状态 | 原因 |
|------|------|------|
| `src/research_pipeline.py` | ❌ 废弃 | 旧研究流，标签合同可能不一致 |
| `src/research_orchestrator.py` | ❌ 废弃 | 旧编排流，有未来函数风险 |
| `scripts/*_ic_test.py` | ❌ 废弃 | 旧标签口径 pct_change(20).shift(-20) |
| `scripts/batch_calculate_*.py` | ❌ 废弃 | 未跟踪脚本，标签合同未验证 |
| `scripts/complete_*.py` | ❌ 废弃 | 一次性脚本，不可复现 |

这些入口**不能**用于：
- 产出IC/IR结论
- 与正式实验比较
- 作为production配置

---

## Diagnostic入口 (仅用于诊断)

Diagnostic配置用于**因子探索和对比研究**，结论不可直接用于生产。

```
python -m src.cli experiment --config configs/experiments/diagnostic/<name>.yaml
```

Diagnostic配置特征：
- `registry_stage: diagnostic` 或 `allow_rejected_factors: true`
- 路径包含 `diagnostic/`
- 结论需经过正式研究流程验证后才能用于生产

**当前Diagnostic配置：**
- `configs/experiments/diagnostic/hs300_technical_factors.yaml`
- `configs/experiments/diagnostic/hs300_allowed_factors.yaml`

---

## 配置分类规则

### 研究协议阶段

所有正式实验配置必须显式或隐式归入 `research_protocol.stage`：

| 阶段 | 用途 | 是否可产出策略结论 |
|------|------|------------------|
| `diagnostic` | 数据、因子、管线诊断 | 否 |
| `discovery` | 候选策略发现 | 否，只能叫 candidate |
| `validation` | 冻结逻辑后的独立验证 | 否，只能叫 validated candidate |
| `holdout` | 锁定样本最终证据 | 否，作为晋级证据 |
| `production` | 已晋级策略 | 是 |

`model.registry_stage: research` 且未写 `research_protocol` 时，会自动降级为 `discovery`。  
`model.registry_stage: diagnostic` 且未写 `research_protocol` 时，会自动归入 `diagnostic`。  
`model.registry_stage: production` 必须显式写安全的 `research_protocol`，否则加载失败。

### Production配置 (可产出正式结论)

```yaml
# ✅ 正确
name: hs300_ridge_baseline
model:
  registry_stage: production
allow_rejected_factors: false
research_protocol:
  stage: production
  hypothesis_id: <stable_hypothesis_id>
  candidate_id: <validated_candidate_id>
  data_mined: false
  frozen_after: 'YYYY-MM-DD'
  validation_window:
    start: 'YYYY-MM-DD'
    end: 'YYYY-MM-DD'
  holdout_window:
    start: 'YYYY-MM-DD'
    end: 'YYYY-MM-DD'
  allowed_change_after_freeze: none
```

```yaml
# ❌ 错误 - rejected因子不允许在production
name: hs300_diagnostic
model:
  registry_stage: production
allow_rejected_factors: true  # 禁止！
```

```yaml
# ❌ 错误 - data-mined结果不能直接production
model:
  registry_stage: production
research_protocol:
  stage: production
  data_mined: true  # 禁止！
```

### Diagnostic配置 (仅探索)

```yaml
# ✅ 正确
name: hs300_technical_factors
description: [DIAGNOSTIC ONLY] 技术因子探索
registry_stage: diagnostic
allow_rejected_factors: true
```

### Discovery配置 (候选生成)

```yaml
name: hs300_candidate_example
model:
  registry_stage: research
research_protocol:
  stage: discovery
  hypothesis_id: <economic_mechanism_id>
  candidate_id: <candidate_strategy_id>
  data_mined: true
  discovery_window:
    start: 'YYYY-MM-DD'
    end: 'YYYY-MM-DD'
  validation_window:
    start: 'YYYY-MM-DD'
    end: 'YYYY-MM-DD'
  holdout_window:
    start: 'YYYY-MM-DD'
    end: 'YYYY-MM-DD'
  allowed_change_after_freeze: research_only
```

Discovery结果只能用于提出候选，不允许写成“优秀策略”“推荐实盘”。

### 策略晋级顺序

```
diagnostic → discovery → validation → holdout → production
```

晋级要求：
- discovery可以改逻辑，但必须承认 `data_mined: true`
- validation开始前必须冻结因子、模型、组合、成本和窗口
- validation后不得因为结果继续调因子或调参数
- holdout只能作为最终证据，不允许边看边改
- production必须 `data_mined: false`，并保留 validation/holdout 窗口记录
- production必须通过因子档案、Gate、Artifact、成本压力、年度分段、多重检验、风险归因审查

### 多重检验 / 数据挖掘惩罚

只要试过多个因子、多个窗口、多个模型、多个参数或多个组合规则，就必须记录搜索空间：

```yaml
multiple_testing:
  enabled: true
  search_space_id: <search_space_id>
  candidate_count: <候选策略数量>
  factor_trial_count: <因子试验数量>
  parameter_trial_count: <参数/模型试验数量>
  correction_method: <bonferroni|fdr|deflated_sharpe|locked_holdout|...>
  random_baseline: true
  permutation_test: true
  white_noise_baseline: true
  stability_over_best: true
```

production/holdout 结论不允许只引用“最好一次回测”。必须说明这个结果相对于搜索空间、随机基准和稳定性是否仍然成立。

### 风险模型 / 暴露归因

production/holdout 结论必须配置：

```yaml
risk_attribution:
  enabled: true
  industry_exposure: true
  style_exposure: true
  beta_exposure: true
  benchmark_relative_active_risk: true
  neutralized_ic: true
  return_attribution: true
```

结论里必须回答：收益来自 alpha，还是行业、size、beta、流动性、主题暴露。

### Execution Enhancement vs Overlay Strategy

`PositionBuffer`、`WeightSmoother`、`CostFilter` 属于 execution enhancement。  
`RegimeExposure`、仓位择时、风险开关属于 overlay strategy。

Overlay 必须单独声明：

```yaml
overlay:
  enabled: true
  regime_exposure_enabled: true
  hypothesis_id: <overlay_hypothesis_id>
  data_mined: false
  frozen_after: 'YYYY-MM-DD'
```

禁止把 overlay 混成“增强器”后直接把收益算作原 alpha。

---

## 标签合同 (必须遵守)

**正式研究合同** (src/labels/definitions.py):

```
fwd_return_20d = price[T+20] / price[T+1] - 1
```

禁止使用：
- `pct_change(20).shift(-20)` ❌ (标签从T开始，漏了一天)
- `shift(-20) / shift(0) - 1` ❌ (T日收盘价成交，不是T+1)

---

## 因子真相源

| 来源 | 数量 | 状态 |
|------|------|------|
| `src/features/simple_definitions.py` | 203个 | ✅ **可计算主因子库** |
| `src/features/factor_catalog.py` | 144个 | ✅ 有研究状态的因子档案 |
| `src/features/factor_pool.py` | 667个 (637唯一名, 25重复) | ⚠️ 描述池，**不可直接实验** |

### 667池去重

```python
from src.features.factor_pool import get_pool_duplicates, check_pool_name

# 检查重复
dups = get_pool_duplicates()  # 25个重复名

# 检查因子名
result = check_pool_name('earnings_yield')
# {'is_duplicate': True, 'sources': ['barra', 'extended_financial'], ...}
```

### 因子真相源说明

**Production可用**: `simple_definitions.py` (203因子) + `factor_catalog.py` (144因子档案)
- 203个因子可直接计算
- 144个因子有完整档案(IC、IR、状态、失败模式)
- Core因子: 6个 (earnings_yield, roe, operating_margin, ocf_per_share, mom250, close_to_high250)

**描述池** (`factor_pool.py`): ⚠️ 不能直接使用
- 667条记录包含25个重复名
- 74个因子与simple_definitions交集
- 129个simple因子不在667池中
- 主要用于因子探索时的参考描述，不能作为实验依据

**Production配置规则**：
- 因子必须在 `simple_definitions.py` 可计算
- 因子应在 `factor_catalog.py` 有状态记录 (Core/Backup/Observe)
- 不允许使用 `REJECT` 状态的因子做生产实验
- 诊断配置允许使用 `REJECT` 因子，但结论不可用于生产

---

## 脚本分类

```
scripts/
├── production/           # ❌ 不存在，不允许自定义脚本做正式研究
├── diagnostic/           # ⚠️ 一次性诊断脚本，需标注标签合同
├── archive/              # 🗃️ 废弃脚本，不可产出正式结论
│   └── deprecated_label_contract/
└── *.py                  # 仅数据获取、启动看板等基础设施脚本
```

**禁止**：
- 自行编写脚本计算IC/IR用于正式结论
- 使用未归档的旧脚本与正式实验比较
- 修改研究管线代码绕过本规范
- 把 discovery/diagnostic 的好结果称为正式策略
- 用同一段样本反复搜索后再声称通过验证

---

## Artifact审计要求

正式实验的artifact必须包含：

```yaml
artifact:
  entrypoint: src.cli experiment          # 入口
  label_contract: fwd_return_20d          # 标签公式
  label_horizon: 20                       # 持有期
  execution_delay: 1                      # T+1成交
  factor_contract_status:                 # 因子状态
    - factor: earnings_yield
      status: core
    - factor: mom120
      status: diagnostic_only  # 不能直接用于生产
  diagnostic_flags:
    allow_rejected_factors: false         # 是否使用rejected因子
    enhancer_enabled: true                # 是否启用增强器
  gate_passed: true                       # Gate评估结果
```

---

## Agent任务模板

### 新因子探索

```yaml
# configs/experiments/diagnostic/<new_factor_name>.yaml
name: diagnostic_<new_factor>
description: [DIAGNOSTIC ONLY] 探索XXX因子
registry_stage: diagnostic
allow_rejected_factors: true  # 新因子默认为rejected

features:
  names:
    - <new_factor>  # 需先在factor_catalog登记
```

### 新策略候选

1. **第一步**: Diagnostic验证IC > 0.02, IR > 0.1
2. **第二步**: Walk-Forward OOS验证
3. **第三步**: 更新factor_catalog状态
4. **第四步**: 创建production配置

---

## 因子退化监控

**⚠️ 重要**: 不要叫"实时IC"，要用"滞后兑现IC"

### 核心约束

```
IC需要未来收益，在调仓日D最多只能更新已经兑现的IC。

已兑现条件: signal_date + horizon + trade_delay <= current_date
```

### 模块

| 模块 | 路径 | 职责 |
|------|------|------|
| RealizedICMonitor | `src/features/ic_monitor.py` | 滞后兑现IC计算 + 状态机 |
| FactorResponseMonitor | `src/features/factor_response.py` | 状态机 |
| FactorDecayAnalyzer | `src/features/factor_decay.py` | 离线分析 |

### 快速响应规则

| 规则 | 触发 | 动作 |
|------|------|------|
| ic_consecutive_negative_5_obs | 5次调仓IC<0 | 观察 |
| ic_consecutive_negative_10_obs | 10次调仓IC<0 | 降权50% |
| ic_consecutive_negative_15_obs | 15次调仓IC<0 | 下线 |
| ic_below_minus_003 | IC<-0.03 | 立即下线 |

### 权重应用

- **simple_average/weighted_average**: 直接按权重调整得分
- **ridge_regression**: 下次训练剔除offline因子
- **lightgbm**: 下次训练禁用offline因子，不强行改单列权重

### 快速检查清单

Agent执行任务前检查：

- [ ] 配置入口是否为 `python -m src.cli experiment --config`
- [ ] `research_protocol.stage` 是否匹配任务目的
- [ ] production是否显式 `data_mined: false`
- [ ] production是否包含 `frozen_after`、`validation_window`、`holdout_window`
- [ ] holdout/production是否包含 `multiple_testing`
- [ ] holdout/production是否包含 `risk_attribution`
- [ ] 是否把 RegimeExposure/择时开关声明为 `overlay`
- [ ] 配置中无 `allow_rejected_factors: true`（除非路径含diagnostic）
- [ ] 因子在 `simple_definitions.py` 可计算
- [ ] 因子在 `factor_catalog.py` 有状态
- [ ] 脚本未使用旧标签口径
- [ ] **新增IC计算时使用滞后兑现IC，不是实时IC**

---

## 违规处理

如果发现违反本规范的实验：
1. 标记为 `diagnostic_only` 结论
2. 通知维护者审查
3. 不纳入正式策略对比基准
