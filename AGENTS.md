# QMT Investment Assistant - Agent Instructions

本文件是项目级操作规范。目标是防止后续修改、研究、评估时误用旧脚本、旧标签、旧因子结论或不一致的训练/回测口径。所有研究结论必须以当前正式管线和当前研究合同为准。

## 0. 项目定位

本项目是沪深300多因子选股的量化研究与回测平台。核心任务是验证策略研究是否严谨、可复现、成本后是否仍有效；当前不定位为实盘自动下单系统。

核心管线：

```text
数据 -> PIT股票池 -> 因子 -> 标签 -> 模型 -> 信号 -> 组合 -> 回测 -> 评估 -> 门控 -> Artifact
```

研究目标：

- 构建可复现的多因子选股研究流程。
- 避免 look-ahead bias、data leakage、survivorship bias、过拟合和交易成本低估。
- 优先找稳定、可解释、成本后仍有价值的策略，而不是只追求漂亮收益曲线。

## 1. 唯一正式入口

正式实验只能使用：

```bash
python -m src.cli experiment --config configs/experiments/<name>.yaml
```

配置审计使用：

```bash
python -m src.cli audit-config --config configs/experiments/<name>.yaml
```

测试使用：

```bash
pytest -q
```

禁止用以下入口产出正式研究结论：

- `src/research_pipeline.py`
- `src/research_orchestrator.py`
- `scripts/archive/deprecated_label_contract/*`
- `scripts/*_ic_test.py`
- 任何临时 notebook 或一次性脚本
- 顶层 `scripts/run_best_strategy.py`
- 顶层 `scripts/true_walk_forward_oos_v2.py`

如果必须参考旧脚本，只能作为历史背景，不能引用其中 IC、Sharpe、收益率作为当前结论。

## 2. 结论分级

所有策略结论必须明确属于以下哪一类：

```text
diagnostic: 诊断探索，只能用于发现问题或候选方向
research: 研究候选，不可用于正式结论
production: 可作为正式研究候选，但仍不是实盘承诺
```

规则：

- `registry_stage: diagnostic` 的实验不能说“策略有效”，只能说“发现候选现象”。
- `registry_stage: research` 的实验不能说“可用于正式策略”，只能说“研究候选”。
- `registry_stage: production` 也必须通过 `audit-config` 和 `strategy_gate.md` 后，才能说“通过当前研究门控”。
- 即使 production 通过门控，也不能说“可以直接实盘投资”，只能说“在当前回测合同下通过研究准入”。

`audit-config` 预期：

- `research` 应返回 `Research only - NOT for formal conclusions`。
- `diagnostic` 应返回 `Diagnostic only; not production candidate`。
- `production + reject因子` 必须 reject。
- `production + observe因子` 默认 reject，除非显式设置 `allow_observe_factors: true`，且只能给 warning。

## 2.1 研究协议与策略晋级

项目现在以 `research_protocol` 作为策略生成管线的主协议。`registry_stage` 只说明模型登记层级，`research_protocol.stage` 才说明研究生命周期位置。

允许阶段：

```text
diagnostic -> discovery -> validation -> holdout -> production
```

阶段含义：

| stage | 含义 | 允许结论 |
|-------|------|----------|
| diagnostic | 数据、因子、管线诊断 | 不产出策略结论 |
| discovery | 候选策略发现 | 只能称为 candidate |
| validation | 冻结逻辑后的独立验证 | 只能称为 validated candidate |
| holdout | 锁定样本最终证据 | 只能作为晋级证据 |
| production | 已晋级策略 | 可称为当前研究合同下的正式策略 |

配置规则：

- `model.registry_stage: research` 且未写 `research_protocol` 时，自动视为 `stage: discovery`。
- `model.registry_stage: diagnostic` 且未写 `research_protocol` 时，自动视为 `stage: diagnostic`。
- `model.registry_stage: production` 必须显式写安全的 `research_protocol`，否则加载失败。
- production 必须 `data_mined: false`。
- production 必须包含 `hypothesis_id`、`candidate_id`、`frozen_after`、`validation_window`、`holdout_window`。
- production 的 `allowed_change_after_freeze` 不能是 `research_only`，通常应为 `none`。

标准 production 协议：

```yaml
research_protocol:
  stage: production
  hypothesis_id: <validated_hypothesis_id>
  candidate_id: <promoted_candidate_id>
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

禁止行为：

- 把 discovery/diagnostic 的好结果称为“优秀策略”或“可投资策略”。
- 在同一段样本里反复试因子、试参数、试增强器，然后把最好结果写成 validation。
- 看完 validation 或 holdout 结果后继续调策略逻辑，再声称仍是同一个候选。
- 用文件名里的 `best`、`final`、`strategy` 判断配置可信度。

## 2.2 多重检验、风险归因、Overlay 分离

Gate 通过不等于策略有效。如果研究过程中试过多个因子、窗口、模型、参数或组合规则，必须承认这是 multiple testing 场景。production/holdout 结论必须配置并通过以下控制：

```yaml
multiple_testing:
  enabled: true
  search_space_id: <search_space_id>
  candidate_count: <尝试过的策略候选数>
  factor_trial_count: <尝试过的因子数>
  parameter_trial_count: <尝试过的参数/模型设定数>
  correction_method: <bonferroni|fdr|deflated_sharpe|locked_holdout|...>
  random_baseline: true
  permutation_test: true
  white_noise_baseline: true
  stability_over_best: true
```

要求：

- 必须记录候选数量、因子试验次数、参数搜索空间。
- 必须有多重检验校正或锁定 holdout 解释。
- 必须至少包含随机基准、permutation 或白噪声基线之一。
- 评价优先看稳定性，不优先看单次最好结果。

风险归因必须单独成层，不能只看收益、Sharpe 和 IC：

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

必须回答：

- 收益是否只是行业偏离？
- 收益是否只是 size/beta/liquidity 暴露？
- 因子中性化后是否仍有效？
- 主动风险是否集中在少数行业、风格或个股？

组合后处理必须分成两类：

| 类型 | 例子 | 含义 |
|------|------|------|
| execution enhancement | PositionBuffer、WeightSmoother、CostFilter | 降低换手和执行摩擦，不应创造新 alpha |
| overlay strategy | RegimeExposure、仓位择时、风险开关 | 新的预测/择时层，必须单独验证和归因 |

如果启用市场状态仓位控制，必须写：

```yaml
overlay:
  enabled: true
  regime_exposure_enabled: true
  hypothesis_id: <overlay_hypothesis_id>
  data_mined: false
  frozen_after: 'YYYY-MM-DD'
```

禁止把 `RegimeExposure` 混在“组合增强”里直接提升策略表现。它改变 alpha/beta 结构，必须单独验证。

## 3. 数据层规则

数据层文件主要在：

- `src/data_sources/`
- `src/data_store/`
- `src/universe/`

必须遵守：

- 股票池必须优先使用 point-in-time universe。
- 不允许用当前沪深300成分股倒推历史表现。
- 财务数据必须考虑真实披露日期或至少明确披露日期近似规则。
- 缺失数据策略必须在配置或代码中可追溯。
- 数据快照必须写入 artifact，不能只口头说明数据范围。

检查项：

- `trade_calendar` 是否覆盖实验区间。
- `daily_bar` 是否覆盖 warmup + research period。
- `universe_membership` 是否按日期合并。
- `tradability` 是否参与组合过滤和回测执行约束。
- `corporate_actions` 是否传入回测引擎。

禁止：

- 直接用未来股票池做历史回测。
- 用未注明披露日期的财务因子做 production 结论。
- 只凭本地 artifacts 中旧结果判断当前策略有效。

## 4. 标签层规则

当前正式标签：

```text
fwd_return_20d = price[T+20] / price[T+1] - 1
```

实现位置：

- `src/labels/definitions.py`

含义：

- T 日产生信号。
- T+1 是理论可执行起点。
- 标签使用 T+1 到 T+20 的 forward return。

禁止：

- `pct_change(20).shift(-20)`
- `shift(-20) / shift(0) - 1`
- 用 T 日收盘价作为可执行价格后再评估未来收益
- 在训练集中使用尚未兑现的 forward return

改标签前必须同步更新：

- `src/labels/definitions.py`
- `configs/research_contract_v1.yaml`
- `docs/AGENT_WORKFLOW.md`
- 相关测试
- 既有实验结论的有效性说明

## 5. 因子层规则

因子来源分三类：

| 来源 | 用途 |
|------|------|
| `src/features/simple_definitions.py` | 可计算主因子库 |
| `src/features/factor_catalog.py` | 因子档案和状态真相源 |
| `src/features/factor_pool.py` | 候选描述池，不可直接实验 |

`factor_pool.py` 是描述池，不是 production 可用因子库。不得因为一个因子出现在 `factor_pool.py` 就直接加入正式实验。

因子状态：

```text
core: 当前核心池，可作为 production 主因子
backup: 备用池，可作为候选加入研究
observe: 观察池，不得默认进入 production
reject: 拒绝池，禁止进入 production
deprecated: 已退役，禁止使用
```

新增因子流程：

1. 确认能在 `simple_definitions.py` 或对应 registry 中计算。
2. 明确数据依赖和可观测时点。
3. 在 `factor_catalog.py` 登记 family、economic mechanism、expected signal、lookback、inputs、status。
4. 先用 diagnostic/research 配置验证，不得直接 production。
5. 验证 IC、RankIC、IC IR、分组单调性、覆盖率、成本后表现、换手率、冗余关系。
6. 通过后才能进入 backup/core。

拒绝原则：

- IC 长期为负。
- IC IR 过低且无清晰经济解释。
- 分组收益不单调。
- 成本后贡献为负。
- 与核心因子高度冗余且无边际贡献。
- 明显依赖少数年份、少数股票或少数事件。
- 数据时点不干净。

## 6. 因子退化监控规则

不要使用“实时 IC”这个说法。IC 需要未来收益，只能使用已经兑现的 forward return。

正确逻辑：

```text
调仓日 D
找到已兑现信号日 S，使 S + horizon + delay <= D
使用 S 日因子值 + S 日对应已兑现 forward return
计算 RankIC
更新 FactorResponseMonitor 状态机
输出 factor_ic_history / factor_health_snapshot / factor_decay_report
```

相关模块：

- `src/features/ic_monitor.py`
- `src/features/factor_response.py`
- `src/features/factor_decay.py`

快速响应规则按“调仓次数”计算，不是自然日：

| 触发条件 | 动作 |
|----------|------|
| 连续5次调仓 IC < 0 | observe |
| 连续10次调仓 IC < 0 | downweight 50% |
| 连续15次调仓 IC < 0 | offline |
| 单次 IC < -0.03 | immediate offline |
| 连续20次调仓 IC > 0 | 可逐步恢复 |

实现约束：

- IC 监控的 factor panel 必须来自 `model_dataset` 或原始 `feature_panel`，不能来自只含 score 的 `signal_scores`。
- IC 监控日期必须是实际调仓信号日，优先使用 `signal_scores['trade_date']` 或 `portfolio_result.target_weights['signal_date']`，不能使用全量交易日。
- 退化监控如果在回测后运行，只能作为 health report，不能声称它影响了当次组合。
- 如果要让退化监控影响当次策略，必须把它前移到模型训练、信号生成或组合构建之前，并写清楚如何剔除/降权。

Artifact 要求：

- `features/factor_ic_history.parquet`
- `features/factor_health_snapshot.parquet`
- `reports/factor_decay_report.md`

即使没有 offline 因子，也应该写 `factor_decay_report.md`，明确说明 `no offline factors` 或 `IC records = 0`。

## 7. 模型层规则

模型位置：

- `src/models/simple_average.py`
- `src/models/weighted_average.py`
- `src/models/ridge_regression.py`
- `src/models/lightgbm_cross_sectional.py`

允许模型：

- `simple_average`
- `ic_weighted_average`
- `ridge_regression`
- `lightgbm_regression`

原则：

- 先 baseline，再 ML。
- ML 用于检验非线性和交互，不用于替代信号理解。
- 每个模型必须使用同一套标签合同。
- 模型比较必须使用一致的训练窗口、embargo、调仓频率、股票池、成本和评估区间。
- 如果不同模型的窗口协议不同，必须在结论里标注“不可直接横向比较”。

Walk-Forward 要求：

- 训练只能使用信号日前已经安全兑现的数据。
- 训练集必须避开 label horizon 对应的未来收益泄漏。
- LightGBM、Ridge、ICWeighted 等模型应尽量使用相同的 `train_window_days`、`training_embargo_days`、`label_horizon`。
- 不得用全样本训练后再在同一时期回测作为正式结论。

ML 失败模式：

- 训练内表现强，样本外弱。
- 特征重要性集中在 reject/observe 噪音因子。
- 高换手导致成本后收益消失。
- 性能由少数股票或少数日期驱动。
- 参数轻微变化后结果大幅波动。
- OOS Sharpe 显著低于 IS Sharpe。

## 8. 信号层规则

信号层位置：

- `src/signals/`

规则：

- 信号层只负责把模型输出转成统一 score/rank。
- 不得在信号层重新引入未来标签。
- 信号 score 必须可追溯到模型输出。
- 评估 RankIC 时，`signal_scores.trade_date` 必须与 `label_panel.trade_date` 同一信号日口径。

检查项：

- `signal_scores` 是否包含 `trade_date`、`symbol`、`score`。
- 同一交易日 score 是否有足够横截面离散度。
- fallback 使用比例是否过高。
- 最新信号是否和目标组合一致。

## 9. 组合层规则

组合层位置：

- `src/portfolio/construction.py`
- `src/portfolio/enhancer.py`

组合逻辑：

```text
signal_date 产生信号
execution_date = signal_date 后第 trade_delay_days 个交易日
按 score 选择 eligible 股票
生成 target_weight
可选增强：buffer -> smoother -> cost filter
```

必须检查：

- 股票必须在 signal_date 属于 PIT 股票池。
- 严格模式下，tradability 应使用 execution_date 的可交易状态。
- execution_date 必须晚于 signal_date。
- 组合权重不能超过 `gross_exposure` 和 `max_single_weight` 约束。
- enhancer.enabled=false 时不得创建或应用增强器。
- enhancer 如果启用，要在报告里明确换手、过滤和缓冲影响。

限制：

- 当前组合增强里的 equity 估算属于工程近似，只用于增强器成本过滤，不等同回测真实 equity。
- 正式绩效必须以 `src/backtest/engine.py` 输出为准。

## 10. 回测层规则

回测层位置：

- `src/backtest/engine.py`

回测假设：

- 用 execution_date 执行目标权重。
- 买卖使用当日 open，缺失时 fallback 到 close/last close。
- 买入计 commission 和 slippage。
- 卖出计 commission、stamp duty 和 slippage。
- 支持 lot_size 和 min_trade_value。
- 使用 tradability 限制买卖。
- corporate_actions 参与现金分红和送转处理。

必须检查：

- `target_weights` 是否使用 enhanced 后的权重进入回测。
- `trade_date`、`signal_date`、`execution_date` 是否没有倒置。
- 成本单位必须是 bps，不得和比例混用。
- turnover 口径必须说明是成交额 / equity，还是组合选股换手。

不得宣称：

- 当前回测完全等同真实成交。
- 当前滑点模型能覆盖容量、冲击成本和盘口深度。
- 当前结果可直接指导实盘。

## 11. 评估层规则

评估层位置：

- `src/evaluation/reporting.py`
- `src/evaluation/diagnostics.py`
- `src/evaluation/suites.py`
- `src/evaluation/strategy_gate.py`

必须输出和检查：

- NAV
- benchmark NAV
- trades
- positions
- drawdown
- monthly returns
- rank_ic
- quantile returns
- selection turnover
- signal coverage
- strategy_gate.md
- experiment_artifact.json

Strategy Gate 默认标准：

- IC 均值 > 0.02
- IC IR > 0.15
- 分组单调性 > 0.6
- 成本后超额收益 > 0
- 最大回撤 < 30%
- 平均换手 < 50%
- 年度胜率 > 60%
- Sharpe 高于基准

注意：

- `run_summary.json`、`strategy_gate.md`、`experiment_artifact.json` 的收益、超额、Sharpe、换手必须使用同一 active backtest window。不得把 warmup/full benchmark 与 active strategy NAV 混算。
- `total_return` 必须使用 `nav_end / nav_start - 1`，不能假设 active NAV 第一行等于 1。
- `benchmark_total_return` 必须使用与 active NAV 同日期窗口的 benchmark 首尾值。
- Gate 的分组单调性、IC 是信号层诊断；如果组合启用了 enhancer，还必须说明 raw signal diagnostics 与 enhanced portfolio backtest 不是同一个对象。
- 若 enhancer 改变权重，正式结论必须额外检查 raw vs enhanced 的收益、换手和回撤差异；否则只能说“增强后组合通过当前 Gate”，不能说“原始因子组合本身通过”。
- `cost_sensitivity` 是诊断近似，不是完整二次回测。
- `regime_breakdown` 是市场状态近似，不是正式宏观归因。
- 如果 Gate 出现 skipped，不得说策略完全通过。
- 如果 benchmark 获取失败并 fallback 到等权基准，结论必须说明基准口径。

## 11.1 样本分离和防数据挖掘规则

任何“选出策略”的研究必须拆成至少三层：

```text
discovery: 发现候选因子/参数，只能产生假设
validation: 固定候选后验证是否稳健
locked holdout: 不参与任何筛选，只用于最后一次验收
```

Agent 限制：

- 不得在同一个样本上反复试配置后，再用同一窗口 Gate 通过就宣布“优秀策略”。
- 如果策略是在本轮多次试错后发现的，必须标记为 `data-mined candidate`。
- data-mined candidate 即使 Gate 通过，也只能是 research candidate，不能升 production。
- 升 production 前必须固定配置、固定参数、固定因子方向，然后在未参与筛选的 holdout 或新时间段重新跑。
- 如果没有 locked holdout，结论必须写“没有独立样本验证”。
- 报告中必须拆分至少：发现期、验证期、最近一年、最近半年；任一关键区间 RankIC 为负或显著输基准，必须列为风险。
- 不得把“看完很多失败策略后挑出的最好策略”表述为无偏发现。

## 12. Artifact 和复现规则

每次正式实验必须保留：

```text
config/resolved_experiment.yaml
metadata/data_snapshot.json
metadata/dataset_summary.json
metadata/experiment_artifact.json
metadata/experiment_manifest.json
metadata/artifact_inventory.json
features/feature_panel.parquet
labels/label_panel.parquet
datasets/model_dataset.parquet
models/split_metrics.csv
models/feature_importance.csv
signals/predictions.parquet
signals/signal_scores.parquet
signals/target_weights.parquet
signals/target_weights_enhanced.parquet
backtest/nav.parquet
backtest/trades.parquet
backtest/positions.parquet
backtest/rank_ic.parquet
reports/run_report.md
reports/factor_diagnostics.md
reports/strategy_gate.md
```

因子退化监控启用时还应保留：

```text
features/factor_ic_history.parquet
features/factor_health_snapshot.parquet
reports/factor_decay_report.md
```

任何结论必须能指向对应 run_id 和 config hash。没有 artifact 的结果只能算临时观察。

## 13. 配置文件规则

实验配置位于：

- `configs/experiments/*.yaml`
- `configs/experiments/diagnostic/*.yaml`
- `configs/experiments/ARCHIVE/*.yaml`

规则：

- production 配置应放在 `configs/experiments/`。
- diagnostic 配置应放在 `configs/experiments/diagnostic/`。
- archive 配置不可作为当前结论。
- 配置中必须明确 features、label、model、portfolio、backtest、evaluation、enhancer。
- `allow_rejected_factors: true` 只能用于 diagnostic。
- `allow_observe_factors: true` 用于 production 时必须在报告中解释原因。

修改配置后必须运行：

```bash
python -m src.cli audit-config --config <config>
```

如果修改核心合同或代码，还要运行：

```bash
pytest -q
```

## 14. 文档和历史结果规则

历史结果只能作为背景，不得覆盖当前合同。

禁止引用旧标签合同结果作为当前结论：

- `pct_change(20).shift(-20)` 口径的 IC。
- `scripts/archive/deprecated_label_contract/` 中的结果。
- 旧 `alpha_017`、旧 `alpha_006`、简化版 `alpha_004` 的高 IC 结论。

当前可引用结论必须满足：

- 使用当前标签 `price[T+20] / price[T+1] - 1`。
- 由正式 CLI 入口生成。
- 有 `resolved_experiment.yaml`。
- 有 `strategy_gate.md`。
- 有 `experiment_artifact.json`。
- 通过 `audit-config`。

## 15. 常见任务处理方式

新因子研究：

1. 先说明经济机制。
2. 确认数据可观测时点。
3. 实现可计算因子。
4. 注册到 `factor_catalog.py`，初始状态不得直接 core。
5. 创建 diagnostic/research 配置。
6. 跑 CLI 实验。
7. 检查 IC、RankIC、IC IR、分组收益、覆盖率、换手、成本后收益、冗余。
8. 通过后再考虑进入 backup/core。

新模型研究：

1. 保留 simple baseline。
2. 使用相同股票池、标签、成本、调仓频率和评估区间。
3. 确认训练窗口和 embargo 与配置一致。
4. 比较 OOS 指标，不只比较 IS。
5. 检查特征重要性是否集中在 reject/observe 因子。
6. 不得因单次高收益直接替换当前基线。

组合调整：

1. 说明改的是选股逻辑还是组合映射。
2. 同时报告收益、回撤、换手、成本。
3. 检查 min_trade_value、lot_size、max_single_weight、gross_exposure。
4. enhancer 改动必须验证 enabled=false 时完全不生效。
5. 不得把组合增强收益误认为模型预测能力提升。

回测或评估修复：

1. 明确修复的是成交、成本、净值、基准还是指标。
2. 添加或更新测试。
3. 对比修复前后关键指标，说明口径变化。
4. 如果指标口径变了，旧实验结论必须降级为 historical。

完整策略审查：

1. 先看 audit-config。
2. 再看 strategy_gate。
3. 再看 rank_ic 和 quantile_summary。
4. 再看 NAV、drawdown、turnover、cost。
5. 再看 feature_importance 和 fallback_rate。
6. 最后判断策略是 production candidate、research candidate 还是 diagnostic only。

## 16. 禁止事项

- 不得用废弃脚本产出正式结论。
- 不得绕过 `audit-config`。
- 不得把 research/diagnostic 结果称为 production。
- 不得用旧标签口径结果证明当前因子有效。
- 不得把 `factor_pool.py` 描述池当作可计算因子库。
- 不得在 production 配置中默认使用 reject/observe 因子。
- 不得只看收益率，不看回撤、换手、成本、IC 和稳定性。
- 不得用同一回测窗口完成策略搜索、参数选择和最终验收。
- 不得忽略 `audit-config` 中的 unknown factors；production 配置中 unknown factor 必须先补 catalog。
- 不得在 summary、Gate、artifact 三处使用不同 benchmark/active window 口径。
- 不得把回测结果表述为实盘收益承诺。
- 不得在未说明基准口径时声称“跑赢市场”。
- 不得把事后 IC 监控说成影响了当次回测，除非代码确实前置接入。

## 17. 最小完成标准

任何“修好了”“可以了”“策略有效”的回答，至少要说明：

- 运行了哪些命令。
- `pytest` 是否通过。
- `audit-config` 状态是什么。
- 是否产生正式 artifact。
- 是否通过 Strategy Gate。
- 还有哪些近似或剩余风险。

如果没有跑实验，只能说“代码/配置层面检查通过”，不能说“策略有效”。
