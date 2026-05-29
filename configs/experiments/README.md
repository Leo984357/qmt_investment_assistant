# Experiments 配置说明

本目录只保存实验配置。不要根据文件名里的 `best`、`final`、`strategy` 判断策略质量，正式结论只看 `research_protocol.stage`、Gate、Artifact 和验证记录。

## 唯一入口

```bash
python -m src.cli experiment --config configs/experiments/<name>.yaml
python -m src.cli audit-config --config configs/experiments/<name>.yaml
```

禁止用一次性脚本、旧研究入口或手写 IC 结果产出正式结论。

## 研究阶段

| stage | 含义 | 允许结论 |
|------|------|----------|
| `diagnostic` | 数据、因子、管线诊断 | 不产出策略结论 |
| `discovery` | 候选策略发现 | 只能称为 candidate |
| `validation` | 冻结逻辑后的独立验证 | 只能称为 validated candidate |
| `holdout` | 锁定样本最终证据 | 只能作为晋级证据 |
| `production` | 已晋级策略 | 可以称为正式策略 |

`model.registry_stage: research` 默认视为 `discovery`。  
`model.registry_stage: production` 必须显式提供安全的 `research_protocol`，否则加载失败。

## 标准配置骨架

```yaml
name: hs300_example_candidate
description: >
  用经济机制解释候选策略，不要用回测收益倒推故事。

model:
  registry_stage: research

research_protocol:
  stage: discovery
  hypothesis_id: <mechanism_id>
  candidate_id: <candidate_id>
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

Production 配置必须使用：

```yaml
model:
  registry_stage: production

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

multiple_testing:
  enabled: true
  search_space_id: <search_space_id>
  candidate_count: <candidate_count>
  factor_trial_count: <factor_trial_count>
  parameter_trial_count: <parameter_trial_count>
  correction_method: <correction_method>
  random_baseline: true
  permutation_test: true
  white_noise_baseline: true
  stability_over_best: true

risk_attribution:
  enabled: true
  industry_exposure: true
  style_exposure: true
  beta_exposure: true
  benchmark_relative_active_risk: true
  neutralized_ic: true
  return_attribution: true
```

如果启用市场状态仓位控制，必须把它当作 overlay strategy，而不是 execution enhancement：

```yaml
overlay:
  enabled: true
  regime_exposure_enabled: true
  hypothesis_id: <overlay_hypothesis_id>
  data_mined: false
  frozen_after: 'YYYY-MM-DD'
```

## 当前候选

以下配置只能作为 discovery 候选，不能称为正式策略：

| 配置 | 阶段 | 说明 |
|------|------|------|
| `hs300_single_close_to_high250.yaml` | discovery | 单因子价格位置/长动量候选 |
| `hs300_mom_volume.yaml` | discovery | 动量 + 成交量确认候选，含未入档因子 |

## 晋级要求

候选进入 production 前必须满足：

- 因子全部可计算，且在 `factor_catalog.py` 中登记为 CORE/BACKUP/允许的 OBSERVE
- 不含 rejected 或 unknown 因子
- validation/holdout 样本独立，且策略逻辑在 `frozen_after` 后未改变
- 记录多重检验搜索空间，并有随机基准、permutation 或白噪声基线
- 完成行业、风格、beta、主动风险、中性化 IC 和收益归因
- Gate 通过，并同时输出 raw vs enhanced、年度分段、成本后超额、换手、最大回撤
- holdout 不是反复搜索后的最好结果
- Artifact 可复现，配置 hash、数据快照、标签合同、成本假设完整

## 配置目录建议

后续新配置按阶段放置：

```text
configs/experiments/
  diagnostic/
  discovery/
  validation/
  holdout/
  production/
  ARCHIVE/
```

历史散落在根目录的配置保留兼容，但不再作为推荐入口。
