# Experiments 配置说明

## 推荐配置 ⭐

| 配置 | 特点 | 总收益 | Sharpe | 最大回撤 | 推荐场景 |
|------|------|--------|--------|----------|----------|
| `hs300_ridge_baseline.yaml` | 平衡型 | 34.7% | 0.715 | -14.0% | 默认推荐 |
| `hs300_baseline_rank_sum.yaml` | 简单基线 | 5.3% | 0.333 | -20.5% | 对比基线 |

> **注意**: Ridge回归配置已移除 vol20（rejected池因子），修复了财务因子 point-in-time 和标签 embargo 问题。修复后 IC 更可信但略有下降。

---

## 运行命令

```bash
# 推荐: Ridge回归基线
python -m src.cli experiment --config configs/experiments/hs300_ridge_baseline.yaml

# 简单平均基线 (对比用)
python -m src.cli experiment --config configs/experiments/hs300_baseline_rank_sum.yaml

# 查看实验结果
python -m src.cli runs --limit 10
```

---

## 配置对比

| 配置 | 模型 | 因子数 | 持仓 | 风控 |
|------|------|--------|------|------|
| Ridge基线 | Ridge | 9 | 15 | 无 |
| Ridge+Support | Ridge | 11 | 15 | 无 |
| 四档风控 | Ridge | 9 | 15 | qmt_style_ladder |
| 简单平均 | SimpleAverage | 3 | 40 | 无 |

---

## 因子配置

### 基础因子 (9个)
```yaml
features:
  names:
    - roe              # 净资产收益率
    - earnings_yield   # 盈利收益率
    - operating_margin # 营业利润率
    - equity_growth   # 净资产增长率
    - ocf_per_share   # 每股经营现金流
    - revenue_growth   # 营收增长率
    - asset_turnover  # 资产周转率
    - gross_margin    # 毛利率
    - cash_ratio      # 现金比率
```

### Support因子 (可选)
```yaml
features:
  names:
    # 基础因子
    - ...
    # + Support因子
    - mom120          # 动量120天
    - vol20           # 波动率20天
```

---

## 成本配置

| 参数 | 值 | 说明 |
|------|-----|------|
| commission_bps | 0.75 | 佣金 0.075% |
| stamp_duty_bps | 10 | 印花税 0.1% (仅卖出) |
| slippage_bps | 5 | 滑点 0.05% |
| lot_size | 100 | A股最小交易单位 |

---

## 归档配置 (ARCHIVE/)

已废弃的实验配置，保留供参考：
- `hs300_lightgbm_*.yaml` - LightGBM实验
- `hs300_all_*.yaml` - 因子探索
- `hs300_alpha_*.yaml` - Alpha因子

---

## 修改配置建议

### 增加因子
```yaml
features:
  names:
    - roe
    - earnings_yield
    # 添加新因子
    - new_factor
```

### 调整持仓
```yaml
portfolio:
  top_n: 20  # 增加持仓数
```

### 调整调仓周期
```yaml
backtest:
  rebalance_frequency_days: 20  # 改为20天
```
