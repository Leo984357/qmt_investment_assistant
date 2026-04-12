# Configs 目录说明

## 目录结构

```
configs/
├── README.md                    # 本文件
├── experiments/                 # 实验配置
│   ├── README.md              # 实验配置详解
│   ├── hs300_ridge_baseline.yaml           # ⭐ 推荐
│   ├── hs300_ridge_with_support.yaml       # 最高Sharpe
│   ├── hs300_ridge_full_risk.yaml          # 保守风控
│   ├── hs300_baseline_rank_sum.yaml        # 简单基线
│   └── ARCHIVE/              # 归档配置
│
├── baseline_frozen.py          # 冻结基线指标
├── feature_set_v2_corrected.py  # 修正特征集
├── recommended_factors.py       # 推荐因子
├── research_contract_v1.yaml    # 研究合同
└── ...                         # 其他配置
```

---

## 核心配置文件

| 文件 | 说明 | 状态 |
|------|------|------|
| `baseline_frozen.py` | 冻结基线指标 | 重要 |
| `feature_set_v2_corrected.py` | 修正特征集 (9因子) | 重要 |
| `recommended_factors.py` | 推荐因子清单 | 重要 |
| `research_contract_v1.yaml` | 研究合同 | 重要 |

---

## 推荐实验配置

### 1. Ridge回归基线 (默认推荐)

```yaml
# configs/experiments/hs300_ridge_baseline.yaml
model:
  family: ridge_regression
  params:
    alpha: 1.0

features:
  names:
    - roe
    - earnings_yield
    - operating_margin
    - equity_growth
    - ocf_per_share
    - revenue_growth
    - asset_turnover
    - gross_margin
    - cash_ratio

portfolio:
  top_n: 15
  rebalance_days: 10
  risk_model: none  # 无风控

backtest:
  initial_cash: 250000
  commission_bps: 0.75
  stamp_duty_bps: 10
  slippage_bps: 5
```

**结果**: 总收益 64.5%, Sharpe 1.066, 最大回撤 -13.4%

### 2. Ridge+Support因子 (最高Sharpe)

```yaml
# configs/experiments/hs300_ridge_with_support.yaml
features:
  names:
    # 9个基础因子
    - roe
    - earnings_yield
    - operating_margin
    - equity_growth
    - ocf_per_share
    - revenue_growth
    - asset_turnover
    - gross_margin
    - cash_ratio
    # + 2个Support因子
    - mom120
    - vol20
```

**结果**: 总收益 62.7%, Sharpe 1.277, 最大回撤 -9.0%

### 3. Ridge+四档风控 (保守)

```yaml
# configs/experiments/hs300_ridge_full_risk.yaml
portfolio:
  risk_model: qmt_style_ladder
  risk_ma_short_window: 60
  risk_ma_long_window: 120
  risk_momentum_window: 20
  risk_mid_exposure: 0.80
  risk_low_exposure: 0.55
  risk_crash_exposure: 0.40
```

**结果**: 总收益 35.1%, Sharpe 0.700, 最大回撤 -19.1%

---

## 基线冻结配置

`baseline_frozen.py` 定义了标准基线性能指标：

```python
BASELINE_CONFIG = {
    'baseline_id': 'frozen_baseline_v1',
    'frozen_metrics': {
        'total_return': 0.212,      # 21.2%
        'sharpe': 0.594,
        'max_drawdown': -0.182,     # -18.2%
        'avg_rank_ic': 0.068,
        'ic_ir': 0.550,
        'total_cost': 1092,         # 元
    }
}
```

---

## 研究合同

`research_contract_v1.yaml` 冻结了实验标准协议：

| 项目 | 值 |
|------|-----|
| 股票池 | HS300, point-in-time |
| 时间范围 | 2022-09-01 ~ 2026-03-28 |
| 调仓周期 | 10天 |
| 标签定义 | fwd_return_20d |
| 成本 | 佣金0.75bp + 印花税10bp + 滑点5bp |
