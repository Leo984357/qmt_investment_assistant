# QMT Investment Assistant - Agent Instructions

## 项目概述
量化投资回测系统，使用沪深300股票。核心流程：数据 → 因子 → 模型 → 信号 → 组合 → 回测 → 评估。

## 核心原则（非negotiable）

1. **协议优先于性能** - 弱协议的好看结果 = 低质量研究
2. **三层分离** - 研究层、回测层、执行层必须分离
3. **小容量买方逻辑** - 资本有限、工程时间稀缺、专注窄而深
4. **基线优先** - 先单因子基线，再ML
5. **因子退化是生命周期事件** - 不是失败，是正常

---

## 因子池管理系统 (Factor Pool Management)

### 核心理念转变
从"因子筛选" → "因子池管理"
- 问: 这个因子在现有池子里还有没有**边际价值**
- 判断: 主因子/辅助因子/冗余因子/条件因子
- 目标: **Pool-level research**，不只是factor-level research

---

## 标准研究流程 (7阶段)

### Stage 1: 数据合同与因子注册

**目标**: 管好"什么时候可用"，不只是"能不能算"

```
1.1 数据源梳理
├── 财务数据 (akshare) → 季度更新
├── 价格数据 (baostock) → 日频更新
├── 资金流数据 (akshare) → 日频更新
└── 分析师预期 (akshare) → 周频更新

1.2 时点审计 (DataTimingAudit)
├── 原始发布日期 (pub_date)
├── 可交易使用日期 (tradeable_date)
├── Lag规则 (lag_days)
└── 是否Point-in-Time (PIT)

1.3 因子注册到元数据表 (FactorMetadataRegistry)
├── 名称、家族、公式
├── 数据依赖、lookback
├── 更新频率
├── 当前状态 (候选/研究/生产/退役)
└── 评估指标 (IC、单调性、换手率)
```

**关键**: 不是只登记"这个字段来自哪里"，而是登记"这列数据什么时候可用"

---

### Stage 2: 单因子原型测试

**目标**: 基本筛选 + 软打分

```
2.1 硬淘汰 (只淘汰明显脏因子)
├── IC长期为负
├── 覆盖率太差 (< 50%)
├── 时点不干净 (有未来函数)
└── 极度依赖个别年份或个别股票

2.2 软打分 (不淘汰，进入候选池)
├── IC均值、IC IR
├── 单调性
├── IC正年份率
├── 中性化后IC改善
└── IC decay特征

2.3 Neutralization测试
├── raw IC
├── industry-neutral IC
├── size-neutral IC
└── 双重中性 IC
```

---

### Stage 3: 单因子健康与可交易性审查

```
3.1 分组单调性
├── Q5 > Q4 > Q3 > Q2 > Q1
└── 单调性 > 0.6

3.2 IC频率匹配
├── 调仓周期: 10天
├── 测试持有期: 5/10/20/40/60天
├── 选择IC最高持有期
└── 计算半衰期

3.3 成本敏感性 (组合层验证)
├── ⚠️ 不是 IC - 成本！
├── 必须在组合层计算
├── 估算: 调仓收益 - 实际交易成本
└── 成本后收益 > 0

3.4 多维度冗余检测
├── 因子值截面相关
├── IC时间序列相关
├── 行业暴露相似度
└── |avg_corr| > 0.7 → 冗余

3.5 极端行情稳定性 (改进定义)
├── 波动率regime (高/低)
├── 趋势regime (上行/下行/震荡)
└── 流动性regime (宽松/收缩)
```

---

### Stage 4: 因子池管理

```
4.1 家族划分
├── Value (价值)
├── Profitability (盈利)
├── Growth (成长)
├── Leverage (杠杆)
├── Momentum (动量)
└── Pattern (形态)

4.2 边际贡献分析
├── 因子加入前后的OOS变化
├── 因子加入前后的成本变化
└── 因子加入前后的暴露变化

4.3 状态管理
├── CANDIDATE: 候选
├── RESEARCH: 研究中
├── PRODUCTION: 生产中
├── OBSERVATION: 观察中
└── RETIRED: 已退役

4.4 Champion-Challenger机制
├── Champion: 当前线上组合
├── Challenger: 新因子/新版本组合
└── 连续通过OOS检验才替换
```

---

### Stage 5: 组合基线与ML对照

```
5.1 简单平均基线 (必须保留!)
├── 横截面Z-score标准化
├── 家族内等权平均
├── 家族间等权平均
└── 生成综合信号

5.2 行业/风格暴露控制
├── 行业中性约束
├── size/beta/liquidity暴露
├── 单票上限 (4%)
└── 单行业上限

5.3 ML对照 (可选)
├── LightGBM横截面回归
├── Walk-Forward验证
└── OOS增量 > 0才使用
```

---

### Stage 6: Walk-Forward滚动验证

**⚠️ 不是一次切分，而是滚动样本外**

```
6.1 滚动窗口配置
├── Expanding/Sliding窗口
├── 训练窗口: 500天
├── 测试窗口: 60天
├── 步长: 30天
└── 最小样本要求

6.2 验证指标
├── OOS平均表现
├── OOS最差窗口
├── OOS稳定性
├── 不同regime下表现
└── IS/OOS比值

6.3 通过标准
├── Sharpe一致性 > 60%
├── OOS/IS比值 > 50%
├── 最差窗口 Sharpe > 0
└── 无显著regime偏好
```

---

### Stage 7: 上线监控与退役机制

```
7.1 信号健康监控
├── IC滚动均值
├── 信号覆盖率
└── 异常交易信号

7.2 组合健康监控
├── 因子暴露变化
├── 换手率异常
└── 行业偏离

7.3 执行偏差监控 (实盘)
├── 目标权重 vs 实际成交
├── 预估成本 vs 实际成本
└── 未成交/部分成交比例

7.4 研究-实盘漂移监控
├── Live IC vs Backtest IC
├── Live turnover vs Simulated
└── Live exposure vs Expected

7.5 因子退役规则
├── 连续N窗口IC弱 → 观察
├── 成本后贡献连续负 → 降权
├── 重写表达后仍无恢复 → 退役
└── 每月Postmortem复盘
```

---

## 因子研究

### 因子池规模 (2026-04-12更新)
**总计667个已注册因子**，覆盖11个类别：

| 类别 | 数量 | 说明 |
|------|------|------|
| Extended Financial | 100 | 扩展财务因子 (盈利能力、成长、估值等) |
| WorldQuant Alpha | 97 | WorldQuant 101 Alpha |
| Extended Technical | 82 | 扩展技术因子 (均线、动量、波动率等) |
| Academic | 74 | 学术因子 (Fama-French, q-factor等) |
| Pattern | 62 | 形态学因子 (K线形态、突破等) |
| Barra Style | 56 | Barra风格因子 |
| Analyst | 50 | 分析师预期因子 |
| Sector | 46 | 行业轮动因子 |
| Money Flow | 40 | 资金流因子 |
| Sentiment | 35 | 情绪/另类因子 |
| Macro | 25 | 宏观因子 |

### 因子库文件
- `src/features/academic_factors.py` - 学术因子 (Fama-French, q-factor, Novy-Marx等)
- `src/features/barra_factors.py` - Barra风格因子
- `src/features/worldquant_alphas.py` - WorldQuant 101 Alpha
- `src/features/sector_factors.py` - 行业轮动因子
- `src/features/pattern_factors.py` - 形态学因子
- `src/features/factor_pool.py` - **统一因子库整合 (667因子)**
- `src/features/extended_financial_factors.py` - **扩展财务因子 (100个)** ⭐
- `src/features/extended_technical_factors.py` - **扩展技术因子 (82个)** ⭐
- `src/features/analyst_expectation_factors.py` - **分析师预期因子 (50个)** ⭐
- `src/features/money_flow_factors.py` - **资金流因子 (40个)** ⭐
- `src/features/sentiment_factors.py` - **情绪因子 (35个)** ⭐
- `src/features/macro_factors.py` - **宏观因子 (25个)** ⭐

### 扩展财务因子 (100个)
覆盖七大类：

| 类别 | 数量 | 因子示例 |
|------|------|---------|
| 盈利能力 | 20 | roe_ttm, roce, roa_ttm, cash_roe |
| 成长能力 | 15 | revenue_growth_yoy, profit_growth_qoq, eps_growth |
| 估值因子 | 20 | pe_ttm, pb, ev_ebitda, peg |
| 运营效率 | 10 | asset_turnover, inventory_turnover, cash_cycle |
| 杠杆/偿债 | 12 | debt_ratio, current_ratio, interest_coverage |
| 现金流 | 10 | ocf_per_share, fcf_per_share, cash_conversion |
| 质量因子 | 13 | accrual_ratio, earnings_volatility, roe_consistency |

### 扩展技术因子 (82个)
覆盖五大类：

| 类别 | 数量 | 因子示例 |
|------|------|---------|
| 均线系统 | 20 | ma5/10/20/60/120/250, 均线交叉 |
| 动量指标 | 20 | rsi6/12/24, macd, kdj, roc |
| 波动率 | 15 | atr14/28, 布林带, hv20/60/120 |
| 成交量 | 15 | obv, vr, 量比, 换手率 |
| 趋势指标 | 10 | adx14/28, plus_di, trend_strength |

### 分析师预期因子 (50个)
覆盖五大类：

| 类别 | 数量 | 因子示例 |
|------|------|---------|
| 盈利预测 | 12 | eps_forecast_1y, profit_forecast_growth |
| 评级 | 10 | rating_score, rating_upgrades_recent |
| 预测修订 | 10 | eps_revision_1m/3m, target_price_upside |
| 一致性预期 | 10 | consensus_eps_std, forecast_curvature |
| 目标价 | 8 | target_irr, upside_to_consensus |

### 资金流因子 (40个)
覆盖四大类：

| 类别 | 数量 | 因子示例 |
|------|------|---------|
| 大单资金流 | 12 | super_large_net_flow, large_net_flow |
| 订单分类流 | 10 | medium_net_flow, small_net_flow |
| 资金流向指标 | 10 | main_force_net_flow, flow_acceleration |
| 资金博弈 | 8 | buy_sell_imbalance, consecutive_inflow_days |

⚠️ **注意**: 样本量小(20只)，需扩大样本验证

### 财务因子最新实验结果 (21因子全量测试)
**2026年4月11日更新**：

| Metric | 基线3因子 | Top3财务 | **全量21财务** |
|--------|----------|----------|----------------|
| Total Return | 19.3% | 25.2% | **42.0%** |
| Sharpe | 0.333 | 0.821 | **0.898** |
| IC | 0.025 | 0.059 | **0.073** |
| IC IR | 0.096 | 0.362 | **0.592** |
| Turnover | 4.0% | 2.8% | **1.4%** |
| Cost | 19.8k | 15.5k | **7.5k** |
| **Excess Return** | - | +1.0% | **+7.7%** |

**关键发现**：
1. 财务因子显著优于价格因子 - Sharpe 0.898 vs 0.333
2. 因子越多越好（在此场景） - 21因子 > 3因子
3. 换手率随因子数增加而降低 - 1.4%为历史最低
4. 超额收益+7.7%，成功跑赢基准

**数据获取修复**：
- 修复symbol格式: `sz.000001` → `000001.SZ`, `sh.600000` → `600000.SH`
- 覆盖率: 100% (大部分因子), 83-84% (margin类因子)

### 关键教训
⚠️ **LightGBM被"噪音因子"误导**：
- LightGBM给vol120最高权重(214.67)，但IC=-0.03
- vol类因子在A股市场IC显著为负
- 必须使用因子档案筛选后的小池子，不能直接用原始池子

### A股市场特点
根据2026年4月因子健康检查和实验验证：

**有效因子 (IC > 0.02, IR > 0.1):**
| 因子 | IC | IC IR | 说明 |
|------|-----|-------|------|
| mom250 | 0.038 | 0.179 | 长期动量 |
| alpha_006 | 0.027 | 0.261 | 开盘量价关系 |
| alpha_003 | 0.024 | 0.245 | 开盘量rank相关 |
| close_to_high250 | 0.023 | 0.124 | 收盘价位置 |
| alpha_016 | 0.023 | 0.225 | 高价成交量关系 |

**无效/负向因子:**
| 因子 | IC | 问题 |
|------|-----|------|
| alpha_013 | -0.003 | IC为负 |
| alpha_018 | -0.009 | IC为负 |
| vol类 | -0.03~-0.06 | 波动率溢价不成立 |
| high_low_pos | -0.025 | IC显著为负 |

**结论**: A股波动率溢价不成立，长期动量有效，短期反转有效

### 行业轮动调整形态因子 (2026年4月新增)
形态因子与行业轮动状态挂钩，根据市场环境动态调整信号强度：

**实现逻辑**:
- 行业强势时, 趋势跟随型形态(如连续上涨)信号增强1.2倍
- 行业弱势时, 反转型形态信号增强

**已注册因子**:
| 类别 | 因子 | 说明 |
|------|------|------|
| Sector | sector_mom_20d, sector_mom_60d | 行业动量 |
| Sector | sector_rs_20d | 行业相对强弱 |
| Sector | sector_regime | 市场状态分类 |
| Sector-Adj | sector_adj_trend_strength | 行业调整趋势强度 |
| Sector-Adj | sector_adj_close_position | 行业调整收盘位置 |
| Sector-Adj | sector_adj_volume_momentum | 行业调整成交量动量 |

**实验结果**:
| 实验 | Total Return | Sharpe | IC | IC IR | 换手率 | 超额收益 |
|------|-------------|--------|-----|-------|--------|----------|
| 21财务因子 | 42.0% | 0.898 | 0.073 | 0.592 | 1.4% | +7.7% |
| **财务+行业调整形态** | 38.4% | **1.040** | 0.067 | 0.522 | 2.6% | **+16.4%** |

**关键发现**:
1. 行业调整形态因子单独使用IC为负(-0.004)
2. 但与财务因子组合后Sharpe从0.898提升到1.040
3. 超额收益从+7.7%提升到+16.4%
4. **形态因子提供增量信息，增强财务因子的效果**

### 统一因子库使用
```python
from src.features.factor_pool import get_all_factors, get_factors_by_source, inventory

# 获取所有因子
factors = get_all_factors()

# 按来源获取
worldquant_factors = get_factors_by_source('worldquant')
academic_factors = get_factors_by_source('academic')

# 导出清单
df = inventory()
df.to_csv('artifacts/factor_pool_unified.csv')
```

---

## 单因子健康检查协议

使用 `src/features/factor_health_check.py` 对每个因子进行全面检查：

### 核心检查项
1. **横截面排序能力** - IC/RankIC是否为正，IR > 0.1
2. **分组单调性** - Q5 > Q4 > Q3 > Q2 > Q1，单调性 > 0.6
3. **IC频率匹配** - IC周期与调仓频率匹配（10天调仓测试5/10/20天持有期）
4. **成本敏感性** - 成本后是否还有正收益
5. **换手率匹配** - 因子换手率与交易成本匹配
6. **极端行情稳定性** - 正常/极端行情下IC差异 < 5%
7. **相关性去重** - |corr| > 0.8 的因子只保留换手率低的

### 淘汰条件
| 条件 | 阈值 | 操作 |
|------|------|------|
| IC均值 | < 0 | 淘汰 |
| IC IR | < 0.1 | 降权/观察 |
| 单调性 | < 0.4 | 淘汰 |
| 相关性 | > 0.8 | 去重，保留低换手 |
| IC正年份率 | < 60% | 降权 |
| 成本后收益 | < 0 | 淘汰 |
| 极端行情IC差异 | > 10% | 降权 |

### 当前因子评估结果 (24因子)

**通过 (IC > 0.02, 单调性 > 0.6):**
| 因子 | IC | 单调性 | 状态 |
|------|-----|--------|------|
| roe | 0.036 | 0.97 | ✅ 最佳 |
| earnings_yield | 0.031 | 0.87 | ✅ |
| total_roa | 0.029 | 0.67 | ✅ |
| operating_margin | 0.026 | 0.70 | ✅ |
| gross_margin | 0.009 | 0.93 | ✅ |

**冗余需去重 (相关性 > 0.8):**
| 组别 | 保留 | 剔除 |
|------|------|------|
| ROE组 | roe | roe_weighted |
| Margin组 | operating_margin | net_margin |
| Liquidity组 | current_ratio | quick_ratio |

**极端行情稳定性: 全部 ✅**

---

## 因子更新机制

### 各因子更新频率

| 因子类型 | 数据源 | 更新频率 | 说明 |
|----------|--------|----------|------|
| **财务因子** | akshare | **季度** | 财报发布后更新 |
| **价格因子** | baostock | **日频** | 每日收盘后 |
| **形态因子** | baostock | **日频** | 每日收盘后 |
| **行业因子** | 计算 | **日频** | 每日收盘后 |

### 财务因子更新时机
- Q1: 4月底前
- Q2: 8月底前
- Q3: 10月底前
- Q4/年报: 次年4月底前

### 更新命令
```bash
# 财务因子更新 (每季度)
python -m src.data_sources.batch_financial_factors --symbols-file data/symbols.txt --start-year 2024

# 因子健康检查 (每月)
python -c "from src.features.factor_health_check import batch_check, print_health_report"
```
- 单调性差（得分 < 0.5）
- 成本30bp后无效
- 换手率 > 50%

### 2026年4月审查结果
```
通过: 0个
条件通过: 2个 (mom250, high_low_pos120)
失败: 69个
```

**CONDITIONAL - mom250**：
- IC均值: 0.034, IR: 0.14, 单调性: 0.99
- 成本30bp后: +0.41% ✅
- 换手率: 9.3% ✅
- ⚠️ IC集中在2020年(贡献67%)
- ⚠️ 2021/2018/2026年IC为负

**CONDITIONAL - high_low_pos120**：
- IC均值: 0.017, IR: 0.09, 单调性: 0.98
- 成本30bp后: +0.14% ⚠️ (勉强)
- ⚠️ IC集中在2017年
- ⚠️ 50bp后可能无效

### 失败原因分类
| 原因 | 因子数 |
|------|--------|
| IC均值负 | 45 |
| IC不稳定 | 40 |
| 单调性差 | 38 |
| 成本30bp后无效 | 53 |
| 换手率过高 | 18 |

### 结论
- 71个候选因子中，只有2个通过条件检查
- **先洗干净池子，再做多因子组合**
- 使用 `batch_check()` 和 `print_health_report()` 定期检查

---

## 实验结果对比 (2026年4月)

### 基线策略对比
| 实验 | Total Return | Sharpe | IC | IC IR | Turnover | 因子数 |
|------|-------------|--------|-----|-------|----------|--------|
| 基线3因子 | 19.3% | 0.333 | 0.025 | 0.096 | 4.0% | 3 |
| +4 Alpha | 18.6% | 0.365 | 0.035 | 0.141 | 6.9% | 7 |
| +11 Alpha | 4.0% | 0.112 | 0.017 | 0.091 | 9.0% | 13 |
| Top5 | -1.7% | -0.052 | 0.043 | 0.214 | 88% | 5 |
| **IC加权3因子** | 17.7% | 0.361 | **0.052** | **0.204** | 6.6% | 3 |
| Top3财务因子 | 25.2% | 0.821 | 0.059 | 0.362 | 2.8% | 3 |
| 全量21财务因子 | 42.0% | 0.898 | 0.073 | 0.592 | 1.4% | 21 |
| **财务+行业调整形态** | 38.4% | **1.040** | 0.067 | 0.522 | 2.6% | **24** |
| 基准 | 31.9% | - | - | - | - | - |

### 关键发现
1. **财务因子显著优于价格因子** - Sharpe 0.898 vs 0.333
2. **形态因子提供增量收益** - Sharpe从0.898提升到1.040
3. **因子数量与换手率负相关** - 21因子换手仅1.4%
4. **财务+形态因子实现+16.4%超额收益** - 历史最佳

### WorldQuant Alpha IC排名 (47个Alpha完整分析)
| 排名 | Alpha | IC | IC IR | 结论 |
|------|-------|-----|-------|------|
| 1 | alpha_006 | 0.027 | 0.261 | **有效** |
| 2 | alpha_003 | 0.024 | 0.245 | **有效** |
| 3 | alpha_016 | 0.023 | 0.225 | **有效** |
| 4 | alpha_014 | 0.020 | 0.214 | **有效** |
| 5 | alpha_036 | 0.017 | 0.184 | **有效** |
| 6 | alpha_020 | 0.016 | 0.125 | 勉强 |
| 7 | alpha_027 | 0.014 | 0.105 | 勉强 |
| 8 | alpha_077 | 0.012 | 0.110 | 噪声 |
| 9 | alpha_086 | 0.012 | 0.081 | 噪声 |
| 10 | alpha_067 | 0.011 | 0.068 | 噪声 |
| 11-20 | alpha_068/023/074/094/075/001/029/073/026/076 | <0.01 | <0.1 | 低效 |
| 21-27 | alpha_043/093/070/046/072/010/071/009/002 | <0.005 | ~0 | 噪声 |
| 28-34 | alpha_012/092/028/047/008/004/013 | <0 | <0 | 负向 |
| 35-40 | alpha_032/031/087/018/088 | <-0.01 | <0 | 显著负向 |
| 41-44 | alpha_039/022/033/044/034 | <-0.02 | <0 | **强负向** |

### 因子相关性矩阵 (关键发现)
```
                  mom250  alpha_006  alpha_003  alpha_016  close_to_high250
mom250             1.000      0.003     -0.013      0.022             0.527
alpha_006          0.003      1.000      0.596      0.340            -0.001
alpha_003         -0.013      0.596      1.000      0.334            -0.026
alpha_016          0.022      0.340      0.334      1.000             0.008
close_to_high250   0.527     -0.001     -0.026      0.008             1.000
```

**关键相关性发现**：
- alpha_006 ↔ alpha_014: 0.834 (极高冗余)
- alpha_006 ↔ alpha_003: 0.596 (高冗余)
- mom250 ↔ close_to_high250: 0.527 (中等冗余)

### 推荐配置
```yaml
# configs/experiments/hs300_baseline_rank_sum.yaml
features:
  names:
    - mom250          # IC=0.038, 长期动量
    - close_to_high250 # IC=0.023, 收盘位置
    - mom120          # IC=0.017, 中期动量
model:
  family: simple_average  # 不使用ML
```

### 最终实验汇总
| 实验 | Total Return | Sharpe | IC | IC IR | Turnover | 成本 |
|------|-------------|--------|-----|-------|----------|------|
| 基线3因子 | **19.3%** | **0.333** | 0.025 | 0.096 | 4.0% | 19.8k |
| +4 Alpha | 18.6% | 0.365 | 0.035 | 0.141 | 6.9% | 33.7k |
| Top6 Alpha | -1.0% | -0.034 | 0.025 | 0.177 | 81% | - |
| 去冗余Alpha | 0.3% | 0.024 | 0.019 | 0.146 | 85% | 52k |
| 动量+Alpha | 0.0% | 0.003 | 0.033 | 0.135 | 64% | 38k |
| 基准 | 31.9% | - | - | - | - | - |

**核心发现**：
1. **原始3因子基线仍是最佳** - 最低换手(4%)带来最高收益
2. **纯Alpha策略换手过高** - 因子信号不稳定
3. **不要盲目增加因子** - 噪声因子损害组合
4. **相关性是冗余的根源** - alpha_006与alpha_014相关性0.83

### 因子退化检测清单
- [ ] 滚动IC/RankIC
- [ ] 单调性稳定性
- [ ] 分周期贡献
- [ ] 成本调整后贡献
- [ ] 换手率负担
- [ ] 与核心因子相关性
- [ ] 极端名称依赖
- [ ] 特定体制弱化

### 退化判断分类
1. 经济机制退化
2. 交易层退化
3. 表达/规格退化
4. 与现有因子冗余
5. 暂时体制不匹配
6. 数据质量或合约问题

### 退化处理操作
- keep - 保持
- observe - 观察
- downweight - 降权
- isolate - 隔离
- rewrite - 重写
- retire - 退役

---

## 快速响应规则

| 触发条件 | 操作 |
|---------|------|
| IC连续5天<0 | 进入观察 |
| IC连续10天<0 | 降权50% |
| IC连续15天<0 | 下线，从备用池替换 |
| IC连续20天>0 | 可重新上线 |

**止损规则：**
- IC<-0.03 → 立即下线
- 单因子回撤>2% → 强制降权
- 贡献为负 → 权重归零

---

## 关键命令

```bash
# 运行实验
python -m src.cli experiment --config configs/experiments/xxx.yaml

# 启动GUI
python -m src.cli dashboard
```

---

## 目录结构

- `src/features/` - 因子定义和研究工具
  - `academic_factors.py` - 学术因子 (74个)
  - `barra_factors.py` - Barra风格因子 (56个)
  - `worldquant_alphas.py` - WorldQuant Alpha (97个)
  - `sector_factors.py` - 行业轮动因子 (46个)
  - `pattern_factors.py` - 形态学因子 (62个)
  - `factor_pool.py` - **统一因子库整合** (335个)
  - `factor_catalog.py` - 因子档案系统
  - `factor_health_check.py` - 单因子健康检查工具
  - `signal_research.py` - 信号层研究工具
  - `factor_decay.py` - 因子退化分析
  - `factor_response.py` - 因子快速响应机制
  - `simple_definitions.py` - 可计算因子注册表 (71个已注册)
- `src/backtest/` - 回测引擎
- `src/models/` - 模型
  - `lightgbm_cross_sectional.py` - LightGBM横截面模型
  - `simple_average.py` - 简单平均模型
  - `weighted_average.py` - IC加权平均模型
- `src/ui/dashboard.py` - Streamlit可视化界面

---

## ML使用原则

ML用于：
- 学习非线性
- 学习交互
- 自适应条件权重
- 组合弱但有用的信号

ML不用于避免信号理解。

### ML失败模式检测
- [ ] 泄漏
- [ ] 不稳定增益来自弱特征
- [ ] 参数调优过度敏感
- [ ] 模型分数到组合的差映射
- [ ] 强内样本拟合但弱外样本排序
- [ ] 高换手假alpha
- [ ] 依赖窄日期范围
- [ ] 性能由少数名称或少数事件驱动

---

## 响应模式

### Mode A: 新研究想法
1. 研究框架
2. 机制假设
3. 候选信号族
4. 最小基线测试
5. 可能风险
6. 下一步实验

### Mode B: 因子审查
1. 因子定义
2. 可观测时点
3. 可能重叠
4. 验证清单
5. 生产风险
6. 保留/拒绝/细化判断

### Mode C: ML实验审查
1. 标签设计
2. 分割设计
3. 基线比较
4. 可能泄漏风险
5. ML预期增量值
6. go/no-go判断

### Mode D: 组合审查
1. 分数到仓位映射
2. 换手和成本含义
3. 暴露含义
4. 可能脆弱性
5. 改进

### Mode E: 退化诊断
1. 症状
2. 可能退化类型
3. 需要支持的证据
4. 立即风险行动
5. 中期重写或退役计划

### Mode F: 完整策略验尸
1. 什么有效
2. 什么是偶然
3. 什么是脆弱
4. 什么可能驱动损失
5. 什么应该保留
6. 什么应该移除
7. 下次迭代优先级列表

---

## 核心原则

- **宁可错杀，不可放过**
- **快速降权，慢速加回**
- **备份因子永远在线**
- **定期检查因子健康状态**

---

## 模型参数建议

- 调仓周期: 10天
- 持仓数量: 40只
- 初始资金: 100万
- 特征: 优先使用IC为正的因子

---

## 新增模块 (2026年4月)

### 核心模块

| 模块 | 路径 | 说明 |
|------|------|------|
| DataTimingAudit | `src/features/data_timing_audit.py` | 数据时点审计，防未来函数 |
| FactorMetadataRegistry | `src/features/factor_metadata_registry.py` | 因子元数据注册表 |
| WalkForwardValidator | `src/features/walk_forward_validator.py` | Walk-Forward滚动验证 |
| ICDecayAnalyzer | `src/features/factor_ic_decay_neutralization.py` | IC Decay分析和半衰期估计 |
| FactorNeutralizer | `src/features/factor_ic_decay_neutralization.py` | 因子中性化 (行业/Size) |
| MarketRegimeClassifier | `src/features/market_regime_and_correlation.py` | 市场状态分类 |
| MultiDimensionalCorrelationAnalyzer | `src/features/market_regime_and_correlation.py` | 多维度相关性分析 |
| MarginContributionAnalyzer | `src/features/margin_contribution.py` | 因子边际贡献分析 |

### 使用示例

```python
# 数据时点审计
from src.features.data_timing_audit import get_data_timing_audit

audit = get_data_timing_audit()
result = audit.audit_factor_timing('roe', datetime(2024, 6, 1))
print(f"roe可交易日期: {result['tradeable_date']}")

# Walk-Forward验证
from src.features.walk_forward_validator import WalkForwardValidator, WalkForwardConfig

validator = WalkForwardValidator(WalkForwardConfig(
    train_window_days=500,
    test_window_days=60,
    step_days=30,
))
report = validator.run(factor_data, label_col='fwd_return_20d')
print(report.to_summary())

# IC Decay分析
from src.features.factor_ic_decay_neutralization import ICDecayAnalyzer

analyzer = ICDecayAnalyzer()
result = analyzer.analyze(data, 'roe', horizons=[5, 10, 20, 40])
print(f"半衰期: {result.half_life}天")
print(f"推荐调仓周期: {result.recommended_rebalance}天")

# 因子中性化
from src.features.factor_ic_decay_neutralization import FactorNeutralizer

neutralizer = FactorNeutralizer()
industry_neutral = neutralizer.industry_neutral(factor, industry)

# 多维度相关性分析
from src.features.market_regime_and_correlation import MultiDimensionalCorrelationAnalyzer

analyzer = MultiDimensionalCorrelationAnalyzer()
redundancy_df = analyzer.analyze_redundancy(factor_data, factors)
groups = analyzer.get_redundancy_groups(redundancy_df)

# 边际贡献分析
from src.features.margin_contribution import MarginContributionAnalyzer

analyzer = MarginContributionAnalyzer()
result = analyzer.analyze('new_factor', baseline_pool, new_pool, data)
print(f"建议: {result.recommendation}")
```

---

## 后半段研究流程 (2026年4月)

### 核心原则

> **模型负责排序，组合负责赚钱，风控负责不把前两者毁掉。**

---

### Phase 1: 冻结研究合同

所有实验必须在**同一套合同**下比较。固定六件事：

| 固定项 | 值 | 说明 |
|--------|-----|------|
| 股票池 | HS300 | point-in-time成分股 |
| 时间频率 | 日频 | 训练/标签/组合统一 |
| 标签定义 | fwd_return_20d | 未来20日收益 |
| 调仓规则 | 每10日 | 固定锚点 |
| 成交假设 | next VWAP | 简化close代理 |
| 成本口径 | 佣金0.75+印花税10+滑点5 | min_trade=2000 |

**配置文件**: `configs/research_contract_v1.yaml`

---

### Phase 2: 特征层定版

把因子池变成**可训练的feature_set_v1**：

```
feature_set_v1 (9因子)
├── Production (9个): roe, earnings_yield, operating_margin, equity_growth
│                     ocf_per_share, revenue_growth, asset_turnover, gross_margin, cash_ratio
├── Support (0个): 待验证
└── Quarantine (5个): profit_growth, sector_adj_close_position, inv_turnover
                       debt_ratio, sector_adj_trend_strength
```

**预处理规则**:
- 盈利类因子: winsorize[0.01,0.99] + 行业中性
- 成长类因子: winsorize[0.05,0.95] + 市值中性
- 价值类因子: winsorize[0.01,0.99]

**配置文件**: `configs/feature_set_v1.yaml`

---

### Phase 3: 样本与标签层

独立模块，统一构造协议：

```python
from src.data.sample_constructor import SampleConstructor

constructor = SampleConstructor(
    label='fwd_return_20d',
    horizon=20,
    clip_range=[-0.30, 0.30],
    walk_forward_config=WalkForwardConfig(
        train_window_days=500,
        test_window_days=60,
        step_days=30,
    )
)
```

---

### Phase 4: 模型层（三层架构）

```
基线模型 (Champion)
├── 简单平均 - Sharpe=0.594 (当前基线)
└── Ridge回归 - 待测试

主模型 (Challenger)
└── LightGBM - Sharpe=1.093 ✅ (当前最佳)

备选
├── XGBoost
└── ElasticNet
```

**Champion-Challenger机制**:
- 当前线上版本是 champion
- 新版本是 challenger
- challenger必须持续占优才晋升

---

### Phase 5: 分数层

模型输出(score) → 组合之间加一层：

```python
from src.signals.score_layer import ScoreValidator, CompositeScore

# 分数清洗
validator = ScoreValidator()
is_healthy = validator.check_health(scores)

# 分数融合
composite = CompositeScore()
final_score = composite.blend([
    ('lightgbm', 0.7),
    ('simple_average', 0.3),
])
```

---

### Phase 6: 组合层

**组件**:
1. **PositionBuffer** - 持仓缓冲区
2. **WeightSmoother** - 权重平滑
3. **CostAlphaFilter** - 成本-收益过滤

```python
from src.portfolio.enhancer import PortfolioEnhancer, BufferConfig, SmootherConfig, CostFilterConfig

enhancer = PortfolioEnhancer(
    buffer_config=BufferConfig(retain_threshold_rank=50, max_retain_ratio=0.6),
    smoother_config=SmootherConfig(step_ratio=0.5),
    cost_config=CostFilterConfig(min_alpha_threshold=0.002, cost_to_alpha_ratio=0.3),
)

enhanced_weights, summary = enhancer.enhance(
    candidates=candidates,
    current_positions=current_positions,
    target_weights=target_weights,
    prices=prices,
    execution_date=execution_date,
    total_equity=250000,
)
```

---

### Phase 7: 风控层（四层）

```
Layer 1: 模型健康风控
├── Rolling IC监控
├── 分数离散度检测
└── Fallback到基线

Layer 2: 组合结构风控
├── 单票/行业上限
└── 流动性门槛

Layer 3: 总仓位风控
├── 市场状态切换
└── 风险等级对应仓位

Layer 4: 成本风控
├── 换手率上限
└── 边际alpha过滤
```

---

### Phase 8: Walk-Forward验证

滚动样本外，不是单次切分：

```
Window 1: Train[2020-2022) → Test[2022) → OOS_1
Window 2: Train[2020-2022) → Test[2023) → OOS_2
Window 3: Train[2020-2022) → Test[2024) → OOS_3
...
聚合: mean(OOS), worst(OOS), regime_breakdown
```

**验证指标**:
- OOS平均表现
- OOS最差窗口
- 不同regime下表现
- IS/OOS比值

**Ridge+Support Walk-Forward OOS结果 (2026-04-12):**

| 指标 | 值 | 说明 |
|------|-----|------|
| 窗口数 | 54 | Train 500d, Test 60d, Step 30d |
| OOS Sharpe | 0.150 | 显著低于IS Sharpe=1.277 |
| OOS Annual Return | 1.03% | 年化收益较低 |
| OOS Annual Vol | 6.86% | 波动率 |
| Mean IC | 0.030 | 测试窗口内平均IC |
| Mean IC IR | 0.308 | IC稳定性 |
| Win Rate | 59.3% | 正收益窗口比例 |
| Excess Win Rate | 64.8% | 跑赢市场比例 |
| Bull Return | 11.00% | 牛市窗口平均收益 |
| Bear Return | -2.44% | 熊市窗口平均收益 |
| Worst Window | -23.27% | 最差窗口 |
| Best Window | 54.17% | 最佳窗口 |

**⚠️ 关键发现:**
- IS Sharpe=1.277 → OOS Sharpe=0.150, 下降87%
- IS与OOS存在显著差距，说明存在过拟合
- IC IR=0.308 仍为正，说明因子有效，但收益不稳定
- 熊市平均-2.44%，有一定保护能力

---

### Phase 9: 监控层

上线前就按上线思维设计：

| 监控组 | 指标 | 频率 |
|--------|------|------|
| 信号健康 | Rolling IC, 覆盖率, 单调性 | 日 |
| 组合健康 | 换手, 集中度, 行业暴露 | 日 |
| 版本健康 | Champion退化, Challenger增量 | 周 |

---

### 当前最佳配置 (2026年4月)

```yaml
# 小资金(25万)最优配置 - 经策略发现流程验证
model:
  family: ridge_regression
  params:
    alpha: 1.0

features:
  names: [earnings_yield, operating_margin, gross_margin, 
          roe_change, profit_growth, revenue_growth]

portfolio:
  top_n: 15
  min_trade_value: 2000
  risk_model: none  # 小资金禁用

# 策略发现结果 (2026-04-12)
# 因子筛选: 从26个候选因子 → 6个精选因子
# 筛选条件: IC IR > 0.05, 去冗余(corr < 0.7)

# OOS验证结果 (54窗口)
# - OOS Sharpe: 0.160
# - OOS Annual Return: 1.22%
# - Win Rate: 59.3%
# - Excess Win Rate: 61.1%
# - Bull Return: 12.30%
# - Bear Return: -1.45%
```

### 策略发现流程结果 (2026-04-12)

**单因子IC排名 (Top 10):**

| 排名 | 因子 | IC均值 | IC IR | IC正率 |
|------|------|--------|-------|--------|
| 1 | earnings_yield | 0.041 | 0.368 | 63.9% |
| 2 | roe | 0.041 | 0.363 | 65.7% |
| 3 | operating_margin | 0.027 | 0.307 | 63.6% |
| 4 | ocf_per_share | 0.032 | 0.266 | 60.1% |
| 5 | asset_turnover | 0.018 | 0.198 | 55.4% |
| 6 | equity_growth | 0.019 | 0.190 | 55.8% |
| 7 | gross_margin | 0.018 | 0.169 | 58.5% |
| 8 | roe_change | 0.015 | 0.130 | 54.9% |
| 9 | profit_growth | 0.012 | 0.110 | 56.0% |
| 10 | revenue_growth | 0.011 | 0.092 | 50.2% |

**关键发现:**
- **财务因子主导**: Top 10全部为财务因子，无一技术因子
- **波动率因子全负**: vol类因子IC显著为负 (A股波动率溢价不成立)
- **均线因子全负**: ma类因子IC为负，说明趋势跟随在A股无效
- **价值+盈利+成长**: 三维度均衡覆盖是最佳组合

---

### 新增组件 (2026年4月)

| 组件 | 路径 | 说明 |
|------|------|------|
| PositionBuffer | `src/portfolio/enhancer.py` | 持仓缓冲区 |
| WeightSmoother | `src/portfolio/enhancer.py` | 权重平滑 |
| CostAlphaFilter | `src/portfolio/enhancer.py` | 成本-收益过滤(已校准) |
| PortfolioEnhancer | `src/portfolio/enhancer.py` | 组合增强器 |
| WalkForwardValidator | `src/walk_forward_validator.py` | **真实OOS验证** |
| FactorFrequencyManager | `src/features/factor_frequency_manager.py` | **因子频率对齐** |
| SupportFactors | `src/features/support_factors.py` | **条件/交互因子** |
| FeaturePreprocessor | `src/features/feature_preprocessor.py` | **缺失值标记** |
| RiskConstraintEngine | `src/risk/risk_engine.py` | **风控约束引擎** |
| CostSensitivityAnalyzer | `src/risk/risk_engine.py` | **成本敏感性分析** |
| ResearchPipeline | `src/research_pipeline.py` | **完整研究流程** |

### 配置文件

| 文件 | 说明 |
|------|------|
| `configs/research_contract_v1.yaml` | 研究合同 |
| `configs/feature_set_v1.yaml` | 特征集v1 |

### 研究流程状态 (2026年4月11日)

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 1: 研究合同 | ✅ 完成 | 冻结6项固定口径 |
| Phase 2: 特征层定版 | ✅ 完成 | 9因子+3个Support因子 |
| Phase 3: 样本标签层 | ✅ 完成 | Walk-Forward协议 |
| Phase 4: 模型层 | ✅ 完成 | 简单平均 vs LightGBM |
| Phase 5: 分数层 | ⚠️ 待完善 | 需集成ScoreValidator |
| Phase 6: 组合层 | ✅ 完成 | Buffer+Smoother+CostFilter |
| Phase 7: 风控层 | ✅ 完成 | 四层约束引擎 |
| Phase 8: OOS验证 | ✅ 完成 | 真实滚动OOS |
| Phase 9: 监控层 | ⚠️ 待完善 | 需集成监控仪表盘 |

### 关键发现 (2026-04-12更新)

**三模型对比 (CLI实验验证):**
```
实验                     总收益     Sharpe       IC        IC IR      成本
-----------------------------------------------------------------------------------
简单平均                 5.3%      0.333    0.025     0.096     19,794
Ridge回归               64.5%      1.066    0.080     0.529     2,361
Ridge+Support因子       62.7%      1.277    0.106     0.615     4,579
LightGBM               70.0%      1.093    0.036     0.195     5,913
```

**关键结论:**
- **Ridge回归显著优于简单平均** - Sharpe 1.066 vs 0.333
- **Ridge+Support最大回撤最小** - -9.0%，夏普最高1.277
- **财务因子+技术因子组合有效** - IC从0.025提升到0.106

**⚠️ 重要发现:**
- alpha_004简化版IC=0.144是**错误的** - 原版IC=-0.001
- 已从因子池中剔除简化版alpha_004

**推荐策略:**
- 主模型: Ridge回归 (稳定、可解释、低换手)
- 备选: Ridge+Support因子 (最高Sharpe 1.277)
- 监控: 定期Walk-Forward验证，发现退化及时降权

### 下一步研究

1. ✅ Ridge回归基线 - **已完成**
2. ✅ Support因子集成 - **已完成**
3. ✅ Walk-Forward OOS验证 - **已完成**
4. ✅ 成本敏感性分析 - **已完成**
5. ✅ CLI实验验证 - **已完成** (Ridge Sharpe=1.066, Ridge+Support Sharpe=1.277)
6. **实盘监控仪表盘** - Streamlit dashboard

---

## 第二次策略发现结果 (2026-04-12)

### 完整IC测试结果 (100因子)

| 排名 | 因子 | IC | IC IR | 正率 | 来源 |
|------|------|-----|-------|------|------|
| 1 | alpha_017 | 0.094 | 0.588 | 73% | WorldQuant |
| 2 | earnings_yield | 0.041 | 0.368 | 64% | 财务 |
| 3 | roe | 0.041 | 0.363 | 66% | 财务 |
| 4 | operating_margin | 0.027 | 0.307 | 64% | 财务 |
| 5 | alpha_006 | 0.035 | 0.277 | 61% | WorldQuant |
| 6 | ocf_per_share | 0.032 | 0.266 | 60% | 财务 |
| 7 | alpha_003 | 0.034 | 0.257 | 61% | WorldQuant |
| 8 | alpha_002 | 0.030 | 0.227 | 61% | WorldQuant |
| 9 | asset_turnover | 0.018 | 0.198 | 55% | 财务 |
| 10 | equity_growth | 0.019 | 0.190 | 56% | 财务 |

**⚠️ alpha_004已剔除** - 简化版IC=0.144是错误的，原版IC=-0.001

**负向有效因子 (可做空):**
| 因子 | IC | IC IR | 说明 |
|------|-----|-------|------|
| volatility_120 | -0.048 | -0.187 | 波动率越高收益越低 |
| volatility_20 | -0.042 | -0.183 | 短期波动率 |
| mom20 | -0.025 | -0.139 | 短期反转 |

### 选中因子 (Top 10)

```
alpha_004, alpha_017, earnings_yield, roe, alpha_006,
alpha_002, ocf_per_share, operating_margin, asset_turnover, equity_growth
```

**去冗余后:** 剔除alpha_003 (与alpha_006相关性0.97)

### 新策略OOS验证结果 (正确版本)

| 指标 | Ridge基线 | 新策略(Top 10) | 说明 |
|------|----------|----------------|------|
| 窗口数 | 27 | 42 | 相同时间段2022-09后 |
| 平均单期收益 | 5.93% | 2.94% (做多) | Ridge更高 |
| Sharpe | 0.774 | 1.233 (做多) | **新策略+59%** |
| 胜率 | 63.0% | 71.4% (做多) | 新策略更高 |

**注意**: 之前的"Sharpe 2.787"是spread（假设完美做空），不是真实组合收益。

### 关键发现

1. **WorldQuant Alpha主导** - alpha_017是最佳Alpha因子 (IC=0.094, IR=0.588)
2. **财务因子仍是基石** - earnings_yield, roe, operating_margin提供稳定信号
3. **Ridge回归显著优于简单平均** - Sharpe 1.066 vs 0.333

### 重要提醒

⚠️ **alpha_004已剔除** - 简化版IC=0.144是错误的，原版IC=-0.001

### 脚本清单

- `scripts/full_factor_ic_test.py` - 完整IC测试 (100因子)
- `scripts/strategy_discovery_v2.py` - 第二次策略发现
- `scripts/verify_new_strategy.py` - 新策略验证
