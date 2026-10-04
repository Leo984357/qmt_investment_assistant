# 首页结果口径核对

核对日期：2026-10-04。范围：当前公开仓库中的代码、配置和文档；本次没有重新运行历史行情实验。

## 旧版业绩摘要

旧 README 曾列出 Ridge 累计收益 `42.3%`、LightGBM 累计收益 `81.87%`，以及相应夏普、回撤和 IC。精确数值检索未找到与这些汇总行对应的公开运行档案。研究文档和配置的日期也不能代替某次运行的回测区间。

因此，首页改为展示可检查的研究流程和产物结构。恢复业绩展示时，应提供：

- 配置文件、提交版本和 `run_id`；
- 数据快照、股票池口径与数据可得时间；
- 研究区间、实际回测区间及独立验证窗口；
- 成本、基准、净值、交易和指标计算记录；
- 研究阶段与策略 Gate 结果。

## 当前实现

- [`src/services/execute.py`](../src/services/execute.py) 的执行分支只支持 `mock`。
- [`src/adapters/qmt/`](../src/adapters/qmt/) 的账户、执行与对账接口仍抛出 `NotImplementedError`。
- [`hs300_single_close_to_high250.yaml`](../configs/experiments/hs300_single_close_to_high250.yaml) 标记为 `discovery`，配置数据范围为 `2022-09-01` 至 `2026-03-28`。
- [`src/experiment/runner.py`](../src/experiment/runner.py) 定义运行档案、因子诊断和策略评审输出。

仓库的 `pyproject.toml` 声明了 MIT，但当前没有独立 `LICENSE` 文件。首页已移除失效的许可证徽章；本次没有新增或更改授权条款。
