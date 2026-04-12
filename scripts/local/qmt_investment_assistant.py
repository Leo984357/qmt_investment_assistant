# strategy_turtle_equity.py
"""
HS300 LightGBM 多因子策略 v3（进攻型参数）

- 价量多因子（mom20 / mom60 / mom120 / rev5 / vol20 / vol60 / liq20）
- 目标：未来 20 日收益（剪切极端值）
- 模型：LightGBM 回归，滚动面板训练
- 调仓：每 5 个交易日调仓一次，持有 TOP_N=25 只
- 风控：指数（成分股均价）60/120 日均线 + 20 日动量  → 控制整体仓位（较 v2 更激进）
- fallback：训练失败或预测退化时退回 60 日动量排序
"""

from atrader import *
import numpy as np
import os
import lightgbm as lgb

# ==================== 超参数 ====================

LABEL_HORIZON = 20          # 预测未来 20 日收益
WIN_LEN = 121               # 单次建特征窗口长度（>=121）
TRAIN_DAYS = 380            # 训练面板覆盖的交易日数（越大越稳，但更贴合样本期）
REBALANCE_GAP = 5           # 每 5 日调仓一次（更激进）
TOP_N = 25                  # 持仓股票数量（更集中）


# ==================== 工具函数 ====================

def _zscore(x):
    x = np.asarray(x, dtype='float64')
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        s = 1.0
    return (x - m) / s


def _fill_nan_2d(arr):
    """简单 NaN 填补：按行均值填补，整行 NaN 用全局均值。"""
    x = np.array(arr, dtype='float64')
    row_mean = np.nanmean(x, axis=1)
    global_mean = np.nanmean(x)
    row_mean = np.where(np.isnan(row_mean), global_mean, row_mean)
    idxs = np.where(np.isnan(x))
    x[idxs] = np.take(row_mean, idxs[0])
    return x


def build_features_window(close_win, vol_win):
    """
    基于一个固定长度窗口构造特征。
    close_win, vol_win: shape = (n_stocks, L)，L >= WIN_LEN(=121)
    使用的是最后一个横截面的信息。
    """
    price_t = close_win[:, -1]

    # 动量
    mom20 = price_t / close_win[:, -21] - 1.0
    mom60 = price_t / close_win[:, -61] - 1.0
    mom120 = price_t / close_win[:, -121] - 1.0

    # 近 5 日反转
    rev5 = -(price_t / close_win[:, -6] - 1.0)

    # 波动率：用 20 / 60 日收益的标准差
    seg20 = close_win[:, -21:]          # 共 21 个价格 → 20 个收益
    ret20 = seg20[:, 1:] / seg20[:, :-1] - 1.0
    vol20 = np.std(ret20, axis=1)

    seg60 = close_win[:, -61:]          # 共 61 个价格 → 60 个收益
    ret60 = seg60[:, 1:] / seg60[:, :-1] - 1.0
    vol60 = np.std(ret60, axis=1)

    # 流动性：20 日均量 / 60 日均量
    vol20_mean = np.mean(vol_win[:, -20:], axis=1)
    vol60_mean = np.mean(vol_win[:, -60:], axis=1)
    liq20 = vol20_mean / (vol60_mean + 1e-8)

    feats = np.column_stack(
        [mom20, mom60, mom120, rev5, vol20, vol60, liq20]
    )
    return feats


def build_panel_lightgbm(close, volume):
    """
    从完整 K 线面板构造 LightGBM 训练面板 + 当日特征。

    close, volume: shape = (n_stocks, T)
    返回:
      X_panel: (N_samples, n_features)
      y_panel: (N_samples,)
      X_today: (n_stocks, n_features)
    """
    n_stocks, T = close.shape

    # 至少要有 121 日窗口 + 20 日标签
    if T < WIN_LEN + LABEL_HORIZON:
        return None, None, None

    # t 表示“作为特征的那天”的索引
    min_t = WIN_LEN - 1                      # 最早的一天：保证左边有 120 天历史
    max_t_allowed = T - 1 - LABEL_HORIZON    # 最晚的一天：右侧留 20 天标签
    max_t = min(max_t_allowed, min_t + TRAIN_DAYS)

    if max_t <= min_t:
        return None, None, None

    X_list, y_list = [], []

    for t in range(min_t, max_t + 1):
        # 固定窗口 [t-120, t]
        close_win = close[:, t - (WIN_LEN - 1): t + 1]
        vol_win = volume[:, t - (WIN_LEN - 1): t + 1]

        feats_t = build_features_window(close_win, vol_win)
        y_t = close[:, t + LABEL_HORIZON] / close[:, t] - 1.0

        X_list.append(feats_t)
        y_list.append(y_t)

    X_panel = np.vstack(X_list)
    y_panel = np.concatenate(y_list)

    # 当日特征
    close_win_today = close[:, -WIN_LEN:]
    vol_win_today = volume[:, -WIN_LEN:]
    X_today = build_features_window(close_win_today, vol_win_today)

    return X_panel, y_panel, X_today


def compute_market_risk_mult(close):
    """
    用成分股均价构造“伪指数”，根据 60/120 日均线 + 20 日动量决定整体仓位倍数。
    """
    idx_price = np.nanmean(close, axis=0)
    if len(idx_price) < 120:
        return 1.0

    idx_now = idx_price[-1]
    ma60 = np.mean(idx_price[-60:])
    ma120 = np.mean(idx_price[-120:])
    mom20 = idx_now / idx_price[-21] - 1.0

    # 默认先满仓
    risk_mult = 1.0

    if idx_now >= ma60:
        # 强势：在 60 日均线上方 → 满仓
        risk_mult = 1.0
    elif idx_now >= ma120:
        # 中性偏弱：60 下方但 120 上方 → 0.85
        risk_mult = 0.85
    else:
        # 弱势：跌破 120 日
        risk_mult = 0.65
        if mom20 < 0:
            # 且近 20 日动量为负 → 最低 0.45（比 v2 的 0.2 高）
            risk_mult = 0.45

    return float(risk_mult)


def rebalance_equal_weight(context, close_today, scores, risk_mult=1.0):
    """
    根据 scores 做等权调仓，并用 risk_mult 调整体仓位：
      - risk_mult ∈ (0,1] 决定总资金中有多少用于持股；
      - 排名前 TOP_N 的股票等权持有，其余目标仓位为 0。
    """
    n = len(scores)
    order_idx = np.argsort(-scores)
    pick_idx = order_idx[:TOP_N]
    long_set = set(pick_idx.tolist())

    acc = context.account()
    positions = acc.positions

    # 当前持仓数量
    try:
        if hasattr(positions, 'empty') and ('volume_long' in positions.columns) and not positions.empty:
            vol_series = positions['volume_long']
            volume_long = np.array(vol_series, dtype='float64')
            if len(volume_long) != n:
                volume_long = np.zeros(n, dtype='float64')
        else:
            volume_long = np.zeros(n, dtype='float64')
    except Exception:
        volume_long = np.zeros(n, dtype='float64')

    # 现金
    try:
        raw_cash = acc.cash
        cash_arr = np.array(raw_cash, dtype='float64')
        cash_val = float(cash_arr.sum())
    except Exception:
        cash_val = float(getattr(context, 'initial_cash', 1e8))

    # 持仓市值
    try:
        mv = float(np.sum(volume_long * close_today))
    except Exception:
        mv = 0.0

    total_value = cash_val + mv
    if not np.isfinite(total_value) or total_value <= 0:
        return

    # 根据市场过滤调整可用于持仓的总资金
    invest_cap = total_value * max(min(risk_mult, 1.0), 0.0)
    target_weight = 1.0 / max(len(long_set), 1)

    for i in range(n):
        price = float(close_today[i])
        if price <= 0 or np.isnan(price):
            continue

        if i in long_set:
            target_value = invest_cap * target_weight
            target_vol = int(target_value / price / 100) * 100  # 100 股一手
            if target_vol < 0:
                target_vol = 0
        else:
            target_vol = 0

        order_target_volume(
            account_idx=0,
            target_idx=i,
            target_volume=int(target_vol),
            side=1,
            order_type=2
        )


# ==================== 策略主函数 ====================

def init(context):
    # 注册日 K 线
    reg_kdata(frequency='day', fre_num=1)

    # 回测参数
    context.initial_cash = 1_0000_0000
    set_backtest(
        initial_cash=context.initial_cash,
        margin_rate=1.0,
        slide_price=0,
        price_loc=1,
        deal_type=0,
        limit_type=0
    )

    context.Tlen = len(context.target_list)
    context.bar = 0
    context.last_rebalance_bar = -REBALANCE_GAP


def on_data(context):
    context.bar += 1

    # 历史长度不足
    if context.bar < WIN_LEN + LABEL_HORIZON + 5:
        return

    # 控制调仓频率
    if (context.bar - context.last_rebalance_bar) < REBALANCE_GAP:
        return

    # 取最近一段数据
    max_len = WIN_LEN + LABEL_HORIZON + TRAIN_DAYS + 5
    length = min(context.bar, max_len)

    data = get_reg_kdata(
        reg_idx=context.reg_kdata[0],
        length=length,
        fill_up=True,
        df=True
    )
    if data is None or data.empty:
        return

    close_raw = data['close'].values.reshape(context.Tlen, -1)
    vol_raw = data['volume'].values.reshape(context.Tlen, -1)

    close = _fill_nan_2d(close_raw)
    volume = _fill_nan_2d(vol_raw)

    n_stocks, T = close.shape
    close_today = close[:, -1]

    # ===== 1. 构建训练面板 =====
    X_panel, y_panel, X_today = build_panel_lightgbm(close, volume)

    use_ml = True
    if X_panel is None or y_panel is None or X_today is None:
        use_ml = False

    preds_ml = None
    if use_ml:
        # 剪切极端收益
        y_panel = np.clip(y_panel, -0.3, 0.3)
        mask = np.isfinite(np.sum(X_panel, axis=1)) & np.isfinite(y_panel)
        X_train = X_panel[mask]
        y_train = y_panel[mask]

        if len(y_train) < 600: 
            use_ml = False
        else:
            try:
                dtrain = lgb.Dataset(X_train, label=y_train)
                params = dict(
                    objective='regression',
                    metric='l2',
                    learning_rate=0.04,
                    num_leaves=95,
                    max_depth=6,
                    min_data_in_leaf=35,
                    subsample=0.85,
                    subsample_freq=1,
                    colsample_bytree=0.9,
                    lambda_l2=2.0,
                    seed=42
                )
                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=220,
                    verbose_eval=False
                )
                preds_ml = model.predict(X_today)
                preds_ml = np.nan_to_num(preds_ml, nan=0.0)
                if np.allclose(preds_ml, preds_ml[0]):
                    use_ml = False
            except Exception:
                use_ml = False

    # ===== 2. fallback：60 日动量 =====
    mom60_today = close_today / close[:, -61] - 1.0
    mom60_today = np.nan_to_num(mom60_today, nan=0.0)

    if not use_ml:
        scores = _zscore(mom60_today)
    else:
        # 进攻型：更偏向模型预测
        scores = 0.85 * _zscore(preds_ml) + 0.15 * _zscore(mom60_today)

    # ===== 3. 市场风格过滤：控制整体仓位 =====
    risk_mult = compute_market_risk_mult(close)

    # ===== 4. 等权调仓 =====
    rebalance_equal_weight(context, close_today, scores, risk_mult=risk_mult)
    context.last_rebalance_bar = context.bar


# ==================== 本地直接回测（可选） ====================

if __name__ == '__main__':
    codes = get_code_list('hs300')['code']
    run_backtest(
        strategy_name='ML_LightGBM_HS300_v3',
        file_path=os.path.basename(__file__),
        target_list=codes,
        frequency='day',
        fre_num=1,
        begin_date='2020-09-01',
        end_date='2025-09-01',
        fq=1
    )