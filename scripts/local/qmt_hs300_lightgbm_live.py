# -*- coding: utf-8 -*-
"""
HS300 LightGBM 多因子策略（QMT / MiniQMT 外部 Python 版）

说明：
1. 本脚本基于你给的 ATrader prototype 改写，保留核心逻辑：
   - 因子：mom20 / mom60 / mom120 / rev5 / vol20 / vol60 / liq20
   - 标签：未来 20 日收益
   - 模型：LightGBM 回归
   - 调仓：每 5 个交易日调仓一次，TOP_N=25 等权
   - 风控：伪指数 60/120 日均线 + 20 日动量 控制总仓位
   - fallback：模型不可用时退回 60 日动量排序
2. 运行方式：在本地 Python IDE 中运行，依赖 MiniQMT + xtquant。
3. 默认执行模式：
   - 启动后先“预热训练”一次；
   - 每个交易日到 RUN_TIME 后判断一次；
   - 使用“上一完整交易日”的日线数据产生信号，在当天执行交易。
4. 实盘前请先仿真盘验证。
"""

from __future__ import annotations

import os
import time
import math
import json
import traceback
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from xtquant import xtdata
from xtquant import xtconstant
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount


# =========================
# 配置区：你只需要先改这里
# =========================

QMT_PATH = r"D:\QMT\userdata_mini"          # 改成你本机 MiniQMT userdata_mini 路径
ACCOUNT_ID = "12345678"                      # 改成你的证券账号
ACCOUNT_TYPE = "STOCK"                       # 股票账户用 STOCK
SESSION_ID = 2026031301                       # 不同策略给不同 session_id
STRATEGY_NAME = "HS300_LGBM_V3_QMT"

SECTOR_NAME = "沪深300"                      # QMT 板块名称
RUN_TIME = "09:35:00"                        # 每个交易日几点执行；默认用昨日完整日线，今天 09:35 下单
POLL_SECONDS = 15                             # 主循环轮询间隔
PRICE_SLIPPAGE_PCT = 0.002                    # 限价滑点 0.2%
MIN_ORDER_LOT = 100                           # A股买入一手 = 100 股

# 历史下载 / 训练范围
DOWNLOAD_START = "20200101"                  # 首次运行建议更长；后续可增量下载
ANCHOR_DATE = "20220901"                     # 用于定义“每 5 个交易日调仓”的锚点
PERIOD = "1d"
DIVIDEND_TYPE = "front"

# 核心超参数（与 prototype 保持一致）
LABEL_HORIZON = 20
WIN_LEN = 121
TRAIN_DAYS = 380
REBALANCE_GAP = 5
TOP_N = 25

# 本地状态文件
STATE_PATH = os.path.join(os.path.dirname(__file__), "qmt_hs300_lightgbm_state.json")


# =========================
# 工具函数
# =========================

def log(*args):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]", *args)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        s = 1.0
    return (x - m) / s


def _fill_nan_2d(arr: np.ndarray) -> np.ndarray:
    x = np.array(arr, dtype="float64")
    row_mean = np.nanmean(x, axis=1)
    global_mean = np.nanmean(x)
    if np.isnan(global_mean):
        global_mean = 0.0
    row_mean = np.where(np.isnan(row_mean), global_mean, row_mean)
    idxs = np.where(np.isnan(x))
    if len(idxs[0]) > 0:
        x[idxs] = np.take(row_mean, idxs[0])
    return x


def build_features_window(close_win: np.ndarray, vol_win: np.ndarray) -> np.ndarray:
    """
    close_win, vol_win: (n_stocks, L), L >= 121
    使用窗口最后一个截面生成特征
    """
    price_t = close_win[:, -1]

    mom20 = price_t / close_win[:, -21] - 1.0
    mom60 = price_t / close_win[:, -61] - 1.0
    mom120 = price_t / close_win[:, -121] - 1.0
    rev5 = -(price_t / close_win[:, -6] - 1.0)

    seg20 = close_win[:, -21:]
    ret20 = seg20[:, 1:] / seg20[:, :-1] - 1.0
    vol20 = np.std(ret20, axis=1)

    seg60 = close_win[:, -61:]
    ret60 = seg60[:, 1:] / seg60[:, :-1] - 1.0
    vol60 = np.std(ret60, axis=1)

    vol20_mean = np.mean(vol_win[:, -20:], axis=1)
    vol60_mean = np.mean(vol_win[:, -60:], axis=1)
    liq20 = vol20_mean / (vol60_mean + 1e-8)

    feats = np.column_stack([mom20, mom60, mom120, rev5, vol20, vol60, liq20])
    return feats


def build_panel_lightgbm(close: np.ndarray, volume: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    n_stocks, T = close.shape

    if T < WIN_LEN + LABEL_HORIZON:
        return None, None, None

    min_t = WIN_LEN - 1
    max_t_allowed = T - 1 - LABEL_HORIZON
    max_t = min(max_t_allowed, min_t + TRAIN_DAYS)

    if max_t <= min_t:
        return None, None, None

    X_list, y_list = [], []

    for t in range(min_t, max_t + 1):
        close_win = close[:, t - (WIN_LEN - 1): t + 1]
        vol_win = volume[:, t - (WIN_LEN - 1): t + 1]
        feats_t = build_features_window(close_win, vol_win)
        y_t = close[:, t + LABEL_HORIZON] / close[:, t] - 1.0
        X_list.append(feats_t)
        y_list.append(y_t)

    X_panel = np.vstack(X_list)
    y_panel = np.concatenate(y_list)

    close_win_today = close[:, -WIN_LEN:]
    vol_win_today = volume[:, -WIN_LEN:]
    X_today = build_features_window(close_win_today, vol_win_today)

    return X_panel, y_panel, X_today


def compute_market_risk_mult(close: np.ndarray) -> float:
    idx_price = np.nanmean(close, axis=0)
    if len(idx_price) < 120:
        return 1.0

    idx_now = idx_price[-1]
    ma60 = np.mean(idx_price[-60:])
    ma120 = np.mean(idx_price[-120:])
    mom20 = idx_now / idx_price[-21] - 1.0

    risk_mult = 1.0
    if idx_now >= ma60:
        risk_mult = 1.0
    elif idx_now >= ma120:
        risk_mult = 0.85
    else:
        risk_mult = 0.65
        if mom20 < 0:
            risk_mult = 0.45
    return float(risk_mult)


def floor_lot_buy(volume: int) -> int:
    if volume <= 0:
        return 0
    return int(volume / MIN_ORDER_LOT) * MIN_ORDER_LOT


def ceil_price_buy(p: float) -> float:
    return round(p * (1.0 + PRICE_SLIPPAGE_PCT), 3)


def floor_price_sell(p: float) -> float:
    return round(p * (1.0 - PRICE_SLIPPAGE_PCT), 3)


def safe_getattr(obj, names: List[str], default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def load_json(path: str, default: dict) -> dict:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# QMT 封装
# =========================

class TraderCallback(XtQuantTraderCallback):
    def on_disconnected(self):
        log("交易连接断开")

    def on_stock_order(self, order):
        try:
            log("委托回报:", order.stock_code, safe_getattr(order, ["order_volume", "m_nVolume"], "?"), safe_getattr(order, ["order_status", "m_nOrderStatus"], "?"))
        except Exception:
            log("委托回报(打印失败)")

    def on_stock_trade(self, trade):
        try:
            log("成交回报:", trade.stock_code, safe_getattr(trade, ["traded_volume", "m_nVolume"], "?"), safe_getattr(trade, ["traded_price", "m_dPrice"], "?"))
        except Exception:
            log("成交回报(打印失败)")

    def on_order_error(self, order_error):
        log("委托错误:", order_error)

    def on_cancel_error(self, cancel_error):
        log("撤单错误:", cancel_error)


@dataclass
class StrategyState:
    trader: Optional[XtQuantTrader] = None
    account: Optional[StockAccount] = None
    stock_list: List[str] = field(default_factory=list)
    model: Optional[object] = None
    last_signal_trade_day: Optional[str] = None
    last_run_date: Optional[str] = None
    last_warmup_time: Optional[str] = None
    latest_scores: Optional[np.ndarray] = None
    latest_risk_mult: float = 1.0


class QMTLightGBMStrategy:
    def __init__(self):
        self.state = StrategyState()
        self.local_state = load_json(STATE_PATH, default={})

    # ---------- 连接 ----------
    def connect(self):
        log("连接交易端...")
        trader = XtQuantTrader(QMT_PATH, SESSION_ID)
        callback = TraderCallback()
        trader.register_callback(callback)
        trader.start()

        ret = trader.connect()
        if ret != 0:
            raise RuntimeError(f"交易连接失败，返回码={ret}")

        account = StockAccount(ACCOUNT_ID, ACCOUNT_TYPE)
        sub_ret = trader.subscribe(account)
        if sub_ret != 0:
            raise RuntimeError(f"账号订阅失败，返回码={sub_ret}")

        self.state.trader = trader
        self.state.account = account
        log("交易连接成功")

    def prepare_universe(self):
        log("下载板块信息...")
        xtdata.download_sector_data()
        stock_list = xtdata.get_stock_list_in_sector(SECTOR_NAME)
        if not stock_list:
            raise RuntimeError(f"板块 {SECTOR_NAME} 为空，请检查 QMT 板块数据")

        stock_list = sorted(set(stock_list))
        self.state.stock_list = stock_list
        log(f"股票池加载完成，数量={len(stock_list)}")

    # ---------- 行情 ----------
    def warmup_download(self):
        log("开始预热下载历史数据...（首次可能较慢）")
        for i, code in enumerate(self.state.stock_list, start=1):
            try:
                xtdata.download_history_data(code, period=PERIOD, start_time=DOWNLOAD_START, incrementally=True)
            except Exception as e:
                log(f"下载失败 {code}: {e}")
            if i % 50 == 0:
                log(f"历史数据下载进度 {i}/{len(self.state.stock_list)}")
        self.state.last_warmup_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log("历史数据预热下载完成")

    def get_daily_panel(self, count: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        codes = self.state.stock_list
        data = xtdata.get_market_data(
            field_list=["close", "volume"],
            stock_list=codes,
            period=PERIOD,
            count=count,
            dividend_type=DIVIDEND_TYPE,
            fill_data=True,
        )
        if not data or "close" not in data or "volume" not in data:
            raise RuntimeError("get_market_data 返回为空")

        close_df: pd.DataFrame = data["close"].reindex(codes)
        vol_df: pd.DataFrame = data["volume"].reindex(codes)

        close = close_df.to_numpy(dtype="float64")
        volume = vol_df.to_numpy(dtype="float64")
        close = _fill_nan_2d(close)
        volume = _fill_nan_2d(volume)
        return close, volume, list(close_df.columns)

    def get_latest_tick_prices(self, codes: List[str]) -> Dict[str, dict]:
        try:
            return xtdata.get_full_tick(codes)
        except Exception:
            return {}

    def get_trade_dates(self, start_time: str, end_time: str) -> List[str]:
        ts_list = xtdata.get_trading_dates("SH", start_time=start_time, end_time=end_time, count=-1)
        out = []
        for x in ts_list:
            try:
                d = dt.datetime.fromtimestamp(x / 1000).strftime("%Y%m%d")
            except Exception:
                try:
                    d = dt.datetime.fromtimestamp(x).strftime("%Y%m%d")
                except Exception:
                    continue
            out.append(d)
        return sorted(set(out))

    def get_previous_completed_trade_day(self, today: str) -> Optional[str]:
        trade_dates = self.get_trade_dates(ANCHOR_DATE, today)
        if not trade_dates:
            return None
        if today in trade_dates:
            idx = trade_dates.index(today)
            if idx == 0:
                return None
            return trade_dates[idx - 1]
        return trade_dates[-1]

    def is_rebalance_day(self, today: str) -> bool:
        trade_dates = self.get_trade_dates(ANCHOR_DATE, today)
        if today not in trade_dates:
            return False
        idx = trade_dates.index(today)
        return idx % REBALANCE_GAP == 0

    # ---------- 模型 ----------
    def train_and_score(self) -> Tuple[np.ndarray, float]:
        needed = WIN_LEN + LABEL_HORIZON + TRAIN_DAYS + 10
        close, volume, columns = self.get_daily_panel(count=needed)
        if close.shape[1] < WIN_LEN + LABEL_HORIZON:
            raise RuntimeError(f"日线长度不足，当前 T={close.shape[1]}")

        X_panel, y_panel, X_today = build_panel_lightgbm(close, volume)
        close_today = close[:, -1]
        mom60_today = close_today / close[:, -61] - 1.0
        mom60_today = np.nan_to_num(mom60_today, nan=0.0)

        use_ml = True
        preds_ml = None

        if X_panel is None or y_panel is None or X_today is None:
            use_ml = False
        else:
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
                        objective="regression",
                        metric="l2",
                        learning_rate=0.04,
                        num_leaves=95,
                        max_depth=6,
                        min_data_in_leaf=35,
                        subsample=0.85,
                        subsample_freq=1,
                        colsample_bytree=0.9,
                        lambda_l2=2.0,
                        seed=42,
                        verbosity=-1,
                    )
                    model = lgb.train(params, dtrain, num_boost_round=220)
                    preds_ml = model.predict(X_today)
                    preds_ml = np.nan_to_num(preds_ml, nan=0.0)
                    if np.allclose(preds_ml, preds_ml[0]):
                        use_ml = False
                    else:
                        self.state.model = model
                except Exception:
                    traceback.print_exc()
                    use_ml = False

        if not use_ml:
            scores = _zscore(mom60_today)
            log("模型不可用，使用 fallback: 60 日动量")
        else:
            scores = 0.85 * _zscore(preds_ml) + 0.15 * _zscore(mom60_today)
            log("使用 LightGBM + mom60 融合得分")

        risk_mult = compute_market_risk_mult(close)
        self.state.latest_scores = scores
        self.state.latest_risk_mult = risk_mult

        signal_day = str(columns[-1])
        self.state.last_signal_trade_day = signal_day
        log(f"信号日={signal_day} risk_mult={risk_mult:.2f}")
        return scores, risk_mult

    # ---------- 账户 / 下单 ----------
    def query_positions(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        trader = self.state.trader
        account = self.state.account
        positions = trader.query_stock_positions(account)
        total_dict = {}
        available_dict = {}
        for p in positions:
            code = p.stock_code
            total_dict[code] = int(safe_getattr(p, ["m_nVolume", "volume"], 0) or 0)
            available_dict[code] = int(safe_getattr(p, ["m_nCanUseVolume", "can_use_volume"], 0) or 0)
        return total_dict, available_dict

    def query_cash(self) -> float:
        trader = self.state.trader
        account = self.state.account
        asset = trader.query_stock_asset(account)
        cash = float(safe_getattr(asset, ["m_dCash", "cash"], 0.0) or 0.0)
        return cash

    def compute_total_value(self, position_total: Dict[str, int], latest_price: Dict[str, float], cash: float) -> float:
        mv = 0.0
        for code, vol in position_total.items():
            p = latest_price.get(code, 0.0)
            if p > 0 and vol > 0:
                mv += p * vol
        return cash + mv

    def submit_target_portfolio(self, scores: np.ndarray, risk_mult: float):
        codes = self.state.stock_list
        trader = self.state.trader
        account = self.state.account

        order_idx = np.argsort(-scores)
        pick_idx = order_idx[:TOP_N]
        target_codes = [codes[i] for i in pick_idx]
        target_set = set(target_codes)

        tick_map = self.get_latest_tick_prices(codes)
        latest_price = {}
        buy_price = {}
        sell_price = {}

        for code in codes:
            tick = tick_map.get(code, {}) if isinstance(tick_map, dict) else {}
            last_price = float(tick.get("lastPrice", 0.0) or 0.0)
            ask1 = 0.0
            bid1 = 0.0
            try:
                ask1 = float((tick.get("askPrice") or [0])[0] or 0.0)
            except Exception:
                ask1 = 0.0
            try:
                bid1 = float((tick.get("bidPrice") or [0])[0] or 0.0)
            except Exception:
                bid1 = 0.0

            ref_price = last_price if last_price > 0 else max(ask1, bid1, 0.0)
            latest_price[code] = ref_price
            buy_ref = ask1 if ask1 > 0 else ref_price
            sell_ref = bid1 if bid1 > 0 else ref_price
            buy_price[code] = ceil_price_buy(buy_ref) if buy_ref > 0 else 0.0
            sell_price[code] = floor_price_sell(sell_ref) if sell_ref > 0 else 0.0

        position_total, position_available = self.query_positions()
        cash = self.query_cash()
        total_value = self.compute_total_value(position_total, latest_price, cash)
        if total_value <= 0:
            raise RuntimeError("总资产异常，无法调仓")

        invest_cap = total_value * max(min(risk_mult, 1.0), 0.0)
        target_weight = 1.0 / max(len(target_codes), 1)

        target_volume = {}
        for code in codes:
            p = latest_price.get(code, 0.0)
            if code in target_set and p > 0:
                tv = invest_cap * target_weight
                target_volume[code] = floor_lot_buy(int(tv / p))
            else:
                target_volume[code] = 0

        # 先卖后买
        sell_orders = []
        buy_orders = []
        for code in codes:
            cur = int(position_total.get(code, 0))
            can_sell = int(position_available.get(code, 0))
            tgt = int(target_volume.get(code, 0))
            diff = tgt - cur

            if diff < 0:
                sell_vol = min(abs(diff), can_sell)
                if sell_vol > 0:
                    sell_orders.append((code, sell_vol))
            elif diff > 0:
                buy_vol = floor_lot_buy(diff)
                if buy_vol > 0:
                    buy_orders.append((code, buy_vol))

        log(f"目标持仓数={len(target_codes)} 卖单数={len(sell_orders)} 买单数={len(buy_orders)}")

        for code, vol in sell_orders:
            px = sell_price.get(code, 0.0)
            if px <= 0:
                log(f"跳过卖出 {code}：无有效价格")
                continue
            order_id = trader.order_stock(
                account,
                code,
                xtconstant.STOCK_SELL,
                int(vol),
                xtconstant.FIX_PRICE,
                float(px),
                STRATEGY_NAME,
                "rebalance_sell",
            )
            log(f"卖出委托 {code} vol={vol} px={px} order_id={order_id}")
            time.sleep(0.2)

        # 给卖出一点时间释放资金；你可以按需要调大
        if sell_orders:
            time.sleep(3)

        for code, vol in buy_orders:
            px = buy_price.get(code, 0.0)
            if px <= 0:
                log(f"跳过买入 {code}：无有效价格")
                continue
            order_id = trader.order_stock(
                account,
                code,
                xtconstant.STOCK_BUY,
                int(vol),
                xtconstant.FIX_PRICE,
                float(px),
                STRATEGY_NAME,
                "rebalance_buy",
            )
            log(f"买入委托 {code} vol={vol} px={px} order_id={order_id}")
            time.sleep(0.2)

    # ---------- 调度 ----------
    def do_warmup_training(self):
        log("开始预热训练...")
        scores, risk_mult = self.train_and_score()
        log(f"预热训练完成，scores_len={len(scores)} risk_mult={risk_mult:.2f}")

    def should_run_now(self) -> bool:
        now = dt.datetime.now()
        today = now.strftime("%Y%m%d")
        now_hms = now.strftime("%H:%M:%S")

        # 仅交易日运行
        trade_dates = self.get_trade_dates(today, today)
        if today not in trade_dates:
            return False

        if now_hms < RUN_TIME:
            return False

        last_run_date = self.local_state.get("last_run_date")
        if last_run_date == today:
            return False

        return True

    def run_once(self):
        now = dt.datetime.now()
        today = now.strftime("%Y%m%d")
        prev_trade_day = self.get_previous_completed_trade_day(today)
        if prev_trade_day is None:
            log("无上一完整交易日，跳过")
            return

        # 今天是否轮到调仓
        if not self.is_rebalance_day(today):
            log(f"{today} 非调仓日，跳过")
            self.local_state["last_run_date"] = today
            save_json(STATE_PATH, self.local_state)
            return

        log(f"{today} 为调仓日，基于上一完整交易日 {prev_trade_day} 重新训练并调仓")
        scores, risk_mult = self.train_and_score()
        self.submit_target_portfolio(scores, risk_mult)

        self.local_state["last_run_date"] = today
        self.local_state["last_signal_trade_day"] = self.state.last_signal_trade_day
        self.local_state["last_warmup_time"] = self.state.last_warmup_time
        save_json(STATE_PATH, self.local_state)
        log("本次执行完成")

    def loop_forever(self):
        log("进入主循环")
        while True:
            try:
                if self.should_run_now():
                    self.run_once()
            except Exception as e:
                log("主循环异常:", e)
                traceback.print_exc()
            time.sleep(POLL_SECONDS)


# =========================
# main
# =========================

def main():
    s = QMTLightGBMStrategy()
    s.connect()
    s.prepare_universe()
    s.warmup_download()      # 预热：先把历史数据拉到本地
    s.do_warmup_training()   # 预热：启动先训练一版模型
    s.loop_forever()         # 到时间自动运行


if __name__ == "__main__":
    main()
