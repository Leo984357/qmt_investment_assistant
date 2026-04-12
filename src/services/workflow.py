from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.adapters.mock.account import read_account, read_positions
from src.core.config import AppConfig, load_app_config
from src.core.decision_engine import build_decision_packet
from src.core.execution_plan import build_execution_plan
from src.core.schemas import DecisionPacket, ExecutionPlan, StrategyOutput
from src.strategy.adapter import run_strategy


@dataclass
class WorkflowState:
    cfg: AppConfig
    research: StrategyOutput | None = None
    latest_prices: pd.Series | None = None
    account_snapshot: dict | None = None
    current_positions: pd.DataFrame | None = None
    packet: DecisionPacket | None = None
    plan: ExecutionPlan | None = None
    execution_result: dict | None = None


def create_workflow_state(cfg: AppConfig | None = None) -> WorkflowState:
    return WorkflowState(cfg=cfg or load_app_config())


def ensure_research(state: WorkflowState) -> StrategyOutput:
    if state.research is None:
        state.research = run_strategy(state.cfg)
    if state.latest_prices is None:
        state.latest_prices = state.research.signals.set_index('ticker')['latest_price']
    return state.research


def ensure_account_snapshot(state: WorkflowState) -> tuple[dict, pd.DataFrame]:
    ensure_research(state)
    if state.account_snapshot is None:
        state.account_snapshot = read_account()
    if state.current_positions is None:
        state.current_positions = read_positions(state.latest_prices)
    return state.account_snapshot, state.current_positions


def ensure_decision_packet(state: WorkflowState) -> DecisionPacket:
    if state.packet is None:
        research = ensure_research(state)
        account, current_positions = ensure_account_snapshot(state)
        state.packet = build_decision_packet(
            current_portfolio=current_positions,
            target_portfolio=research.target_portfolio,
            latest_prices=state.latest_prices,
            cash=account['cash'],
            total_equity=account['total_equity'],
            as_of_date=research.as_of_date,
            cfg=state.cfg.decision,
        )
    return state.packet


def ensure_execution_plan(state: WorkflowState) -> ExecutionPlan:
    if state.plan is None:
        packet = ensure_decision_packet(state)
        account, _ = ensure_account_snapshot(state)
        state.plan = build_execution_plan(
            packet=packet,
            latest_prices=state.latest_prices,
            total_equity=account['total_equity'],
            cfg=state.cfg.execution,
        )
    return state.plan
