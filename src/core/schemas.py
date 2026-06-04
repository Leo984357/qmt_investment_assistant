from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class StrategyOutput:
    as_of_date: str
    signals: pd.DataFrame
    target_portfolio: pd.DataFrame
    diagnostics: dict[str, Any]
    equity_curve: pd.DataFrame
    metrics: dict[str, Any]


@dataclass
class DecisionPacket:
    as_of_date: str
    action: str
    rationale: list[str]
    current_portfolio: pd.DataFrame
    target_portfolio: pd.DataFrame
    rebalance_delta: pd.DataFrame
    risk_summary: dict[str, Any]
    constraints_hit: list[str]


@dataclass
class ExecutionPlan:
    as_of_date: str
    broker: str
    orders: pd.DataFrame
    cash_required: float
    warnings: list[str]


@dataclass
class PostTradeReview:
    as_of_date: str
    target_vs_actual: pd.DataFrame
    execution_summary: dict[str, Any]
    deviation_reasons: list[str]
    next_actions: list[str]
