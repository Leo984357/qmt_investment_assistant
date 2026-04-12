from __future__ import annotations

from src.adapters.mock.account import reset_mock_account
from src.services.decision import run_decision
from src.services.execute import run_execute
from src.services.research import run_research
from src.services.review import run_review
from src.services.workflow import create_workflow_state


def run_pipeline(reset_account: bool = False) -> dict:
    if reset_account:
        reset_mock_account()
    state = create_workflow_state()
    research = run_research(state)
    decision = run_decision(state)
    execute = run_execute(state)
    review = run_review(state)
    return {
        'research': research,
        'decision': decision,
        'execute': execute,
        'review': review,
    }
