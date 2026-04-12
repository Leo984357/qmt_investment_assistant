from src.adapters.mock.account import reset_mock_account
from src.services.research import run_research
from src.services.decision import run_decision
from src.services.execute import run_execute
from src.services.review import run_review


def test_smoke_end_to_end():
    reset_mock_account()
    r = run_research()
    assert 'run_id' in r
    d = run_decision()
    assert d['packet'].action in {'hold', 'rebalance'}
    e = run_execute()
    assert 'execution_result' in e
    assert e['execution_result']['cash'] >= 0
    rv = run_review()
    assert 'review' in rv
