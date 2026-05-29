import pandas as pd

from src.evaluation.reporting import performance_summary


def test_performance_summary_uses_active_window_start_nav():
    nav = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=3),
        'nav': [0.95, 1.00, 1.14],
        'daily_return': [0.0, 0.0526316, 0.14],
        'turnover': [0.0, 0.1, 0.0],
        'trade_cost': [0.0, 10.0, 0.0],
    })
    benchmark = pd.DataFrame({
        'trade_date': pd.date_range('2024-01-01', periods=3),
        'benchmark_nav': [1.20, 1.25, 1.32],
    })
    rank_ic = pd.DataFrame({'rank_ic': [0.03, 0.04]})

    summary = performance_summary(nav, pd.DataFrame({'x': [1]}), rank_ic, benchmark)

    assert summary['total_return'] == 0.2
    assert summary['benchmark_total_return'] == 0.1
    assert summary['excess_total_return'] == 0.1
