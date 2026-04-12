from src.core.config import load_app_config
from src.core.experiment_store import latest_runs
from src.services.pipeline import run_pipeline


def main():
    result = run_pipeline(reset_account=True)
    research = result['research']
    decision = result['decision']
    execute = result['execute']
    review = result['review']
    runs = latest_runs(load_app_config().db_path, limit=10)
    print('research_ok', 'run_id' in research)
    print('decision_ok', decision['packet'].action in {'hold', 'rebalance'})
    print('execute_ok', execute['execution_result']['cash'] >= 0)
    print('review_ok', 'review' in review)
    print('latest_runs', len(runs))


if __name__ == '__main__':
    main()
