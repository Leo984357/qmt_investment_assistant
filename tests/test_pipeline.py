from src.services.pipeline import run_pipeline


def test_pipeline_preserves_as_of_date_across_stages():
    result = run_pipeline(reset_account=True)

    as_of_date = result['research']['as_of_date']
    assert result['decision']['packet'].as_of_date == as_of_date
    assert result['execute']['plan'].as_of_date == as_of_date
    assert result['review']['review'].as_of_date == as_of_date
