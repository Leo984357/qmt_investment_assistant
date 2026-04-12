import json
import subprocess
import sys

from src.cli import main


def test_cli_pipeline_command(capsys):
    exit_code = main(['pipeline', '--reset-mock'])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload['research']['run_id'].startswith('research_')
    assert payload['decision']['packet']['as_of_date'] == payload['research']['as_of_date']


def test_cli_module_entrypoint():
    proc = subprocess.run(
        [sys.executable, '-m', 'src.cli', 'runs', '--limit', '1'],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert 'runs' in payload
