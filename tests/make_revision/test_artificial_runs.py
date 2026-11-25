"""G4 follow-up experiment (c): artificial_runs, the artificial family's settings applied to extra_runs without editing it.
Every check that needs the settings runs in a subprocess, so this session's extra_runs keeps its committed values."""
import json
import subprocess
import sys
from pathlib import Path
import pytest
from experiments.make_revision import extra_ablation as ea
from experiments.make_revision import extra_runs as er
from experiments.make_revision.evaluation import config_id
from experiments.make_revision.run_revision import environment_record

REPO = Path(__file__).resolve().parents[2]
DEDUP_COMMIT = 'ecc7fd81f9b09a5181ef256625a2eb86812d7d89'          # the commit the dedup production run started from


def wrapped(code):
    """Run `code` in a subprocess that imported artificial_runs; return the JSON its last stdout line prints."""
    completed = subprocess.run([sys.executable, '-W', 'ignore', '-c', 'import json\nimport experiments.make_revision.artificial_runs\n' + code],
                               cwd=REPO, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_settings_change_the_artificial_entries_only_and_this_session_keeps_the_committed_values():
    result = wrapped("from experiments.make_revision import extra_runs as er, extra_ablation as ea\n"
                     "from experiments.make_revision.evaluation import config_id\n"
                     "a = er.draft_protocol('artificial')\n"
                     "print(json.dumps({'workers': er.WORKERS, 'cap': er.CAP_HOURS, 'dedup': config_id(er.draft_protocol('dedup')),\n"
                     "                  'artificial': [a['workers'], a['wallclock_cap_hours'], a['registry']],\n"
                     "                  'sealed': ['experiments.make_revision.artificial_runs' in m.SOURCE_MODULES for m in (er, ea)]}))")
    assert result['workers'] == {'dedup': 16, 'artificial': 16} and result['cap'] == {'dedup': 9, 'artificial': 6}
    assert result['artificial'] == [16, 6, er.REGISTRY] and result['sealed'] == [True, True]
    assert result['dedup'] == config_id(er.draft_protocol('dedup'))
    assert er.WORKERS == {'dedup': 16, 'artificial': 8} and er.CAP_HOURS == {'dedup': 9, 'artificial': 4}
    assert 'experiments.make_revision.artificial_runs' not in er.SOURCE_MODULES + ea.SOURCE_MODULES


def test_the_routed_draft_differs_from_the_committed_draft_in_workers_and_cap_only(tmp_path):
    output = tmp_path/'draft.json'
    subprocess.run([sys.executable, '-W', 'ignore', '-m', 'experiments.make_revision.artificial_runs', 'extra_runs', 'draft',
                    '--family', 'artificial', '--output', str(output)], cwd=REPO, check=True, capture_output=True)
    routed, committed = json.loads(output.read_text()), er.draft_protocol('artificial')
    assert sorted(key for key in set(routed) | set(committed) if routed.get(key) != committed.get(key)) == ['wallclock_cap_hours', 'workers']
    assert (routed['workers'], routed['wallclock_cap_hours']) == (16, 6)


def test_the_wrapper_refuses_another_family_and_other_modules(tmp_path):
    runs = [['extra_runs', 'draft', '--family', 'dedup', '--output', str(tmp_path/'draft.json')],
            ['compare_extra', 'analyse', '--family', 'dedup', '--run', str(tmp_path), '--ablation', str(tmp_path), '--output', str(tmp_path/'o')],
            ['extra_runs', 'run', '--protocol', str(er.PROTOCOL_FILES['dedup']), '--output', str(tmp_path/'run')],
            ['holistic', 'benchmark', '--output', str(tmp_path/'h')], []]
    for args in runs:
        completed = subprocess.run([sys.executable, '-W', 'ignore', '-m', 'experiments.make_revision.artificial_runs', *args],
                                   cwd=REPO, capture_output=True, text=True)
        assert completed.returncode != 0 and ('refused' in completed.stderr or 'usage' in completed.stderr), args
    assert sorted(path.name for path in tmp_path.iterdir()) == []


def test_every_file_the_dedup_run_and_its_ablation_seal_is_byte_identical_to_the_dedup_commit():
    try:
        subprocess.run(['git', 'cat-file', '-e', DEDUP_COMMIT], cwd=REPO, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('the dedup commit is not in this repository')
    sealed = set(environment_record(er.REGISTRY)['source_hashes']) | set(ea.environment()['source_hashes'])
    assert 'experiments/make_revision/extra_runs.py' in sealed and 'experiments/make_revision/extra_ablation.py' in sealed
    for path in sorted(sealed):
        blob = subprocess.run(['git', 'show', f'{DEDUP_COMMIT}:arrowflow_repo/{path}'], cwd=REPO, check=True, capture_output=True).stdout
        assert (REPO/path).read_bytes() == blob, path


@pytest.mark.skipif(not er.PROTOCOL_FILES['artificial'].is_file(), reason='the artificial protocol is not frozen yet')
def test_the_frozen_artificial_protocol_validates_under_the_settings_only():
    path = er.PROTOCOL_FILES['artificial']
    protocol = json.loads(path.read_text())
    with pytest.raises(ValueError, match='wallclock_cap_hours, workers'):
        er.validate_extra_protocol(protocol)                   # the committed extra_runs values
    result = wrapped(f"from experiments.make_revision import extra_runs as er\np = json.loads(open({str(path)!r}).read())\n"
                     "er.validate_extra_protocol(p)\nd = er.draft_protocol('artificial')\n"
                     "print(json.dumps({'equal': {k: v for k, v in p.items() if k not in er.FREEZE_FIELDS} == "
                     "{k: v for k, v in d.items() if k not in er.FREEZE_FIELDS}, 'frozen': p['frozen'], 'hours': p['projection']['decision_hours'], "
                     "'workers': p['projection']['workers'], 'cap': p['projection']['cap_hours'], 'id': p['protocol_id']}))")
    assert result['equal'] and result['frozen'] and result['id'] == 'arrowflow-v3-artificial-1'
    assert (result['workers'], result['cap']) == (16, 6) and 0 < result['hours'] <= 6
