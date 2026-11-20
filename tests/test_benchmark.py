import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).parents[1] / 'examples/benchmark_backends.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_missing_cloud_configuration_does_not_send_request(tmp_path, monkeypatch):
    for key in ('ARK_API_KEY', 'DOUBAO_MODEL'):
        monkeypatch.delenv(key, raising=False)
    def unexpected_call(**kwargs):
        raise AssertionError('Must not construct cloud client without configuration')
    monkeypatch.setattr(benchmark, 'OpenAICompatibleBackend', unexpected_call)
    references = tmp_path / 'reference.json'
    references.write_text(json.dumps({'samples': []}))
    output = tmp_path / '.private' / 'doubao'
    code = benchmark.main(['--provider', 'doubao', '--references', str(references), '--output', str(output)])
    assert code == 2
    availability = json.loads((output / 'availability.json').read_text())
    assert availability['requests_sent'] == 0
    assert availability['status'] == 'not_run_missing_configuration'
    assert not (output / 'results.json').exists()


def test_total_and_triple_scoring_do_not_count_missing_reference_as_success():
    rows = [
        {'category': 'vat', 'seconds': 1., 'checks': {'total': True, 'amount': False, 'tax': True},
         'total_reference_printed': True, 'has_triple_reference': True, 'amounts_complete': False, 'amounts_balanced': False},
        {'category': 'rail', 'seconds': .2, 'checks': {'total': True}, 'total_reference_printed': True,
         'has_triple_reference': False, 'amounts_complete': False, 'amounts_balanced': False},
        {'category': 'vat', 'seconds': 1., 'checks': {'total': True}, 'total_reference_printed': False,
         'has_triple_reference': False, 'amounts_complete': False, 'amounts_balanced': False},
    ]
    result = benchmark.summarize(rows)
    assert result['all']['total_correct'] == 2 and result['all']['total_checked'] == 2
    assert result['all']['triple_checked'] == 1 and result['all']['triple_correct'] == 0
