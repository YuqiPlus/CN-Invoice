"""Run an explicit backend against a frozen local reference set; raw data stays private."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, median

from cninvoice import FieldVisionBackend, InvoiceReader, OpenAICompatibleBackend, RapidOCRBackend, Source


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(data, field):
    for part in field.split('.'):
        if isinstance(data, dict):
            data = data.get(part)
        elif isinstance(data, list) and part.isdigit() and int(part) < len(data):
            data = data[int(part)]
        else:
            return None
    return data


def same(actual, expected, field):
    if actual is None or expected is None:
        return actual is None and expected is None
    if field in {'amount', 'tax', 'total', 'air.fare', 'air.airport_tax'}:
        try:
            return Decimal(str(actual)) == Decimal(str(expected))
        except InvalidOperation:
            return False
    a, b = str(actual).strip(), str(expected).strip()
    if field.endswith('_station'):
        a, b = [re.sub(r'(?:火车站|车站|站)$', '', re.sub(r'\s', '', v)) for v in (a, b)]
    if field.endswith('_time') and re.fullmatch(r'\d{2}:\d{2}:00', a):
        a = a[:5]
    if field.endswith('carriage') or field.endswith('seat_number'):
        a, b = a.lstrip('0'), b.lstrip('0')
    return a == b


class Recorder:
    def __init__(self, backend):
        self.backend, self.payload = backend, None
        self.source = getattr(backend, 'source', Source.MODEL)

    def recognize(self, images):
        self.payload = self.backend.recognize(images)
        return self.payload


def summarize(cases):
    groups = defaultdict(list)
    for case in cases:
        groups['all'].append(case)
        groups[case['category']].append(case)
    summary = {}
    for name, group in groups.items():
        seconds = sorted(c['seconds'] for c in group)
        triples = [c for c in group if c['has_triple_reference']]
        summary[name] = {
            'images': len(group), 'errors': sum(bool(c.get('error')) for c in group),
            'total_correct': sum(c['checks'].get('total', False) for c in group if c['total_reference_printed']),
            'total_checked': sum(c['total_reference_printed'] for c in group),
            'triple_correct': sum(all(c['checks'].get(k, False) for k in ('amount', 'tax', 'total')) for c in triples),
            'triple_checked': len(triples),
            'fields_correct': sum(sum(c['checks'].values()) for c in group),
            'fields_checked': sum(len(c['checks']) for c in group),
            'complete_amount_triples': sum(c['amounts_complete'] for c in group),
            'balanced_amount_triples': sum(c['amounts_balanced'] for c in group),
            'mean_seconds': round(mean(seconds), 4), 'median_seconds': round(median(seconds), 4),
            'p95_seconds': seconds[math.ceil(.95 * len(seconds)) - 1],
            'api_usage_total': {key: sum(c.get('api_usage', {}).get(key, 0) for c in group)
                                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=['field-vision', 'rapidocr', 'doubao', 'huawei'], required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--ocr-model-dir', type=Path)
    parser.add_argument('--provider-config', type=Path, help='Explicit private YAML; credential values are never written to the manifest')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4 or (args.provider in {'rapidocr','field-vision'} and args.workers != 1):
        parser.error('Cloud workers must be between 1 and 4; local baseline requires one worker')
    if '.private' not in args.output.resolve().parts:
        parser.error('Raw benchmark output must be inside an explicit .private directory')
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(args.references.read_text())
    root = Path(__file__).resolve().parents[1]
    manifest = {
        'provider': args.provider, 'reference_sha256': digest(args.references),
        'sample_count': len(reference['samples']), 'platform': platform.platform(),
        'workers': args.workers,
        'code_sha256': {str(p.relative_to(root)): digest(p) for p in sorted((root / 'src/cninvoice').glob('*.py'))},
        'runner_sha256': digest(__file__),
        'timing': 'Per-request wall time includes image loading, encoding and field extraction; cloud includes network. No cold-start subtraction. Worker count is recorded separately.',
    }
    if args.provider in {'rapidocr','field-vision'}:
        if args.model_dir is None:
            parser.error('--model-dir is required for rapidocr')
        ocr_dir=args.ocr_model_dir if args.provider=='field-vision' else args.model_dir
        if ocr_dir is None:parser.error('--ocr-model-dir is required for field-vision')
        manifest['model_sha256'] = {name: digest(ocr_dir / name) for name in RapidOCRBackend.MODEL_FILES}
        manifest['versions'] = {name: importlib.metadata.version(name) for name in ('rapidocr', 'onnxruntime')}
        manifest['execution'] = 'CPU, 4 intra-op threads, 1 inter-op thread'
        if args.provider=='field-vision':
            manifest['model_sha256'].update({name:digest(args.model_dir/name) for name in ('detector.onnx','model.json')})
            backend=FieldVisionBackend(args.model_dir,ocr_dir)
        else:backend = RapidOCRBackend(args.model_dir)
    else:
        config = {}
        if args.provider_config is not None:
            import yaml
            config = yaml.safe_load(args.provider_config.read_text()) or {}
            if not isinstance(config, dict):
                parser.error('Provider configuration must be a mapping')
        required = ['ARK_API_KEY', 'DOUBAO_MODEL'] if args.provider == 'doubao' else ['HUAWEI_OCR_ENDPOINT', 'HUAWEI_PROJECT_ID', 'HUAWEI_AUTH_TOKEN']
        settings = {key: os.environ.get(key) for key in required}
        if args.provider == 'doubao':
            settings['ARK_API_KEY'] = settings['ARK_API_KEY'] or config.get('api_key')
            settings['DOUBAO_MODEL'] = settings['DOUBAO_MODEL'] or config.get('model_name')
        missing = [key for key in required if not settings.get(key)]
        if missing:
            result = {'status': 'not_run_missing_configuration', 'provider': args.provider, 'missing': missing, 'requests_sent': 0}
            (args.output / 'availability.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(json.dumps(result))
            return 2
        if args.provider == 'doubao':
            endpoint = config.get('base_url') or 'https://ark.cn-beijing.volces.com/api/v3'
            backend = OpenAICompatibleBackend(base_url=endpoint,
                                            model=settings['DOUBAO_MODEL'], api_key=settings['ARK_API_KEY'],
                                            extra_body={'thinking': {'type': 'disabled'}},
                                            image_max_side=2000, image_format='JPEG')
            manifest['model'] = settings['DOUBAO_MODEL']
            manifest['endpoint'] = backend.base_url
            manifest['thinking'] = 'disabled'
            manifest['max_tokens'] = backend.max_tokens
            manifest['image_preprocessing'] = {'max_side': 2000, 'format': 'JPEG', 'quality': 95}
        else:
            from cninvoice.cloud import HuaweiOCRBackend
            backend = HuaweiOCRBackend(endpoint=os.environ['HUAWEI_OCR_ENDPOINT'],
                project_id=os.environ['HUAWEI_PROJECT_ID'], token=os.environ['HUAWEI_AUTH_TOKEN'])
    manifest_file = args.output / 'manifest.json'
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        parser.error('Frozen benchmark configuration changed; use a new output directory')
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    report_file = args.output / 'results.json'
    report = json.loads(report_file.read_text()) if report_file.exists() else {'status': 'running', 'cases': []}
    done = {case['id'] for case in report['cases']}
    def process(sample):
        recorder = Recorder(backend)
        reader = InvoiceReader(recorder)
        path = args.references.parent / sample['file']
        if digest(path) != sample['sha256']:
            raise ValueError('Sample hash changed')
        tick = time.perf_counter()
        recorder.payload = None
        data, error = {}, None
        try:
            invoice = reader.read(path)
            data = invoice.model_dump(mode='json')
            if any(i.code in {'backend_failed', 'rendering_unavailable'} for i in invoice.issues):
                error = 'recognition_failed'
        except Exception as exc:
            error = type(exc).__name__
        seconds = round(time.perf_counter() - tick, 4)
        (args.output / (sample['id'] + '.json')).write_text(json.dumps({'invoice': data, 'backend_payload': recorder.payload}, ensure_ascii=False))
        values = [data.get(key) for key in ('amount', 'tax', 'total')]
        complete = all(value is not None for value in values)
        case = {'id': sample['id'], 'category': sample['category'], 'seconds': seconds,
                'checks': {field: same(get(data, field), expected, field) for field, expected in sample['expected'].items()},
                'total_reference_printed': sample['expected'].get('total') is not None,
                'has_triple_reference': all(sample['expected'].get(key) is not None for key in ('amount', 'tax', 'total')),
                'amounts_complete': complete,
                'amounts_balanced': complete and Decimal(values[0]) + Decimal(values[1]) == Decimal(values[2]),
                'issue_codes': sorted({i['code'] for i in data.get('issues', [])})}
        if error:
            case['error'] = error
        if recorder.payload and isinstance(recorder.payload.get('_backend_usage'), dict):
            case['api_usage'] = recorder.payload['_backend_usage']
            case['backend_model'] = recorder.payload.get('_backend_model')
        return case

    pending = [sample for sample in reference['samples'] if sample['id'] not in done]
    started = time.perf_counter()
    consecutive_errors, aborted = 0, False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, sample) for sample in pending]
        for future in as_completed(futures):
            if future.cancelled():
                continue
            case = future.result()
            report['cases'].append(case)
            report['summary'] = summarize(report['cases'])
            temporary = report_file.with_suffix('.next')
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            temporary.replace(report_file)
            if len(report['cases']) % 10 == 0:
                print(json.dumps({'completed': len(report['cases']), 'samples': len(reference['samples'])}), flush=True)
            consecutive_errors = consecutive_errors + 1 if case.get('error') else 0
            if consecutive_errors >= 3 and not aborted:
                aborted = True
                for queued in futures:
                    queued.cancel()
                print(json.dumps({'status': 'stopping_after_three_consecutive_errors'}), flush=True)
    report['status'] = 'stopped_after_errors' if aborted else 'complete'
    report['last_run_wall_seconds'] = round(time.perf_counter() - started, 4)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'status': report['status'], 'summary': report['summary']}, ensure_ascii=False), flush=True)
    return 1 if aborted else 0


if __name__ == '__main__':
    raise SystemExit(main())
