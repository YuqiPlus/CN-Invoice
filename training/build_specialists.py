"""Split an already isolated teacher corpus into strict document-family datasets.

No new labels are invented. Legacy railway tickets and taxi receipts cannot train
new electronic railway invoices or ride itinerary sheets. Uncertain cases stay
in a private audit file. Preserve the original train/validation groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from cninvoice.profiles import DocumentProfile, SPECIALIST_FIELDS, route_profile


EXPECTED_CATEGORY = {
    DocumentProfile.VAT: {'vat_invoice'},
    DocumentProfile.RAIL_EINVOICE: {'rail_ticket','rail_einvoice'},
    DocumentProfile.RIDE_ITINERARY: {'ride_itinerary'},
    DocumentProfile.AIR_ITINERARY: {'air_itinerary'},
}


def ocr_path(row):
    direct = Path(row.get('ocr_file', '/not-found'))
    if direct.is_file():
        return direct
    return Path(row['teacher_result_file']).parent / 'ocr' / (row['id'] + '.json')


def partition(manifest, dataset_root, load_texts):
    selected = {profile: [] for profile in SPECIALIST_FIELDS}
    audit = []
    groups = {'train': set(), 'val': set()}
    for row in manifest['records']:
        if row['split'] not in groups:
            raise ValueError('Only previously isolated train/validation records are accepted')
        groups[row['split']].add(row['group'])
    if groups['train'] & groups['val']:
        raise ValueError('Known identities overlap train and validation')
    for row in manifest['records']:
        texts = load_texts(row)
        decision = route_profile(texts)
        profile = decision.profile
        reason = decision.reason
        # Screenshots of invoice-management UIs are not isolated invoice images.
        if any(word in ''.join(texts) for word in ['上传发票', '发票管理', '发票列表', '确认发票']):
            reason = 'invoice_management_screenshot'
        elif profile in selected and row['category'] not in EXPECTED_CATEGORY[profile]:
            reason = 'teacher_category_conflict_requires_reannotation'
        elif profile in selected:
            annotations = [a for a in row['annotations'] if a['field'] in SPECIALIST_FIELDS[profile]]
            if annotations:
                path = (dataset_root / row['file']).resolve()
                selected[profile].append({**row, 'file': str(path), 'annotations': annotations,
                    'document_profile': profile.value, 'routing_basis': decision.reason})
                reason = 'selected'
            else:
                reason = 'no_matching_teacher_fields'
        audit.append({'id': row['id'], 'old_category': row['category'],
                      'profile': profile.value, 'reason': reason})
    return selected, audit


def readiness(records):
    counts = Counter(row['split'] for row in records)
    groups = {s: len({row['group'] for row in records if row['split'] == s}) for s in ('train','val')}
    # A minimum for a pilot run only, not a claim of production sufficiency.
    ready = counts['train'] >= 48 and counts['val'] >= 12 and groups['train'] >= 12 and groups['val'] >= 3
    return {'images':len(records), 'train':counts['train'], 'val':counts['val'],
            'known_groups':groups, 'status':'pilot_ready' if ready else 'insufficient_data'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if '.private' not in args.output.resolve().parts:
        parser.error('Raw training data must remain private')
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.dataset.read_bytes()
    manifest = json.loads(source)
    def load(row):
        path = ocr_path(row)
        if not path.is_file():
            raise ValueError('Missing OCR provenance; cannot classify a teacher image')
        data = json.loads(path.read_text())
        return [line['text'] for line in data]
    selected, audit = partition(manifest, args.dataset.parent, load)
    summary = {'source_dataset_sha256':hashlib.sha256(source).hexdigest(),
        'source_images':len(manifest['records']), 'profiles':{},
        'audit_reasons':dict(Counter(row['reason'] for row in audit)),
        'notes':['Titles are conservative OCR-derived family labels, not a trained classifier or human-verified truth.',
                 'Field values/boxes are preserved Lite supervision; no arithmetic completion or relabelling.',
                 'Original known-identity splits and test exclusions are inherited without moving images.',
                 'Pilot minimum: 48/12 images and 12/3 known groups in train/val. This is not business accuracy.']}
    for profile, records in selected.items():
        directory = args.output / profile.value
        directory.mkdir(exist_ok=True)
        data = {'version':3, 'document_profile':profile.value, 'fields':list(SPECIALIST_FIELDS[profile]),
            'records':records, 'counts':dict(Counter(r['split']+':'+r['document_profile'] for r in records)),
            'source_dataset_sha256':summary['source_dataset_sha256'], 'notes':summary['notes']}
        path = directory / 'dataset.json'
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
        summary['profiles'][profile.value] = {**readiness(records),
            'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'field_counts':dict(Counter(a['field'] for r in records for a in r['annotations']))}
    (args.output / 'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    (args.output / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
