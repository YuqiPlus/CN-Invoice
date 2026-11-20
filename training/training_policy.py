"""Training-only balancing and validation policy, independent of inference data."""
from collections import Counter, defaultdict


def validate_specialist_manifest(manifest):
    """Missing new-format data must not be replaced with legacy document rows."""
    profile = manifest.get('document_profile')
    if not profile:
        return
    rows = manifest['records']
    if any(row.get('document_profile') != profile for row in rows):
        raise ValueError('Specialist dataset contains a different document family')
    counts = Counter(row['split'] for row in rows)
    groups = {s: {row['group'] for row in rows if row['split'] == s} for s in ('train','val')}
    if groups['train'] & groups['val']:
        raise ValueError('Specialist train and validation identities overlap')
    if (counts['train'] < 48 or counts['val'] < 12 or
            len(groups['train']) < 12 or len(groups['val']) < 3):
        raise ValueError('Insufficient target-family data; legacy tickets are not substitutes')


def category_weights(categories):
    """Square-root balancing keeps common categories represented without erasing rare ones."""
    counts = Counter(categories)
    return [counts[category] ** -.5 for category in categories]


def validation_score(validation, fields, policy):
    if policy == 'micro':
        return validation['recall_iou50'] - .1 * validation['unmatched_detections'] / max(1, validation['fields'])
    if policy != 'macro-category':
        raise ValueError('Unknown checkpoint selection policy')
    kinds = defaultdict(lambda: [0, 0, 0])
    for index, field in enumerate(fields, 1):
        hits, count = validation['by_class'].get(index, [0, 0])
        row = kinds[field.split('.')[0]]
        row[0] += hits
        row[1] += count
        row[2] += validation['unmatched_by_class'].get(index, 0)
    scores = [(hits - .1 * false) / count for hits, count, false in kinds.values() if count]
    if not scores:
        raise ValueError('Validation contains no supervised fields')
    return sum(scores) / len(scores)
