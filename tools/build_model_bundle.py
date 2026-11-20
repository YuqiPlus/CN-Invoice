"""Build a portable local VAT model bundle; this command never publishes files."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from cninvoice import __version__
from cninvoice.ocr import RapidOCRBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--ocr-dir', type=Path, required=True)
    parser.add_argument('--ocr-license-dir', type=Path, required=True)
    parser.add_argument('--model-card', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', default='cninvoice-vat-models-' + __version__)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}', args.name):
        parser.error('Bundle name must be a simple directory name')
    metadata = json.loads((args.model_dir / 'model.json').read_text())
    if metadata.get('document_profile') != 'vat_invoice':
        parser.error('This builder packages the VAT specialist only')
    dest = args.output.resolve() / args.name
    archive = args.output.resolve() / (args.name + '.zip')
    if dest.exists() or archive.exists():
        parser.error('Output already exists; choose a fresh output directory')
    sources = [(args.model_dir / 'detector.onnx', 'models/vat_invoice/detector.onnx'),
               (args.model_card, 'MODEL_CARD.md')]
    sources += [(args.ocr_dir / name, 'ocr/' + name) for name in RapidOCRBackend.MODEL_FILES]
    sources += [(args.ocr_license_dir / name, 'licenses/ocr/' + name) for name in ['LICENSE', 'MODEL_LICENSES.md']]
    if any(not path.is_file() for path, _ in sources):
        parser.error('A required model, model card or OCR license file is missing')
    dest.mkdir(parents=True)
    for source, relative in sources:
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    runtime = {k: metadata[k] for k in ['architecture', 'document_profile', 'fields', 'input_size',
                                      'score_threshold', 'parameters', 'onnx_bytes']}
    (dest / 'models/vat_invoice/model.json').write_text(json.dumps(runtime, indent=2))
    files = {str(path.relative_to(dest).as_posix()): {
        'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path in sorted(dest.rglob('*')) if path.is_file()}
    manifest = {'format': 'cninvoice-model-bundle', 'version': 1,
                'minimum_cninvoice_version': '0.1.0',
                'specialists': {'vat_invoice': 'models/vat_invoice'}, 'ocr_models': 'ocr',
                'model_card': 'MODEL_CARD.md', 'files': files}
    (dest / 'bundle.json').write_text(json.dumps(manifest, indent=2))
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(dest.rglob('*')):
            if path.is_file():zipped.write(path, arcname=args.name + '/' + path.relative_to(dest).as_posix())
    print(json.dumps({'bundle': str(dest), 'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'published': False}))


if __name__ == '__main__':
    main()
