"""Load an explicitly downloaded local model bundle without network access."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from .ocr import RapidOCRBackend
from .profiles import DocumentProfile, SPECIALIST_FIELDS


def bundle_paths(directory: str | Path) -> tuple[dict[str, Path], Path]:
    root = Path(directory).expanduser().resolve()
    manifest = root / 'bundle.json'
    if not manifest.is_file() or not manifest.resolve().is_relative_to(root) or manifest.stat().st_size > 128 * 1024:
        raise ValueError('A local model bundle with bundle.json is required')
    try:
        data = json.loads(manifest.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError):
        raise ValueError('Invalid model bundle manifest') from None
    if not isinstance(data, dict) or data.get('format') != 'cninvoice-model-bundle' or type(data.get('version')) is not int or data['version'] != 1:
        raise ValueError('Unsupported model bundle format')

    def local_path(name):
        if not isinstance(name, str) or not name or '\\' in name or ':' in name:
            raise ValueError('Model bundle paths must be relative to the bundle')
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Model bundle paths must stay inside the bundle')
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Model bundle paths must stay inside the bundle')
        return path

    models = data.get('specialists')
    files = data.get('files')
    if not isinstance(models, dict) or not models or len(models) > 4:
        raise ValueError('Model bundle must declare supported specialists')
    if not isinstance(files, dict) or not 1 <= len(files) <= 64:
        raise ValueError('Model bundle must include file checksums')
    verified = set()
    for name, info in files.items():
        path = local_path(name)
        if not isinstance(info, dict) or not re.fullmatch(r'[a-f0-9]{64}', str(info.get('sha256', ''))):
            raise ValueError('Invalid model bundle file checksum')
        size = info.get('bytes')
        if type(size) is not int or not 0 < size <= 256 * 1024**2:
            raise ValueError('Invalid model bundle file size')
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError('A model bundle file is missing or has changed')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != info['sha256']:
            raise ValueError('A model bundle file checksum does not match')
        verified.add(path)
    directories = {}
    for name, value in models.items():
        try:
            profile = DocumentProfile(name)
        except ValueError:
            raise ValueError('Unsupported specialist in model bundle') from None
        if profile not in SPECIALIST_FIELDS:
            raise ValueError('Unsupported specialist in model bundle')
        path = local_path(value)
        if not {(path / 'detector.onnx').resolve(), (path / 'model.json').resolve()} <= verified:
            raise ValueError('Specialist files must have verified checksums')
        directories[name] = path
    ocr = local_path(data.get('ocr_models'))
    if not {(ocr / name).resolve() for name in RapidOCRBackend.MODEL_FILES} <= verified:
        raise ValueError('OCR model files must have verified checksums')
    return directories, ocr
