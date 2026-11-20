import hashlib
import json
import shutil

import pytest

from cninvoice import ReaderConfig
from cninvoice.ocr import RapidOCRBackend


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'models'
    paths = ['models/vat_invoice/detector.onnx', 'models/vat_invoice/model.json']
    paths += ['ocr/' + name for name in RapidOCRBackend.MODEL_FILES]
    files = {}
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'model')
        files[name] = {'bytes': 5, 'sha256': hashlib.sha256(b'model').hexdigest()}
    manifest = {'format': 'cninvoice-model-bundle', 'version': 1,
                'specialists': {'vat_invoice': 'models/vat_invoice'}, 'ocr_models': 'ocr',
                'files': files}
    (root / 'bundle.json').write_text(json.dumps(manifest))
    return root, manifest


def save(root, manifest):
    (root / 'bundle.json').write_text(json.dumps(manifest))


def test_bundle_is_portable_and_only_configures_local_models(bundle, tmp_path):
    root, _ = bundle
    relocated = tmp_path / 'another-directory' / 'downloaded-models'
    shutil.copytree(root, relocated)
    config = ReaderConfig.from_bundle(relocated)
    assert config.recognizer == 'specialists'
    assert config.specialist_model_dirs == {'vat_invoice': relocated / 'models/vat_invoice'}
    assert config.ocr_model_dir == relocated / 'ocr'
    assert not config.ai_review and config.endpoint is None and config.api_key is None


def test_changed_model_bytes_are_rejected_even_if_size_is_unchanged(bundle):
    root, _ = bundle
    (root / 'ocr' / RapidOCRBackend.MODEL_FILES[0]).write_bytes(b'wrong')
    with pytest.raises(ValueError, match='checksum does not match'):
        ReaderConfig.from_bundle(root)


def test_required_runtime_files_cannot_be_omitted_from_checksums(bundle):
    root, manifest = bundle
    del manifest['files']['models/vat_invoice/model.json']
    save(root, manifest)
    with pytest.raises(ValueError, match='verified checksums'):
        ReaderConfig.from_bundle(root)


@pytest.mark.parametrize('path', ['../outside', '/outside', 'C:/outside', 'ocr\\outside'])
def test_manifest_cannot_escape_the_bundle(bundle, path):
    root, manifest = bundle
    manifest['ocr_models'] = path
    save(root, manifest)
    with pytest.raises(ValueError, match='paths must'):
        ReaderConfig.from_bundle(root)


def test_checksum_file_symlink_cannot_escape_the_bundle(bundle, tmp_path):
    root, _ = bundle
    external = tmp_path / 'external-model'
    external.write_bytes(b'model')
    path = root / 'models/vat_invoice/detector.onnx'
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(ValueError, match='stay inside'):
        ReaderConfig.from_bundle(root)


def test_manifest_symlink_cannot_read_from_outside_bundle(bundle, tmp_path):
    root, manifest = bundle
    external = tmp_path / 'external.json'
    external.write_text(json.dumps(manifest))
    (root / 'bundle.json').unlink()
    (root / 'bundle.json').symlink_to(external)
    with pytest.raises(ValueError, match='local model bundle'):
        ReaderConfig.from_bundle(root)


@pytest.mark.parametrize('size', [True, -1, 0, 300 * 1024**2])
def test_invalid_or_oversized_file_declarations_are_rejected(bundle, size):
    root, manifest = bundle
    manifest['files']['models/vat_invoice/detector.onnx']['bytes'] = size
    save(root, manifest)
    with pytest.raises(ValueError, match='file size'):
        ReaderConfig.from_bundle(root)


def test_missing_ocr_file_is_rejected(bundle):
    root, _ = bundle
    (root / 'ocr' / RapidOCRBackend.MODEL_FILES[0]).unlink()
    with pytest.raises(ValueError, match='missing or has changed'):
        ReaderConfig.from_bundle(root)


def test_unknown_model_profile_is_rejected(bundle):
    root, manifest = bundle
    manifest['specialists'] = {'taxi_receipt': 'models/vat_invoice'}
    save(root, manifest)
    with pytest.raises(ValueError, match='Unsupported specialist'):
        ReaderConfig.from_bundle(root)


@pytest.mark.parametrize('content', ['[]', 'null', '{broken'])
def test_invalid_manifest_has_a_clear_error(bundle, content):
    root, _ = bundle
    (root / 'bundle.json').write_text(content)
    with pytest.raises(ValueError, match='model bundle'):
        ReaderConfig.from_bundle(root)
