"""One shared local OCR pass, followed by exactly one independent field model."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

from .errors import RecognitionError
from .models import Invoice, InvoiceKind, Source
from .ocr import RapidOCRBackend
from .profiles import DocumentProfile, SPECIALIST_FIELDS, route_profile
from .vision import FieldVisionBackend


PROFILE_KINDS = {
    DocumentProfile.VAT: InvoiceKind.VAT,
    DocumentProfile.RAIL_EINVOICE: InvoiceKind.RAIL,
    DocumentProfile.RIDE_ITINERARY: InvoiceKind.RIDE,
    DocumentProfile.AIR_ITINERARY: InvoiceKind.AIR,
}


class SpecialistVisionBackend:
    """Explicit local experts; missing/ambiguous profiles abstain without cloud calls.

    Model directories are independent. The old merged detector is not a fallback.
    Sessions load on demand; OCR weights and OCR results are shared.
    """
    source = Source.OCR

    def __init__(self, model_dirs: Mapping[str, str | Path], ocr_model_dir: str | Path, *,
                 threads: int = 4, document_profile: str | None = None):
        self.model_dirs = {}
        for name, path in model_dirs.items():
            profile = DocumentProfile(name)
            if profile not in SPECIALIST_FIELDS:
                raise ValueError('Only the four target document families can configure specialists')
            directory = Path(path)
            if not (directory / 'model.json').is_file() or not (directory / 'detector.onnx').is_file():
                raise RecognitionError('Specialists require explicit local model files')
            meta = json.loads((directory / 'model.json').read_text())
            fields = meta.get('fields', [])
            if (meta.get('document_profile') != profile.value or meta.get('input_size') != 320 or not fields
                    or len(set(fields)) != len(fields)
                    or not set(fields) <= set(SPECIALIST_FIELDS[profile])):
                raise RecognitionError('Specialist profile or field metadata does not match its configured role')
            self.model_dirs[profile] = directory
        self.document_profile = DocumentProfile(document_profile) if document_profile else None
        if self.document_profile is not None and self.document_profile not in SPECIALIST_FIELDS:
            raise ValueError('Document profile must be one of the four target families')
        self.ocr = RapidOCRBackend(ocr_model_dir, threads=threads)
        self.threads = threads
        self._experts = {}
        self._lock = threading.Lock()

    def recognize(self, images: Sequence[Image.Image]) -> dict:
        if len(images) != 1:
            raise RecognitionError('Specialist recognition currently requires one document page')
        with self._lock:
            baseline = self.ocr.recognize(images)
            decision = route_profile([line['text'] for line in baseline['_ocr_lines']])
            profile = self.document_profile or decision.profile
            conflict = decision.reason == 'ambiguous_titles' or (
                self.document_profile is not None and decision.profile != DocumentProfile.UNKNOWN
                and decision.profile != self.document_profile)
            route = {'profile': profile.value, 'basis': 'explicit_configuration' if self.document_profile else decision.reason}
            if conflict or profile not in self.model_dirs:
                invoice = Invoice(kind=PROFILE_KINDS.get(profile, InvoiceKind.UNKNOWN))
                result = invoice.model_dump(mode='json', exclude={'sources','issues','status','needs_review','raw_text'})
                code = 'specialist_route_conflict' if conflict else (
                    'specialist_unavailable' if profile in SPECIALIST_FIELDS else 'specialist_unsupported_document')
                messages = {
                    'specialist_route_conflict': '票据标题冲突或与指定票种不符，未运行字段模型，请核对文件',
                    'specialist_unavailable': '已识别票据类型，但尚未配置该类独立模型，字段保留为空',
                    'specialist_unsupported_document': '无法确定目标票型，或属于旧票型，未套用其他模型',
                }
                result.update(_ocr_evidence={}, _ocr_issues=[{'code':code,'message':messages[code],'field':None}],
                              _ocr_lines=baseline['_ocr_lines'])
                route['status'] = code
            else:
                if profile not in self._experts:
                    self._experts[profile] = FieldVisionBackend(self.model_dirs[profile], '.',
                        threads=self.threads, ocr_backend=self.ocr)
                result = self._experts[profile].recognize_with_ocr(images, baseline, known_kind=PROFILE_KINDS[profile])
                if result.get('kind') == InvoiceKind.UNKNOWN:
                    result['kind'] = PROFILE_KINDS[profile].value
                route['status'] = 'expert_used'
            result['_specialist_route'] = route
            return result
