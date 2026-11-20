from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .review import Reviewer


@dataclass(frozen=True)
class ReaderConfig:
    """Local visual recognition is the configured default; AI review is opt-in."""
    recognizer: Literal['vision','specialists','ocr','native','qwen','http']='vision'
    model_dir: str|Path|None=None
    ocr_model_dir: str|Path|None=None
    model_name: str|None=None
    endpoint: str|None=None
    api_key: str|None=field(default=None,repr=False)
    device: str='cpu'
    ai_review: bool=False
    review_when: Literal['problems','always']='problems'
    specialist_model_dirs: dict[str,str|Path]=field(default_factory=dict)
    document_profile: str|None=None

    def __post_init__(self):
        if not isinstance(self.ai_review,bool):raise ValueError('ai_review must be a boolean')

    @classmethod
    def from_bundle(cls, directory: str | Path) -> ReaderConfig:
        """Verify an extracted local bundle and configure its local experts."""
        from .bundles import bundle_paths
        models, ocr = bundle_paths(directory)
        return cls(recognizer='specialists', specialist_model_dirs=models,
                   ocr_model_dir=ocr, ai_review=False)

    def create_reader(self, *, reviewer: Reviewer|None=None):
        from .backends import OpenAICompatibleBackend, QwenBackend
        from .ocr import RapidOCRBackend
        from .reader import InvoiceReader
        from .vision import FieldVisionBackend
        if self.recognizer=='specialists':
            from .specialists import SpecialistVisionBackend
            if self.ocr_model_dir is None:
                raise ValueError('A local OCR model directory is required')
            backend=SpecialistVisionBackend(self.specialist_model_dirs,self.ocr_model_dir,
                document_profile=self.document_profile)
        elif self.recognizer=='vision':
            if self.model_dir is None or self.ocr_model_dir is None:
                raise ValueError('Local field detector and OCR model directories must be configured')
            backend=FieldVisionBackend(self.model_dir,self.ocr_model_dir)
        elif self.recognizer=='ocr':
            if self.ocr_model_dir is None:raise ValueError('A local OCR model directory is required')
            backend=RapidOCRBackend(self.ocr_model_dir)
        elif self.recognizer=='native':backend=None
        elif self.recognizer=='qwen':
            if self.model_dir is None:raise ValueError('An explicit local large-model directory is required')
            backend=QwenBackend(self.model_dir,device=self.device)
        elif self.recognizer=='http':
            if not self.endpoint or not self.model_name:raise ValueError('An explicit endpoint and model are required')
            backend=OpenAICompatibleBackend(base_url=self.endpoint,model=self.model_name,api_key=self.api_key)
        else:raise ValueError('Unknown recognizer')
        if self.ai_review and reviewer is None:raise ValueError('AI review requires an explicitly configured reviewer')
        if reviewer is not None and not self.ai_review:raise ValueError('Enable ai_review explicitly to use a reviewer')
        return InvoiceReader(backend=backend,reviewer=reviewer if self.ai_review else None,review_when=self.review_when)
