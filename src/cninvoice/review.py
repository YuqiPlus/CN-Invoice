from __future__ import annotations

from typing import Protocol, Sequence

from PIL import Image

from ._merge import apply_payload
from .backends import Backend
from .models import Invoice, Issue, Source


class Reviewer(Protocol):
    def review(self, invoice: Invoice, images: Sequence[Image.Image]) -> Sequence[Issue]:
        """Return advisory findings; must not replace the extracted invoice."""
        ...


class ConsistencyReviewer:
    """Optional independent extraction comparison using an explicitly supplied backend.

    No backend is created implicitly. Findings never overwrite or approve fields.
    """
    def __init__(self, backend: Backend):self.backend=backend

    def review(self, invoice: Invoice, images: Sequence[Image.Image]) -> list[Issue]:
        if not images:
            return [Issue(code='ai_review_unavailable',message='没有可供图像审核的页面，保留原读取结果')]
        payload=self.backend.recognize(images)
        if payload.get('document_count',1)!=1:
            return [Issue(code='ai_review_multiple_documents',message='可选审核未能确认文件只包含一张票据')]
        other=Invoice();apply_payload(other,payload,Source.MODEL)
        findings=[];checked=0
        for field in ('kind','invoice_number','amount','tax','total'):
            original,reviewed=getattr(invoice,field),getattr(other,field)
            if original is None:continue
            if reviewed is None:
                findings.append(Issue(code='ai_review_missing',message='可选审核未读取到该字段，未自动修改原值',field=field))
            elif original!=reviewed:
                findings.append(Issue(code='ai_review_disagreement',message='可选审核与原结果不一致，需人工核对；未自动改写',field=field))
            else:checked+=1
        if not checked and not findings:
            findings.append(Issue(code='ai_review_unavailable',message='没有可交叉核对的字段'))
        return findings
