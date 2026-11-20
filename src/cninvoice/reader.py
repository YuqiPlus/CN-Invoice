from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import ValidationError

from . import documents
from ._merge import apply_payload, issue, merge, validate_totals
from .backends import Backend
from .errors import InvalidDocumentError, RecognitionError
from .models import Evidence, Invoice, InvoiceKind, Issue, Source
from .parsers import parse_text
from .review import Reviewer


class InvoiceReader:
    def __init__(self, backend: Backend | None = None, *, ocr: Literal["auto", "always", "never"] = "auto",
                 include_raw_text: bool = False, max_bytes: int = 20 * 1024**2, max_pages: int = 8,
                 reviewer: Reviewer | None = None, review_when: Literal['problems','always']='problems'):
        if ocr not in {"auto", "always", "never"}:
            raise ValueError("ocr must be auto, always or never")
        if max_bytes <= 0 or not 1 <= max_pages <= 32:
            raise ValueError("Invalid document limits")
        if review_when not in {'problems','always'}:raise ValueError('review_when must be problems or always')
        self.backend, self.ocr = backend, ocr
        self.include_raw_text = include_raw_text
        self.max_bytes, self.max_pages = max_bytes, max_pages
        self.reviewer,self.review_when=reviewer,review_when

    def read(self, file: str | Path | bytes | BinaryIO) -> Invoice:
        document = documents.load(file, max_bytes=self.max_bytes, max_pages=self.max_pages)
        result = Invoice()
        for candidate in document.structured:
            merge(result, candidate)
        page_numbers: set[str] = set()
        for page in document.pages:
            candidate = parse_text(page.text, page.source, page.number)
            if candidate.invoice_number:
                page_numbers.add(candidate.invoice_number)
            merge(result, candidate)
        if len(page_numbers) > 1:
            issue(result, "multiple_documents", "文件包含多张不同发票，请拆分后读取")
        if self.include_raw_text:
            result.raw_text = "\n\n".join(page.text for page in document.pages) or None
        native_complete = result.kind != InvoiceKind.UNKNOWN and result.total is not None and (
            result.invoice_number is not None or result.rail is not None or result.air is not None
        )
        if result.kind == InvoiceKind.VAT:
            native_complete = native_complete and result.amount is not None and result.tax is not None
        needs_model = document.kind != "xml" and self.ocr != "never" and (self.ocr == "always" or not native_complete)
        # Ambiguous documents are not combined into a fabricated invoice by the model.
        ambiguous = any(w.code in {"multiple_documents", "multiple_contexts", "conflicting_field"} for w in result.issues)
        if needs_model and not ambiguous:
            if self.backend is None:
                issue(result, "backend_required", "需要配置识图后端才能读取或补全此文件")
            else:
                images = []
                try:
                    images = documents.render(document, self.max_pages)
                    payload = self.backend.recognize(images)
                    if not isinstance(payload, dict):
                        raise RecognitionError("Backend must return a JSON object")
                    count = payload.get("document_count", 1)
                    if count != 1:
                        issue(result, "multiple_documents", "模型未能确认文件只包含一张票据，请拆分或人工复核")
                    else:
                        source = Source.OCR if getattr(self.backend, "source", None) == Source.OCR else Source.MODEL
                        apply_payload(result, payload, source)
                        if source == Source.OCR:
                            for field, data in payload.get("_ocr_evidence", {}).items():
                                base=result.sources.get(field) or result.sources.get(field.split('.')[0])
                                if base is not None and base.source == Source.OCR:
                                    current=result
                                    for name in field.split('.'):
                                        current=getattr(current,name,None)
                                        if current is None:break
                                    if current is None:continue
                                    try:
                                        result.sources[field] = Evidence.model_validate({**data, "source": Source.OCR})
                                    except (ValidationError, TypeError, ValueError):
                                        issue(result, "invalid_ocr_evidence", "OCR 字段位置或原文证据无效，请核对票面", field)
                            for warning in payload.get("_ocr_issues", []):
                                issue(result, warning["code"], warning["message"], warning.get("field"))
                            issue(result, "ocr_extraction", "包含 OCR 识别结果，请对照票面复核")
                        else:
                            issue(result, "model_extraction", "包含模型识别结果，请对照票面复核")
                except InvalidDocumentError:
                    issue(result, "rendering_unavailable", "无法将文件完整渲染为图片，已保留直接读取结果")
                except RecognitionError:
                    issue(result, "backend_failed", "识图后端未返回可用结果，已保留直接读取结果")
                finally:
                    for image in images:
                        image.close()
        validate_totals(result)
        problems=result.kind==InvoiceKind.UNKNOWN or result.total is None or any(
            warning.code not in {'ocr_extraction','model_extraction'} for warning in result.issues)
        if self.reviewer is not None and (self.review_when=='always' or problems):
            review_images=[]
            try:
                if document.kind!='xml':review_images=documents.render(document,self.max_pages)
                findings=self.reviewer.review(result.model_copy(deep=True),review_images)
                for finding in findings:
                    item=Issue.model_validate(finding)
                    issue(result,item.code,item.message,item.field)
                issue(result,'ai_review_completed','已执行可选 AI 交叉核对，意见不自动改写或确认票据')
            except Exception:
                issue(result,'ai_review_failed','可选 AI 核对失败，已保留本地读取结果')
            finally:
                for image in review_images:image.close()
        return result


def read_invoice(file: str | Path | bytes | BinaryIO, *, backend: Backend | None = None,
                 ocr: Literal["auto", "always", "never"] = "auto") -> Invoice:
    """Read a local file, bytes, or a binary stream into an Invoice object."""
    return InvoiceReader(backend=backend, ocr=ocr).read(file)
