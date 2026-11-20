from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from ._merge import issue, merge, put
from .errors import RecognitionError
from .models import Evidence, Invoice, InvoiceKind, Source, decimal_value
from .parsers import parse_date, parse_text


@dataclass(frozen=True)
class OCRLine:
    text: str
    bbox: tuple[float, float, float, float]
    score: float
    page: int = 1

    @property
    def compact(self) -> str:
        return re.sub(r"\s", "", unicodedata.normalize("NFKC", self.text))

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> float:
        return max(1, self.bbox[3] - self.bbox[1])


def _same_row(a: OCRLine, b: OCRLine) -> bool:
    return a.page == b.page and abs(a.cy - b.cy) <= 0.55 * max(a.height, b.height)


def _rows(lines: Sequence[OCRLine]) -> list[list[OCRLine]]:
    rows: list[list[OCRLine]] = []
    for line in sorted(lines, key=lambda item: (item.page, item.cy, item.bbox[0])):
        if rows and _same_row(rows[-1][0], line):
            rows[-1].append(line)
        else:
            rows.append([line])
    return [sorted(row, key=lambda item: item.bbox[0]) for row in rows]


def _money(text: str):
    match = re.fullmatch(r"(?:CNY|CN|[¥￥])?(-?\d[\d,]*\.\d{1,2})[元]?", text)
    if not match:
        return None
    try:
        return decimal_value(match.group(1))
    except ValueError:
        return None


def extract_ocr_lines(lines: Sequence[OCRLine]) -> dict:
    """Conservative layout rules; never calculate a missing printed amount."""
    invoice = Invoice()
    proof: dict[str, Evidence] = {}
    for page in sorted({line.page for line in lines}):
        page_lines = [line for line in lines if line.page == page]
        text = "\n".join(" ".join(line.text for line in row) for row in _rows(page_lines))
        parsed = parse_text(text, Source.OCR, page)
        # Linearized text loses column boundaries; select totals spatially below.
        parsed.total = None
        parsed.sources.pop('total', None)
        compact = re.sub(r"\s", "", unicodedata.normalize("NFKC", text))
        if parsed.kind == InvoiceKind.UNKNOWN and any('价税合计' in line.compact for line in page_lines):
            put(parsed, 'kind', InvoiceKind.VAT, Source.OCR, page)
        if parsed.kind == InvoiceKind.UNKNOWN and re.search(r"[GDKTZCSY]\d{1,5}", compact) and re.search(
            r"\d{4}年\d{1,2}月\d{1,2}日", compact
        ) and re.search(r"[¥￥]\d+\.\d{1,2}", compact):
            put(parsed, "kind", InvoiceKind.RAIL, Source.OCR, page)
        merge(invoice, parsed)
        # A printed label and its neighbouring amount are required for totals.
        labels = [line for line in page_lines if re.search(r"小写|价税合计|总金额|合计票价", line.compact)]
        if parsed.kind == InvoiceKind.RAIL:
            labels += [line for line in page_lines if re.search(r"票价|票款", line.compact)]
        if parsed.kind == InvoiceKind.AIR:
            labels += [line for line in page_lines if re.fullmatch(r'(?:合计(?:TOTAL)?|TOTAL)[:：]?', line.compact.upper())]
        if any('小写' in line.compact for line in labels):
            labels = [line for line in labels if '小写' in line.compact]
        for label in labels:
            same = re.search(r"(?:小写[)）]?|价税合计[)）]?|总金额|合计票价|票价|票款|合计)[:：]?[¥￥]?(-?\d[\d,]*\.\d{2})(?![\d.,])", label.compact)
            candidates = [(_money(same.group(1)), label)] if same and _money(same.group(1)) is not None else [
                (_money(line.compact), line) for line in page_lines
                if _same_row(label, line) and line.bbox[0] >= label.bbox[2] and _money(line.compact) is not None
            ]
            if not candidates and parsed.kind == InvoiceKind.AIR:
                candidates = [(_money(line.compact), line) for line in page_lines
                              if line.bbox[1] >= label.bbox[3] and line.bbox[1] - label.bbox[3] <= 3 * label.height
                              and label.bbox[0] <= (line.bbox[0] + line.bbox[2]) / 2 <= label.bbox[2]
                              and _money(line.compact) is not None]
            if len(candidates) == 1:
                value, line = candidates[0]
                put(invoice, 'total', value, Source.OCR, page)
                if invoice.total == value:
                    proof['total'] = Evidence(source=Source.OCR, page=page, text=line.text, bbox=line.bbox, score=line.score)
        if parsed.kind == InvoiceKind.RAIL and invoice.total is None:
            # Legacy railway tickets often have only one currency-labelled price.
            candidates = [(value, line) for line in page_lines if line.compact.startswith(('¥', '￥'))
                          and (value := _money(line.compact)) is not None]
            if len(candidates) == 1:
                value, line = candidates[0]
                put(invoice, 'total', value, Source.OCR, page)
                proof['total'] = Evidence(source=Source.OCR, page=page, text=line.text, bbox=line.bbox, score=line.score)
        if parsed.kind == InvoiceKind.RAIL:
            journey = parsed.rail.model_dump(exclude_none=True) if parsed.rail is not None else {}
            trains = [line for line in page_lines if re.fullmatch(r'[GDKTZCSY]\d{1,5}', line.compact)]
            if len(trains) == 1:
                train = trains[0]
                journey['train_number'] = train.compact
                stations = [line for line in page_lines if _same_row(train, line) and re.fullmatch(r'[\u4e00-\u9fff]{1,12}站', line.compact)]
                for field, choices in [('departure_station', [line for line in stations if line.bbox[2] <= train.bbox[0]]),
                                       ('arrival_station', [line for line in stations if line.bbox[0] >= train.bbox[2]])]:
                    if len(choices) == 1:
                        journey[field] = choices[0].compact
            for line in page_lines:
                value = line.compact
                date = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日)(\d{1,2}:\d{2})', value)
                if date:
                    try:
                        journey['departure_date'] = parse_date(date.group(1))
                        journey['departure_time'] = date.group(2).zfill(5)
                    except ValueError:
                        issue(invoice,'invalid_date','票面日期无法解析，保留其他可读字段','rail.departure_date')
                ticket = re.fullmatch(r'[A-Z]\d{6}', value)
                if ticket:
                    journey['ticket_number'] = value
                seat = re.search(r'(\d{1,3})车(\d{1,3}[A-Z]?)号', value)
                if seat:
                    journey['carriage'], journey['seat_number'] = seat.groups()
                seat_class = re.fullmatch(r'(?:新空调)?(二等座|一等座|特等座|商务座|硬座|软座|硬卧|软卧|无座)', value)
                if seat_class:
                    journey['seat_class'] = seat_class.group(1)
            if journey:
                if invoice.rail is None:
                    put(invoice, 'rail', journey, Source.OCR, page)
                else:
                    for field, value in journey.items():
                        put(invoice, 'rail.' + field, value, Source.OCR, page)
        if parsed.kind == InvoiceKind.VAT:
            for label in page_lines:
                if label.compact.rstrip(':：') != '合计':
                    continue
                values = sorted([line for line in page_lines if _same_row(label, line)
                                 and line.bbox[0] >= label.bbox[2] and _money(line.compact) is not None],
                                key=lambda line: line.bbox[0])
                if len(values) == 2:
                    for field, line in zip(('amount', 'tax'), values):
                        value = _money(line.compact)
                        put(invoice, field, value, Source.OCR, page)
                        if getattr(invoice, field) == value:
                            proof[field] = Evidence(source=Source.OCR, page=page, text=line.text, bbox=line.bbox, score=line.score)
        # Retain literal OCR evidence for amounts found by the text-label parser.
        for field in ('amount', 'tax', 'total'):
            value = getattr(invoice, field)
            if value is None or field in proof:
                continue
            for line in page_lines:
                tokens = re.findall(r"(?<![\d.,])-?\d[\d,]*\.\d{2}(?![\d.,])", line.compact)
                if any(_money(token) == value for token in tokens):
                    proof[field] = Evidence(source=Source.OCR, page=page, text=line.text, bbox=line.bbox, score=line.score)
                    break
    if len({line.page for line in lines}) > 1:
        issue(invoice, 'ocr_multiple_pages', 'OCR 多页结果需要人工核对票据范围')
    payload = invoice.model_dump(mode='json', exclude={'sources', 'issues', 'needs_review', 'status', 'raw_text'})
    payload['_ocr_evidence'] = {key: value.model_dump(mode='json') for key, value in proof.items()}
    payload['_ocr_issues'] = [warning.model_dump(mode='json') for warning in invoice.issues]
    return payload


class RapidOCRBackend:
    """Local PP-OCRv6 small baseline with explicit ONNX files, running on CPU."""

    source = Source.OCR
    MODEL_FILES = ('PP-OCRv6_det_small.onnx', 'ch_ppocr_mobile_v2.0_cls_mobile.onnx', 'PP-OCRv6_rec_small.onnx')

    def __init__(self, model_dir: str | Path, *, threads: int = 4):
        self.model_dir = Path(model_dir)
        if not all((self.model_dir / name).is_file() for name in self.MODEL_FILES):
            raise RecognitionError('Explicit local RapidOCR ONNX files are required; no models were downloaded')
        if not 1 <= threads <= 64:
            raise ValueError('threads must be between 1 and 64')
        self.threads = threads
        self._engine = None
        self._lock = threading.Lock()

    def recognize(self, images: Sequence[Image.Image]) -> dict:
        with self._lock:
            try:
                import numpy as np
                from rapidocr import RapidOCR

                if self._engine is None:
                    self._engine = RapidOCR(params={
                        'Det.model_path': str(self.model_dir / self.MODEL_FILES[0]),
                        'Cls.model_path': str(self.model_dir / self.MODEL_FILES[1]),
                        'Rec.model_path': str(self.model_dir / self.MODEL_FILES[2]),
                        'EngineConfig.onnxruntime.intra_op_num_threads': self.threads,
                        'EngineConfig.onnxruntime.inter_op_num_threads': 1,
                        'Global.log_level': 'warning',
                    })
                lines = []
                for page, image in enumerate(images, 1):
                    output = self._engine(np.asarray(image.convert('RGB'))[:, :, ::-1].copy())
                    if output.boxes is None or output.txts is None:
                        continue
                    for box, text, score in zip(output.boxes, output.txts, output.scores):
                        coords = np.asarray(box)
                        bounds = (max(0, float(coords[:, 0].min())), max(0, float(coords[:, 1].min())),
                                  max(0, float(coords[:, 0].max())), max(0, float(coords[:, 1].max())))
                        lines.append(OCRLine(text, bounds, float(score), page))
                payload = extract_ocr_lines(lines)
                payload['_ocr_lines'] = [dict(text=line.text, bbox=line.bbox, score=line.score, page=line.page) for line in lines]
                return payload
            except RecognitionError:
                raise
            except Exception:
                raise RecognitionError('Local OCR failed; check the optional dependencies and local models') from None
