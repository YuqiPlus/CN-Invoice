from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from PIL import Image

from ._merge import issue, put
from .errors import RecognitionError
from .models import AirJourney, Evidence, Invoice, InvoiceKind, RailJourney, Source, decimal_value
from .ocr import OCRLine, RapidOCRBackend
from .parsers import parse_date
from .vat_anchors import anchored_vat_fields


@dataclass(frozen=True)
class FieldDetection:
    field: str
    bbox: tuple[float, float, float, float]
    score: float
    page: int = 1


def _financial_amount(text):
    """Decode a written financial amount, without combining invoice fields."""
    digits=dict(zip('零壹贰叁肆伍陆柒捌玖',range(10)))
    match=re.fullmatch(r'(负?)([零壹贰叁肆伍陆柒捌玖拾佰仟万亿]+)[元圆](?:([零壹贰叁肆伍陆柒捌玖])角)?(?:零?([壹贰叁肆伍陆柒捌玖])分)?[整正]?',text)
    if not match:return None
    total=section=0;number=None;last_small=10000;last_big=1000000000
    for char in match.group(2):
        if char in digits:
            if number not in (None,0):return None
            number=digits[char]
        elif char in '拾佰仟':
            unit={'拾':10,'佰':100,'仟':1000}[char]
            if unit>=last_small or number==0:return None
            if number is None and (char!='拾' or section or total):return None
            section+=(1 if number is None else number)*unit;number=None
            last_small=unit
        else:
            section+=number or 0;number=None
            unit=10000 if char=='万' else 100000000
            if section==0 or unit>=last_big:return None
            total+=section*unit;section=0;last_small=10000;last_big=unit
    value=Decimal(total+section+(number or 0))
    value+=Decimal(digits.get(match.group(3),0))/10+Decimal(digits.get(match.group(4),0))/100
    if match.group(1):value=-value
    try:return str(decimal_value(value.quantize(Decimal('.01'))))
    except ValueError:return None


def _overlap(box, line):
    x1,y1,x2,y2=box
    a,b,c,d=line.bbox
    intersection=max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))
    return intersection/max(1,(c-a)*(d-b))


def _values(field, line):
    text=line.compact.replace('−','-').replace('－','-')
    if field.rsplit('.', 1)[-1] in {'amount', 'tax', 'total', 'fare'}:
        text=text.replace('·','.')
        values=[]
        for token in re.findall(r'(?<![\d.,])-?\d[\d,]*\.\d{1,2}(?![\d.,])',text):
            try:values.append(str(decimal_value(token)))
            except ValueError:pass
        for token in re.findall(r'负?[零壹贰叁肆伍陆柒捌玖拾佰仟万亿]+[元圆](?:[零壹贰叁肆伍陆柒捌玖]角)?(?:零?[壹贰叁肆伍陆柒捌玖]分)?[整正]?',text):
            value=_financial_amount(token)
            if value is not None:values.append(value)
        # Whole-line integers occur on quota/toll receipts. Require a short
        # number or an explicit currency marker; never pick digits out of IDs.
        integer=re.fullmatch(r'-?(?:0|[1-9]\d{0,4})',text)
        currency=re.fullmatch(r'(?:金额|总金额|价税合计|合计|票价)?[:：]?(?:(?:人民币|RMB|CNY|[¥￥])[:：]?)?(-?(?:0|[1-9]\d{0,8}))(?:[元圆](?:整|正)?)?',text,flags=re.I)
        if integer or (currency and re.search(r'人民币|RMB|CNY|[¥￥元圆]',text,flags=re.I)):
            try:values.append(str(decimal_value(integer.group() if integer else currency.group(1))))
            except ValueError:pass
        return values
    if field in {'vat.invoice_number','rail.invoice_number'}:
        match=re.fullmatch(r'(?:发票号码[:：]?|No[:：]?)?(\d{8}|\d{20})',text,flags=re.I)
        return [match.group(1)] if match else []
    if field in {'taxi.invoice_number', 'other.invoice_number'}:
        match=re.fullmatch(r'(?:发票号码[:：]?|票据号[:：]?|No[:：]?)?(\d{6,30})',text,flags=re.I)
        return [match.group(1)] if match else []
    if field=='air.ticket_number':
        match=re.fullmatch(r'(?:电子客票号码[:：]?|客票号[:：]?|票号[:：]?)?(\d{3}-?\d{10})',text)
        return [match.group(1)] if match else []
    if field=='air.passenger_name':
        return [text] if re.fullmatch(r'[\u4e00-\u9fff·]{2,15}|[A-Z/ .-]{3,60}',text) else []
    if field=='rail.ticket_number':
        return [text] if re.fullmatch(r'[A-Z0-9]{6,30}',text) else []
    if field in {'rail.departure_station','rail.arrival_station'}:
        return [text] if re.fullmatch(r'[\u4e00-\u9fff]{1,12}站?',text) else []
    if field=='rail.train_number':
        return [text] if re.fullmatch(r'[GDKTZCSY]?\d{1,5}',text) else []
    if field=='rail.departure':
        match=re.search(r'(\d{4}年\d{1,2}月\d{1,2}日)(\d{1,2}:\d{2})',text)
        if match:
            try:return [parse_date(match.group(1))+'T'+match.group(2).zfill(5)]
            except ValueError:pass
        return []
    if field=='rail.seat_class':
        match=re.fullmatch(r'(?:新空调)?(二等座|一等座|特等座|商务座|硬座|软座|硬卧|软卧|无座)',text)
        return [match.group(1)] if match else []
    if field=='rail.passenger_name':
        return [text] if re.fullmatch(r'[\u4e00-\u9fff·]{2,10}',text) else []
    return []


def extract_detected_fields(detections: Sequence[FieldDetection], lines: Sequence[OCRLine], *,
                             score_threshold: float = .35, known_kind: str | None = None,
                             trained_fields: Sequence[str] | None = None) -> dict:
    """Read only literal OCR values inside learned regions. No arithmetic completion."""
    invoice=Invoice();proof={}
    usable=[d for d in detections if d.score>=score_threshold]
    kinds={'vat':InvoiceKind.VAT,'rail':InvoiceKind.RAIL,'air':InvoiceKind.AIR,'ride':InvoiceKind.RIDE,
           'taxi':InvoiceKind.TAXI,'other':InvoiceKind.OTHER}
    trained={field.split('.')[0] for field in (trained_fields or [d.field for d in detections])}|{'vat','rail'}
    strengths={prefix:sum(max([d.score for d in usable if d.field==field],default=0)
                          for field in {d.field for d in usable if d.field.startswith(prefix+'.')}) for prefix in kinds if prefix in trained}
    if known_kind==InvoiceKind.AIR and 'air' not in trained:
        put(invoice,'kind',InvoiceKind.AIR,Source.OCR)
        issue(invoice,'vision_kind_unsupported','当前视觉模型尚未训练航空行程单字段')
    else:
        forced=next((key for key,value in kinds.items() if value==known_kind and key!='other'),None)
        prefix=forced if forced in strengths else max(strengths,key=strengths.get)
        if strengths[prefix]>0:
            put(invoice,'kind',kinds[prefix],Source.OCR)
            if prefix=='rail':invoice.rail=RailJourney()
            if prefix=='air':invoice.air=AirJourney()
            for field in sorted({d.field for d in usable if d.field.startswith(prefix+'.')}):
                regions=sorted([d for d in usable if d.field==field],key=lambda d:d.score,reverse=True)
                region=regions[0]
                candidates=[]
                x1,y1,x2,y2=region.bbox;margin=max(2,(y2-y1)*.12)
                box=(x1-margin,y1-margin,x2+margin,y2+margin)
                for line in lines:
                    if line.page!=region.page or _overlap(box,line)<.5:continue
                    candidates.extend((value,line) for value in _values(field,line))
                distinct={value for value,line in candidates}
                if len(distinct)!=1:
                    issue(invoice,'vision_field_unreadable' if not distinct else 'vision_field_ambiguous',
                          '已定位字段区域，但没有读到唯一的票面值，请核对原图',field)
                    continue
                value,line=max(candidates,key=lambda pair:_overlap(box,pair[1])*pair[1].score)
                evidence=Evidence(source=Source.OCR,page=line.page,text=line.text,bbox=line.bbox,score=line.score)
                if field=='rail.departure':
                    day,clock=value.split('T')
                    for name,val in [('rail.departure_date',day),('rail.departure_time',clock)]:
                        put(invoice,name,val,Source.OCR,line.page);proof[name]=evidence
                else:
                    common=field.rsplit('.',1)[-1]
                    name=common if prefix in {'vat','taxi','other','ride'} or common in {'amount','tax','total','invoice_number'} else field
                    put(invoice,name,value,Source.OCR,line.page);proof[name]=evidence
        else:issue(invoice,'vision_no_fields','当前模型未定位到可用字段，未猜测票据内容')
    recoveries={}
    if invoice.kind==InvoiceKind.VAT or known_kind==InvoiceKind.VAT:
        anchors,conflicts=anchored_vat_fields(lines,_values)
        for field in sorted(conflicts):
            issue(invoice,'vat_label_conflict','票面金额标签对应多个不同值，请核对原图',field)
        for field,anchor in anchors.items():
            current=getattr(invoice,field)
            if current is not None:
                expected=anchor.value if field=='invoice_number' else decimal_value(anchor.value)
                if current!=expected:
                    issue(invoice,'vat_label_conflict','定位区域读值与明确字段标签不一致，保留原值供核对',field)
                continue
            put(invoice,'kind',InvoiceKind.VAT,Source.OCR)
            put(invoice,field,anchor.value,Source.OCR,anchor.line.page)
            proof[field]=Evidence(source=Source.OCR,page=anchor.line.page,text=anchor.line.text,
                bbox=anchor.line.bbox,score=anchor.line.score)
            recoveries[field]=anchor.basis
            invoice.issues=[entry for entry in invoice.issues if not
                (entry.field==field and entry.code in {'vision_field_unreadable','vision_field_ambiguous'})]
            issue(invoice,'vat_label_recovery','已按明确票面标签补读字段，请核对原文',field)
    if len({d.page for d in usable})>1:
        issue(invoice,'vision_multiple_pages','视觉结果涉及多个页面，请人工核对票据范围')
    payload=invoice.model_dump(mode='json',exclude={'sources','issues','status','needs_review','raw_text'})
    payload['_ocr_evidence']={key:entry.model_dump(mode='json') for key,entry in proof.items()}
    payload['_ocr_issues']=[entry.model_dump(mode='json') for entry in invoice.issues]
    payload['_vision_detections']=[dict(field=d.field,bbox=d.bbox,score=d.score,page=d.page) for d in usable]
    payload['_vision_anchor_recoveries']=recoveries
    return payload


class FieldVisionBackend:
    """Dedicated ONNX detector + local OCR. Neither Torch nor a large model is loaded."""
    source=Source.OCR

    def __init__(self,model_dir: str|Path,ocr_model_dir: str|Path,*,threads: int=4,
                 ocr_backend: RapidOCRBackend|None=None):
        self.model_dir=Path(model_dir)
        if not (self.model_dir/'detector.onnx').is_file() or not (self.model_dir/'model.json').is_file():
            raise RecognitionError('Explicit local detector.onnx and model.json are required; no models were downloaded')
        self.metadata=json.loads((self.model_dir/'model.json').read_text())
        if not self.metadata.get('fields') or self.metadata.get('input_size')!=320:
            raise RecognitionError('Unsupported visual model metadata')
        if not 1<=threads<=64:raise ValueError('threads must be between 1 and 64')
        self.ocr=ocr_backend if ocr_backend is not None else RapidOCRBackend(ocr_model_dir,threads=threads)
        self.threads=threads;self._session=None;self._lock=threading.Lock()

    def recognize(self,images: Sequence[Image.Image]) -> dict:
        if len(images)!=1:raise RecognitionError('The pilot field detector currently requires one page')
        return self.recognize_with_ocr(images,self.ocr.recognize(images))

    def recognize_with_ocr(self,images: Sequence[Image.Image],baseline: dict,*,known_kind: str|None=None) -> dict:
        """Reuse the routing OCR pass; never run OCR once per specialist."""
        if len(images)!=1:raise RecognitionError('The pilot field detector currently requires one page')
        with self._lock:
            try:
                import numpy as np
                import onnxruntime as ort
                if self._session is None:
                    options=ort.SessionOptions();options.intra_op_num_threads=self.threads;options.inter_op_num_threads=1
                    self._session=ort.InferenceSession(str(self.model_dir/'detector.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
                lines=[OCRLine(entry['text'],tuple(entry['bbox']),entry['score'],entry['page']) for entry in baseline['_ocr_lines']]
                detections=[]
                for page,image in enumerate(images,1):
                    resized=image.convert('RGB').resize((320,320),Image.Resampling.BILINEAR)
                    array=np.asarray(resized,dtype=np.float32).transpose(2,0,1)[None]/255
                    resized.close()
                    boxes,scores,labels=self._session.run(None,{'images':array})
                    for box,score,label in zip(boxes,scores,labels):
                        if not 1<=int(label)<=len(self.metadata['fields']):continue
                        x1,y1,x2,y2=map(float,box)
                        detections.append(FieldDetection(self.metadata['fields'][int(label)-1],
                            (x1*image.width/320,y1*image.height/320,x2*image.width/320,y2*image.height/320),float(score),page))
                result=extract_detected_fields(detections,lines,score_threshold=self.metadata.get('score_threshold',.35),known_kind=known_kind or baseline.get('kind'),trained_fields=self.metadata['fields'])
                result['_ocr_issues'].extend(entry for entry in baseline.get('_ocr_issues',[])
                                             if entry.get('code')=='invalid_date')
                result['_ocr_lines']=baseline['_ocr_lines']
                return result
            except RecognitionError:raise
            except Exception:
                raise RecognitionError('Local field vision failed; check optional dependencies and model files') from None
