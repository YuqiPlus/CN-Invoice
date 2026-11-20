from decimal import Decimal

import pytest

from cninvoice import FieldVisionBackend, InvoiceReader, ReaderConfig, RecognitionError
from cninvoice.models import Issue, Source
from cninvoice.ocr import OCRLine
from cninvoice.review import ConsistencyReviewer
from cninvoice.vision import FieldDetection, extract_detected_fields


def detected(field,box):return FieldDetection(field,box,.9)
def text(value,box):return OCRLine(value,box,.98)


def test_learned_fields_keep_literal_evidence_and_do_not_calculate_missing_total():
    detections=[detected('vat.amount',(0,0,100,30)),detected('vat.tax',(200,0,300,30))]
    lines=[text('￥100.00',(0,0,100,30)),text('￥13.00',(200,0,300,30))]
    data=extract_detected_fields(detections,lines)
    assert data['amount']=='100.00' and data['tax']=='13.00' and data['total'] is None
    assert data['_ocr_evidence']['amount']['text']=='￥100.00'
    assert data['_ocr_evidence']['amount']['bbox']==[0.,0.,100.,30.]


def test_visual_region_with_two_different_amounts_is_not_guessed():
    data=extract_detected_fields([detected('vat.total',(0,0,300,30))],
        [text('￥100.00',(0,0,100,30)),text('￥113.00',(200,0,300,30))])
    assert data['total'] is None
    assert 'vision_field_ambiguous' in {i['code'] for i in data['_ocr_issues']}


def test_money_outside_detected_region_is_not_used():
    data=extract_detected_fields([detected('vat.total',(0,0,100,30))],[text('￥999.00',(500,500,600,530))])
    assert data['total'] is None


def test_trained_airline_keeps_fare_separate_and_does_not_compute_total():
    fields=['air.fare','air.total','air.ticket_number']
    data=extract_detected_fields([detected('air.fare',(0,0,100,30)),
        detected('air.ticket_number',(0,100,300,130))],
        [text('￥820.00',(0,0,100,30)),text('781-1234567890',(0,100,300,130))],
        known_kind='air_itinerary',trained_fields=fields)
    assert data['kind']=='air_itinerary' and data['air']['fare']=='820.00'
    assert data['air']['ticket_number']=='781-1234567890' and data['total'] is None


@pytest.mark.parametrize('printed,expected',[
    ('壹佰元整','100.00'),('贰拾伍元伍角','25.50'),('负壹佰零贰元零伍分','-102.05'),
    ('壹万贰仟叁佰肆拾伍元陆角柒分','12345.67'),
])
def test_written_total_is_decoded_from_learned_region(printed,expected):
    data=extract_detected_fields([detected('other.total',(0,0,300,30))],
        [text(printed,(0,0,300,30))],trained_fields=['other.total'])
    assert data['kind']=='other' and data['total']==expected
    assert data['_ocr_evidence']['total']['text']==printed


@pytest.mark.parametrize('printed',['壹佰拾元整','壹佰佰元整','壹万壹亿元整','壹贰元整'])
def test_malformed_financial_amount_is_not_completed(printed):
    data=extract_detected_fields([detected('other.total',(0,0,300,30))],
        [text(printed,(0,0,300,30))],trained_fields=['other.total'])
    assert data['total'] is None


@pytest.mark.parametrize('printed,expected',[
    ('24','24.00'),('70元','70.00'),('金额：￥100','100.00'),
    ('人民币123456元','123456.00'),('29·00','29.00'),('０','0.00'),
])
def test_whole_printed_money_formats_in_learned_region(printed,expected):
    data=extract_detected_fields([detected('other.total',(0,0,300,30))],
        [text(printed,(0,0,300,30))],trained_fields=['other.total'])
    assert data['total']==expected
    assert data['_ocr_evidence']['total']['text']==printed


@pytest.mark.parametrize('printed',[
    '123456','00123','发票号码12345','2026-11-19','10+3','12.345元','电话12345678',
])
def test_integer_money_does_not_extract_identifiers_dates_or_arithmetic(printed):
    data=extract_detected_fields([detected('other.total',(0,0,300,30))],
        [text(printed,(0,0,300,30))],trained_fields=['other.total'])
    assert data['total'] is None


@pytest.mark.parametrize('sign',['-','−','－'])
def test_ocr_minus_sign_is_preserved_in_printed_credit_amount(sign):
    data=extract_detected_fields([detected('vat.total',(0,0,150,30))],[text('￥'+sign+'113.00',(0,0,150,30))])
    assert data['total']=='-113.00'


def test_visual_rail_ticket_number_stays_in_journey(png):
    data=extract_detected_fields([detected('rail.ticket_number',(0,0,100,30)),detected('rail.total',(0,100,100,130))],
        [text('A123456',(0,0,100,30)),text('￥553.5元',(0,100,100,130))])
    class Backend:
        source=Source.OCR
        def recognize(self,images):return data
    result=InvoiceReader(Backend()).read(png)
    assert result.invoice_number is None and result.rail.ticket_number=='A123456'
    assert result.total==Decimal('553.50')
    assert result.sources['rail.ticket_number'].text=='A123456'


def test_untrained_airline_is_explicitly_unsupported():
    data=extract_detected_fields([detected('vat.total',(0,0,100,30))],[text('￥113.00',(0,0,100,30))],known_kind='air_itinerary')
    assert data['kind']=='air_itinerary' and data['total'] is None
    assert data['_ocr_issues'][0]['code']=='vision_kind_unsupported'


def test_visual_model_requires_local_files(tmp_path):
    with pytest.raises(RecognitionError,match='no models were downloaded'):
        FieldVisionBackend(tmp_path,tmp_path)


def test_config_defaults_to_local_visual_and_no_ai_review():
    config=ReaderConfig()
    assert config.recognizer=='vision' and config.ai_review is False
    with pytest.raises(ValueError,match='directories'):config.create_reader()


def test_native_config_does_not_create_cloud_client(xml_invoice):
    config=ReaderConfig(recognizer='native',endpoint='https://example.test',api_key='private-test-key')
    assert 'private-test-key' not in repr(config)
    reader=config.create_reader()
    assert reader.backend is None and reader.reviewer is None
    assert reader.read(xml_invoice).total==Decimal('113')


def test_ai_review_requires_both_enablement_and_reviewer():
    class Review:
        def review(self,invoice,images):return []
    with pytest.raises(ValueError,match='Enable ai_review'):
        ReaderConfig(recognizer='native').create_reader(reviewer=Review())
    with pytest.raises(ValueError,match='explicitly configured reviewer'):
        ReaderConfig(recognizer='native',ai_review=True).create_reader()


def test_optional_review_only_runs_on_problems_and_cannot_mutate_result(png):
    class Backend:
        def __init__(self,partial):self.partial=partial
        def recognize(self,images):
            return {'kind':'vat_invoice','amount':None if self.partial else '100','tax':'13','total':'113'}
    class Review:
        calls=0
        def review(self,invoice,images):
            self.calls+=1;invoice.total='999'
            return [Issue(code='advisory_only',message='Check the original')]
    reviewer=Review()
    good=InvoiceReader(Backend(False),reviewer=reviewer).read(png)
    assert reviewer.calls==0 and good.total==Decimal('113')
    partial=InvoiceReader(Backend(True),reviewer=reviewer).read(png)
    assert reviewer.calls==1 and partial.total==Decimal('113') and partial.amount is None
    assert 'advisory_only' in {i.code for i in partial.issues}


def test_failed_optional_review_keeps_native_fields_and_redacts_error(xml_invoice):
    class Review:
        def review(self,invoice,images):raise RuntimeError('secret-test-key')
    result=InvoiceReader(reviewer=Review(),review_when='always').read(xml_invoice)
    assert result.total==Decimal('113') and 'ai_review_failed' in {i.code for i in result.issues}
    assert 'secret-test-key' not in result.model_dump_json()


def test_consistency_reviewer_never_overwrites_or_requests_without_images(png):
    class Backend:
        calls=0
        def recognize(self,images):
            self.calls+=1;return {'kind':'vat_invoice','total':'999'}
    backend=Backend();review=ConsistencyReviewer(backend)
    from cninvoice import Invoice
    from PIL import Image
    invoice=Invoice(kind='vat_invoice',total='113')
    assert review.review(invoice,[])[0].code=='ai_review_unavailable' and backend.calls==0
    with Image.new('RGB',(32,32)) as image:findings=review.review(invoice,[image])
    assert backend.calls==1 and invoice.total==Decimal('113')
    assert any(f.field=='total' and f.code=='ai_review_disagreement' for f in findings)
