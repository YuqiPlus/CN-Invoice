from decimal import Decimal

import pytest
from pydantic import ValidationError

from cninvoice import Evidence, InvoiceReader, RapidOCRBackend, RecognitionError, Source
from cninvoice.ocr import OCRLine, extract_ocr_lines


def line(text, x, y, width=100, page=1):
    return OCRLine(text, (x, y, x + width, y + 20), .97, page)


def vat_lines():
    return [line('增值税专用发票', 0, 0), line('合计', 0, 200),
            line('¥100.00', 300, 200), line('¥13.00', 500, 200),
            line('价税合计（大写）壹佰壹拾叁元整', 0, 250, 250),
            line('（小写）', 300, 250), line('¥113.00', 500, 250)]


def test_ocr_spatial_totals_have_original_text_and_box(png):
    payload = extract_ocr_lines(vat_lines())

    class Backend:
        source = Source.OCR
        def recognize(self, images):
            return payload

    invoice = InvoiceReader(Backend()).read(png)
    assert (invoice.amount, invoice.tax, invoice.total) == (Decimal('100'), Decimal('13'), Decimal('113'))
    assert invoice.sources['total'].text == '¥113.00'
    assert invoice.sources['total'].bbox == (500, 250, 600, 270)
    assert invoice.sources['total'].source == Source.OCR
    assert invoice.needs_review
    assert {i.code for i in invoice.issues} == {'ocr_extraction'}


def test_bad_optional_ocr_evidence_keeps_recognized_values(png):
    class Backend:
        source = Source.OCR
        def recognize(self, images):
            return {'kind': 'vat_invoice', 'total': '113.00',
                    '_ocr_evidence': {'total': {'bbox': [-1, 0, 100, 20]}}}
    invoice = InvoiceReader(Backend()).read(png)
    assert invoice.total == Decimal('113.00')
    assert 'invalid_ocr_evidence' in {i.code for i in invoice.issues}


def test_ocr_does_not_use_detail_row_or_invent_total():
    lines = [line('增值税专用发票', 0, 0), line('商品', 0, 100),
             line('¥100.00', 300, 100), line('¥13.00', 500, 100)]
    payload = extract_ocr_lines(lines)
    assert payload['amount'] is None and payload['tax'] is None and payload['total'] is None


def test_ocr_aggregate_without_total_is_not_calculated():
    payload = extract_ocr_lines(vat_lines()[:4])
    assert payload['amount'] == '100.00' and payload['tax'] == '13.00'
    assert payload['total'] is None


def test_ocr_ambiguous_same_row_does_not_pick_arbitrary_amount():
    lines = [line('增值税专用发票', 0, 0), line('小写', 0, 100),
             line('¥100.00', 200, 100), line('¥113.00', 400, 100)]
    assert extract_ocr_lines(lines)['total'] is None


def test_ocr_legacy_train_single_currency_price():
    lines = [line('G123', 200, 20), line('2026年11月19日', 0, 60), line('¥553.00', 0, 100)]
    payload = extract_ocr_lines(lines)
    assert payload['kind'] == 'rail_ticket' and payload['total'] == '553.00'
    assert payload['tax'] is None and payload['amount'] is None


def test_ocr_broken_vat_title_can_use_explicit_tax_total_label():
    lines = vat_lines()[1:] + [line('广东增值', 0, 0), line('用发票', 200, 0)]
    payload = extract_ocr_lines(lines)
    assert payload['kind'] == 'vat_invoice'
    assert payload['amount'] == '100.00' and payload['tax'] == '13.00'


def test_ocr_legacy_train_one_decimal_and_unlabelled_journey():
    lines = [line('G123', 200, 40), line('北京南站', 0, 40), line('上海虹桥站', 400, 40),
             line('2026年11月19日08:30开', 0, 90, 300), line('￥553.5元', 0, 120),
             line('A123456', 0, 0), line('二等座', 400, 120), line('01车02A号', 400, 90)]
    payload = extract_ocr_lines(lines)
    assert payload['total'] == '553.50' and payload['kind'] == 'rail_ticket'
    assert payload['rail']['ticket_number'] == 'A123456'
    assert payload['rail']['departure_station'] == '北京南站'
    assert payload['rail']['arrival_station'] == '上海虹桥站'
    assert payload['rail']['departure_date'] == '2026-11-19'
    assert payload['rail']['departure_time'] == '08:30'
    assert payload['rail']['carriage'] == '01' and payload['rail']['seat_number'] == '02A'


def test_ocr_air_total_in_next_table_row():
    lines = [line('航空运输电子客票行程单', 0, 0), line('合计 TOTAL', 300, 100), line('CNY113.00', 300, 130)]
    payload = extract_ocr_lines(lines)
    assert payload['total'] == '113.00'


def test_ocr_multiple_pages_with_conflicting_totals():
    lines = [line('增值税发票', 0, 0), line('小写¥113.00', 0, 100),
             line('增值税发票', 0, 0, page=2), line('小写¥226.00', 0, 100, page=2)]
    payload = extract_ocr_lines(lines)
    assert payload['total'] == '113.00'
    assert payload['_ocr_evidence']['total']['page'] == 1
    assert {'conflicting_field', 'ocr_multiple_pages'} <= {i['code'] for i in payload['_ocr_issues']}


def test_ocr_missing_models_never_download(tmp_path):
    with pytest.raises(RecognitionError, match='no models were downloaded'):
        RapidOCRBackend(tmp_path)


@pytest.mark.parametrize('box', [(-1, 0, 2, 3), (2, 0, 1, 3), (0, 2, 1, 1), (0, 0, float('nan'), 2)])
def test_invalid_evidence_coordinates_rejected(box):
    with pytest.raises(ValidationError):
        Evidence(source=Source.OCR, bbox=box)
