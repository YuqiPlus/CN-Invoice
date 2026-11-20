import pytest

from cninvoice.models import Source
from cninvoice.ocr import OCRLine, extract_ocr_lines
from cninvoice.parsers import parse_text
from cninvoice.vision import FieldDetection, extract_detected_fields


def line(text, box=(0,0,300,30), page=1):
    return OCRLine(text,box,.98,page)


def read(lines, detections=()):
    return extract_detected_fields(detections,lines,known_kind='vat_invoice',
                                  trained_fields=['vat.amount','vat.tax','vat.total'])


def test_explicit_lowercase_total_survives_a_missing_visual_region():
    result=read([line('（小写）￥113.00')])
    assert result['total']=='113.00'
    assert result['_ocr_evidence']['total']['text']=='（小写）￥113.00'
    assert result['_vision_anchor_recoveries']['total']=='explicit_lowercase_total'


@pytest.mark.parametrize('minus',['-','−','－'])
def test_label_recovery_preserves_printed_credit_sign(minus):
    result=read([line('（小写）￥'+minus+'113.00')])
    assert result['total']=='-113.00'
    assert result['_ocr_evidence']['total']['text']=='（小写）￥'+minus+'113.00'


def test_skewed_aggregate_row_uses_column_heading_alignment():
    result=read([line('金额',(600,80,660,110)),line('税额',(1000,60,1060,90)),
        line('合计',(50,300,110,330)),line('￥100.00',(600,272,720,302)),
        line('￥13.00',(1000,252,1120,282))])
    assert result['amount']=='100.00' and result['tax']=='13.00'
    assert result['total'] is None


def test_combined_aggregate_row_does_not_impute_missing_total():
    result=read([line('合计￥100.00￥13.00')])
    assert result['amount']=='100.00' and result['tax']=='13.00'
    assert result['total'] is None


def test_spaced_aggregate_characters_can_label_a_currency_pair():
    result=read([line('合',(0,100,30,130)),line('计',(200,100,230,130)),
        line('￥100.00',(500,100,650,130)),line('￥13.00',(800,100,950,130))])
    assert result['amount']=='100.00' and result['tax']=='13.00'
    assert result['total'] is None
    assert result['_vision_anchor_recoveries']['tax']=='split_aggregate_row'


def test_aggregate_characters_on_different_rows_are_not_joined():
    result=read([line('合',(0,0,30,30)),line('计',(200,100,230,130)),
        line('￥100.00',(500,100,650,130)),line('￥13.00',(800,100,950,130))])
    assert result['amount'] is None and result['tax'] is None


def test_lone_aggregate_character_does_not_infer_a_totals_row():
    result=read([line('计',(200,100,230,130)),line('￥100.00',(500,100,650,130)),
        line('￥13.00',(800,100,950,130))])
    assert result['amount'] is None and result['tax'] is None


@pytest.mark.parametrize('second_box',[(200,35,300,45),(500,35,600,45)])
def test_two_neighbouring_rows_are_not_an_aggregate_pair(second_box):
    result=read([line('合计',(0,0,100,60)),line('￥100.00',(200,5,300,15)),
        line('￥13.00',second_box)])
    assert result['amount'] is None and result['tax'] is None


def test_written_total_is_literal_fallback_for_unreadable_lowercase():
    result=read([line('价税合计（大写）',(0,100,250,130)),
        line('⊗贰佰贰拾陆圆整',(260,100,650,130)),line('（小写）￥22G.00',(700,100,1000,130))])
    assert result['total']=='226.00'
    assert result['_ocr_evidence']['total']['text']=='⊗贰佰贰拾陆圆整'


def test_disagreeing_printed_total_variants_do_not_select_one():
    result=read([line('价税合计（大写）壹佰壹拾元整'),line('（小写）￥100.00',(400,0,650,30))])
    assert result['total'] is None
    assert any(i['code']=='vat_label_conflict' for i in result['_ocr_issues'])


@pytest.mark.parametrize('value',['100.00','113.00'])
def test_multiple_total_labels_even_equal_values_are_ambiguous(value):
    result=read([line('（小写）￥100.00'),line('（小写）￥'+value,(0,500,300,530))])
    assert result['total'] is None


def test_anchor_conflict_keeps_existing_visual_read_for_review():
    result=read([line('￥999.00'),line('价税合计：￥113.00',(0,100,300,130))],
        [FieldDetection('vat.total',(0,0,300,30),.95)])
    assert result['total']=='999.00'
    assert any(i['code']=='vat_label_conflict' for i in result['_ocr_issues'])


def test_bare_quantities_and_exempt_tax_are_not_amount_tax_pair():
    result=read([line('合计',(0,0,100,30)),line('2',(200,0,250,30)),
        line('￥100.00',(500,0,650,30)),line('免税',(800,0,900,30)),
        line('（小写）￥100.00',(500,50,900,80))])
    assert result['total']=='100.00' and result['tax'] is None


@pytest.mark.parametrize('tax_text', ['***', '免税', ''])
@pytest.mark.parametrize('amount', ['100.00', '0.00'])
def test_aggregate_amount_column_is_read_without_inventing_star_tax(tax_text, amount):
    result=read([line('金额',(500,0,560,30)),line('税额',(800,0,860,30)),
        line('合计',(0,100,100,130)),line('￥'+amount,(500,100,650,130)),
        line(tax_text,(800,100,900,130))])
    assert result['amount']==amount
    assert result['tax'] is None and result['total'] is None
    assert result['_vision_anchor_recoveries']['amount']=='explicit_aggregate_amount_column'


def test_single_tax_column_value_is_not_an_aggregate_amount():
    result=read([line('金额',(500,0,560,30)),line('税额',(800,0,860,30)),
        line('合计',(0,100,100,130)),line('￥13.00',(800,100,950,130))])
    assert result['amount'] is None and result['tax'] is None


def test_single_aggregate_value_without_unique_headings_stays_unassigned():
    result=read([line('合计',(0,100,100,130)),line('￥100.00',(500,100,650,130))])
    assert result['amount'] is None and result['tax'] is None


@pytest.mark.parametrize('text,expected', [
    ('发票号码：00123456','00123456'),
    ('No. 00123456','00123456'),
    ('发票号码：26312000000012345678','26312000000012345678'),
])
def test_explicit_invoice_number_survives_missing_visual_region(text, expected):
    result=read([line(text)])
    assert result['invoice_number']==expected
    assert result['_ocr_evidence']['invoice_number']['text']==text
    assert result['amount'] is None and result['total'] is None


def test_separate_number_label_requires_same_page_row_and_unique_value():
    result=read([line('发票号码',(0,100,140,130)),line('00123456',(150,100,300,130))])
    assert result['invoice_number']=='00123456'
    ambiguous=read([line('发票号码',(0,100,140,130)),line('00123456',(150,100,300,130)),
                    line('20261120',(350,100,500,130))])
    assert ambiguous['invoice_number'] is None
    other_page=read([line('发票号码',(0,100,140,130)),line('00123456',(150,100,300,130),page=2)])
    assert other_page['invoice_number'] is None


def test_invoice_code_or_unlabelled_number_does_not_become_invoice_number():
    result=read([line('发票代码：00123456'),line('00123456',(0,100,150,130))])
    assert result['invoice_number'] is None


def test_conflicting_invoice_labels_keep_field_empty_and_report_conflict():
    result=read([line('发票号码：00123456'),line('No 00987654',(0,100,300,130))])
    assert result['invoice_number'] is None
    assert any(x['code']=='vat_label_conflict' and x['field']=='invoice_number' for x in result['_ocr_issues'])


def test_conflicting_invoice_anchor_keeps_existing_visual_value_for_review():
    result=read([line('00123456'),line('发票号码：00987654',(0,100,300,130))],
        [FieldDetection('vat.invoice_number',(0,0,300,30),.95)])
    assert result['invoice_number']=='00123456'
    assert any(x['code']=='vat_label_conflict' and x['field']=='invoice_number' for x in result['_ocr_issues'])


def test_detail_rows_and_tax_rates_do_not_become_aggregate():
    result=read([line('￥100.00'),line('13%',(400,0,450,30)),line('￥113.00',(600,0,800,30))])
    assert result['amount'] is None and result['tax'] is None and result['total'] is None


@pytest.mark.parametrize('bad',['.50','壹元壹元','12.345元','100+13','发票号码12345678'])
def test_invalid_written_label_is_not_coerced_to_money(bad):
    assert read([line('小写：'+bad)])['total'] is None


def test_labels_on_another_page_cannot_supply_a_value():
    result=read([line('（小写）',(0,0,100,30)),line('￥100.00',(110,0,250,30),page=2)])
    assert result['total'] is None


def test_non_vat_reader_does_not_apply_vat_fallback():
    result=extract_detected_fields([], [line('（小写）￥100.00')],known_kind='taxi_invoice',
                                  trained_fields=['taxi.total'])
    assert result['total'] is None


def test_bad_issue_date_keeps_other_readable_ocr_fields():
    result=parse_text('增值税电子普通发票\n开票日期：2026年19月01日\n价税合计：113.00',Source.OCR)
    assert result.issue_date is None and str(result.total)=='113.00'
    assert any(i.code=='invalid_date' for i in result.issues)


def test_bad_rail_date_does_not_abort_ocr_price_extraction():
    result=extract_ocr_lines([line('铁路客票'),line('2026年19月01日12:30',(0,50,300,80)),
        line('￥100.00',(0,100,300,130))])
    assert result['total']=='100.00'
    assert any(i['code']=='invalid_date' for i in result['_ocr_issues'])
