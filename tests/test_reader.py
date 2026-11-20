from __future__ import annotations

import io
import json
from decimal import Decimal

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter

from cninvoice import Invoice, InvoiceKind, InvoiceReader, InvalidDocumentError, LineItem, RecognitionError, Source, read_invoice
from cninvoice.backends import OpenAICompatibleBackend, parse_response
from cninvoice.cli import main
from cninvoice.parsers import parse_text
from conftest import ofd_archive, pdf_with_xml


class FakeBackend:
    def __init__(self, payload=None, error=False):
        self.payload = payload or {"document_count": 1, "kind": "vat_invoice", "invoice_number": "00012345", "total": "113.00"}
        self.calls = 0
        self.error = error
        self.page_count = 0

    def recognize(self, images):
        self.calls += 1
        self.page_count = len(images)
        assert all(image.mode == "RGB" for image in images)
        if self.error:
            raise RecognitionError("not available")
        return self.payload


def codes(invoice):
    return {issue.code for issue in invoice.issues}


def test_optional_fields_and_decimal_json():
    invoice = Invoice(total="0", amount="-1,234.56")
    assert invoice.buyer.name is None and invoice.rail is None
    assert invoice.total == Decimal("0.00")
    assert invoice.amount == Decimal("-1234.56")
    assert json.loads(invoice.model_dump_json())["total"] == "0.00"
    assert LineItem(unit_price="0.123456", quantity="1.25").unit_price == Decimal("0.123456")


@pytest.mark.parametrize("value", [True, "NaN", "Infinity", "1e3", "1.234", "12元", "1.2.3", "1,23"])
def test_unsafe_amounts_are_not_coerced(value):
    with pytest.raises(ValidationError):
        Invoice(total=value)


def test_xml_all_fields_and_no_backend_call(xml_invoice):
    backend = FakeBackend()
    invoice = InvoiceReader(backend).read(xml_invoice)
    assert invoice.total == Decimal("113.00") and invoice.amount == Decimal("100.00")
    assert invoice.tax == Decimal("13.00") and invoice.issue_date.isoformat() == "2026-11-19"
    assert invoice.invoice_number == "00000001234567890123"
    assert invoice.buyer.name == "Example Buyer" and invoice.seller.name == "Example Seller"
    assert invoice.items[0].amount == Decimal("100.00")
    assert invoice.sources["total"].source == Source.XML
    assert backend.calls == 0 and invoice.status == "parsed"


def test_file_path_bytes_stream_and_magic(tmp_path, xml_invoice):
    file = tmp_path / "renamed.jpg"
    file.write_bytes(xml_invoice)
    stream = io.BytesIO(xml_invoice)
    for source in [file, str(file), xml_invoice, stream]:
        assert read_invoice(source).total == Decimal("113.00")
    assert stream.tell() == 0


def test_utf16_xml(xml_invoice):
    content = xml_invoice.decode().replace('<?xml version="1.0"?>', '<?xml version="1.0" encoding="UTF-16"?>')
    assert read_invoice(content.encode("utf-16")).total == Decimal("113.00")


def test_unknown_schema_is_not_guessed():
    result = read_invoice(b'<evil:Invoice xmlns:evil="urn:unrelated"><evil:TaxInclusiveTotalAmount>999</evil:TaxInclusiveTotalAmount></evil:Invoice>')
    assert result.total is None and "unknown_xml_schema" in codes(result)


@pytest.mark.parametrize("content", [b'<!DOCTYPE x [<!ENTITY xx SYSTEM "file:///etc/passwd">]><Invoice>&xx;</Invoice>', b'<!DOCTYPE x><Invoice/>', b'<Invoice><invalid>'])
def test_unsafe_or_invalid_xml_is_rejected(content):
    with pytest.raises(InvalidDocumentError):
        read_invoice(content)


def test_conflicting_amount_and_arithmetic(xml_invoice):
    bad = xml_invoice.replace(b'</Invoice>', b'<TaxInclusiveTotalAmount>115.00</TaxInclusiveTotalAmount></Invoice>')
    result = read_invoice(bad)
    assert result.total == Decimal("113.00") and "conflicting_field" in codes(result)
    result = read_invoice(xml_invoice.replace(b'<TaxTotalAmount>13.00', b'<TaxTotalAmount>12.00'))
    assert "amount_mismatch" in codes(result) and result.needs_review


@pytest.mark.parametrize("missing", [
    ("amount",), ("tax",), ("total",),
    ("amount", "tax"), ("amount", "total"), ("tax", "total"),
    ("amount", "tax", "total"),
])
def test_incomplete_vat_amounts_are_not_calculated(xml_invoice, missing):
    elements = {
        "amount": b'<TaxExclusiveTotalAmount>100.00</TaxExclusiveTotalAmount>',
        "tax": b'<TaxTotalAmount>13.00</TaxTotalAmount>',
        "total": b'<TaxInclusiveTotalAmount>113.00</TaxInclusiveTotalAmount>',
    }
    data = xml_invoice
    for field in missing:
        data = data.replace(elements[field], b'')
    result = read_invoice(data)
    assert {i.field for i in result.issues if i.code == "amount_incomplete"} == set(missing)
    assert all(getattr(result, field) is None for field in missing)
    assert result.status == "needs_review"


def test_one_cent_discrepancy_requires_review(xml_invoice):
    result = read_invoice(xml_invoice.replace(b'<TaxTotalAmount>13.00', b'<TaxTotalAmount>12.99'))
    assert "amount_mismatch" in codes(result) and result.needs_review
    assert result.tax == Decimal("12.99") and result.total == Decimal("113.00")


@pytest.mark.parametrize("net,tax,total", [("0.00", "0.00", "0.00"), ("-100.00", "-13.00", "-113.00")])
def test_zero_and_credit_vat_amounts_are_valid(xml_invoice, net, tax, total):
    data = xml_invoice.replace(b'<TaxExclusiveTotalAmount>100.00', b'<TaxExclusiveTotalAmount>' + net.encode())
    data = data.replace(b'<TaxTotalAmount>13.00', b'<TaxTotalAmount>' + tax.encode())
    data = data.replace(b'<TaxInclusiveTotalAmount>113.00', b'<TaxInclusiveTotalAmount>' + total.encode())
    result = read_invoice(data)
    assert "amount_incomplete" not in codes(result) and "amount_mismatch" not in codes(result)


def test_partial_vat_native_fields_trigger_image_completion(xml_invoice):
    data = xml_invoice.replace(b'<TaxExclusiveTotalAmount>100.00</TaxExclusiveTotalAmount>', b'')
    backend = FakeBackend({"kind": "vat_invoice", "amount": "100.00", "tax": "13.00", "total": "113.00"})
    result = InvoiceReader(backend).read(pdf_with_xml(data))
    assert backend.calls == 1 and result.amount == Decimal("100.00")
    assert result.sources["amount"].source == Source.MODEL
    assert result.sources["total"].source == Source.PDF_XML
    assert "amount_incomplete" not in codes(result)


@pytest.mark.parametrize("kind", ["rail_ticket", "air_itinerary"])
def test_transport_without_printed_tax_does_not_require_vat_triple(png, kind):
    result = InvoiceReader(FakeBackend({"kind": kind, "total": "113.00"})).read(png)
    assert result.amount is None and result.tax is None
    assert "amount_incomplete" not in codes(result)


def test_balanced_model_amounts_still_require_review(png):
    result = InvoiceReader(FakeBackend({
        "kind": "vat_invoice", "amount": "3495.00", "tax": "507.82", "total": "4002.82",
    })).read(png)
    assert "amount_mismatch" not in codes(result)
    assert result.needs_review and "model_extraction" in codes(result)


def test_xbrl_currency_and_context():
    data = b'''<x:xbrl xmlns:x="http://www.xbrl.org/2003/instance" xmlns:i="http://xbrl.mof.gov.cn/taxonomy/2023-12-31/einv">
    <x:unit id="u"><x:measure>iso4217:USD</x:measure></x:unit>
    <i:InvoiceNumber>00000001</i:InvoiceNumber><i:TotalTax-includedAmount unitRef="u" contextRef="c">10</i:TotalTax-includedAmount></x:xbrl>'''
    result = read_invoice(data)
    assert result.total is None and "unknown_currency" in codes(result)
    result = read_invoice(data.replace(b'USD', b'CNY'))
    assert result.total == Decimal("10") and result.currency == "CNY"


def test_pdf_embedded_xml_first(xml_invoice):
    backend = FakeBackend()
    invoice = InvoiceReader(backend).read(pdf_with_xml(xml_invoice))
    assert invoice.total == Decimal("113") and backend.calls == 0
    assert invoice.sources["total"].source == Source.PDF_XML


def test_pdf_image_fallback_and_review():
    backend = FakeBackend()
    invoice = InvoiceReader(backend).read(pdf_with_xml())
    assert backend.calls == 1 and backend.page_count == 1
    assert invoice.total == Decimal("113") and invoice.needs_review
    assert invoice.sources["total"].source == Source.MODEL


def test_pdf_limits_and_encryption():
    with pytest.raises(InvalidDocumentError):
        InvoiceReader(max_pages=1).read(pdf_with_xml(pages=2))
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("test-only")
    output = io.BytesIO();writer.write(output)
    with pytest.raises(InvalidDocumentError):
        read_invoice(output.getvalue())


def test_ofd_structured_and_scan_fallback(xml_invoice, png):
    backend = FakeBackend()
    invoice = InvoiceReader(backend).read(ofd_archive(xml_invoice))
    assert invoice.total == Decimal("113") and backend.calls == 0
    assert invoice.sources["total"].source == Source.OFD_XML
    invoice = InvoiceReader(backend).read(ofd_archive(png=png))
    assert backend.calls == 1 and invoice.total == Decimal("113")


@pytest.mark.parametrize("name", ["../outside.xml", "/absolute.xml", "dir\\evil.xml", "http:evil.xml"])
def test_ofd_path_traversal_rejected(name):
    with pytest.raises(InvalidDocumentError):
        read_invoice(ofd_archive(extra={name: b'<x/>'}))


def test_complex_ofd_does_not_ocr_a_logo():
    invoice = InvoiceReader(FakeBackend()).read(ofd_archive())
    assert "rendering_unavailable" in codes(invoice)


def test_image_requires_explicit_backend(png):
    result = read_invoice(png)
    assert result.total is None and result.status == "unrecognized"
    assert "backend_required" in codes(result)
    result = InvoiceReader(FakeBackend()).read(png)
    assert result.total == Decimal("113") and result.needs_review


def test_partial_backend_values_do_not_discard_valid_fields(png):
    backend = FakeBackend({"kind": "vat_invoice", "invoice_number": "00123456", "total": "113", "tax": "nonsense", "buyer": {"name": "Test", "unknown": "ignored"}, "issue_date": "2026-99-99"})
    invoice = InvoiceReader(backend).read(png)
    assert invoice.total == Decimal("113") and invoice.invoice_number == "00123456"
    assert invoice.tax is None and invoice.issue_date is None and invoice.buyer.name == "Test"
    assert "invalid_field" in codes(invoice)


def test_model_never_overwrites_direct_fields(xml_invoice):
    result = InvoiceReader(FakeBackend(), ocr="always").read(pdf_with_xml(xml_invoice))
    assert result.invoice_number == "00000001234567890123"
    assert result.sources["total"].source == Source.PDF_XML
    assert "conflicting_field" in codes(result)


def test_backend_failure_keeps_partial_results(xml_invoice):
    result = InvoiceReader(FakeBackend(error=True), ocr="always").read(pdf_with_xml(xml_invoice))
    assert result.total == Decimal("113") and "backend_failed" in codes(result)


def test_no_ocr_mode(png):
    backend = FakeBackend()
    result = InvoiceReader(backend, ocr="never").read(png)
    assert backend.calls == 0 and result.total is None


def test_model_multiple_invoices_not_merged(png):
    result = InvoiceReader(FakeBackend({"document_count": 2, "total": "226"})).read(png)
    assert result.total is None and "multiple_documents" in codes(result)


def test_misnested_common_fields_preserve_journey_and_are_marked(png):
    payload = {"kind": "rail_ticket", "rail": {"train_number": "G123", "total": "553.00", "unknown": "unused"}}
    result = InvoiceReader(FakeBackend(payload)).read(png)
    assert result.total == Decimal("553.00") and result.rail.train_number == "G123"
    assert "model_schema_normalized" in codes(result) and "unknown_model_field" in codes(result)
    assert payload["rail"]["total"] == "553.00"  # caller payload is not mutated


def test_misnested_conflicts_are_not_overwritten(png):
    payload = {"kind": "rail_ticket", "total": "553.00", "rail": {"train_number": "G123", "total": "999.00"}}
    result = InvoiceReader(FakeBackend(payload)).read(png)
    assert result.total == Decimal("553.00") and "conflicting_field" in codes(result)


def test_vat_text_aggregate_and_party_fields():
    text = '''电子发票（普通发票）
发票号码：00000001234567890123
开票日期：2026年11月19日
购买方名称：测试购买方
销售方名称：测试销售方
合 计 ￥100.00 ￥13.00
价税合计（大写）壹佰壹拾叁元整 （小写）￥113.00'''
    invoice = parse_text(text, Source.PDF_TEXT)
    assert invoice.total == Decimal("113") and invoice.amount == Decimal("100") and invoice.tax == Decimal("13")
    assert invoice.buyer.name == "测试购买方" and invoice.seller.name == "测试销售方"
    assert invoice.invoice_number.startswith("000") and invoice.issue_date.isoformat() == "2026-11-19"


@pytest.mark.parametrize("value", ["1.234", "1e3", "1,23"])
def test_text_amount_does_not_truncate_invalid_decimal(value):
    invoice = parse_text("电子发票\n价税合计（小写）：￥" + value, Source.PDF_TEXT)
    assert invoice.total is None


def test_rail_optional_journey():
    text = '''电子发票（铁路电子客票）
发票号码：00000001234567890123
旅客姓名：张测试
车次：G123
出发站：北京南站
到达站：上海虹桥站
乘车日期：2026年11月19日
开车时间：08:30
席别：二等座
票价：￥553.00'''
    invoice = parse_text(text, Source.PDF_TEXT)
    assert invoice.kind == InvoiceKind.RAIL and invoice.total == Decimal("553")
    assert invoice.rail.train_number == "G123" and invoice.rail.passenger_name == "张测试"
    assert invoice.rail.departure_station == "北京南站" and invoice.rail.seat_number is None


def test_air_fare_is_not_total():
    text = '''航空运输电子客票行程单
旅客姓名：张测试
电子客票号码：123-1234567890
航班号：CA1234
票价：800.00
民航发展基金：50.00
燃油附加费：40.00
合计：890.00'''
    invoice = parse_text(text, Source.PDF_TEXT)
    assert invoice.kind == InvoiceKind.AIR and invoice.total == Decimal("890")
    assert invoice.air.fare == Decimal("800") and invoice.air.airport_tax == Decimal("50")
    assert invoice.air.segments[0].flight_number == "CA1234"
    assert invoice.air.ticket_number == "123-1234567890"


@pytest.mark.parametrize("data", [b'', b'not a file', b'<Invoice/>more junk', b'%PDF-broken'])
def test_invalid_inputs(data):
    with pytest.raises(InvalidDocumentError):
        read_invoice(data)


def test_max_file_size(png):
    with pytest.raises(InvalidDocumentError):
        InvoiceReader(max_bytes=10).read(png)


def test_json_response_and_invalid_responses():
    assert parse_response('```json\n{"total":"1"}\n```')["total"] == "1"
    for raw in ["[]", "not json", '{"total":']:
        with pytest.raises(RecognitionError):
            parse_response(raw)


@pytest.mark.parametrize("url", ["file:///tmp/model", "https://user:secret@example.test/v1", "https://example.test/v1?key=secret"])
def test_endpoint_requires_explicit_valid_url(url):
    with pytest.raises(ValueError):
        OpenAICompatibleBackend(base_url=url, model="test")


def test_cli(tmp_path, xml_invoice, capsys):
    file = tmp_path / "test.xml";file.write_bytes(xml_invoice)
    assert main([str(file)]) == 0
    assert json.loads(capsys.readouterr().out)["total"] == "113.00"
    assert main([str(tmp_path / "missing")]) == 1
    captured = capsys.readouterr()
    assert "missing" not in captured.err


def test_raw_text_disabled_by_default():
    result = InvoiceReader(include_raw_text=True).read(pdf_with_xml())
    assert result.raw_text is None
    assert Invoice().raw_text is None
