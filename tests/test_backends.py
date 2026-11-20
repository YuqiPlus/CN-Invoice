from __future__ import annotations

import json
import base64
import io

import httpx
import pytest
from PIL import Image

from cninvoice import InvoiceReader, RecognitionError
from cninvoice.backends import OpenAICompatibleBackend
from cninvoice.models import InvoiceKind, Source
from cninvoice.parsers import parse_xml


def test_http_backend_protocol(monkeypatch):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"kind":"rail_ticket","total":"553.00"}'}}]})

    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    backend = OpenAICompatibleBackend(base_url="http://127.0.0.1:8000/v1", model="local-model", api_key="test-only")
    with Image.new("RGB", (16, 16)) as image:
        result = backend.recognize([image])
    assert result["total"] == "553.00"
    assert captured[0].url.path == "/v1/chat/completions"
    payload = json.loads(captured[0].content)
    assert payload["model"] == "local-model"
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_http_provider_options_usage_and_truncation(monkeypatch):
    original = httpx.Client
    def handler(request):
        data = json.loads(request.content)
        assert data['thinking'] == {'type': 'disabled'} and data['max_tokens'] == 1024
        return httpx.Response(200, json={'model': 'test-model', 'usage': {'total_tokens': 123},
            'choices': [{'finish_reason': 'stop', 'message': {'content': '{"total":"113.00"}'}}]})
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    backend = OpenAICompatibleBackend(base_url='https://example.test', model='test-model', max_tokens=1024,
                                    extra_body={'thinking': {'type': 'disabled'}})
    with Image.new('RGB', (16, 16)) as image:
        result = backend.recognize([image])
    assert result['_backend_usage']['total_tokens'] == 123 and result['total'] == '113.00'
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        'choices': [{'finish_reason': 'length', 'message': {'content': '{"total":"113.00"}'}}]}))
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: original(transport=transport, **kwargs))
    with Image.new('RGB', (16, 16)) as image, pytest.raises(RecognitionError, match='truncated'):
        backend.recognize([image])


def test_http_provider_options_cannot_override_document_or_model():
    with pytest.raises(ValueError):
        OpenAICompatibleBackend(base_url='https://example.test', model='test', extra_body={'messages': []})


def test_teacher_prompt_is_per_instance_and_does_not_modify_default(monkeypatch):
    from cninvoice.backends import PROMPT
    captured=[];original=httpx.Client
    def handler(request):
        captured.append(json.loads(request.content)['messages'][0]['content'][0]['text'])
        return httpx.Response(200,json={'choices':[{'message':{'content':'{"total":"113.00"}'}}]})
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    teacher=OpenAICompatibleBackend(base_url='https://example.test',model='test',prompt='Explicit teacher task')
    default=OpenAICompatibleBackend(base_url='https://example.test',model='test')
    with Image.new('RGB',(16,16)) as image:
        teacher.recognize([image]);default.recognize([image])
    assert captured==['Explicit teacher task',PROMPT]


def test_http_image_preprocessing_is_explicit_and_preserves_original(monkeypatch):
    original = httpx.Client
    def handler(request):
        uri = json.loads(request.content)['messages'][0]['content'][1]['image_url']['url']
        assert uri.startswith('data:image/jpeg;base64,')
        with Image.open(io.BytesIO(base64.b64decode(uri.split(',', 1)[1]))) as uploaded:
            assert uploaded.size == (200, 100)
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"total":"113"}'}}]})
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    backend = OpenAICompatibleBackend(base_url='https://example.test', model='test', image_max_side=200, image_format='JPEG')
    with Image.new('RGB', (400, 200)) as image:
        backend.recognize([image])
        assert image.size == (400, 200)


@pytest.mark.parametrize("status,body", [(500, b'internal error'), (302, b''), (200, b'{}')])
def test_http_failures_do_not_expose_credentials_or_payload(monkeypatch, status, body):
    original = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(status, content=body, headers={"location": "https://unwanted.test"}))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    backend = OpenAICompatibleBackend(base_url="http://127.0.0.1:8000/v1", model="test", api_key="secret-test-key")
    with Image.new("RGB", (16, 16)) as image:
        with pytest.raises(RecognitionError) as error:
            backend.recognize([image])
    assert "secret-test-key" not in str(error.value) and "internal error" not in str(error.value)


def test_xml_transport_fields():
    content = '''<Invoice><InvoiceTypeName>电子发票（铁路电子客票）</InvoiceTypeName>
    <InvoiceNumber>00000001234567890123</InvoiceNumber><TaxInclusiveTotalAmount>553</TaxInclusiveTotalAmount>
    <PassengerName>张测试</PassengerName><TrainNumber>G123</TrainNumber><DepartureStation>北京南</DepartureStation>
    <DepartureDate>2026年11月20日</DepartureDate></Invoice>'''
    invoice = parse_xml(content.encode())
    assert invoice.kind == InvoiceKind.RAIL
    assert invoice.rail.train_number == "G123"
    assert invoice.rail.departure_date.isoformat() == "2026-11-20"
    assert invoice.sources["rail"].source == Source.XML


def test_invoice_serialization_round_trip(xml_invoice):
    from cninvoice import Invoice

    invoice = InvoiceReader().read(xml_invoice)
    restored = Invoice.model_validate_json(invoice.model_dump_json(round_trip=True))
    assert restored.total == invoice.total and restored.sources == invoice.sources
