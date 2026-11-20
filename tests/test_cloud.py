import json

import httpx
import pytest
from PIL import Image

from cninvoice import InvoiceReader, RecognitionError
from cninvoice.cloud import HuaweiOCRBackend, map_huawei_response


def response(kind, value):
    return {'result': [{'status': {'error_code': 'AIS.0000'}, 'type': kind, 'content': value}]}


def test_cloud_vat_uses_explicit_aggregates_not_subtotals_or_computation(png):
    data = response('vat_invoice', {'number': '00123456', 'total': '￥113.00', 'subtotal_amount': '￥100.00',
                                  'subtotal_tax': '￥13.00', 'sum_amount': '90.00', 'sum_tax': '12.00'})
    payload = map_huawei_response(data)
    assert (payload['amount'], payload['tax'], payload['total']) == ('￥100.00', '￥13.00', '￥113.00')
    assert payload['invoice_number'] == '00123456'
    data['result'][0]['content'].pop('total')
    assert map_huawei_response(data)['total'] is None


def test_cloud_train_serial_is_not_invoice_number():
    payload = map_huawei_response(response('train_ticket', {
        'ticket_id': 'K090841', 'ticket_price': '164.5元', 'train_number': 'G1138',
        'departure_time': '2017年12月12日20:39', 'seat_number': '03车03C号',
    }))
    assert payload['rail']['ticket_number'] == 'K090841' and payload.get('invoice_number') is None
    assert payload['rail']['departure_date'] == '2017-12-12' and payload['rail']['departure_time'] == '20:39'
    assert payload['rail']['carriage'] == '03' and payload['rail']['seat_number'] == '03C'
    assert payload['total'] == '164.5' and payload.get('tax') is None


def test_cloud_air_fare_is_not_total_and_incomplete_date_is_not_guessed():
    data = response('flight_itinerary', {'fare': 'CNY 800.00', 'caac_development_fund': 'CN 50.00',
        'total': 'CNY 890.00', 'itinerary_list': [{'date': '19NOV', 'flight': 'CA1234'}]})
    payload = map_huawei_response(data)
    assert payload['total'] == '890.00' and payload['air']['fare'] == '800.00'
    assert payload['air']['airport_tax'] == '50.00'
    assert payload['air']['segments'][0]['departure_date'] is None
    assert payload['air'].get('booking_reference') is None


def test_cloud_multiple_tickets_not_combined():
    assert map_huawei_response({'result': [{}, {}]}) == {'document_count': 2}


@pytest.mark.parametrize('data', [{}, {'result': [{}]}, response('unknown', {}),
                                 {'result': [{'status': {'error_code': 'failed'}, 'content': {}}]}])
def test_cloud_malformed_results_rejected(data):
    with pytest.raises(RecognitionError):
        map_huawei_response(data)


@pytest.mark.parametrize('endpoint', ['http://example.test', 'https://user:token@example.test', 'https://example.test/path', 'https://example.test?token=x'])
def test_cloud_endpoint_validation(endpoint):
    with pytest.raises(ValueError):
        HuaweiOCRBackend(endpoint=endpoint, project_id='a' * 32, token='test-only')


def test_cloud_request_and_failed_http_do_not_expose_token(monkeypatch, png):
    real_client = httpx.Client
    calls = []
    def handle(request):
        calls.append(request)
        assert request.url.path == '/v2/' + 'a' * 32 + '/ocr/auto-classification'
        assert request.headers['X-Auth-Token'] == 'test-only-secret'
        assert json.loads(request.content)['type_list'] == ['vat_invoice', 'train_ticket', 'flight_itinerary']
        return httpx.Response(200, json=response('vat_invoice', {'subtotal_amount': '100', 'subtotal_tax': '13', 'total': '113'}))
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    backend = HuaweiOCRBackend(endpoint='https://ocr.example.test', project_id='a' * 32, token='test-only-secret')
    invoice = InvoiceReader(backend).read(png)
    assert str(invoice.total) == '113.00' and invoice.sources['total'].source == 'ocr'
    assert invoice.needs_review and len(calls) == 1
    def denied(request):
        return httpx.Response(401, text='test-only-secret')
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: real_client(transport=httpx.MockTransport(denied), **kwargs))
    with pytest.raises(RecognitionError) as exc:
        backend.recognize([Image.new('RGB', (100, 100))])
    assert 'test-only-secret' not in str(exc.value)


def test_cloud_multiple_pages_do_not_send_requests():
    backend = HuaweiOCRBackend(endpoint='https://ocr.example.test', project_id='a' * 32, token='test-only')
    with pytest.raises(RecognitionError, match='one page'):
        backend.recognize([Image.new('RGB', (100, 100)), Image.new('RGB', (100, 100))])
