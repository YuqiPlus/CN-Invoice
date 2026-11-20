from __future__ import annotations

import base64
import io
import json
import re
from datetime import date
from typing import Sequence
from urllib.parse import urlsplit

from PIL import Image

from .errors import RecognitionError
from .models import Source
from .parsers import parse_date


def _money(value):
    if not isinstance(value, str):
        return value
    return re.sub(r'元$', '', re.sub(r'^(?:CNY|CN)\s*', '', value.strip()))


def _date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(parse_date(value)).isoformat()
    except (ValueError, TypeError):
        return None


def map_huawei_response(data: dict) -> dict:
    """Map documented OCR fields; never synthesize totals or reuse ticket serials."""
    rows = data.get('result')
    if not isinstance(rows, list):
        raise RecognitionError('Ticket OCR returned no classification result')
    if len(rows) != 1:
        return {'document_count': len(rows)}
    row = rows[0]
    if row.get('status', {}).get('error_code') != 'AIS.0000':
        raise RecognitionError('Ticket OCR did not recognize the document')
    value = row.get('content')
    if not isinstance(value, dict):
        raise RecognitionError('Ticket OCR returned invalid content')
    kind = row.get('type')
    payload = {'document_count': 1, 'currency': 'CNY'}
    if kind == 'vat_invoice':
        payload.update({
            'kind': 'vat_invoice', 'invoice_number': value.get('number'), 'invoice_code': value.get('code'),
            'invoice_type': value.get('title'), 'issue_date': _date(value.get('issue_date')),
            'check_code': value.get('check_code'), 'total_in_words': value.get('total_in_words'),
            'amount': _money(value.get('subtotal_amount')), 'tax': _money(value.get('subtotal_tax')),
            'total': _money(value.get('total')), 'remarks': value.get('remarks'),
            'issuer': value.get('issuer'), 'reviewer': value.get('reviewer'), 'payee': value.get('receiver'),
        })
        for party in ('buyer', 'seller'):
            payload[party] = {'name': value.get(party + '_name'), 'tax_id': value.get(party + '_id'),
                              'address_phone': value.get(party + '_address'), 'bank_details': value.get(party + '_bank')}
        payload['items'] = [{key: _money(item.get(key)) if key in {'amount', 'tax'} else item.get(key)
                             for key in ('name', 'specification', 'unit', 'quantity', 'unit_price', 'amount', 'tax_rate', 'tax')}
                            for item in value.get('item_list', []) if isinstance(item, dict)]
    elif kind == 'train_ticket':
        journey = {canonical: value.get(raw) for canonical, raw in {
            'ticket_number': 'ticket_id', 'train_number': 'train_number', 'departure_station': 'departure_station',
            'arrival_station': 'destination_station', 'seat_number': 'seat_number', 'seat_class': 'seat_category',
            'passenger_name': 'name', 'passenger_id': 'id_number',
        }.items()}
        timestamp = str(value.get('departure_time') or '')
        match = re.fullmatch(r'(\d{4}年\d{1,2}月\d{1,2}日)(\d{1,2}:\d{2})', timestamp)
        if match:
            journey.update(departure_date=_date(match.group(1)), departure_time=match.group(2).zfill(5))
        seat = re.fullmatch(r'(\d{1,3})车(\d{1,3}[A-Z]?)号', str(value.get('seat_number') or ''))
        if seat:
            journey['carriage'], journey['seat_number'] = seat.groups()
        payload.update(kind='rail_ticket', total=_money(value.get('ticket_price')), rail=journey)
    elif kind == 'flight_itinerary':
        air = {canonical: value.get(raw) for canonical, raw in {
            'ticket_number': 'e_ticket_number', 'passenger_name': 'passenger_name',
            'passenger_id': 'id_number', 'order_number': 'gp_number',
        }.items()}
        for canonical, raw in {'fare': 'fare', 'airport_tax': 'caac_development_fund',
                               'fuel_surcharge': 'fuel_surcharge', 'other_taxes': 'other_taxes', 'insurance': 'insurance'}.items():
            air[canonical] = _money(value.get(raw))
        air['segments'] = [
            {'departure_airport': segment.get('departure_station'), 'arrival_airport': segment.get('destination_station'),
             'carrier': segment.get('carrier'), 'flight_number': segment.get('flight'), 'cabin_class': segment.get('cabin_class'),
             'departure_date': _date(segment.get('date')), 'departure_time': segment.get('time')}
            for segment in value.get('itinerary_list', []) if isinstance(segment, dict)
        ]
        payload.update(kind='air_itinerary', invoice_number=value.get('number'),
                       issue_date=_date(value.get('issue_date')), total=_money(value.get('total')),
                       tax=_money(value.get('tax')), air=air)
    else:
        raise RecognitionError('Unsupported ticket OCR document type')
    return payload


class HuaweiOCRBackend:
    """Explicit Huawei Cloud auto-classification endpoint using a caller-provided IAM token.

    No account creation, token acquisition, automatic retries or implicit network calls.
    Field mappings follow the vendor's documented VAT/train/flight APIs.
    """

    source = Source.OCR

    def __init__(self, *, endpoint: str, project_id: str, token: str, timeout: float = 60):
        url = urlsplit(endpoint)
        if (url.scheme != 'https' or not url.hostname or url.username or url.password
                or url.query or url.fragment or url.path not in {'', '/'}):
            raise ValueError('An explicit HTTPS service endpoint without path or credentials is required')
        if not re.fullmatch(r'[a-fA-F0-9]{32}', project_id) or not token or '\n' in token or '\r' in token:
            raise ValueError('A project ID and IAM token are required')
        self.endpoint, self.project_id = endpoint.rstrip('/'), project_id
        self._token, self.timeout = token, timeout

    def recognize(self, images: Sequence[Image.Image]) -> dict:
        if len(images) != 1:
            raise RecognitionError('Ticket cloud OCR currently requires one page at a time')
        try:
            import httpx
            buffer = io.BytesIO()
            images[0].save(buffer, format='PNG')
            if buffer.tell() > 7_500_000:
                buffer = io.BytesIO()
                images[0].convert('RGB').save(buffer, format='JPEG', quality=95)
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            if len(encoded) > 10 * 1024**2:
                raise RecognitionError('Ticket OCR image exceeds the service input limit')
            with httpx.Client(timeout=self.timeout, follow_redirects=False, trust_env=False) as client:
                with client.stream('POST', self.endpoint + '/v2/' + self.project_id + '/ocr/auto-classification',
                    headers={'X-Auth-Token': self._token},
                    json={'image': encoded, 'type_list': ['vat_invoice', 'train_ticket', 'flight_itinerary']}) as response:
                    response.raise_for_status()
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > 4 * 1024**2:
                            raise RecognitionError('Ticket OCR response exceeds the size limit')
            return map_huawei_response(json.loads(body))
        except RecognitionError:
            raise
        except Exception:
            raise RecognitionError('Ticket OCR request failed; check endpoint, service access and token') from None
