"""Document families for independent specialists, distinct from accounting kinds."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class DocumentProfile(str, Enum):
    VAT = 'vat_invoice'
    RAIL_EINVOICE = 'rail_einvoice'
    RIDE_ITINERARY = 'ride_itinerary'
    AIR_ITINERARY = 'air_itinerary'
    RAIL_LEGACY = 'rail_legacy'
    TAXI_RECEIPT = 'taxi_receipt'
    UNKNOWN = 'unknown'


SPECIALIST_FIELDS = {
    DocumentProfile.VAT: ('vat.amount', 'vat.tax', 'vat.total', 'vat.invoice_number'),
    DocumentProfile.RAIL_EINVOICE: (
        'rail.amount', 'rail.tax', 'rail.total', 'rail.invoice_number',
        'rail.ticket_number', 'rail.train_number', 'rail.departure_station',
        'rail.arrival_station', 'rail.departure', 'rail.seat_class', 'rail.passenger_name'),
    DocumentProfile.RIDE_ITINERARY: ('ride.total',),
    DocumentProfile.AIR_ITINERARY: ('air.total', 'air.fare', 'air.ticket_number', 'air.passenger_name'),
}


@dataclass(frozen=True)
class ProfileDecision:
    profile: DocumentProfile
    reason: str


def route_profile(texts: Sequence[str]) -> ProfileDecision:
    """Use explicit document titles; never infer a trip sheet from a brand alone.

    This is a conservative OCR title router, not a trained visual classifier.
    Ambiguous or unrecognised titles abstain instead of trying every expert.
    """
    lines = [re.sub(r'\s', '', unicodedata.normalize('NFKC', text)).upper() for text in texts]
    joined = '\n'.join(lines)
    # Joining only consecutive, short title fragments supports split title OCR
    # without letting a remark elsewhere turn an ordinary VAT invoice into rail.
    titles = lines + [''.join(lines[i:i + 3]) for i in range(len(lines))
                      if all(len(t) <= 24 for t in lines[i:i + 3])]
    rail = any(re.search(r'电子发票[:：]?[([]?铁路电子客票', t) for t in titles)
    air = any(re.search(r'航空(?:运输)?电子客票行程单|航空运输电子发票', t) for t in titles)
    ride_words = r'滴滴(?:出行|企业版)?|高德打车|曹操出行|T3出行|网约车|用车'
    ride = any(re.search(rf'(?:{ride_words}).{{0,12}}(?:行程单|行程报销单|行程报销凭证)', t)
               for t in titles)
    vat = any(re.search(r'增值税.{0,12}发票|数电发票|电子发票[([]?(?:普通发票|专用发票)', t)
              for t in titles)
    # The electronic rail/air title is itself a specialised invoice title.
    # A second ordinary VAT title must still be treated as multiple documents.
    hits = ([DocumentProfile.RAIL_EINVOICE] if rail else [])
    hits += [DocumentProfile.AIR_ITINERARY] if air else []
    hits += [DocumentProfile.RIDE_ITINERARY] if ride else []
    hits += [DocumentProfile.VAT] if vat else []
    if len(hits) > 1:
        return ProfileDecision(DocumentProfile.UNKNOWN, 'ambiguous_titles')
    if hits:
        return ProfileDecision(hits[0], 'explicit_title')
    # Old VAT title characters are often embossed or fragmented. Require the
    # tax table AND party identifiers, not merely the words "total"/"invoice".
    if ('纳税人识别号' in joined and
            (('税额' in joined and '价税合计' in joined) or
             ('货物或应税劳务' in joined and '税率' in joined))):
        return ProfileDecision(DocumentProfile.VAT, 'vat_tax_table')
    if ('航空运输' in joined and '电子客票号码' in joined and
            re.search(r'合计(?:TOTAL)?|TOTAL', joined)):
        return ProfileDecision(DocumentProfile.AIR_ITINERARY, 'air_ticket_table')
    if re.search(r'铁路电子客票|火车票|火车客票|铁路客票', joined):
        return ProfileDecision(DocumentProfile.RAIL_LEGACY, 'legacy_title')
    if re.search(r'出租汽车|出租车', joined):
        return ProfileDecision(DocumentProfile.TAXI_RECEIPT, 'legacy_title')
    return ProfileDecision(DocumentProfile.UNKNOWN, 'no_supported_title')
