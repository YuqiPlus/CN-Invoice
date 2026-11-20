"""Official Ministry of Finance 2023-12-31 transport and VAT instance mappings.

Mappings follow the Chinese labels in the published 2025 taxonomy packages.
Accounting entries and receiver-only missing facts are never used to infer totals.
"""
from __future__ import annotations

from typing import Any

from ._merge import issue, put
from .models import AirJourney, FlightSegment, Invoice, InvoiceKind, RailJourney, Source

BASE = "http://xbrl.mof.gov.cn/taxonomy/2023-12-31/"
FAMILIES = {BASE + "rai": InvoiceKind.RAIL, BASE + "atr": InvoiceKind.AIR, BASE + "inv": InvoiceKind.VAT}


def read_xbrl(root, source: Source) -> Invoice | None:
    namespaces = {element.tag[1:].split("}", 1)[0] for element in root.iter() if element.tag.startswith("{")}
    families = namespaces & FAMILIES.keys()
    if not families:
        return None
    result = Invoice()
    if len(families) != 1:
        issue(result, "multiple_documents", "XML 包含多个票据命名空间，请拆分或复核")
        return result
    ns = families.pop()
    kind = FAMILIES[ns]
    put(result, "kind", kind, source)
    put(result, "is_electronic", True, source)
    facts: dict[str, list[Any]] = {}
    for element in root.iter():
        if element.tag.startswith("{" + ns + "}") and not len(element):
            facts.setdefault(element.tag.split("}", 1)[1], []).append(element)
    units = {element.attrib.get("id", ""): "".join(element.itertext()).strip()
             for element in root.iter("{http://www.xbrl.org/2003/instance}unit")}
    contexts: set[str] = set()

    def values(name: str) -> list[str]:
        return [(element.text or "").strip() for element in facts.get(name, []) if (element.text or "").strip()]

    def journey_fields(name: str, schema, values_map: dict[str, Any]) -> None:
        value = schema()
        for key, item in values_map.items():
            try:
                setattr(value, key, item)
            except ValueError:
                issue(result, "invalid_field", "有旅程字段无法解析，已保留其他字段", name + "." + key)
        put(result, name, value, source)

    def field(path: str, *names: str, monetary: bool = False) -> None:
        for name in names:
            for element in facts.get(name, []):
                value = (element.text or "").strip()
                if not value:
                    continue
                if monetary:
                    unit = element.attrib.get("unitRef")
                    if unit and units.get(unit, "").split(":")[-1] != "CNY":
                        issue(result, "unknown_currency", "结构化金额的币种无法确认", path)
                        continue
                    if unit:
                        put(result, "currency", "CNY", source)
                    if element.attrib.get("contextRef"):
                        contexts.add(element.attrib["contextRef"])
                put(result, path, value, source)

    common = {
        "issue_date": ["DateOfIssue", "IssueDate"], "buyer.name": ["NameOfPurchaser"],
        "buyer.tax_id": ["UnifiedSocialCreditCodeOfPurchaser", "TaxpayerIdentificationNumberUnifiedSocialCreditCodeOfPurchaser"],
        "buyer.address_phone": ["AddressPhoneNumberOfPurchaser"], "buyer.bank_details": ["DepositBankAndAccountNumberOfPurchaser"],
        "seller.name": ["NameOfSeller"],
        "seller.tax_id": ["UnifiedSocialCreditCodeOfSeller", "TaxpayerIdentificationNumberUnifiedSocialCreditCodeOfSeller"],
        "seller.address_phone": ["AddressPhoneNumberOfSeller"], "seller.bank_details": ["DepositBankAndAccountNumberOfSeller"],
        "remarks": ["Remarks"], "issuer": ["Issuer"], "payee": ["Payee"], "reviewer": ["Reviewer"],
    }
    for path, aliases in common.items():
        field(path, *aliases)
    if kind == InvoiceKind.VAT:
        for path, aliases in {
            "invoice_type": ["TypeOfInvoice"], "invoice_number": ["NumberOfInvoice"], "invoice_code": ["CodeOfInvoice"],
            "unique_code": ["UniqueCodeOfInvoice"], "check_code": ["IdentifyingCode"], "total_in_words": ["TaxIncludedAmountInWords"],
        }.items():
            field(path, *aliases)
        for path, alias in [("amount", "TotalAmountExcludingTax"), ("tax", "TotalTaxAmount"), ("total", "TaxIncludedAmountInFigures")]:
            field(path, alias, monetary=True)
        items = []
        aliases = {"NameOfTaxableGoodsOrServices": "name", "UnitOfGoodsOrServices": "unit", "QuantityOfGoodsOrServices": "quantity", "UnitPriceOfGoodsOrServices": "unit_price", "AmountExcludingTax": "amount", "TaxRate": "tax_rate", "TaxAmount": "tax"}
        for element in root.iter("{" + ns + "}InformationOfTaxableGoodsOrServicesDetailItemsTuple"):
            values_map = {aliases[child.tag.split("}", 1)[-1]]: child.text for child in element
                          if child.tag.startswith("{" + ns + "}") and child.tag.split("}", 1)[-1] in aliases and child.text}
            if values_map:
                items.append(values_map)
        put(result, "items", items, source)
    elif kind == InvoiceKind.RAIL:
        field("invoice_number", "ElectronicInvoiceRailwayETicketNumber")
        field("amount", "TotalAmountExcludingTax", monetary=True)
        field("tax", "TaxAmount", monetary=True)
        field("total", "Fare", monetary=True)
        aliases = {
            "Name": "passenger_name", "IdNumber": "passenger_id", "ETicketNumber": "ticket_number",
            "TrainNumber": "train_number", "DepartureStation": "departure_station", "DestinationStation": "arrival_station",
            "TravelDate": "departure_date", "DepartureTime": "departure_time", "SeatLevel": "seat_class",
            "Carriage": "carriage", "Seat": "seat_number",
        }
        journey = {}
        for tag, key in aliases.items():
            options = set(values(tag))
            if len(options) == 1:
                journey[key] = options.pop()
            elif options:
                issue(result, "conflicting_field", "旅程字段存在多个值，请核对", "rail." + key)
        if journey:
            journey_fields("rail", RailJourney, journey)
    else:
        field("invoice_number", "ElectronicInvoiceAirTransportReceiptNumber")
        field("tax", "VatTaxAmount", monetary=True)
        field("total", "TotalAmount", monetary=True)
        journey = {}
        aliases = {"PassengerName": "passenger_name", "ValidIdNumber": "passenger_id", "ETicketNumber": "ticket_number", "NumberOfGpOrder": "order_number", "Fare": "fare", "CivilAviationDevelopmentFund": "airport_tax", "FuelSurcharge": "fuel_surcharge", "OtherTaxes": "other_taxes", "Insurance": "insurance"}
        for tag, key in aliases.items():
            options = set(values(tag))
            if len(options) == 1:
                journey[key] = options.pop()
            elif options:
                issue(result, "conflicting_field", "航空字段存在多个值，请核对", "air." + key)
        segments = []
        segment_aliases = {"Carrier": "carrier", "Flight": "flight_number", "DepartureStation": "departure_airport", "DestinationStation": "arrival_airport", "CarrierDate": "departure_date", "DepartureTime": "departure_time", "Class": "cabin_class"}
        for element in root.iter("{" + ns + "}DetailInformationOfAirTicketTuple"):
            segment = FlightSegment()
            for child in element:
                key = segment_aliases.get(child.tag.split("}", 1)[-1]) if child.tag.startswith("{" + ns + "}") else None
                if key and child.text:
                    try:
                        setattr(segment, key, child.text)
                    except ValueError:
                        issue(result, "invalid_field", "有航段字段无法解析，已保留其他字段", "air.segments." + key)
            segments.append(segment)
        if segments:
            journey["segments"] = segments
        if journey:
            journey_fields("air", AirJourney, journey)
    if len(contexts) > 1:
        issue(result, "multiple_contexts", "文件包含多个金额上下文，请核对票据范围")
    return result
