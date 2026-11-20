from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any

from defusedxml import ElementTree as SafeET

from ._merge import issue, put
from .errors import InvalidDocumentError
from .models import AirJourney, FlightSegment, Invoice, InvoiceKind, LineItem, RailJourney, Source

OFD_NS = "http://www.ofdspec.org/2016"
INVOICE_NAMESPACES = {
    "http://www.edrm.org.cn/schema/e-invoice/2016",
    "http://www.edrm.org.cn/schema/e-invoice/2019",
    "http://xbrl.mof.gov.cn/taxonomy/2023-12-31/einv",
}
XML_FIELDS = {
    "invoice_number": {"InvoiceNumber", "InvoiceNo", "发票号码"},
    "invoice_code": {"InvoiceCode", "发票代码"},
    "invoice_type": {"InvoiceTypeName", "InvoiceName", "InvoiceType", "发票类型"},
    "issue_date": {"IssueDate", "InvoiceDate", "DateOfIssue", "RequestTime", "开票日期"},
    "check_code": {"CheckCode", "InvoiceCheckCode", "校验码"},
    "buyer.name": {"BuyerName", "PurchaserName", "购买方名称"},
    "buyer.tax_id": {"BuyerTaxID", "BuyerTaxId", "BuyerIdNum", "PurchaserTaxID", "购买方识别号"},
    "buyer.address": {"BuyerAddress", "PurchaserAddress"},
    "buyer.phone": {"BuyerPhone", "PurchaserPhone"},
    "buyer.bank_name": {"BuyerBankName"},
    "buyer.bank_account": {"BuyerBankAccount"},
    "seller.name": {"SellerName", "销售方名称"},
    "seller.tax_id": {"SellerTaxID", "SellerTaxId", "SellerIdNum", "销售方识别号"},
    "seller.address": {"SellerAddress"},
    "seller.phone": {"SellerPhone"},
    "seller.bank_name": {"SellerBankName"},
    "seller.bank_account": {"SellerBankAccount"},
    "amount": {"TaxExclusiveTotalAmount", "TotalAmWithoutTax", "AmountWithoutTax", "合计金额"},
    "tax": {"TaxTotalAmount", "TotalTaxAm", "合计税额"},
    "total": {"TaxInclusiveTotalAmount", "TotalTax-includedAmount", "TotalAmountWithTax", "价税合计", "价税合计(小写)", "价税合计（小写）"},
    "total_in_words": {"TotalAmountInWords", "TotalTaxIncludedAmountInWords", "价税合计大写"},
    "currency": {"Currency", "CurrencyCode"},
    "remarks": {"Remarks", "Remark", "备注"},
    "issuer": {"Issuer", "Drawer", "开票人"},
    "reviewer": {"Reviewer", "复核人"},
    "payee": {"Payee", "收款人"},
}
ALIASES = {alias: field for field, aliases in XML_FIELDS.items() for alias in aliases}
MONEY = r"-?\d[\d,]*(?:\.\d{1,2})?(?![\d.,A-Za-z])"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def namespace(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def safe_xml(data: bytes):
    if len(data) > 2 * 1024**2:
        raise InvalidDocumentError("XML exceeds the size limit")
    try:
        root = SafeET.fromstring(data, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except Exception:
        raise InvalidDocumentError("Invalid or unsafe XML") from None
    if sum(1 for _ in root.iter()) > 50_000:
        raise InvalidDocumentError("XML contains too many elements")
    return root


def parse_date(value: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?", value.strip()):
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return value
    match = re.fullmatch(r"(\d{4})[年/.-](\d{1,2})[月/.-](\d{1,2})日?", value.strip())
    if match:
        return date(*map(int, match.groups())).isoformat()
    return value


def classify(title: str) -> InvoiceKind:
    from .profiles import DocumentProfile, route_profile
    decision = route_profile(title.splitlines())
    profile = decision.profile
    if profile == DocumentProfile.RIDE_ITINERARY:
        return InvoiceKind.RIDE
    if decision.reason == 'ambiguous_titles':
        return InvoiceKind.UNKNOWN
    compact = re.sub(r"\s", "", title)
    if any(word in compact for word in ["铁路", "火车票", "火车客票"]):
        return InvoiceKind.RAIL
    if any(word in compact for word in ["航空", "机票", "行程单"]):
        return InvoiceKind.AIR
    if any(word in compact for word in ["出租车", "出租汽车"]):
        return InvoiceKind.TAXI
    if any(word in compact for word in ["增值税", "电子发票", "数电发票"]):
        return InvoiceKind.VAT
    return InvoiceKind.UNKNOWN


def parse_xml(data: bytes, source: Source = Source.XML) -> Invoice:
    root = safe_xml(data)
    from .xbrl import read_xbrl
    structured = read_xbrl(root, source)
    if structured is not None:
        return structured
    result = Invoice()
    root_ns = namespace(root.tag)
    plain = not root_ns and local_name(root.tag) in {"Invoice", "EInvoice", "DigitalInvoice"}
    supported = root_ns in INVOICE_NAMESPACES or plain or any(namespace(e.tag) in INVOICE_NAMESPACES for e in root.iter())
    contexts: set[str] = set()
    units: dict[str, str] = {}
    for element in root.iter():
        if element.tag == "{http://www.xbrl.org/2003/instance}unit":
            units[element.attrib.get("id", "")] = "".join(element.itertext()).strip()
    for element in root.iter():
        name, ns = local_name(element.tag), namespace(element.tag)
        is_custom = ns == OFD_NS and name == "CustomData"
        if not (ns in INVOICE_NAMESPACES or (plain and not ns) or is_custom):
            continue
        field = ALIASES.get(element.attrib.get("Name", "") if is_custom else name)
        if field is None or len(element):
            continue
        value = (element.text or "").strip()
        if not value:
            continue
        if field in {"amount", "tax", "total"}:
            context = element.attrib.get("contextRef")
            if context:
                contexts.add(context)
            unit = element.attrib.get("unitRef")
            if unit and not re.fullmatch(r"(?:[\w-]+:)?CNY", units.get(unit, "")):
                issue(result, "unknown_currency", "结构化金额的币种无法确认", field)
                continue
            if unit:
                put(result, "currency", "CNY", source)
        if field == "issue_date":
            value = parse_date(value)
        put(result, field, value, source)
    if len(contexts) > 1:
        issue(result, "multiple_contexts", "文件包含多个金额上下文，请核对发票范围")
    if result.invoice_type:
        put(result, "kind", classify(result.invoice_type), source)
        if "电子" in result.invoice_type or "数电" in result.invoice_type:
            put(result, "is_electronic", True, source)
    if result.kind == InvoiceKind.UNKNOWN and supported and result.invoice_number is not None:
        put(result, "kind", InvoiceKind.VAT, source)
    if supported and result.kind in {InvoiceKind.RAIL, InvoiceKind.AIR}:
        transport_aliases = {
            "PassengerName": "passenger_name", "PassengerID": "passenger_id",
            "PassengerId": "passenger_id", "TicketNumber": "ticket_number", "TicketNo": "ticket_number",
        }
        if result.kind == InvoiceKind.RAIL:
            transport_aliases.update({
                "TrainNumber": "train_number", "TrainNo": "train_number",
                "DepartureStation": "departure_station", "ArrivalStation": "arrival_station",
                "DepartureDate": "departure_date", "DepartureTime": "departure_time",
                "SeatClass": "seat_class", "Carriage": "carriage", "SeatNumber": "seat_number",
            })
        else:
            transport_aliases.update({"BookingReference": "booking_reference", "Fare": "fare", "AirportTax": "airport_tax", "FuelSurcharge": "fuel_surcharge", "Insurance": "insurance"})
        transport: dict[str, Any] = {}
        for element in root.iter():
            name = local_name(element.tag)
            if namespace(element.tag) not in {root_ns, *INVOICE_NAMESPACES} or name not in transport_aliases or len(element):
                continue
            value = (element.text or "").strip()
            field = transport_aliases[name]
            if field.endswith("date"):
                value = parse_date(value)
            if value:
                if field in transport and transport[field] != value:
                    issue(result, "multiple_tickets", "结构化旅程字段不一致，请核对票据范围")
                else:
                    transport[field] = value
        if transport:
            put(result, "rail" if result.kind == InvoiceKind.RAIL else "air", transport, source)
    # Only explicit item containers are accepted; aggregate/accounting entries are not line items.
    item_aliases = {
        "GoodsName": "name", "ItemName": "name", "Specification": "specification",
        "Unit": "unit", "Quantity": "quantity", "UnitPrice": "unit_price",
        "Amount": "amount", "TaxRate": "tax_rate", "TaxAmount": "tax",
    }
    items: list[LineItem] = []
    if supported:
        for element in root.iter():
            if namespace(element.tag) not in {root_ns, *INVOICE_NAMESPACES}:
                continue
            if local_name(element.tag) not in {"Item", "GoodsItem", "InvoiceItem"}:
                continue
            values = {item_aliases[local_name(child.tag)]: child.text for child in element
                      if local_name(child.tag) in item_aliases and child.text}
            if not values:
                continue
            try:
                items.append(LineItem.model_validate(values))
            except ValueError:
                issue(result, "invalid_item", "有明细字段无法解析", "items")
        put(result, "items", items, source)
    if not result.sources and root_ns != OFD_NS:
        issue(result, "unknown_xml_schema", "XML 未匹配已支持的发票字段，未猜测其含义")
    return result


def _label(text: str, labels: str, pattern: str, *, flags: int = 0) -> list[str]:
    # A bounded optional line break allows label/value layouts without searching unrelated lines.
    return re.findall(r"(?:" + labels + r")[ \t]*[:：]?[ \t]*(?:\n[ \t]*)?(" + pattern + ")", text, flags)


def parse_text(text: str, source: Source, page: int = 1) -> Invoice:
    result = Invoice()
    text = unicodedata.normalize("NFKC", text).replace("−", "-")
    rows = [re.sub(r"[ \t]+", "", line) for line in text.splitlines()]
    compact = "\n".join(rows)
    title = next((row for row in rows[:20] if any(word in row for word in ["发票", "行程单", "火车票", "铁路客票"])), "")
    kind = classify(title)
    put(result, "kind", kind, source, page)
    if title:
        # Avoid including adjacent ticket fields in the type label.
        put(result, "invoice_type", title[:100], source, page)
        if "电子" in title or "数电" in title:
            put(result, "is_electronic", True, source, page)
    total_labels = "小写[)）]?|价税合计(?:\\(小写\\))?|合计票价|总金额"
    if kind == InvoiceKind.RAIL:
        total_labels += "|票价|票款"
    if kind == InvoiceKind.AIR:
        total_labels += "|合计"
    for field, labels, pattern in [
        ("invoice_number", "发票号码|发票号", r"\d{8,20}(?!\d)"),
        ("invoice_code", "发票代码", r"\d{10,12}(?!\d)"),
        ("check_code", "校验码", r"\d{6,20}(?!\d)"),
        ("issue_date", "开票日期|填开日期", r"\d{4}[年/.-]\d{1,2}[月/.-]\d{1,2}日?"),
        ("total", total_labels, r"[¥￥]?(" + MONEY + r")"),
        ("amount", "合计金额|不含税金额", r"[¥￥]?(" + MONEY + r")"),
        ("tax", "合计税额", r"[¥￥]?(" + MONEY + r")"),
        ("total_in_words", "大写[)）]?", r"[零壹贰叁肆伍陆柒捌玖拾佰仟万亿圆元角分整正负]+"),
    ]:
        for value in _label(compact, labels, pattern):
            if isinstance(value, tuple):
                value = value[-1]
            if field == "issue_date":
                try:
                    value = parse_date(value)
                except ValueError:
                    issue(result, 'invalid_date', '票面日期无法解析，保留其他可读字段', field)
                    continue
            put(result, field, value, source, page)
    # Standard VAT totals row: never interpret detail rows as aggregate totals.
    for row in rows:
        match = re.fullmatch(r"合计[¥￥](" + MONEY + r")[¥￥](" + MONEY + r")", row)
        if match:
            put(result, "amount", match.group(1), source, page)
            put(result, "tax", match.group(2), source, page)
    # Horizontal buyer/seller blocks have two labels on the same physical row.
    for field, labels in [("name", "名称"), ("tax_id", "纳税人识别号|统一社会信用代码(?:/纳税人识别号)?")]:
        for row in rows:
            values = re.findall(r"(?:" + labels + r")[:：]([^:：\n]+?)(?=(?:名称|纳税人识别号|统一社会信用代码)[:：]|$)", row)
            if len(values) == 2 and kind == InvoiceKind.VAT:
                for party, value in zip(["buyer", "seller"], values):
                    if field == "tax_id":
                        match = re.match(r"[0-9A-Z]{15,20}", value)
                        value = match.group(0) if match else ""
                    put(result, party + "." + field, value, source, page)
    # Explicitly labelled party fields work for vertical and simplified layouts.
    for party, label in [("buyer", "购买方"), ("seller", "销售方")]:
        for field, suffix, pattern in [("name", "名称", r"[^\n:：]+"), ("tax_id", "(?:纳税人识别号|税号)", r"[0-9A-Z]{15,20}")]:
            for value in _label(compact, label + suffix, pattern):
                put(result, party + "." + field, value, source, page)
    for field, labels in [("issuer", "开票人"), ("reviewer", "复核人|复核"), ("payee", "收款人"), ("remarks", "备注")]:
        for value in _label(compact, labels, r"[^\n:：]+"):
            put(result, field, value[:500], source, page)
    if kind == InvoiceKind.RAIL:
        fields: dict[str, Any] = {}
        for key, labels, pattern in [
            ("passenger_name", "旅客姓名|乘车人|旅客", r"[^\n:：]+"),
            ("passenger_id", "证件号码|身份证号", r"[0-9A-Za-z*]{6,25}"),
            ("ticket_number", "电子客票号|车票号|票号", r"[0-9A-Za-z]{6,30}"),
            ("train_number", "车次", r"[GDKTZCSY]?\d{1,5}"),
            ("departure_station", "出发站|始发站|发站", r"[^\n:：]+"),
            ("arrival_station", "到达站|到站", r"[^\n:：]+"),
            ("departure_date", "乘车日期|出发日期", r"\d{4}[年/.-]\d{1,2}[月/.-]\d{1,2}日?"),
            ("departure_time", "开车时间|出发时间", r"\d{1,2}:\d{2}"),
            ("seat_class", "席别", r"[^\n:：]+"),
            ("carriage", "车厢", r"\d{1,3}"),
            ("seat_number", "座位号|座位", r"[0-9A-Z]{1,8}"),
        ]:
            values = _label(compact, labels, pattern)
            if len(set(values)) == 1:
                try:
                    fields[key] = parse_date(values[0]) if key.endswith("date") else values[0]
                except ValueError:
                    issue(result, 'invalid_date', '票面日期无法解析，保留其他可读字段', 'rail.' + key)
            elif len(set(values)) > 1:
                issue(result, "multiple_tickets", "同一文件可能有多张车票，未合并旅程信息", "rail")
        if fields:
            put(result, "rail", fields, source, page)
    elif kind == InvoiceKind.AIR:
        fields = {}
        for key, labels, pattern in [
            ("passenger_name", "旅客姓名|旅客", r"[^\n:：]+"),
            ("passenger_id", "证件号码|身份证号", r"[0-9A-Za-z*]{6,25}"),
            ("ticket_number", "电子客票号码|电子客票号|票号", r"[\d-]{10,20}"),
            ("booking_reference", "订座记录编号|订座编号", r"[A-Z0-9]{5,8}"),
            ("fare", "票价", r"[¥￥]?(" + MONEY + r")"),
            ("airport_tax", "民航发展基金|机场建设费", r"[¥￥]?(" + MONEY + r")"),
            ("fuel_surcharge", "燃油附加费", r"[¥￥]?(" + MONEY + r")"),
            ("insurance", "保险费", r"[¥￥]?(" + MONEY + r")"),
        ]:
            values = _label(compact, labels, pattern)
            if values:
                value = values[0]
                fields[key] = value[-1] if isinstance(value, tuple) else value
        segments = []
        for number in _label(compact, "航班号|航班", r"[A-Z0-9]{2}\d{3,4}"):
            segments.append(FlightSegment(flight_number=number))
        if segments:
            fields["segments"] = segments
        if fields:
            put(result, "air", fields, source, page)
    return result
