from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from .models import AirJourney, Evidence, Invoice, InvoiceKind, Issue, Model, RailJourney, Source


def issue(invoice: Invoice, code: str, message: str, field: str | None = None) -> None:
    item = Issue(code=code, message=message, field=field)
    if item not in invoice.issues:
        invoice.issues.append(item)


def put(invoice: Invoice, path: str, value: Any, source: Source, page: int | None = None) -> None:
    if value is None or value == "" or value == [] or value == InvoiceKind.UNKNOWN:
        return
    parts = path.split(".")
    target: Model = invoice
    for part in parts[:-1]:
        target = getattr(target, part)
        if target is None:
            return
    key = parts[-1]
    old = getattr(target, key)
    try:
        # Validate using a copy, so a bad value cannot mutate an existing result.
        candidate = target.model_copy(deep=True)
        setattr(candidate, key, value)
        value = getattr(candidate, key)
    except (ValidationError, ValueError, TypeError):
        issue(invoice, "invalid_field", "字段格式无法确认，已保留为空或原值", path)
        return
    if old is not None and old != [] and old != InvoiceKind.UNKNOWN:
        if old != value:
            issue(invoice, "conflicting_field", "多个来源的字段值不一致，请核对原件", path)
        return
    setattr(target, key, value)
    invoice.sources[path] = Evidence(source=source, page=page)


def merge(invoice: Invoice, other: Invoice) -> None:
    for path, evidence in other.sources.items():
        value: Any = other
        for part in path.split("."):
            value = getattr(value, part)
        put(invoice, path, value, evidence.source, evidence.page)
    for warning in other.issues:
        if warning not in invoice.issues:
            invoice.issues.append(warning)


def apply_payload(invoice: Invoice, payload: dict[str, Any], source: Source, page: int | None = None) -> None:
    payload = dict(payload)
    # Some vision models nest common invoice fields under the transport object.
    # Preserve their explicit values, validate normally, and expose the correction.
    nested_common: list[tuple[str, Any]] = []
    for name, schema in [("rail", RailJourney), ("air", AirJourney)]:
        if not isinstance(payload.get(name), dict):
            continue
        value = dict(payload[name])
        for field in {"invoice_number", "invoice_code", "issue_date", "total", "currency"}:
            if field in value:
                nested_common.append((field, value.pop(field)))
                issue(invoice, "model_schema_normalized", "模型字段位置已规范化，请对照票面复核", name + "." + field)
        for field in set(value) - set(schema.model_fields):
            value.pop(field)
            issue(invoice, "unknown_model_field", "模型返回了未支持的旅程字段，未自动猜测其含义", name)
        payload[name] = value or None
    allowed = set(Invoice.model_fields) - {"sources", "issues", "raw_text"}
    for key, value in payload.items():
        if key not in allowed:
            continue
        if key in {"buyer", "seller"} and isinstance(value, dict):
            for field, item in value.items():
                if field in type(getattr(invoice, key)).model_fields:
                    put(invoice, key + "." + field, item, source, page)
        else:
            put(invoice, key, value, source, page)
    for field, value in nested_common:
        put(invoice, field, value, source, page)


def validate_totals(invoice: Invoice) -> None:
    if invoice.kind == InvoiceKind.VAT:
        for field, label in (("amount", "不含税合计金额"), ("tax", "合计税额"), ("total", "价税合计")):
            if getattr(invoice, field) is None:
                issue(invoice, "amount_incomplete", f"未读取到{label}，金额识别不完整；不通过计算补全，请核对票面", field)
    if all(v is not None for v in (invoice.amount, invoice.tax, invoice.total)):
        # All three monetary fields are validated to cents before this check.
        if invoice.amount + invoice.tax != invoice.total:
            issue(invoice, "amount_mismatch", "不含税金额与税额之和不等于价税合计", "total")
    if invoice.items and invoice.amount is not None and all(item.amount is not None for item in invoice.items):
        if abs(sum((item.amount for item in invoice.items), Decimal(0)) - invoice.amount) > Decimal("0.01"):
            issue(invoice, "item_amount_mismatch", "明细金额与不含税合计不一致", "items")
