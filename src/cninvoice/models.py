from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, computed_field, field_validator


class InvoiceKind(str, Enum):
    VAT = "vat_invoice"
    RAIL = "rail_ticket"
    AIR = "air_itinerary"
    TAXI = "taxi_invoice"
    RIDE = "ride_itinerary"
    OTHER = "other"
    UNKNOWN = "unknown"


class Source(str, Enum):
    XML = "xml"
    PDF_XML = "pdf_xml"
    PDF_TEXT = "pdf_text"
    OFD_XML = "ofd_xml"
    OFD_TEXT = "ofd_text"
    MODEL = "model"
    OCR = "ocr"


def decimal_value(value: Any, *, money: bool = True) -> Decimal | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not an amount")
    text = str(value).strip().replace("−", "-").replace("－", "-")
    text = re.sub(r"[¥￥\s]", "", text).replace("，", ",")
    if "," in text and not re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        raise ValueError("Invalid thousands separators")
    text = text.replace(",", "")
    pattern = r"-?\d{1,16}(?:\.\d{1,2})?" if money else r"-?\d{1,20}(?:\.\d{1,12})?"
    if not re.fullmatch(pattern, text):
        raise ValueError("Invalid decimal")
    try:
        result = Decimal(text)
    except InvalidOperation:
        raise ValueError("Invalid decimal") from None
    return result.quantize(Decimal("0.01")) if money else result


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator("*", mode="before")
    @classmethod
    def empty_as_none(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if value in {"", "null", "None", "未知", "未识别", "无法识别", "看不清"}:
                return None
        return value


class Party(Model):
    name: str | None = None
    tax_id: str | None = None
    address: str | None = None
    phone: str | None = None
    address_phone: str | None = None
    bank_name: str | None = None
    bank_account: str | None = None
    bank_details: str | None = None


class LineItem(Model):
    name: str | None = None
    specification: str | None = None
    unit: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None
    tax_rate: str | None = None
    tax: Decimal | None = None

    @field_validator("quantity", "unit_price", mode="before")
    @classmethod
    def precise_decimal(cls, value: Any) -> Decimal | None:
        return decimal_value(value, money=False)

    @field_validator("amount", "tax", mode="before")
    @classmethod
    def monetary_decimal(cls, value: Any) -> Decimal | None:
        return decimal_value(value)


class RailJourney(Model):
    passenger_name: str | None = None
    passenger_id: str | None = None
    ticket_number: str | None = None
    train_number: str | None = None
    departure_station: str | None = None
    arrival_station: str | None = None
    departure_date: date | None = None
    departure_time: str | None = None
    seat_class: str | None = None
    carriage: str | None = None
    seat_number: str | None = None


class FlightSegment(Model):
    carrier: str | None = None
    flight_number: str | None = None
    departure_airport: str | None = None
    arrival_airport: str | None = None
    departure_date: date | None = None
    departure_time: str | None = None
    cabin_class: str | None = None


class AirJourney(Model):
    passenger_name: str | None = None
    passenger_id: str | None = None
    ticket_number: str | None = None
    booking_reference: str | None = None
    order_number: str | None = None
    fare: Decimal | None = None
    airport_tax: Decimal | None = None
    fuel_surcharge: Decimal | None = None
    other_taxes: Decimal | None = None
    insurance: Decimal | None = None
    segments: list[FlightSegment] = Field(default_factory=list, max_length=64)

    @field_validator("fare", "airport_tax", "fuel_surcharge", "other_taxes", "insurance", mode="before")
    @classmethod
    def monetary_decimal(cls, value: Any) -> Decimal | None:
        return decimal_value(value)


class Evidence(Model):
    source: Source
    page: int | None = None
    text: str | None = None
    bbox: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] | None = None
    score: float | None = Field(default=None, ge=0, le=1)

    @field_validator("bbox")
    @classmethod
    def valid_box(cls, value):
        if value is not None and (value[0] < 0 or value[1] < 0 or value[2] < value[0] or value[3] < value[1]):
            raise ValueError("Invalid evidence box")
        return value


class Issue(Model):
    code: str
    message: str
    field: str | None = None


class Invoice(Model):
    """One invoice/ticket; absent fields are None, amounts are Decimal.

    Extraction is not a verification of authenticity or a calibrated confidence score.
    """

    kind: InvoiceKind = InvoiceKind.UNKNOWN
    invoice_type: str | None = None
    is_electronic: bool | None = None
    invoice_number: str | None = None
    invoice_code: str | None = None
    unique_code: str | None = None
    issue_date: date | None = None
    check_code: str | None = None
    currency: str | None = None
    buyer: Party = Field(default_factory=Party)
    seller: Party = Field(default_factory=Party)
    amount: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal | None = None
    total_in_words: str | None = None
    items: list[LineItem] = Field(default_factory=list, max_length=1000)
    rail: RailJourney | None = None
    air: AirJourney | None = None
    remarks: str | None = None
    issuer: str | None = None
    reviewer: str | None = None
    payee: str | None = None
    sources: dict[str, Evidence] = Field(default_factory=dict)
    issues: list[Issue] = Field(default_factory=list)
    raw_text: str | None = None

    @field_validator("amount", "tax", "total", mode="before")
    @classmethod
    def monetary_decimal(cls, value: Any) -> Decimal | None:
        return decimal_value(value)

    @computed_field
    @property
    def needs_review(self) -> bool:
        return bool(self.issues) or self.kind == InvoiceKind.UNKNOWN or self.total is None or any(
            evidence.source in {Source.MODEL, Source.OCR} for evidence in self.sources.values()
        )

    @computed_field
    @property
    def status(self) -> str:
        meaningful = self.total is not None or self.invoice_number is not None or self.rail is not None or self.air is not None
        return "unrecognized" if not meaningful else "needs_review" if self.needs_review else "parsed"
