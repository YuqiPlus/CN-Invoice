from decimal import Decimal

from cninvoice import InvoiceKind, read_invoice


def instance(family, facts, *, units=True):
    return (f'<x:xbrl xmlns:x="http://www.xbrl.org/2003/instance" xmlns:i="http://xbrl.mof.gov.cn/taxonomy/2023-12-31/{family}">'
            + ('<x:unit id="u"><x:measure>iso4217:CNY</x:measure></x:unit>' if units else '')
            + facts + '</x:xbrl>').encode()


def test_rail_issuer_explicit_fare_and_ticket_number_are_distinct():
    result = read_invoice(instance('rai', '''
      <i:ElectronicInvoiceRailwayETicketNumber>00000000000000000001</i:ElectronicInvoiceRailwayETicketNumber>
      <i:ETicketNumber>OLDTICKET001</i:ETicketNumber><i:DateOfIssue>2026-11-19</i:DateOfIssue>
      <i:Name>测试旅客</i:Name><i:TrainNumber>G123</i:TrainNumber><i:TravelDate>2026-11-20</i:TravelDate>
      <i:Fare unitRef="u">109.00</i:Fare><i:TotalAmountExcludingTax unitRef="u">100</i:TotalAmountExcludingTax><i:TaxAmount unitRef="u">9</i:TaxAmount>'''))
    assert result.kind == InvoiceKind.RAIL and result.total == Decimal('109')
    assert result.invoice_number == '00000000000000000001'
    assert result.rail.ticket_number == 'OLDTICKET001' and result.rail.train_number == 'G123'
    assert result.rail.departure_date.isoformat() == '2026-11-20'


def test_receiver_does_not_invent_missing_total():
    result = read_invoice(instance('rai', '''
      <i:ElectronicInvoiceRailwayETicketNumber>00000000000000000001</i:ElectronicInvoiceRailwayETicketNumber>
      <i:TotalAmountExcludingTax>100</i:TotalAmountExcludingTax><i:TaxAmount>9</i:TaxAmount>'''))
    assert result.amount == Decimal('100') and result.tax == Decimal('9')
    assert result.total is None and result.needs_review


def test_air_fare_is_not_invoice_total_and_placeholder_keeps_journey():
    result = read_invoice(instance('atr', '''
      <i:ElectronicInvoiceAirTransportReceiptNumber>00000000000000000002</i:ElectronicInvoiceAirTransportReceiptNumber>
      <i:PassengerName>测试旅客</i:PassengerName><i:ETicketNumber>1230000000000</i:ETicketNumber>
      <i:Fare>800</i:Fare><i:CivilAviationDevelopmentFund>50</i:CivilAviationDevelopmentFund>
      <i:FuelSurcharge>40</i:FuelSurcharge><i:TotalAmount>890</i:TotalAmount><i:Insurance>XXX</i:Insurance>
      <i:DetailInformationOfAirTicketTuple><i:Flight>CA1234</i:Flight><i:DepartureStation>北京</i:DepartureStation><i:DestinationStation>上海</i:DestinationStation><i:CarrierDate>2026-11-20</i:CarrierDate></i:DetailInformationOfAirTicketTuple>
      <i:DetailInformationOfAirTicketTuple><i:Flight>CA4321</i:Flight><i:DepartureStation>上海</i:DepartureStation><i:DestinationStation>北京</i:DestinationStation></i:DetailInformationOfAirTicketTuple>'''))
    assert result.kind == InvoiceKind.AIR and result.total == Decimal('890')
    assert result.air.fare == Decimal('800') and result.air.insurance is None
    assert len(result.air.segments) == 2 and result.air.segments[1].flight_number == 'CA4321'
    assert any(issue.field == 'air.insurance' for issue in result.issues)


def test_air_receiver_fare_is_not_promoted_to_total():
    result = read_invoice(instance('atr', '<i:Fare>800</i:Fare><i:ElectronicInvoiceAirTransportReceiptNumber>00000000000000000002</i:ElectronicInvoiceAirTransportReceiptNumber>'))
    assert result.air.fare == Decimal('800') and result.total is None


def test_vat_unique_code_is_not_invoice_number():
    result = read_invoice(instance('inv', '<i:UniqueCodeOfInvoice>00000000000000000003</i:UniqueCodeOfInvoice><i:TaxIncludedAmountInFigures>113</i:TaxIncludedAmountInFigures>'))
    assert result.unique_code == '00000000000000000003' and result.invoice_number is None
    assert result.total == Decimal('113') and result.kind == InvoiceKind.VAT


def test_detail_tax_does_not_conflict_with_total_tax():
    result = read_invoice(instance('inv', '''<i:TotalTaxAmount>13</i:TotalTaxAmount><i:TotalAmountExcludingTax>100</i:TotalAmountExcludingTax><i:TaxIncludedAmountInFigures>113</i:TaxIncludedAmountInFigures>
    <i:InformationOfTaxableGoodsOrServicesDetailItemsTuple><i:NameOfTaxableGoodsOrServices>测试商品</i:NameOfTaxableGoodsOrServices><i:AmountExcludingTax>100</i:AmountExcludingTax><i:TaxAmount>13</i:TaxAmount></i:InformationOfTaxableGoodsOrServicesDetailItemsTuple>'''))
    assert result.tax == Decimal('13') and result.items[0].amount == Decimal('100')
    assert not result.issues


def test_digital_invoice_xbrl_root_and_issue_timestamp():
    result = read_invoice(instance('einv', '<i:InvoiceNumber>00000000000000000004</i:InvoiceNumber><i:RequestTime>2026-11-19T10:15:30+08:00</i:RequestTime><i:TotalTax-includedAmount>113</i:TotalTax-includedAmount>'))
    assert result.kind == InvoiceKind.VAT and result.issue_date.isoformat() == '2026-11-19'
    assert result.total == Decimal('113')


def test_unknown_currency_preserves_other_fields():
    result = read_invoice(instance('rai', '<i:TrainNumber>G123</i:TrainNumber><i:Fare unitRef="missing">109</i:Fare>'))
    assert result.total is None and result.rail.train_number == 'G123'
    assert any(issue.code == 'unknown_currency' for issue in result.issues)


def test_issue_time_with_space_is_normalized_to_date():
    result = read_invoice(instance('einv', '<i:InvoiceNumber>00000000000000000004</i:InvoiceNumber><i:RequestTime>2026-11-19 20:05:10</i:RequestTime><i:TotalTax-includedAmount>113</i:TotalTax-includedAmount>'))
    assert result.issue_date.isoformat() == '2026-11-19'
