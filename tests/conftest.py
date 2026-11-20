from __future__ import annotations

import io
from zipfile import ZipFile

import pytest
from PIL import Image
from pypdf import PdfWriter


@pytest.fixture
def xml_invoice():
    return b'''<?xml version="1.0"?><Invoice>
      <InvoiceTypeName>VAT</InvoiceTypeName><InvoiceNumber>00000001234567890123</InvoiceNumber>
      <IssueDate>2026-11-19</IssueDate><BuyerName>Example Buyer</BuyerName><SellerName>Example Seller</SellerName>
      <TaxExclusiveTotalAmount>100.00</TaxExclusiveTotalAmount><TaxTotalAmount>13.00</TaxTotalAmount>
      <TaxInclusiveTotalAmount>113.00</TaxInclusiveTotalAmount>
      <Item><GoodsName>Test goods</GoodsName><Quantity>2</Quantity><UnitPrice>50</UnitPrice><Amount>100</Amount><TaxAmount>13</TaxAmount><TaxRate>13%</TaxRate></Item>
    </Invoice>'''


@pytest.fixture
def png():
    output = io.BytesIO()
    with Image.new("RGB", (128, 128), "white") as image:
        image.save(output, "PNG")
    return output.getvalue()


def pdf_with_xml(data: bytes | None = None, pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    if data is not None:
        writer.add_attachment("invoice.xml", data)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def ofd_archive(xml: bytes | None = None, png: bytes | None = None, extra: dict[str, bytes] | None = None) -> bytes:
    ns = "http://www.ofdspec.org/2016"
    image_object = '<ofd:ImageObject ResourceID="99" Boundary="0 0 210 297"/>' if png else ''
    files = {
        "OFD.xml": f'<ofd:OFD xmlns:ofd="{ns}"><ofd:DocBody><ofd:DocRoot>Doc_0/Document.xml</ofd:DocRoot></ofd:DocBody></ofd:OFD>'.encode(),
        "Doc_0/Document.xml": f'<ofd:Document xmlns:ofd="{ns}"><ofd:Pages><ofd:Page ID="1" BaseLoc="Pages/Page_0/Content.xml"/></ofd:Pages></ofd:Document>'.encode(),
        "Doc_0/Pages/Page_0/Content.xml": f'<ofd:Page xmlns:ofd="{ns}"><ofd:Content><ofd:Layer>{image_object}</ofd:Layer></ofd:Content></ofd:Page>'.encode(),
    }
    if xml:
        files["Doc_0/invoice.xml"] = xml
    if png:
        files["Doc_0/DocumentRes.xml"] = f'<ofd:Res xmlns:ofd="{ns}"><ofd:MultiMedias><ofd:MultiMedia ID="99"><ofd:MediaFile>Res/page.png</ofd:MediaFile></ofd:MultiMedia></ofd:MultiMedias></ofd:Res>'.encode()
        files["Doc_0/Res/page.png"] = png
    files.update(extra or {})
    output = io.BytesIO()
    with ZipFile(output, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return output.getvalue()
