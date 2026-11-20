from cninvoice import InvoiceReader

# Configure a vision backend on InvoiceReader when processing scans or images.
reader = InvoiceReader()
invoice = reader.read("invoice.pdf")
print(invoice.model_dump_json(indent=2))
