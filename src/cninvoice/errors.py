class InvoiceError(Exception):
    """Base error. Messages deliberately exclude invoice contents and credentials."""


class InvalidDocumentError(InvoiceError):
    """Invalid, unsupported, encrypted or oversized input."""


class RecognitionError(InvoiceError):
    """A configured recognition backend could not produce usable structured data."""
