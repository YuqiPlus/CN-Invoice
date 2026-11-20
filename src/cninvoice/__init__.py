from .backends import Backend, OpenAICompatibleBackend, QwenBackend
from .errors import InvalidDocumentError, InvoiceError, RecognitionError
from .models import AirJourney, Evidence, FlightSegment, Invoice, InvoiceKind, Issue, LineItem, Party, RailJourney, Source
from .reader import InvoiceReader, read_invoice
from .ocr import RapidOCRBackend
from .cloud import HuaweiOCRBackend
from .vision import FieldVisionBackend
from .config import ReaderConfig
from .review import Reviewer, ConsistencyReviewer
from .profiles import DocumentProfile
from .specialists import SpecialistVisionBackend

__version__ = "0.1.0"
__all__ = [
    "Invoice", "InvoiceKind", "Party", "LineItem", "RailJourney", "AirJourney", "FlightSegment",
    "Evidence", "Issue", "Source", "InvoiceReader", "read_invoice", "Backend", "QwenBackend",
    "OpenAICompatibleBackend", "RapidOCRBackend", "HuaweiOCRBackend", "InvoiceError", "InvalidDocumentError", "RecognitionError",
    "FieldVisionBackend", "ReaderConfig", "Reviewer", "ConsistencyReviewer",
    "DocumentProfile", "SpecialistVisionBackend",
]
