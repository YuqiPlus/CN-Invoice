from __future__ import annotations

import io
import posixpath
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
from zipfile import BadZipFile, ZipFile

from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import InvalidDocumentError
from .models import Invoice, Source
from .parsers import OFD_NS, local_name, namespace, parse_xml, safe_xml

PDFIUM_LOCK = threading.RLock()
MAX_PIXELS = 20_000_000


@dataclass
class Page:
    number: int
    text: str
    source: Source
    image: bytes | None = None


@dataclass
class Document:
    kind: str
    data: bytes
    pages: list[Page] = field(default_factory=list)
    structured: list[Invoice] = field(default_factory=list)


def image_from_bytes(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > MAX_PIXELS or getattr(source, "n_frames", 1) != 1:
                    raise InvalidDocumentError("Only a single image frame within the pixel limit is supported")
                return ImageOps.exif_transpose(source).convert("RGB")
    except InvalidDocumentError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise InvalidDocumentError("Invalid or oversized image") from None


def load(source: str | Path | bytes | BinaryIO, *, max_bytes: int, max_pages: int) -> Document:
    try:
        if isinstance(source, (str, Path)):
            with Path(source).open("rb") as handle:
                data = handle.read(max_bytes + 1)
        elif isinstance(source, bytes):
            data = source
        else:
            position = source.tell() if source.seekable() else None
            try:
                data = source.read(max_bytes + 1)
            finally:
                if position is not None:
                    source.seek(position)
    except (OSError, AttributeError, TypeError):
        raise InvalidDocumentError("Cannot read input file") from None
    if not isinstance(data, bytes) or not data or len(data) > max_bytes:
        raise InvalidDocumentError("Input must be a nonempty binary file within the size limit")
    if data.lstrip().startswith(b"%PDF-"):
        return _pdf(data, max_pages)
    if data.startswith(b"PK"):
        return _ofd(data, max_pages)
    head = data[:256].replace(b"\x00", b"").lstrip(b"\xef\xbb\xbf\xff\xfe \t\r\n")
    if head.startswith(b"<"):
        return Document("xml", data, structured=[parse_xml(data)])
    # Magic-based detection: filenames and user-supplied extensions are not trusted.
    image = image_from_bytes(data)
    image.close()
    return Document("image", data, pages=[Page(1, "", Source.MODEL, data)])


def _pdf(data: bytes, max_pages: int) -> Document:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise InvalidDocumentError("Encrypted PDFs are not supported")
        if not 1 <= len(reader.pages) <= max_pages:
            raise InvalidDocumentError("PDF page count exceeds the configured limit")
        result = Document("pdf", data)
        for number, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False) or ""
            except Exception:
                # A valid but unusual text layer can still be rendered for vision.
                text = ""
            if len(text) > 200_000:
                raise InvalidDocumentError("PDF text exceeds the extraction limit")
            result.pages.append(Page(number, text, Source.PDF_TEXT))
        if len(reader.attachments) > 32:
            raise InvalidDocumentError("PDF has too many embedded files")
        for name, attachments in reader.attachments.items():
            if name.lower().endswith(".xml"):
                for attachment in attachments:
                    result.structured.append(parse_xml(attachment, Source.PDF_XML))
        return result
    except InvalidDocumentError:
        raise
    except Exception:
        raise InvalidDocumentError("Cannot parse PDF") from None


def _archive_path(base: str, reference: str) -> str:
    if not reference or "\\" in reference or ":" in reference or ".." in reference.split("/"):
        raise InvalidDocumentError("Unsafe OFD resource reference")
    return posixpath.normpath(reference.lstrip("/") if reference.startswith("/") else posixpath.join(base, reference))


def _ofd(data: bytes, max_pages: int) -> Document:
    try:
        with ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 512 or sum(e.file_size for e in entries) > 64 * 1024**2:
                raise InvalidDocumentError("OFD archive exceeds the extraction limit")
            files: dict[str, bytes] = {}
            for entry in entries:
                name = entry.filename
                if name in files or name.startswith("/") or "\\" in name or ":" in name or ".." in name.split("/"):
                    raise InvalidDocumentError("Unsafe OFD archive entry")
                limit = 2 * 1024**2 if name.lower().endswith(".xml") else 16 * 1024**2
                if entry.file_size > limit or entry.flag_bits & 1:
                    raise InvalidDocumentError("Oversized or encrypted OFD entry")
                with archive.open(entry) as handle:
                    content = handle.read(limit + 1)
                if len(content) > limit:
                    raise InvalidDocumentError("OFD entry exceeds the extraction limit")
                files[name] = content
        if "OFD.xml" not in files:
            raise InvalidDocumentError("ZIP is not an OFD document")
        roots = {name: safe_xml(content) for name, content in files.items() if name.lower().endswith(".xml")}
        if roots["OFD.xml"].tag != "{" + OFD_NS + "}OFD":
            raise InvalidDocumentError("Invalid OFD manifest")
        for root in roots.values():
            for element in root.iter():
                if namespace(element.tag) != OFD_NS:
                    continue
                for value in [element.attrib.get("BaseLoc"), element.text if local_name(element.tag) in {"DocRoot", "FileLoc", "MediaFile"} else None]:
                    if value:
                        _archive_path("", value.strip())
        result = Document("ofd", data)
        for name, root in roots.items():
            parsed = parse_xml(files[name], Source.OFD_XML)
            if parsed.sources:
                result.structured.append(parsed)
        pages: list[tuple[str, str]] = []
        for element in roots["OFD.xml"].iter("{" + OFD_NS + "}DocRoot"):
            document_name = _archive_path("", (element.text or "").strip())
            if document_name not in roots:
                raise InvalidDocumentError("OFD document reference is missing")
            document = roots[document_name]
            base = posixpath.dirname(document_name)
            for node in document.iter("{" + OFD_NS + "}Page"):
                if "BaseLoc" in node.attrib:
                    pages.append((_archive_path(base, node.attrib["BaseLoc"]), document_name))
        if not 1 <= len(pages) <= max_pages:
            raise InvalidDocumentError("OFD page count exceeds the configured limit")
        for number, (name, document_name) in enumerate(pages, 1):
            if name not in roots:
                raise InvalidDocumentError("OFD page reference is missing")
            page_root = roots[name]
            content_roots = [page_root]
            template_ids = {node.attrib.get("TemplateID") for node in page_root.iter("{" + OFD_NS + "}Template")}
            for template in roots[document_name].iter("{" + OFD_NS + "}TemplatePage"):
                if template.attrib.get("ID") not in template_ids:
                    continue
                template_name = _archive_path(posixpath.dirname(document_name), template.attrib.get("BaseLoc", ""))
                if template_name in roots:
                    content_roots.append(roots[template_name])
            rows: list[tuple[float, float, str]] = []
            for root in content_roots:
                for element in root.iter("{" + OFD_NS + "}TextObject"):
                    if "CTM" in element.attrib:
                        continue
                    boundary = element.attrib.get("Boundary", "").split()
                    if len(boundary) != 4:
                        continue
                    for code in element.iter("{" + OFD_NS + "}TextCode"):
                        try:
                            rows.append((float(boundary[1]) + float(code.attrib.get("Y", 0)), float(boundary[0]) + float(code.attrib.get("X", 0)), code.text or ""))
                        except ValueError:
                            continue
            lines: list[list[tuple[float, float, str]]] = []
            for box in sorted(rows):
                if lines and abs(lines[-1][0][0] - box[0]) < 1.2:
                    lines[-1].append(box)
                else:
                    lines.append([box])
            text = "\n".join("".join(box[2] for box in sorted(line, key=lambda item: item[1])) for line in lines)
            image = None
            images = list(page_root.iter("{" + OFD_NS + "}ImageObject"))
            has_vectors = any(local_name(node.tag) in {"TextObject", "PathObject", "CompositeObject"} for node in page_root.iter())
            # A scan-only OFD page can provide its full raster directly. Complex vector
            # pages require a separate OFD renderer; resource logos are not whole pages.
            if len(images) == 1 and not has_vectors:
                resource_id = images[0].attrib.get("ResourceID")
                for resource_name, resource_root in roots.items():
                    for media in resource_root.iter("{" + OFD_NS + "}MultiMedia"):
                        if media.attrib.get("ID") == resource_id:
                            location = media.find("{" + OFD_NS + "}MediaFile")
                            if location is not None:
                                resource_base = _archive_path(posixpath.dirname(resource_name), resource_root.attrib.get("BaseLoc", "."))
                                image = files.get(_archive_path(resource_base, (location.text or "").strip()))
            result.pages.append(Page(number, text, Source.OFD_TEXT, image))
        return result
    except InvalidDocumentError:
        raise
    except (BadZipFile, KeyError, OSError, ValueError):
        raise InvalidDocumentError("Cannot parse OFD") from None


def render(document: Document, max_pages: int) -> list[Image.Image]:
    if document.kind == "pdf":
        import pypdfium2 as pdfium

        images: list[Image.Image] = []
        # PDFium calls must be serialized even when different Reader instances are used.
        with PDFIUM_LOCK:
            try:
                with pdfium.PdfDocument(document.data) as pdf:
                    for index in range(min(len(pdf), max_pages)):
                        page = pdf[index]
                        try:
                            width, height = page.get_size()
                            scale = min(2.0, (MAX_PIXELS / max(1, width * height)) ** .5 * .99)
                            bitmap = page.render(scale=scale)
                            try:
                                images.append(bitmap.to_pil().convert("RGB"))
                            finally:
                                bitmap.close()
                        finally:
                            page.close()
            except Exception:
                for image in images:
                    image.close()
                raise InvalidDocumentError("Cannot render PDF pages") from None
        return images
    images = []
    try:
        for page in document.pages:
            if page.image is None:
                raise InvalidDocumentError("A renderer is required for complex vector OFD pages")
            images.append(image_from_bytes(page.image))
        return images
    except Exception:
        for image in images:
            image.close()
        raise
