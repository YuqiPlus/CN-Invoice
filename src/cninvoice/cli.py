from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .backends import OpenAICompatibleBackend, QwenBackend
from .errors import InvoiceError
from .reader import InvoiceReader
from .ocr import RapidOCRBackend
from .vision import FieldVisionBackend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read a Chinese invoice or reimbursement ticket as JSON")
    parser.add_argument("file")
    parser.add_argument("--ocr", choices=["auto", "always", "never"], default="auto")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--vision-model", help="Local dedicated detector directory (ONNX)")
    backend.add_argument("--specialist-models", help="Local specialist registry JSON; routes to one document-family model")
    backend.add_argument("--model-path", help="Optional local Qwen3-VL model directory")
    backend.add_argument("--base-url", help="Explicit OpenAI-compatible vision endpoint, including /v1")
    parser.add_argument("--model", help="Model name for the HTTP endpoint")
    parser.add_argument("--ocr-models", help="Local small OCR model directory; required with --vision-model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--document-profile", choices=['vat_invoice','rail_einvoice','ride_itinerary','air_itinerary'],
                        help="Optional explicit family with --specialist-models; conflicting titles still require review")
    parser.add_argument("--include-raw-text", action="store_true")
    args = parser.parse_args(argv)
    try:
        recognizer = None
        if args.document_profile and not args.specialist_models:
            parser.error('--document-profile requires --specialist-models')
        if args.specialist_models:
            from .specialists import SpecialistVisionBackend
            if not args.ocr_models:parser.error('--ocr-models is required with --specialist-models')
            registry=Path(args.specialist_models)
            try:
                models=json.loads(registry.read_text())['models']
                if not isinstance(models,dict):raise ValueError('Invalid specialist registry')
                directories={name:registry.parent/path for name,path in models.items()}
            except (OSError,KeyError,TypeError,ValueError):
                raise ValueError('Invalid local specialist registry') from None
            recognizer=SpecialistVisionBackend(directories,args.ocr_models,document_profile=args.document_profile)
        elif args.vision_model:
            if not args.ocr_models:parser.error('--ocr-models is required with --vision-model')
            recognizer = FieldVisionBackend(args.vision_model,args.ocr_models)
        elif args.model_path:
            recognizer = QwenBackend(args.model_path, device=args.device)
        elif args.base_url:
            if not args.model:
                parser.error("--model is required with --base-url")
            recognizer = OpenAICompatibleBackend(base_url=args.base_url, model=args.model, api_key=os.environ.get("CNINVOICE_API_KEY"))
        elif args.ocr_models:
            recognizer = RapidOCRBackend(args.ocr_models)
        result = InvoiceReader(recognizer, ocr=args.ocr, include_raw_text=args.include_raw_text).read(args.file)
        print(result.model_dump_json(indent=2))
        return 0 if result.status != "unrecognized" else 2
    except (InvoiceError, ValueError):
        print("Unable to read invoice; check input and backend configuration.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
