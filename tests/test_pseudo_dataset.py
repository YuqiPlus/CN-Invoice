import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "merge_pseudo_dataset", Path(__file__).parents[1] / "training/merge_pseudo_dataset.py"
)
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


def inputs(tmp_path, monkeypatch):
    def source(name):
        path = tmp_path / name
        path.write_bytes(name.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"id": name, "file": str(path), "sha256": digest,
                "pixel_sha256": digest, "width": 100, "height": 100}

    base = {"records": [{**source("manual"), "group": "old", "split": "val",
                         "category": "vat", "origin": "publisher_manual_boxes"}],
            "fields": ["vat.total"], "notes": []}
    cases = []
    for index in range(4):
        row = source(f"view-{index}")
        cases.append({"id": row["id"], "source": row, "accepted": True,
                      "values": {"invoice_number": "12345678"},
                      "annotations": [{"field": "vat.total", "box": [1, 1, 20, 20],
                                       "ocr_score": .9}], "teacher_payload": {}})
    heldout = source("heldout")
    cases.append({**cases[0], "id": "heldout", "source": heldout})
    paths = {}
    for name, value in {"base": base, "labels": {"status": "complete", "cases": cases},
                        "identities": {"manual": "12345678"},
                        "heldout": {"samples": [heldout]}}.items():
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_text(json.dumps(value))
    output = tmp_path / ".private" / "merged"
    monkeypatch.setattr(sys, "argv", ["merge", "--base", str(paths["base"]),
        "--labels", str(paths["labels"]), "--base-identities", str(paths["identities"]),
        "--heldout", str(paths["heldout"]), "--output", str(output)])
    return output, cases


def test_invoice_views_follow_existing_split_and_heldout_is_excluded(tmp_path, monkeypatch):
    output, _ = inputs(tmp_path, monkeypatch)
    merge.main()
    dataset = json.loads((output / "dataset.json").read_text())
    assert len(dataset["records"]) == 4  # One manual source and at most three new views.
    assert {row["split"] for row in dataset["records"]} == {"val"}
    assert len({row["group"] for row in dataset["records"]}) == 1
    assert dataset["pseudo_summary"]["excluded"] == {
        "heldout_image": 1, "excess_views_same_invoice": 1}


def test_changed_teacher_image_is_rejected(tmp_path, monkeypatch):
    _, cases = inputs(tmp_path, monkeypatch)
    Path(cases[0]["source"]["file"]).write_bytes(b"changed since annotation")
    with pytest.raises(ValueError, match="Teacher source image changed"):
        merge.main()
