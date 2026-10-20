"""Pure serialization and integrity checks for BankingEnv v3 packages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "bankingenv.document-case/v3"
FAMILIES = ("settlement", "allocations", "reconciliation", "margin", "permissions",
            "lifecycle", "evidence", "temporal")
CRITERIA = ("authoritative_evidence", "calculation", "exception_diagnosis", "routing",
            "approval_control", "correct_mutation", "outcome_verification", "closure")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def validate_case(case):
    if set(case) != {"schema", "classification", "id", "family", "world_id", "unit", "cutoff_minute",
                     "initial_value", "documents", "task", "provenance"}:
        raise ValueError("unknown or missing case fields")
    if case.get("schema") != SCHEMA or case.get("classification") != "CLEANROOM_SYNTHETIC":
        raise ValueError("unsupported case classification or schema")
    if case.get("family") not in FAMILIES:
        raise ValueError("unknown workflow")
    documents = case.get("documents", [])
    ids = [d["id"] for d in documents]
    if len(ids) != len(set(ids)) or not documents:
        raise ValueError("duplicate or missing document ids")
    for doc in documents:
        if set(doc) != {"id", "kind", "title", "status", "effective_minute", "body", "data",
                        "classification", "origin", "sha256"}:
            raise ValueError("unknown or missing document fields")
        if doc["sha256"] != digest({k: v for k, v in doc.items() if k != "sha256"}):
            raise ValueError("document hash mismatch")
    if case.get("provenance", {}).get("restricted_source_inputs") != 0:
        raise ValueError("restricted-source cases cannot enter this synthetic release")
    return case


def load_package(root):
    root = Path(root).resolve()
    manifest = load_json(root / "manifest.json")
    if manifest.get("schema") != "bankingenv.dataset/v3":
        raise ValueError("unsupported dataset manifest")
    entries = manifest["cases"]
    if manifest.get("case_count") != len(entries) or manifest.get("split") not in ("development", "evaluation"):
        raise ValueError("manifest count or split mismatch")
    if not entries or len({e['id'] for e in entries}) != len(entries):
        raise ValueError("empty or duplicated case index")
    actual = {p.relative_to(root).as_posix() for p in (root / "cases").glob("*.json")}
    if actual != {e['path'] for e in entries}:
        raise ValueError("manifest does not enumerate the exact case set")
    cases = []
    for entry in entries:
        original = root / entry["path"]
        path = original.resolve()
        if not path.is_relative_to(root) or path.suffix != ".json" or original.is_symlink():
            raise ValueError("unsafe case path")
        case = validate_case(load_json(path))
        if digest(case) != entry["sha256"] or case["id"] != entry["id"]:
            raise ValueError("case commitment mismatch")
        cases.append(case)
    if digest(entries) != manifest["case_index_sha256"]:
        raise ValueError("index commitment mismatch")
    return manifest, cases
