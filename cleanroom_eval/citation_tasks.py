"""Load and verify the frozen clean-room semantic citation task bundle."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .citations import CitationValidationError, validate_selection
from .contract import (
    ContractError,
    canonical_bytes,
    digest,
    file_sha256,
    load_json,
    validate_schema,
)


ASSET_ROOT = Path(__file__).with_name("assets")
MANIFEST_PATH = ASSET_ROOT / "citation-tasks.manifest.v1.json"
SOURCE_MANIFEST_PATH = ASSET_ROOT / "sealed-set.manifest.v1.json"
TASK_PATH = ASSET_ROOT / "citation_tasks" / "citation-tasks.v1.jsonl"
ORACLE_PATH = ASSET_ROOT / "citation_tasks" / "citation-task-oracle.v1.jsonl"
EXPECTED_CATEGORIES = {
    "supported_claim": 100,
    "unsupported_addition_guard": 80,
    "unsupported_refusal": 80,
    "polarity_adversary": 80,
    "relation_adversary": 80,
    "qualifier_adversary": 80,
}
FORBIDDEN_PROVIDER_KEYS = frozenset(
    {
        "answer",
        "answer_kind",
        "category",
        "expected",
        "expected_answer_kind",
        "expected_disposition",
        "expected_selection",
        "forbidden_additions",
        "gold",
        "oracle",
        "source_challenge_id",
    }
)


def _hidden_provider_fields(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        found.update(str(key) for key in value if key in FORBIDDEN_PROVIDER_KEYS)
        for item in value.values():
            found.update(_hidden_provider_fields(item))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            found.update(_hidden_provider_fields(item))
    return found


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot load JSONL asset {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line:
            raise ContractError(f"blank JSONL line in {path} at {number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"invalid JSONL in {path} at line {number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ContractError(f"JSONL record in {path} is not an object")
        if canonical_bytes(value) + b"\n" != (line + "\n").encode("utf-8"):
            raise ContractError(f"JSONL record in {path} is not canonical")
        records.append(value)
    return records


def load_provider_tasks(path: Path = TASK_PATH) -> list[dict[str, Any]]:
    """Return only provider-visible task projections, never oracle metadata."""

    tasks = _load_jsonl(path)
    for task in tasks:
        hidden = _hidden_provider_fields(task)
        if hidden:
            raise ContractError(
                f"provider task {task.get('task_id')} contains hidden/gold fields: "
                f"{sorted(hidden)}"
            )
        validate_schema(task, "citation-task.schema.json")
        for evidence in task["evidence"]:
            validate_schema(evidence, "semantic-evidence.schema.json")
    return tasks


def load_task_oracle(path: Path = ORACLE_PATH) -> list[dict[str, Any]]:
    """Load evaluator-only expectations.

    Callers must never merge these records into the object passed to a
    ``CitationSelectionProvider``.
    """

    records = _load_jsonl(path)
    for record in records:
        validate_schema(record, "citation-task-oracle.schema.json")
        validate_schema(
            record["expected_selection"], "claim-selection.schema.json"
        )
    return records


def _verify_assets(
    manifest: Mapping[str, Any],
    *,
    asset_root: Path,
) -> None:
    expected_paths = {
        "citation_tasks/citation-tasks.v1.jsonl",
        "citation_tasks/citation-task-oracle.v1.jsonl",
    }
    assets = {item["path"]: item for item in manifest["assets"]}
    if set(assets) != expected_paths:
        raise ContractError("citation task manifest asset set differs")
    for relative, item in assets.items():
        path = asset_root / relative
        if not path.is_file():
            raise ContractError(f"citation task asset missing: {relative}")
        if path.stat().st_size != item["bytes"]:
            raise ContractError(f"citation task asset byte count differs: {relative}")
        if file_sha256(path) != item["sha256"]:
            raise ContractError(f"citation task asset digest differs: {relative}")


def _assert_rejected_additions(
    task: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> None:
    for sequence, proposition in enumerate(oracle["forbidden_additions"], start=1):
        selection = deepcopy(oracle["expected_selection"])
        selection["claims"].append(
            {
                "claim_id": f"forbidden-addition-{sequence:02d}",
                **deepcopy(proposition),
                "evidence_ids": [task["evidence"][-1]["evidence_id"]],
            }
        )
        try:
            validate_selection(selection, evidence_records=task["evidence"])
        except CitationValidationError:
            continue
        raise ContractError(
            f"forbidden addition is not rejected for {task['task_id']}"
        )


def verify_citation_task_set(
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """Recompute every count, hash and semantic oracle invariant."""

    manifest = load_json(manifest_path)
    validate_schema(manifest, "citation-task-set.schema.json")
    asset_root = manifest_path.parent
    source = manifest["source_sealed_set"]
    expected_source_path = "cleanroom_eval/assets/sealed-set.manifest.v1.json"
    if source["manifest_path"] != expected_source_path:
        raise ContractError("citation tasks reference an unexpected sealed set")
    source_manifest_path = asset_root / "sealed-set.manifest.v1.json"
    if file_sha256(source_manifest_path) != source["manifest_sha256"]:
        raise ContractError("source sealed-set manifest digest differs")
    source_manifest = load_json(source_manifest_path)
    if source_manifest["set_id"] != source["set_id"]:
        raise ContractError("source sealed-set identifier differs")
    _verify_assets(manifest, asset_root=asset_root)

    tasks = load_provider_tasks(
        asset_root / "citation_tasks" / "citation-tasks.v1.jsonl"
    )
    oracles = load_task_oracle(
        asset_root / "citation_tasks" / "citation-task-oracle.v1.jsonl"
    )
    if len(tasks) != 500 or len(oracles) != 500:
        raise ContractError("citation task set does not contain exactly 500 records")
    task_ids = [item["task_id"] for item in tasks]
    oracle_ids = [item["task_id"] for item in oracles]
    if len(set(task_ids)) != 500 or task_ids != oracle_ids:
        raise ContractError("citation task/oracle identifiers are not one-to-one")

    ledger: list[dict[str, Any]] = []
    for sequence, (task, oracle) in enumerate(zip(tasks, oracles), start=1):
        validate_selection(
            oracle["expected_selection"],
            evidence_records=task["evidence"],
        )
        _assert_rejected_additions(task, oracle)
        ledger.append(
            {
                "sequence": sequence,
                "task_id": task["task_id"],
                "task_sha256": digest(task),
                "oracle_sha256": digest(oracle),
                "episode_id": oracle["episode_id"],
                "scenario_family": oracle["scenario_family"],
                "category": oracle["category"],
                "expected_answer_kind": oracle["expected_selection"][
                    "answer_kind"
                ],
            }
        )
    if ledger != manifest["task_ledger"]:
        raise ContractError("citation task manifest ledger does not reproduce")
    task_list = "".join(
        f"{item['task_sha256']}  {item['task_id']}\n" for item in ledger
    )
    if hashlib.sha256(task_list.encode("utf-8")).hexdigest() != manifest[
        "task_list_sha256"
    ]:
        raise ContractError("citation task-list digest differs")

    categories = Counter(item["category"] for item in oracles)
    families = Counter(item["scenario_family"] for item in oracles)
    kinds = Counter(
        item["expected_selection"]["answer_kind"] for item in oracles
    )
    expected_coverage = {
        "categories": dict(sorted(categories.items())),
        "scenario_families": dict(sorted(families.items())),
        "expected_answer_kinds": dict(sorted(kinds.items())),
    }
    if manifest["coverage"] != expected_coverage:
        raise ContractError("citation task coverage distribution differs")
    if dict(categories) != EXPECTED_CATEGORIES:
        raise ContractError("citation task category coverage is incomplete")
    if set(families.values()) - {62, 63}:
        raise ContractError("citation task scenario-family balance differs")
    if len({item["episode_id"] for item in oracles}) != 40:
        raise ContractError("citation tasks do not cover all 40 sealed episodes")

    return {
        "set_id": manifest["set_id"],
        "task_count": len(tasks),
        "episode_count": len({item["episode_id"] for item in oracles}),
        "coverage": expected_coverage,
        "task_list_sha256": manifest["task_list_sha256"],
    }
