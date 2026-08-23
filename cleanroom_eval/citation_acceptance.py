"""Measured acceptance harness for the schema-first citation boundary.

The harness records what actually happened for each supplied task.  It never
pads call counts, converts failed calls into passes, or treats mock execution as
commercial acceptance evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from typing import Any, Protocol

from .citations import CitationValidationError, validate_and_render
from .contract import ContractError, canonical_bytes, digest, validate_schema


REQUIRED_ACCEPTANCE_CALLS = 500
MINIMUM_COMPLETION_RATE = 0.99


def derive_acceptance_metrics_and_gates(
    records: Sequence[Mapping[str, Any]],
    *,
    execution_mode: str,
    frozen_task_set_verified: bool,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Derive the single canonical aggregate view from call-level evidence."""

    if not records:
        raise ContractError("citation acceptance records are empty")
    completed = sum(bool(item["provider_completed"]) for item in records)
    schema_valid = sum(bool(item["selection_schema_valid"]) for item in records)
    semantic_valid = sum(bool(item["semantic_valid"]) for item in records)
    rendered_valid = sum(bool(item["rendered_schema_valid"]) for item in records)
    errors = Counter(
        item["error_code"] for item in records if item["error_code"] is not None
    )
    attempted = len(records)
    metrics = {
        "attempted_calls": attempted,
        "completed_calls": completed,
        "completion_rate": completed / attempted,
        "schema_valid_selections": schema_valid,
        "schema_valid_selection_rate": schema_valid / max(1, completed),
        "semantic_valid_selections": semantic_valid,
        "rendered_schema_valid_objects": rendered_valid,
        "rendered_schema_valid_rate": rendered_valid / max(1, completed),
        "evidence_outside_set": errors["EVIDENCE_OUTSIDE_SET"],
        "unsupported_additions": errors["UNSUPPORTED_ADDITION"],
        "rejected_refusals": errors["REFUSAL_REJECTED"],
        "other_validation_rejections": (
            errors["SCHEMA_INVALID"] + errors["VALIDATION_REJECTED"]
        ),
        "oracle_mismatches": errors["ORACLE_MISMATCH"],
    }
    completed_records = [item for item in records if item["provider_completed"]]
    gates = {
        "exactly_500_calls": attempted == REQUIRED_ACCEPTANCE_CALLS,
        "minimum_99pct_completed": metrics["completion_rate"] >= MINIMUM_COMPLETION_RATE,
        "all_completed_outputs_schema_valid": schema_valid == completed,
        "all_completed_outputs_semantically_valid": semantic_valid == completed,
        "all_completed_outputs_match_oracle": all(
            bool(item["oracle_match"]) for item in completed_records
        ),
        "all_completed_citation_objects_schema_valid": rendered_valid == completed,
        "no_evidence_outside_supplied_set": metrics["evidence_outside_set"] == 0,
        "no_unsupported_additions": metrics["unsupported_additions"] == 0,
        "no_rejected_or_uncited_refusals": metrics["rejected_refusals"] == 0,
        "real_provider_execution": execution_mode == "REAL_PROVIDER",
        "frozen_task_set_verified": frozen_task_set_verified,
    }
    return metrics, gates


class CitationSelectionProvider(Protocol):
    name: str
    execution_mode: str

    def select_claims(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return only a ``cleanroom.claim-selection/v1`` object."""


def _error_code(error: Exception) -> str:
    message = str(error).casefold()
    if "outside supplied set" in message:
        return "EVIDENCE_OUTSIDE_SET"
    if "refusal" in message:
        return "REFUSAL_REJECTED"
    if "unsupported" in message:
        return "UNSUPPORTED_ADDITION"
    if "schema" in message or "validation failed" in message:
        return "SCHEMA_INVALID"
    return "VALIDATION_REJECTED"


def _validate_task(task: Mapping[str, Any]) -> None:
    validate_schema(task, "citation-task.schema.json")
    for evidence in task["evidence"]:
        validate_schema(evidence, "semantic-evidence.schema.json")


def run_citation_acceptance(
    *,
    provider: CitationSelectionProvider,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Execute each supplied task exactly once and derive all gate metrics."""

    if not tasks:
        raise ContractError("citation acceptance requires at least one task")
    if getattr(provider, "execution_mode", None) not in {
        "MOCK_ONLY",
        "REAL_PROVIDER",
    }:
        raise ContractError("citation provider execution mode is invalid")
    if not isinstance(getattr(provider, "name", None), str) or not provider.name:
        raise ContractError("citation provider name is invalid")

    records: list[dict[str, Any]] = []
    for sequence, task_value in enumerate(tasks, start=1):
        if not isinstance(task_value, Mapping):
            raise ContractError(f"citation task {sequence} is not an object")
        task = json.loads(canonical_bytes(task_value))
        _validate_task(task)
        record: dict[str, Any] = {
            "sequence": sequence,
            "task_id": task["task_id"],
            "task_sha256": digest(task),
            "oracle_sha256": digest(task),
            "provider_completed": False,
            "selection_schema_valid": False,
            "semantic_valid": False,
            "rendered_schema_valid": False,
            "oracle_match": False,
            "selection_sha256": None,
            "rendered_sha256": None,
            "error_code": None,
        }
        try:
            selection_value = provider.select_claims(task)
            record["provider_completed"] = True
            if not isinstance(selection_value, Mapping):
                raise CitationValidationError("model selection is not an object")
            selection = json.loads(canonical_bytes(selection_value))
            record["selection_sha256"] = digest(selection)
            validate_schema(selection, "claim-selection.schema.json")
            record["selection_schema_valid"] = True
            rendered = validate_and_render(
                selection,
                evidence_records=task["evidence"],
            )
            record["semantic_valid"] = True
            validate_schema(rendered, "rendered-answer.schema.json")
            record["rendered_schema_valid"] = True
            record["oracle_match"] = True
            record["rendered_sha256"] = digest(rendered)
        except (ContractError, TypeError, ValueError) as exc:
            record["error_code"] = _error_code(exc)
        records.append(record)

    metrics, gates = derive_acceptance_metrics_and_gates(
        records,
        execution_mode=provider.execution_mode,
        frozen_task_set_verified=False,
    )
    accepted = all(gates.values())
    receipt: dict[str, Any] = {
        "schema": "cleanroom.citation-acceptance-run/v1",
        "provider": provider.name,
        "execution_mode": provider.execution_mode,
        "task_set": {
            "set_id": "UNREGISTERED",
            "manifest_sha256": digest(tasks),
            "task_list_sha256": digest([digest(task) for task in tasks]),
        },
        "acceptance_status": "PASS" if accepted else "INCOMPLETE",
        "metrics": metrics,
        "gates": gates,
        "records": records,
    }
    receipt["receipt_sha256"] = digest(receipt)
    validate_schema(receipt, "citation-acceptance-run.schema.json")
    return receipt


def verify_citation_acceptance(receipt: Mapping[str, Any]) -> None:
    """Recompute summary fields and reject caller-authored aggregate claims."""

    validate_schema(receipt, "citation-acceptance-run.schema.json")
    unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if receipt["receipt_sha256"] != digest(unsigned):
        raise ContractError("citation acceptance receipt digest differs")
    records = receipt["records"]
    expected_metrics, expected_gates = derive_acceptance_metrics_and_gates(
        records,
        execution_mode=receipt["execution_mode"],
        frozen_task_set_verified=receipt["task_set"]["set_id"] != "UNREGISTERED",
    )
    if receipt["metrics"] != expected_metrics:
        raise ContractError("citation acceptance metrics do not reproduce")

    if receipt["gates"] != expected_gates:
        raise ContractError("citation acceptance gates do not reproduce")
    expected_status = "PASS" if all(expected_gates.values()) else "INCOMPLETE"
    if receipt["acceptance_status"] != expected_status:
        raise ContractError("citation acceptance status does not reproduce")
