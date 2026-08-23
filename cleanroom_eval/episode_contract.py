"""Declarative, agent-facing episode contracts (#117).

A sealed episode (``schemas/episode.schema.json``) is the evaluator's record:
it carries the oracle (``final_state``, every scripted mutation, expected
receipts, traps). Nothing in it is meant to be handed to an agent as-is.

This module derives a *task card* per sealed episode — the clean-room
equivalent of Harvey LAB's ``task.json``: what the agent is asked to do, on
which surfaces, with which request shape, what it must deliver, which
deterministic checks grade it, and which disclosure barrier applies. The card
is derived, never hand-written, and ``verify_contract`` proves both that it
agrees with its episode and that it leaks no oracle field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contract import CLASSIFICATION, ContractError, digest, load_json, validate_schema

CONTRACT_SCHEMA = "cleanroom.episode-contract/v1"
TASK_SUFFIX = ".task.v1.json"

# Fields of the sealed episode that must never reach an agent.
ORACLE_FIELDS = (
    "final_state",
    "mutations",
    "expected_receipt",
    "reward_traps",
    "adversarial_mutations",
    "evidence_challenges",
    "duplicate_of",
    "ledger_entries",
    "template_event_id",
)

# Deterministic graders in cleanroom_eval.contract.evaluate_episode, plus the
# free-mode completion grader in cleanroom_corpus.eval_adapters.
CHECKS: tuple[dict[str, str], ...] = (
    {"id": "chronology", "grader": "contract.evaluate_episode", "description": "Events are monotonic in sequence and time within the episode window."},
    {"id": "referential_integrity", "grader": "contract.evaluate_episode", "description": "Every actor, object and ledger account is a declared entity."},
    {"id": "authorization", "grader": "contract.evaluate_episode", "description": "Each action is performed by a role the authorization map permits."},
    {"id": "version_monotonicity", "grader": "contract.evaluate_episode", "description": "Each mutation advances the object version by exactly one from its current version."},
    {"id": "idempotency", "grader": "contract.evaluate_episode", "description": "A resubmitted request ID yields the primary receipt and no new mutation."},
    {"id": "ledger_conservation", "grader": "contract.evaluate_episode", "description": "Ledger entries balance to zero per currency on every event."},
    {"id": "evidence_presence", "grader": "contract.evaluate_episode", "description": "Every accepted action cites at least one public evidence reference."},
    {"id": "final_state", "grader": "contract.evaluate_episode", "description": "Every object reaches its sealed terminal state and version."},
    {"id": "evidence_sufficiency", "grader": "contract.evaluate_episode", "description": "Supported claims carry their required evidence; unsupported economic claims are refused."},
    {"id": "adversarial_mutations", "grader": "contract.evaluate_episode", "description": "Each committed adversarial mutation of the episode is rejected with the expected error."},
    {"id": "free_mode_completion", "grader": "eval_adapters.EvaluationSession.is_complete", "description": "In free-action mode the agent reaches the sealed terminal state using only the observable boundary."},
)

WORK_TYPE_BY_FAMILY = {
    "lifecycle": "resolve",
    "booking_allocations": "repair",
    "reconciliation": "reconcile",
    "collateral_margin": "resolve",
    "settlement_exceptions": "repair",
    "permissions_release": "approve",
    "temporal_causality": "investigate",
    "evidence_sufficiency": "adjudicate",
}

BARRIER_POLICY = {
    "policy_id": "mnpi_barrier_default_v1",
    "deliverable": ["existence", "coarse_instrument_class", "operational_status"],
    "behind_wall": [
        "sizes_notionals_counts_positions",
        "desk_level_attribution",
        "sub_annual_dates",
    ],
    "on_violation": "FAIL_EVEN_IF_FACTUALLY_CORRECT",
    "source": "eval/harness.py policy_default",
}


def _family_of(episode: Mapping[str, Any]) -> str:
    family = episode["template_family"].removeprefix("eval_")
    for suffix in ("_r2",):
        family = family.removesuffix(suffix)
    return family


def _instructions(episode: Mapping[str, Any]) -> str:
    objects = ", ".join(
        f"{row['object_id']} (currently {row['state']}, version {row['version']})"
        for row in episode["initial_state"]
    )
    actors = "; ".join(
        f"{e['id']} as {e['role']}" for e in episode["entities"] if e.get("role")
    )
    return (
        f"{episode['title']}. Working between {episode['time_window']['start']} and "
        f"{episode['time_window']['end']}, bring every tracked object to its required "
        f"terminal state through the authorised tool surfaces. Tracked objects: {objects}. "
        f"Available actors: {actors}. Every request must name an authorised actor, supply "
        f"the current object versions it mutates, and cite public evidence references. "
        f"Resubmitting a request ID must be idempotent. Do not infer economic performance "
        f"from operational evidence; refuse unsupported claims."
    )


def build_contract(episode: Mapping[str, Any]) -> dict[str, Any]:
    family = _family_of(episode)
    actions = sorted(episode["authorization"])
    return {
        "schema": CONTRACT_SCHEMA,
        "episode_id": episode["episode_id"],
        "classification": CLASSIFICATION,
        "family": family,
        "work_type": WORK_TYPE_BY_FAMILY.get(family, "resolve"),
        "title": episode["title"],
        "instructions": _instructions(episode),
        "time_window": dict(episode["time_window"]),
        "competencies": list(episode["competencies"]),
        "actors": [
            {
                "id": e["id"],
                "role": e["role"],
                "allowed_actions": sorted(
                    a for a, roles in episode["authorization"].items() if e["role"] in roles
                ),
            }
            for e in episode["entities"]
            if e.get("role")
        ],
        "tracked_objects": [
            {"object_id": e["id"], "kind": e["kind"]}
            for e in episode["entities"]
            if not e.get("role")
        ],
        "initial_state_sha256": digest(episode["initial_state"]),
        "tool_surfaces": [
            {"surface": surface, "request_schema": "schemas/tool-request.schema.json"}
            for surface in episode["tool_surfaces"]
        ],
        "actions": actions,
        "deliverables": {
            "receipts": "cleanroom.tool-receipt/v1 per accepted request",
            "trajectory": "EvaluationSession.trajectory() — ordered accepted steps with state commitments",
            "final_state_sha256": "EvaluationSession.state_sha256() at completion",
        },
        "checks": [dict(check) for check in CHECKS],
        "barrier_policy": dict(BARRIER_POLICY),
        "oracle_separation": {
            "excluded_fields": list(ORACLE_FIELDS),
            "statement": "The sealed episode holds the terminal state and scripted path; this card does not.",
        },
        "transfer_tags": list(episode["transfer_tags"]),
    }


def verify_contract(contract: Mapping[str, Any], episode: Mapping[str, Any]) -> None:
    """Fail closed if the card disagrees with its episode or leaks the oracle."""

    validate_schema(contract, "episode-contract.schema.json")
    if contract["episode_id"] != episode["episode_id"]:
        raise ContractError("contract episode_id differs from episode")
    if contract != build_contract(episode):
        raise ContractError(f"contract drifted from its episode: {episode['episode_id']}")
    encoded = json.dumps(contract, sort_keys=True)
    for field in ORACLE_FIELDS:
        # Oracle fields may be *named* (the check list and the exclusion list
        # do), but never carried as keys with values.
        if f'"{field}": ' in encoded:
            raise ContractError(f"contract leaks oracle field {field!r}")
    terminal = {(row["object_id"], row["state"], row["version"]) for row in episode["final_state"]}
    initial = {(row["object_id"], row["state"], row["version"]) for row in episode["initial_state"]}
    for object_id, state, version in terminal - initial:
        if state in encoded and f'"{object_id}"' in encoded and f"version {version}" in encoded:
            raise ContractError(f"contract names the terminal state of {object_id}")
    for receipt in (e["expected_receipt"] for e in episode["events"]):
        if receipt in encoded:
            raise ContractError("contract leaks an expected receipt id")


def contract_path(episode_path: Path) -> Path:
    name = episode_path.name
    if name.endswith(TASK_SUFFIX):
        raise ValueError(f"already a contract: {name}")
    stem = name.removesuffix(".json").removesuffix(".v1")
    return episode_path.with_name(stem + TASK_SUFFIX)


def write_contracts(episode_dir: Path) -> list[Path]:
    written = []
    for path in sorted(episode_dir.glob("*.json")):
        if path.name.endswith(TASK_SUFFIX):
            continue
        episode = load_json(path)
        card = build_contract(episode)
        verify_contract(card, episode)
        target = contract_path(path)
        target.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(target)
    return written


def verify_contracts(episode_dir: Path) -> dict[str, Any]:
    episodes = [p for p in sorted(episode_dir.glob("*.json")) if not p.name.endswith(TASK_SUFFIX)]
    if not episodes:
        raise ContractError(f"no episodes under {episode_dir}")
    for path in episodes:
        card_path = contract_path(path)
        if not card_path.is_file():
            raise ContractError(f"missing contract for {path.name}")
        verify_contract(load_json(card_path), load_json(path))
    return {"schema": "cleanroom.episode-contract-verification/v1", "status": "PASS", "count": len(episodes)}
