"""Deterministically freeze 500 semantic-citation tasks from sealed episodes.

Only the strict ``cleanroom.citation-task/v1`` projection is provider-visible.
Expected selections and adversarial labels are written to a separate oracle
asset so evaluation metadata can never leak through the provider boundary.
No provider is called and no provider outcome is fabricated by this builder.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contract import canonical_bytes, digest, file_sha256, load_json


ASSET_ROOT = Path(__file__).with_name("assets")
EPISODE_ROOT = ASSET_ROOT / "episodes"
OUTPUT_ROOT = ASSET_ROOT / "citation_tasks"
TASK_PATH = OUTPUT_ROOT / "citation-tasks.v1.jsonl"
ORACLE_PATH = OUTPUT_ROOT / "citation-task-oracle.v1.jsonl"
MANIFEST_PATH = ASSET_ROOT / "citation-tasks.manifest.v1.json"
SOURCE_MANIFEST_PATH = ASSET_ROOT / "sealed-set.manifest.v1.json"

CLASSIFICATION = "CLEANROOM_SYNTHETIC"
CATEGORIES = (
    "supported_claim",
    "unsupported_addition_guard",
    "unsupported_refusal",
    "polarity_adversary",
    "relation_adversary",
    "qualifier_adversary",
)


def _iso_offset(value: str, *, seconds: int) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _proposition(
    subject: str,
    relation: str,
    object_: str,
    *,
    polarity: str = "affirmed",
    qualifiers: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "subject": subject,
        "relation": relation,
        "object": object_,
        "polarity": polarity,
        "qualifiers": dict(qualifiers),
    }


def _coverage(proposition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "subject": proposition["subject"],
        "relation": proposition["relation"],
        "qualifiers": deepcopy(proposition["qualifiers"]),
    }


def _selection_answer(
    task_id: str,
    proposition: Mapping[str, Any],
    evidence_ids: Iterable[str],
) -> dict[str, Any]:
    return {
        "schema": "cleanroom.claim-selection/v1",
        "answer_kind": "answer",
        "claims": [
            {
                "claim_id": f"claim_{task_id}",
                **deepcopy(proposition),
                "evidence_ids": list(evidence_ids),
            }
        ],
    }


def _selection_refusal(
    proposition: Mapping[str, Any],
    evidence_ids: Iterable[str],
) -> dict[str, Any]:
    return {
        "schema": "cleanroom.claim-selection/v1",
        "answer_kind": "refusal",
        "claims": [],
        "refusal": {
            "reason_code": "insufficient_evidence",
            "requested_proposition": deepcopy(proposition),
            "evidence_ids": list(evidence_ids),
        },
    }


def _question(proposition: Mapping[str, Any], *, instruction: str) -> str:
    qualifiers = ", ".join(
        f"{key}={value}" for key, value in sorted(proposition["qualifiers"].items())
    )
    return (
        f"{instruction} Proposition: {proposition['subject']} "
        f"{proposition['relation']} {proposition['object']} "
        f"(polarity={proposition['polarity']}; {qualifiers})."
    )


def _episode_material(
    episode: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    supported = next(
        item
        for item in episode["evidence_challenges"]
        if item["expected_disposition"] == "SUPPORTED"
    )
    unsupported = next(
        item
        for item in episode["evidence_challenges"]
        if item["expected_disposition"] == "REFUSE_UNSUPPORTED"
    )
    subject = episode["initial_state"][0]["object_id"]
    case = episode["initial_state"][1]["object_id"]
    completed_at = episode["events"][-1]["at"]
    approved_at = episode["events"][-2]["at"]
    common = {
        "episode_id": episode["episode_id"],
        "as_of": completed_at,
    }
    primary = _proposition(
        subject,
        "has supported operational finding",
        supported["claim"],
        qualifiers=common,
    )
    case_state = _proposition(
        case,
        "has case state",
        episode["final_state"][1]["state"],
        qualifiers=common,
    )
    approval = _proposition(
        subject,
        "has independently authorised resolution",
        "approved",
        qualifiers={
            "episode_id": episode["episode_id"],
            "as_of": approved_at,
        },
    )
    economic = _proposition(
        subject,
        "will outperform market benchmark",
        "next reporting period",
        qualifiers={
            "episode_id": episode["episode_id"],
            "as_of": completed_at,
        },
    )
    fair_value = _proposition(
        subject,
        "has projected fair value",
        "favourable",
        qualifiers={
            "episode_id": episode["episode_id"],
            "as_of": completed_at,
        },
    )
    polarity_primary = {**deepcopy(primary), "polarity": "negated"}
    polarity_state = {**deepcopy(case_state), "polarity": "negated"}
    relation_primary = {
        **deepcopy(primary),
        "relation": "has unsupported valuation finding",
    }
    relation_state = {
        **deepcopy(case_state),
        "relation": "has profitable economic outcome",
        "object": "confirmed",
    }
    qualifier_primary = deepcopy(primary)
    qualifier_primary["qualifiers"]["as_of"] = _iso_offset(
        primary["qualifiers"]["as_of"], seconds=1
    )
    qualifier_state = deepcopy(case_state)
    qualifier_state["qualifiers"]["as_of"] = episode["time_window"]["start"]

    propositions = {
        "primary": primary,
        "case_state": case_state,
        "approval": approval,
        "economic": economic,
        "fair_value": fair_value,
        "polarity_primary": polarity_primary,
        "polarity_state": polarity_state,
        "relation_primary": relation_primary,
        "relation_state": relation_state,
        "qualifier_primary": qualifier_primary,
        "qualifier_state": qualifier_state,
    }
    evidence_ids = sorted(
        {
            evidence_id
            for event in episode["events"]
            for evidence_id in event["evidence_refs"]
        }
    )
    by_suffix = {value.rsplit("_", 1)[-1]: value for value in evidence_ids}
    selected_ids = {
        "approval": by_suffix["04"],
        "completion": by_suffix["05"],
        "boundary": by_suffix["06"],
    }
    all_coverage = list(
        {
            canonical_bytes(item): item
            for item in (
                _coverage(value)
                for key, value in propositions.items()
                if key not in {"approval"}
            )
        }.values()
    )
    evidence = {
        selected_ids["approval"]: {
            "schema": "cleanroom.semantic-evidence/v1",
            "evidence_id": selected_ids["approval"],
            "propositions": [deepcopy(primary), deepcopy(approval)],
            "coverage": [_coverage(approval), *all_coverage],
        },
        selected_ids["completion"]: {
            "schema": "cleanroom.semantic-evidence/v1",
            "evidence_id": selected_ids["completion"],
            "propositions": [deepcopy(primary), deepcopy(case_state)],
            "coverage": all_coverage,
        },
        selected_ids["boundary"]: {
            "schema": "cleanroom.semantic-evidence/v1",
            "evidence_id": selected_ids["boundary"],
            "propositions": [deepcopy(case_state)],
            "coverage": all_coverage,
        },
    }
    challenge_ids = {
        "supported": supported["challenge_id"],
        "unsupported": unsupported["challenge_id"],
    }
    return propositions, evidence, {**selected_ids, **challenge_ids}


def _task_and_oracle(
    *,
    episode: Mapping[str, Any],
    category: str,
    slot: int,
    propositions: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    ids: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    slug = episode["episode_id"].removeprefix("episode_").removesuffix("_v1")
    task_id = f"citation_{slug}_{category}_{slot:02d}"
    forbidden: list[dict[str, Any]] = []
    if category == "supported_claim":
        proposition_key = ("primary", "case_state", "approval")[slot - 1]
        proposition = propositions[proposition_key]
        evidence_ids = {
            "primary": [ids["approval"], ids["completion"]],
            "case_state": [ids["completion"], ids["boundary"]],
            "approval": [ids["approval"]],
        }[proposition_key]
        selection = _selection_answer(task_id, proposition, evidence_ids)
        source_challenge = ids["supported"]
        instruction = "State the proposition only when supplied evidence supports it."
    elif category == "unsupported_addition_guard":
        proposition_key = ("primary", "case_state")[slot - 1]
        proposition = propositions[proposition_key]
        evidence_ids = {
            "primary": [ids["approval"], ids["completion"]],
            "case_state": [ids["completion"], ids["boundary"]],
        }[proposition_key]
        selection = _selection_answer(task_id, proposition, evidence_ids)
        forbidden = [deepcopy(propositions["economic"])]
        source_challenge = ids["supported"]
        instruction = (
            "State only the supported operational proposition; do not add an "
            "economic-performance proposition."
        )
    else:
        key_by_category = {
            "unsupported_refusal": ("economic", "fair_value"),
            "polarity_adversary": ("polarity_primary", "polarity_state"),
            "relation_adversary": ("relation_primary", "relation_state"),
            "qualifier_adversary": ("qualifier_primary", "qualifier_state"),
        }
        proposition = propositions[key_by_category[category][slot - 1]]
        cited_id = {
            "unsupported_refusal": ids["boundary"],
            "polarity_adversary": ids["approval"],
            "relation_adversary": ids["completion"],
            "qualifier_adversary": ids["completion"],
        }[category]
        selection = _selection_refusal(proposition, [cited_id])
        source_challenge = (
            ids["unsupported"]
            if category == "unsupported_refusal"
            else ids["supported"]
        )
        instruction = (
            "Determine whether the supplied evidence establishes the exact "
            "proposition, including polarity, relation and every qualifier; "
            "select a cited refusal when it does not."
        )

    task = {
        "schema": "cleanroom.citation-task/v1",
        "task_id": task_id,
        "question": _question(proposition, instruction=instruction),
        "evidence": [deepcopy(evidence[key]) for key in sorted(evidence)],
    }
    oracle = {
        "schema": "cleanroom.citation-task-oracle/v1",
        "task_id": task_id,
        "episode_id": episode["episode_id"],
        "source_challenge_id": source_challenge,
        "scenario_family": episode["scenario_lineage"]["scenario_family"],
        "category": category,
        "expected_selection": selection,
        "forbidden_additions": forbidden,
    }
    return task, oracle


def materialize() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = [
        load_json(path) for path in sorted(EPISODE_ROOT.glob("*.json"))
    ]
    if len(episodes) != 40:
        raise ValueError(f"expected 40 sealed episodes, found {len(episodes)}")
    tasks: list[dict[str, Any]] = []
    oracles: list[dict[str, Any]] = []
    for ordinal, episode in enumerate(episodes):
        propositions, evidence, ids = _episode_material(episode)
        slots = {category: 2 for category in CATEGORIES}
        if ordinal % 2 == 0:
            slots["supported_claim"] = 3
        for category in CATEGORIES:
            for slot in range(1, slots[category] + 1):
                task, oracle = _task_and_oracle(
                    episode=episode,
                    category=category,
                    slot=slot,
                    propositions=propositions,
                    evidence=evidence,
                    ids=ids,
                )
                tasks.append(task)
                oracles.append(oracle)
    if len(tasks) != 500 or len(oracles) != 500:
        raise ValueError("citation task generator did not produce exactly 500 tasks")
    return tasks, oracles


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(record) + b"\n" for record in records)


def _asset(path: Path, records: int, *, asset_root: Path = ASSET_ROOT) -> dict[str, Any]:
    return {
        "path": path.relative_to(asset_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "records": records,
    }


def build(output_root: Path | None = None) -> Path:
    """Write the frozen task bundle and return its manifest path."""

    tasks, oracles = materialize()
    root = ASSET_ROOT if output_root is None else output_root
    task_path = root / "citation_tasks" / TASK_PATH.name
    oracle_path = root / "citation_tasks" / ORACLE_PATH.name
    manifest_path = root / MANIFEST_PATH.name
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_bytes(_jsonl_bytes(tasks))
    oracle_path.write_bytes(_jsonl_bytes(oracles))

    category_counts = Counter(item["category"] for item in oracles)
    family_counts = Counter(item["scenario_family"] for item in oracles)
    answer_counts = Counter(
        item["expected_selection"]["answer_kind"] for item in oracles
    )
    ledger = [
        {
            "sequence": sequence,
            "task_id": task["task_id"],
            "task_sha256": digest(task),
            "oracle_sha256": digest(oracle),
            "episode_id": oracle["episode_id"],
            "scenario_family": oracle["scenario_family"],
            "category": oracle["category"],
            "expected_answer_kind": oracle["expected_selection"]["answer_kind"],
        }
        for sequence, (task, oracle) in enumerate(zip(tasks, oracles), start=1)
    ]
    task_list = "".join(
        f"{item['task_sha256']}  {item['task_id']}\n" for item in ledger
    )
    source_manifest = load_json(SOURCE_MANIFEST_PATH)
    manifest = {
        "schema": "cleanroom.citation-task-set/v1",
        "set_id": "cmo_cleanroom_citation_tasks_v1",
        "classification": CLASSIFICATION,
        "status": "FROZEN",
        "source_sealed_set": {
            "set_id": source_manifest["set_id"],
            "manifest_path": SOURCE_MANIFEST_PATH.relative_to(
                Path(__file__).parents[1]
            ).as_posix(),
            "manifest_sha256": file_sha256(SOURCE_MANIFEST_PATH),
        },
        "task_count": len(tasks),
        "episode_count": len({item["episode_id"] for item in oracles}),
        "assets": [
            _asset(task_path, len(tasks), asset_root=root),
            _asset(oracle_path, len(oracles), asset_root=root),
        ],
        "coverage": {
            "categories": dict(sorted(category_counts.items())),
            "scenario_families": dict(sorted(family_counts.items())),
            "expected_answer_kinds": dict(sorted(answer_counts.items())),
        },
        "task_ledger": ledger,
        "task_list_sha256": hashlib.sha256(task_list.encode("utf-8")).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


if __name__ == "__main__":
    build()
