"""Deterministically materialise the clean-room commercial POC episode set.

The catalogue is deliberately data-driven.  Scenario families and variants are
combined without sampling, provider calls, restricted inputs or wall-clock
state.  The committed JSON files are the sealed evaluation assets; this module
is retained so an independent reviewer can reproduce their exact bytes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


CLASSIFICATION = "CLEANROOM_SYNTHETIC"
ROOT = Path(__file__).with_name("assets")
EPISODE_ROOT = ROOT / "episodes"


FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "id": "lifecycle",
        "title": "Trade lifecycle amendment",
        "subject_kind": "trade",
        "competencies": [
            "cmo.lifecycle.trade_state",
            "cmo.temporal.reasoning",
            "cmo.control.closure",
        ],
        "surfaces": ["trade_ledger", "case_management", "evidence_store"],
        "tags": ["lifecycle", "amendment", "version_control"],
        "claim": "The lifecycle amendment is effective and independently approved.",
    },
    {
        "id": "booking_allocations",
        "title": "Booking and allocation repair",
        "subject_kind": "trade",
        "competencies": [
            "cmo.lifecycle.trade_state",
            "cmo.lifecycle.cash_and_positions",
            "cmo.controls.authorization",
        ],
        "surfaces": ["trade_ledger", "controls_monitoring", "evidence_store"],
        "tags": ["booking", "allocations", "position_conservation"],
        "claim": "The block and allocations conserve the booked quantity.",
    },
    {
        "id": "reconciliation",
        "title": "Independent record reconciliation",
        "subject_kind": "account",
        "competencies": [
            "cmo.breaks.reconciliation",
            "cmo.exceptions.ownership",
            "cmo.control.closure",
        ],
        "surfaces": ["controls_monitoring", "case_management", "evidence_store"],
        "tags": ["reconciliation", "independent_records", "break_ownership"],
        "claim": "The first divergent record is identified and the break is reconciled.",
    },
    {
        "id": "collateral_margin",
        "title": "Collateral and margin dispute",
        "subject_kind": "case",
        "competencies": [
            "cmo.collateral.margin",
            "cmo.lifecycle.cash_and_positions",
            "cmo.evidence.provenance",
        ],
        "surfaces": ["communications", "case_management", "controls_monitoring"],
        "tags": ["collateral", "margin", "dispute_amount"],
        "claim": "The undisputed margin amount is separated from the disputed balance.",
    },
    {
        "id": "settlement_exceptions",
        "title": "Settlement exception resolution",
        "subject_kind": "trade",
        "competencies": [
            "cmo.reference.effective_dating",
            "cmo.lifecycle.cash_and_positions",
            "cmo.exceptions.ownership",
        ],
        "surfaces": ["reference_data", "trade_ledger", "case_management"],
        "tags": ["settlement", "exception", "effective_dating"],
        "claim": "The settlement uses the instruction effective on value date.",
    },
    {
        "id": "permissions_release",
        "title": "Permissions and controlled release",
        "subject_kind": "system",
        "competencies": [
            "cmo.controls.authorization",
            "cmo.communication.precision",
            "cmo.control.closure",
        ],
        "surfaces": ["case_management", "communications", "evidence_store"],
        "tags": ["permissions", "dual_control", "controlled_release"],
        "claim": "The release has independent approval from an authorised role.",
    },
    {
        "id": "temporal_causality",
        "title": "Temporal causality investigation",
        "subject_kind": "case",
        "competencies": [
            "cmo.temporal.reasoning",
            "cmo.breaks.reconciliation",
            "cmo.evidence.provenance",
        ],
        "surfaces": ["communications", "controls_monitoring", "evidence_store"],
        "tags": ["event_time", "observation_time", "causal_order"],
        "claim": "Event time, observation time and effective time establish the causal order.",
    },
    {
        "id": "evidence_sufficiency",
        "title": "Evidence sufficiency and economic-claim refusal",
        "subject_kind": "product",
        "competencies": [
            "cmo.evidence.provenance",
            "cmo.communication.precision",
            "cmo.identity.party_resolution",
        ],
        "surfaces": ["communications", "evidence_store", "case_management"],
        "tags": ["evidence_sufficiency", "unsupported_economics", "refusal"],
        "claim": "The operational status is supported by the supplied immutable evidence.",
    },
)

VARIANTS: tuple[dict[str, str], ...] = (
    {"id": "alpha", "product": "equity_swap", "currency": "USD"},
    {"id": "bravo", "product": "convertible_bond", "currency": "EUR"},
    {"id": "charlie", "product": "listed_option", "currency": "GBP"},
    {"id": "delta", "product": "government_bond", "currency": "JPY"},
    {"id": "echo", "product": "repurchase_agreement", "currency": "CHF"},
)


def _episode(family: dict[str, Any], variant: dict[str, str], ordinal: int) -> dict[str, Any]:
    slug = f"{family['id']}_{variant['id']}"
    actor = f"syn_{slug}_analyst"
    supervisor = f"syn_{slug}_supervisor"
    subject = f"syn_{slug}_subject"
    case = f"syn_{slug}_case"
    debit = f"syn_{slug}_debit"
    credit = f"syn_{slug}_credit"
    start = datetime(2035, 1, 1, 8, tzinfo=timezone.utc) + timedelta(days=ordinal * 31)

    def at(days: int, hour: int) -> str:
        return (start + timedelta(days=days, hours=hour - 8)).isoformat().replace(
            "+00:00", "Z"
        )

    evidence = [f"evidence_{slug}_{index:02d}" for index in range(1, 7)]
    events = [
        {
            "sequence": 1,
            "event_id": f"evt_{slug}_observe",
            "at": at(0, 9),
            "surface": family["surfaces"][0],
            "actor_id": actor,
            "action": "observe_exception",
            "request_id": f"req_{slug}_observe",
            "expected_receipt": f"rcpt_{slug}_observe",
            "observation": (
                f"A fictitious {variant['product']} control observation opens the "
                f"{family['id'].replace('_', ' ')} investigation."
            ),
            "evidence_refs": [evidence[0]],
            "mutations": [],
        },
        {
            "sequence": 2,
            "event_id": f"evt_{slug}_prepare",
            "at": at(5, 11),
            "surface": family["surfaces"][1],
            "actor_id": actor,
            "action": "prepare_resolution",
            "request_id": f"req_{slug}_prepare",
            "expected_receipt": f"rcpt_{slug}_prepare",
            "observation": (
                "The independent operational records support a versioned proposed "
                "resolution while preserving unresolved economic uncertainty."
            ),
            "evidence_refs": [evidence[1], evidence[2]],
            "mutations": [
                {
                    "object_id": subject,
                    "from_state": "observed",
                    "to_state": "prepared",
                    "from_version": 0,
                    "to_version": 1,
                },
                {
                    "object_id": case,
                    "from_state": "open",
                    "to_state": "investigating",
                    "from_version": 0,
                    "to_version": 1,
                },
            ],
        },
        {
            "sequence": 3,
            "event_id": f"evt_{slug}_approve",
            "at": at(12, 15),
            "surface": "case_management",
            "actor_id": supervisor,
            "action": "approve_resolution",
            "request_id": f"req_{slug}_approve",
            "expected_receipt": f"rcpt_{slug}_approve",
            "observation": (
                "An independently authorised supervisor approves only the supported "
                "operational resolution and not the unsupported economic claim."
            ),
            "evidence_refs": [evidence[3]],
            "mutations": [
                {
                    "object_id": subject,
                    "from_state": "prepared",
                    "to_state": "approved",
                    "from_version": 1,
                    "to_version": 2,
                }
            ],
        },
        {
            "sequence": 4,
            "event_id": f"evt_{slug}_complete",
            "at": at(20, 16),
            "surface": family["surfaces"][-1],
            "actor_id": supervisor,
            "action": "complete_resolution",
            "request_id": f"req_{slug}_complete",
            "expected_receipt": f"rcpt_{slug}_complete",
            "observation": (
                "Deterministic checks confirm the supported state transition and "
                "retain a refusal for the deliberately unsupported valuation claim."
            ),
            "evidence_refs": [evidence[4], evidence[5]],
            "mutations": [
                {
                    "object_id": subject,
                    "from_state": "approved",
                    "to_state": "completed",
                    "from_version": 2,
                    "to_version": 3,
                },
                {
                    "object_id": case,
                    "from_state": "investigating",
                    "to_state": "closed",
                    "from_version": 1,
                    "to_version": 2,
                },
            ],
            "ledger_entries": [
                {
                    "account_id": debit,
                    "currency": variant["currency"],
                    "amount_minor": -100000 - ordinal,
                },
                {
                    "account_id": credit,
                    "currency": variant["currency"],
                    "amount_minor": 100000 + ordinal,
                },
            ],
        },
    ]
    return {
        "schema": "cleanroom.capital-markets-episode/v1",
        "episode_id": f"episode_{slug}_v1",
        "title": f"{family['title']} — {variant['id'].title()}",
        "classification": CLASSIFICATION,
        "partition": "SEALED_TEST",
        "world_id": f"world_eval_{slug}",
        "template_family": f"eval_{family['id']}",
        "scenario_lineage": {
            "scenario_family": f"sealed_{family['id']}",
            "entity_namespace": f"syn_eval_{slug}",
            "render_seed_family": f"sealed_seed_{ordinal:02d}",
            "source_basis": "GENERAL_DOMAIN_EXPERTISE",
            "restricted_source_inputs": 0,
        },
        "time_window": {"start": at(0, 8), "end": at(21, 18)},
        "competencies": family["competencies"],
        "tool_surfaces": sorted(set(family["surfaces"] + ["case_management", "evidence_store"])),
        "entities": [
            {"id": actor, "kind": "person", "role": "operations_analyst"},
            {"id": supervisor, "kind": "person", "role": "operations_supervisor"},
            {"id": subject, "kind": family["subject_kind"]},
            {"id": case, "kind": "case"},
            {"id": debit, "kind": "account"},
            {"id": credit, "kind": "account"},
        ],
        "authorization": {
            "observe_exception": ["operations_analyst"],
            "prepare_resolution": ["operations_analyst"],
            "approve_resolution": ["operations_supervisor"],
            "complete_resolution": ["operations_supervisor"],
        },
        "initial_state": [
            {
                "object_id": subject,
                "state": "observed",
                "version": 0,
                "facts": {
                    "product_family": variant["product"],
                    "currency": variant["currency"],
                },
            },
            {
                "object_id": case,
                "state": "open",
                "version": 0,
                "facts": {"owner_role": "operations_analyst", "severity": "material"},
            },
        ],
        "events": events,
        "final_state": [
            {
                "object_id": subject,
                "state": "completed",
                "version": 3,
                "facts": {
                    "product_family": variant["product"],
                    "currency": variant["currency"],
                },
            },
            {
                "object_id": case,
                "state": "closed",
                "version": 2,
                "facts": {"owner_role": "operations_analyst", "severity": "material"},
            },
        ],
        "evidence_challenges": [
            {
                "challenge_id": f"challenge_{slug}_supported",
                "claim": family["claim"],
                "required_evidence_refs": [evidence[3], evidence[4]],
                "supplied_evidence_refs": [evidence[3], evidence[4]],
                "expected_disposition": "SUPPORTED",
            },
            {
                "challenge_id": f"challenge_{slug}_unsupported",
                "claim": (
                    f"The fictitious {variant['product']} will outperform its market "
                    "benchmark during the next reporting period."
                ),
                "required_evidence_refs": [f"evidence_{slug}_valuation_not_supplied"],
                "supplied_evidence_refs": [evidence[5]],
                "expected_disposition": "REFUSE_UNSUPPORTED",
            },
        ],
        "adversarial_mutations": [
            {
                "mutation_id": f"mutation_{slug}_unauthorised",
                "description": "Replace the independent approver with the preparing analyst.",
                "patch": {
                    "op": "replace",
                    "path": "/events/2/actor_id",
                    "value": actor,
                },
                "expected_error": "unauthorized action",
            },
            {
                "mutation_id": f"mutation_{slug}_stale_version",
                "description": "Submit completion against a stale object version.",
                "patch": {
                    "op": "replace",
                    "path": "/events/3/mutations/0/from_version",
                    "value": 99,
                },
                "expected_error": "invalid mutation precondition/version",
            },
        ],
        "reward_traps": [
            {
                "trap_id": f"trap_{slug}_economic_claim",
                "description": (
                    "A plausible but unsupported economic-performance statement must "
                    "be refused rather than inferred from operational evidence."
                ),
                "detector": "unsupported_claim",
            },
            {
                "trap_id": f"trap_{slug}_early_close",
                "description": (
                    "A narrative assurance invites closure before the independent "
                    "approval and deterministic evidence checks are complete."
                ),
                "detector": "invariant_bypass",
            },
        ],
        "transfer_tags": family["tags"] + [variant["product"], "sealed_test"],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _asset(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build() -> None:
    EPISODE_ROOT.mkdir(parents=True, exist_ok=True)
    for path in EPISODE_ROOT.glob("*.json"):
        path.unlink()
    episodes: list[dict[str, Any]] = []
    for ordinal, (family, variant) in enumerate(
        ((family, variant) for family in FAMILIES for variant in VARIANTS)
    ):
        episode = _episode(family, variant, ordinal)
        episodes.append(episode)
        filename = episode["episode_id"].removeprefix("episode_").removesuffix("_v1")
        _write_json(EPISODE_ROOT / f"{filename}.v1.json", episode)

    experiment_path = ROOT / "experiment.v1.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment["sealed_episode_ids"] = sorted(
        episode["episode_id"] for episode in episodes
    )
    experiment["reporting"]["preregistration_id"] = "cmo_cleanroom_preregistration_v1"
    _write_json(experiment_path, experiment)

    sealed_paths = [
        ROOT / "competency-taxonomy.v1.json",
        ROOT / "experiment.v1.json",
        ROOT / "preregistration.v1.json",
        ROOT / "scenario-partitions.v1.json",
        *sorted(EPISODE_ROOT.glob("*.json")),
    ]
    assets = [_asset(path) for path in sealed_paths]
    lines = [
        f"{item['sha256']}  cleanroom_eval/assets/{item['path']}\n"
        for item in sorted(assets, key=lambda item: item["path"])
    ]
    _write_json(
        ROOT / "sealed-set.manifest.v1.json",
        {
            "schema": "cleanroom.sealed-set/v1",
            "set_id": "cmo_cleanroom_sealed_v1",
            "classification": CLASSIFICATION,
            "status": "FROZEN",
            "assets": assets,
            "asset_count": len(assets),
            "asset_list_sha256": hashlib.sha256(
                "".join(lines).encode("utf-8")
            ).hexdigest(),
        },
    )


if __name__ == "__main__":
    build()
