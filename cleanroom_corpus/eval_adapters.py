"""Six observable adapters for exported clean-room episodes.

Two session modes share one contract:

``replay`` (default) is teacher-forced: every request must match the next
scripted event exactly. It is a deterministic validator of a known trajectory.

``free`` is an environment: a request is accepted when it names a legal
transition whose preconditions currently hold, regardless of script order.
The transition table is derived from the sealed episode's own mutations, so no
asset changes and no answer-key field is exposed. Authorization, object
versions, idempotency, evidence grounding and per-currency ledger conservation
are enforced identically in both modes; only the ordering constraint differs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .eval_export import TOOL_SURFACES, _canonical


class AdapterError(ValueError):
    """Raised when a tool request violates the episode contract."""


MODES = ("replay", "free")


class EvaluationSession:
    """Version-checked episode simulator with no state-read API."""

    def __init__(
        self, corpus_root: Path, episode: dict[str, Any], *, mode: str = "replay"
    ) -> None:
        if mode not in MODES:
            raise AdapterError(f"unknown session mode: {mode}")
        self.mode = mode
        self._root = corpus_root.resolve()
        self._episode = episode
        self._event_index = 0
        self._transitions = _derive_transitions(episode)
        self._requests: dict[str, dict[str, Any]] = {}
        self._applied_transitions: set[str] = set()
        self._trajectory: list[dict[str, Any]] = []
        self._last_rejection: dict[str, Any] | None = None
        self._states = {
            row["object_id"]: {
                "state": row["state"],
                "version": row["version"],
                "facts": row.get("facts", {}),
            }
            for row in episode["initial_state"]
        }
        self._receipts: dict[str, dict[str, Any]] = {}
        self._entities = {row["id"]: row for row in episode["entities"]}
        self._evidence = {
            row["evidence_id"]: row
            for row in _load_jsonl(
                self._root / "evaluation" / "public" / "evidence_refs.jsonl"
            )
        }
        adapters = {
            "communications": CommunicationsAdapter,
            "case_management": CaseManagementAdapter,
            "trade_ledger": TradeLedgerAdapter,
            "reference_data": ReferenceDataAdapter,
            "controls_monitoring": ControlsMonitoringAdapter,
            "evidence_store": EvidenceStoreAdapter,
        }
        self.adapters = {
            name: adapter_type(self, name)
            for name, adapter_type in adapters.items()
        }

    @classmethod
    def from_file(
        cls, corpus_root: Path, episode_path: Path, *, mode: str = "replay"
    ) -> "EvaluationSession":
        return cls(
            corpus_root,
            json.loads(episode_path.read_text(encoding="utf-8")),
            mode=mode,
        )

    def adapter(self, surface: str) -> "BaseAdapter":
        try:
            return self.adapters[surface]
        except KeyError as exc:
            raise AdapterError(f"unknown surface: {surface}") from exc

    # ------------------------------------------------------------------
    # Free-action environment surface
    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        """True when every object has reached its sealed final state and version."""

        for row in self._episode["final_state"]:
            state = self._states.get(row["object_id"])
            if (
                state is None
                or state["state"] != row["state"]
                or state["version"] != row["version"]
            ):
                return False
        return True

    def trajectory(self) -> list[dict[str, Any]]:
        """Immutable copy of every accepted step, for grading and transcripts."""

        return [dict(step) for step in self._trajectory]

    def state_sha256(self) -> str:
        return hashlib.sha256(
            _canonical(
                [
                    {"object_id": object_id, **state}
                    for object_id, state in sorted(self._states.items())
                ]
            )
        ).hexdigest()

    def state_changes_outside_contract(self) -> int:
        """Diff the live state against an independent recomputation from the
        sealed contract: initial_state plus the mutations of every applied
        transition. Nonzero means state moved outside the contract."""

        expected: dict[str, dict[str, Any]] = {
            row["object_id"]: {
                "state": row["state"],
                "version": row["version"],
                "facts": row.get("facts", {}),
            }
            for row in self._episode["initial_state"]
        }
        for transition in self._transitions:
            if transition["template_event_id"] not in self._applied_transitions:
                continue
            for mutation in transition["mutations"]:
                row = expected.setdefault(
                    mutation["object_id"], {"state": None, "version": -1, "facts": {}}
                )
                # Versions form a strict chain per object, so the contracted
                # end state is the applied mutation with the highest to_version.
                if mutation["to_version"] > row["version"]:
                    row["state"] = mutation["to_state"]
                    row["version"] = mutation["to_version"]
        live = {
            object_id: {
                "state": state["state"],
                "version": state["version"],
                "facts": state["facts"],
            }
            for object_id, state in self._states.items()
        }
        return sum(
            1
            for object_id in set(expected) | set(live)
            if expected.get(object_id) != live.get(object_id)
        )

    def _active_transitions(self) -> list[dict[str, Any]]:
        return [
            transition
            for transition in self._transitions
            if transition["template_event_id"] not in self._applied_transitions
            and self._preconditions_hold(transition)
        ]

    def _preconditions_hold(self, transition: dict[str, Any]) -> bool:
        for mutation in transition["mutations"]:
            state = self._states.get(mutation["object_id"])
            if (
                state is None
                or state["state"] != mutation["from_state"]
                or state["version"] != mutation["from_version"]
            ):
                return False
        return True

    def _observable_free(self) -> dict[str, Any]:
        active = self._active_transitions()
        actors = []
        for entity in self._episode["entities"]:
            role = entity.get("role")
            if entity.get("kind") != "person" or not role:
                continue
            actors.append(
                {
                    "id": entity["id"],
                    "role": role,
                    "allowed_actions": sorted(
                        action
                        for action, roles in self._episode["authorization"].items()
                        if role in roles
                    ),
                }
            )
        return {
            "episode_id": self._episode["episode_id"],
            "mode": "free",
            "step": len(self._trajectory),
            "time_window": dict(self._episode["time_window"]),
            # What the desk currently sees: narrative for every transition whose
            # preconditions hold. This names no action, actor or ordering.
            "observations": sorted({t["observation"] for t in active}),
            "actors": actors,
            "available_tool_surfaces": list(self._episode["tool_surfaces"]),
            # Capability discovery, like a tool list: which actions each
            # surface exposes and for which roles. No ordering, no mutation.
            "surfaces": [
                self.adapters[surface].describe()
                for surface in self._episode["tool_surfaces"]
                if surface in self.adapters
            ],
            "visible_state": [
                {
                    "object_id": object_id,
                    "state": state["state"],
                    "version": state["version"],
                    "facts": state["facts"],
                }
                for object_id, state in sorted(self._states.items())
            ],
            "available_evidence_refs": sorted(
                {ref for t in self._transitions for ref in t["evidence_refs"]}
            ),
            "prior_receipts": list(self._receipts.values()),
            # Feedback: the contract's last rejection, so a stateless policy
            # can correct itself. Names the error text only.
            "last_rejection": dict(self._last_rejection) if self._last_rejection else None,
            "complete": self.is_complete(),
        }

    def _invoke_free(self, surface: str, request: dict[str, Any]) -> dict[str, Any]:
        required = {"request_id", "actor_id", "action", "object_versions", "evidence_refs"}
        if set(request) != required:
            raise AdapterError(f"request fields must be exactly {sorted(required)}")
        request_id = request["request_id"]
        if request_id in self._requests:
            if self._requests[request_id] != request:
                raise AdapterError("request ID reused with a different payload")
            self._trajectory.append(
                {"request_id": request_id, "surface": surface, "status": "IDEMPOTENT"}
            )
            return dict(self._receipts[request_id])
        role = self._entities.get(request["actor_id"], {}).get("role")
        if role not in self._episode["authorization"].get(request["action"], []):
            raise AdapterError("actor is not authorized for action")

        named = [
            t
            for t in self._transitions
            if t["surface"] == surface and t["action"] == request["action"]
        ]
        if not named:
            raise AdapterError(f"action {request['action']!r} is not available on {surface}")
        named = [t for t in named if t["template_event_id"] not in self._applied_transitions]
        if not named:
            raise AdapterError("transition already applied; resubmit the original request ID")
        candidates = [t for t in named if self._preconditions_hold(t)]
        if not candidates:
            raise AdapterError("mutation precondition or version failed")
        if len(candidates) > 1:
            raise AdapterError("ambiguous transition; episode contract is malformed")
        transition = candidates[0]

        expected_versions = {
            mutation["object_id"]: mutation["from_version"]
            for mutation in transition["mutations"]
        }
        asserted = request["object_versions"]
        # Optimistic concurrency: every version the caller asserts must be the
        # object's current version, and every object the transition mutates
        # must be asserted. Asserting extra, current objects is not an error.
        stale = {
            object_id: version
            for object_id, version in asserted.items()
            if object_id not in self._states or self._states[object_id]["version"] != version
        }
        missing = {k: v for k, v in expected_versions.items() if k not in asserted}
        if stale or missing:
            raise AdapterError(
                f"stale or excessive object versions: expected {expected_versions}"
            )
        cited = request["evidence_refs"]
        available = {ref for t in self._transitions for ref in t["evidence_refs"]}
        if (
            not isinstance(cited, list)
            or not cited
            or not set(cited).issubset(available)
        ):
            raise AdapterError("evidence_refs must be a non-empty subset of available evidence")

        totals: dict[str, int] = {}
        for entry in transition["ledger_entries"]:
            totals[entry["currency"]] = totals.get(entry["currency"], 0) + entry["amount_minor"]
        if any(totals.values()):
            raise AdapterError("ledger entries do not balance per currency")

        applied = []
        for mutation in transition["mutations"]:
            if mutation["to_version"] != mutation["from_version"] + 1:
                raise AdapterError("mutation precondition or version failed")
            state = self._states[mutation["object_id"]]
            state["state"] = mutation["to_state"]
            state["version"] = mutation["to_version"]
            applied.append(
                {
                    "object_id": mutation["object_id"],
                    "state": mutation["to_state"],
                    "version": mutation["to_version"],
                }
            )
        receipt = {
            "schema": "cleanroom.tool-receipt/v1",
            "receipt_id": transition["receipt_id"],
            "request_id": request_id,
            "surface": surface,
            "status": "APPLIED",
            "applied_mutations": applied,
            "evidence_refs": sorted(cited),
        }
        receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
        self._receipts[request_id] = receipt
        self._requests[request_id] = dict(request)
        self._applied_transitions.add(transition["template_event_id"])
        self._trajectory.append(
            {
                "request_id": request_id,
                "surface": surface,
                "action": request["action"],
                "actor_id": request["actor_id"],
                "status": "APPLIED",
                "receipt_sha256": receipt["receipt_sha256"],
                "state_sha256": self.state_sha256(),
            }
        )
        return dict(receipt)

    # ------------------------------------------------------------------
    # Teacher-forced replay surface
    # ------------------------------------------------------------------

    def observable_instruction(self) -> dict[str, Any]:
        """Return the provider-safe boundary, excluding every answer-key field."""

        if self.mode == "free":
            return self._observable_free()
        if self._event_index >= len(self._episode["events"]):
            raise AdapterError("episode is complete")
        event = self._episode["events"][self._event_index]
        actor = self._entities[event["actor_id"]]
        role = actor.get("role")
        allowed_actions = sorted(
            action
            for action, roles in self._episode["authorization"].items()
            if role in roles
        )
        current_and_prior_evidence = sorted(
            {
                *event["evidence_refs"],
                *(
                    evidence_id
                    for receipt in self._receipts.values()
                    for evidence_id in receipt["evidence_refs"]
                ),
            }
        )
        return {
            "episode_id": self._episode["episode_id"],
            "event_id": event["event_id"],
            "sequence": event["sequence"],
            "observation_time": event["at"],
            "observation": event["observation"],
            "actor": {"id": actor["id"], "role": role},
            "available_tool_surfaces": list(self._episode["tool_surfaces"]),
            "allowed_actions": allowed_actions,
            "visible_state": [
                {
                    "object_id": object_id,
                    "state": state["state"],
                    "version": state["version"],
                    "facts": state["facts"],
                }
                for object_id, state in sorted(self._states.items())
            ],
            "available_evidence_refs": current_and_prior_evidence,
            "prior_receipts": list(self._receipts.values()),
        }

    def _invoke(self, surface: str, request: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "free":
            try:
                receipt = self._invoke_free(surface, request)
            except AdapterError as exc:
                self._last_rejection = {
                    "surface": surface,
                    "action": request.get("action") if isinstance(request, dict) else None,
                    "error": str(exc),
                }
                raise
            self._last_rejection = None
            return receipt
        if self._event_index >= len(self._episode["events"]):
            raise AdapterError("episode is complete")
        event = self._episode["events"][self._event_index]
        required = {"request_id", "actor_id", "action", "object_versions"}
        if set(request) != required:
            raise AdapterError(f"request fields must be exactly {sorted(required)}")
        if (
            event["surface"] != surface
            or request["request_id"] != event["request_id"]
            or request["actor_id"] != event["actor_id"]
            or request["action"] != event["action"]
        ):
            raise AdapterError("request does not match the next observable event")
        role = self._entities.get(request["actor_id"], {}).get("role")
        if role not in self._episode["authorization"].get(request["action"], []):
            raise AdapterError("actor is not authorized for action")

        expected_versions = {
            mutation["object_id"]: mutation["from_version"]
            for mutation in event["mutations"]
        }
        if request["object_versions"] != expected_versions:
            raise AdapterError(
                f"stale or excessive object versions: expected {expected_versions}"
            )
        if event.get("duplicate_of"):
            cached = self._receipts.get(event["request_id"])
            if cached is None or event["mutations"]:
                raise AdapterError("idempotent replay has no primary receipt")
            self._event_index += 1
            return dict(cached)
        if event["request_id"] in self._receipts:
            raise AdapterError("request ID reused without duplicate event")

        totals: dict[str, int] = {}
        for entry in event.get("ledger_entries", []):
            totals[entry["currency"]] = totals.get(entry["currency"], 0) + entry["amount_minor"]
        if any(totals.values()):
            raise AdapterError("ledger entries do not balance per currency")
        applied = []
        for mutation in event["mutations"]:
            state = self._states.get(mutation["object_id"])
            if (
                state is None
                or state["state"] != mutation["from_state"]
                or state["version"] != mutation["from_version"]
                or mutation["to_version"] != mutation["from_version"] + 1
            ):
                raise AdapterError("mutation precondition or version failed")
            state["state"] = mutation["to_state"]
            state["version"] = mutation["to_version"]
            applied.append(
                {
                    "object_id": mutation["object_id"],
                    "state": mutation["to_state"],
                    "version": mutation["to_version"],
                }
            )
        receipt = {
            "schema": "cleanroom.tool-receipt/v1",
            "receipt_id": event["expected_receipt"],
            "request_id": event["request_id"],
            "surface": surface,
            "status": "APPLIED",
            "applied_mutations": applied,
            "evidence_refs": event["evidence_refs"],
        }
        receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
        self._receipts[event["request_id"]] = receipt
        self._event_index += 1
        return dict(receipt)


class BaseAdapter:
    """One observable tool surface: a capability descriptor plus ``invoke``."""

    request_schema = "schemas/tool-request.schema.json"

    def __init__(self, session: EvaluationSession, surface: str) -> None:
        if surface not in TOOL_SURFACES:
            raise AdapterError(f"unknown adapter surface: {surface}")
        self._session = session
        self.surface = surface

    def describe(self) -> dict[str, Any]:
        """Actions this surface exposes in the bound episode, with permitted roles.

        Derived from the episode's own transition table and authorization map;
        names no ordering, no mutation and no receipt.
        """

        episode = self._session._episode
        actions = sorted(
            {t["action"] for t in self._session._transitions if t["surface"] == self.surface}
        )
        return {
            "surface": self.surface,
            "enabled": self.surface in episode["tool_surfaces"],
            "request_schema": self.request_schema,
            "actions": [
                {"action": action, "permitted_roles": sorted(episode["authorization"].get(action, []))}
                for action in actions
            ],
        }

    def invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.surface not in self._session._episode["tool_surfaces"]:
            raise AdapterError(f"surface {self.surface} is not enabled for this episode")
        return self._session._invoke(self.surface, request)


class CommunicationsAdapter(BaseAdapter):
    """Inbound operational messages; observation-only actions usually start here."""


class CaseManagementAdapter(BaseAdapter):
    """Case ownership, escalation, approval and closure."""


class TradeLedgerAdapter(BaseAdapter):
    """Versioned trade/leg mutations with per-currency balanced ledger entries."""


class ReferenceDataAdapter(BaseAdapter):
    """Effective-dated reference records: accounts, instructions, entitlements."""


class ControlsMonitoringAdapter(BaseAdapter):
    """Break detection, control reviews and conservation confirmations."""


class EvidenceStoreAdapter(BaseAdapter):
    def resolve(self, evidence_id: str) -> dict[str, Any]:
        try:
            record = self._session._evidence[evidence_id]
        except KeyError as exc:
            raise AdapterError(f"unknown evidence reference: {evidence_id}") from exc
        path = (self._session._root / record["relative_path"]).resolve()
        if self._session._root not in path.parents:
            raise AdapterError("evidence path escapes corpus")
        payload = path.read_bytes()[record["byte_start"] : record["byte_end"]]
        if hashlib.sha256(payload).hexdigest() != record["text_sha256"]:
            raise AdapterError("evidence bytes no longer match immutable reference")
        return {
            "schema": "cleanroom.observable-evidence-resolution/v1",
            "evidence_id": evidence_id,
            "surface": record["surface"],
            "artifact_id": record["artifact_id"],
            "span_id": record["span_id"],
            "text": payload.decode("utf-8"),
            "text_sha256": record["text_sha256"],
        }


def _derive_transitions(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive the legal transition table from a sealed episode.

    Each non-duplicate event contributes one transition: the surface and
    action, its state/version preconditions and postconditions, its balanced
    ledger entries, the public evidence it grounds on, and the receipt it
    yields. Order is deliberately dropped; preconditions carry the causality.
    """

    transitions = []
    for event in episode["events"]:
        if event.get("duplicate_of"):
            continue
        transitions.append(
            {
                "template_event_id": event["event_id"],
                "surface": event["surface"],
                "action": event["action"],
                "observation": event["observation"],
                "mutations": [dict(m) for m in event["mutations"]],
                "ledger_entries": [dict(e) for e in event.get("ledger_entries", [])],
                "evidence_refs": list(event["evidence_refs"]),
                "receipt_id": event["expected_receipt"],
            }
        )
    return transitions


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
