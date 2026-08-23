"""Schema-first evidence selection and deterministic citation rendering.

The model is not allowed to write citation markers or final answer prose.  It
selects structured claims and evidence identifiers.  This module validates the
selection against a supplied semantic evidence set and only then renders the
reader-facing answer and citation objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence

from .contract import ContractError, canonical_bytes, digest, validate_schema


SELECTION_SCHEMA = "cleanroom.claim-selection/v1"
EVIDENCE_SCHEMA = "cleanroom.semantic-evidence/v1"
RENDERED_SCHEMA = "cleanroom.rendered-answer/v1"
_SPACE = re.compile(r"\s+")


class CitationValidationError(ContractError):
    """Raised when a model selection is not grounded by supplied evidence."""


def _normalized_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CitationValidationError(f"{field} must be a non-empty string")
    return _SPACE.sub(" ", value.strip())


def _normalized_qualifiers(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise CitationValidationError("qualifiers must be an object")
    normalized: list[tuple[str, str]] = []
    for key, item in value.items():
        normalized.append(
            (
                _normalized_text(key, "qualifier name"),
                _normalized_text(item, f"qualifier {key}"),
            )
        )
    return tuple(sorted(normalized))


@dataclass(frozen=True, order=True)
class Proposition:
    subject: str
    relation: str
    object: str
    polarity: str
    qualifiers: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Proposition":
        return cls(
            subject=_normalized_text(value.get("subject"), "subject"),
            relation=_normalized_text(value.get("relation"), "relation"),
            object=_normalized_text(value.get("object"), "object"),
            polarity=_normalized_text(value.get("polarity"), "polarity"),
            qualifiers=_normalized_qualifiers(value.get("qualifiers")),
        )


@dataclass(frozen=True, order=True)
class Coverage:
    subject: str
    relation: str
    qualifiers: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Coverage":
        return cls(
            subject=_normalized_text(value.get("subject"), "coverage subject"),
            relation=_normalized_text(value.get("relation"), "coverage relation"),
            qualifiers=_normalized_qualifiers(value.get("qualifiers")),
        )

    def covers(self, proposition: Proposition) -> bool:
        return (
            self.subject == proposition.subject
            and self.relation == proposition.relation
            and self.qualifiers == proposition.qualifiers
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    propositions: frozenset[Proposition]
    coverage: frozenset[Coverage]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Evidence":
        validate_schema(value, "semantic-evidence.schema.json")
        return cls(
            evidence_id=_normalized_text(value["evidence_id"], "evidence_id"),
            propositions=frozenset(
                Proposition.from_mapping(item) for item in value["propositions"]
            ),
            coverage=frozenset(
                Coverage.from_mapping(item) for item in value["coverage"]
            ),
        )


@dataclass(frozen=True)
class ValidatedSelection:
    """Opaque proof that a selection passed semantic and set-membership checks."""

    selection: Mapping[str, Any]
    evidence_by_id: Mapping[str, Evidence]


def _evidence_index(
    evidence_records: Sequence[Mapping[str, Any]],
) -> dict[str, Evidence]:
    if not evidence_records:
        raise CitationValidationError("supplied evidence set is empty")
    index: dict[str, Evidence] = {}
    for record in evidence_records:
        evidence = Evidence.from_mapping(record)
        if evidence.evidence_id in index:
            raise CitationValidationError(
                f"duplicate supplied evidence id: {evidence.evidence_id}"
            )
        index[evidence.evidence_id] = evidence
    return index


def _selected_evidence(
    evidence_ids: object,
    *,
    index: Mapping[str, Evidence],
    context: str,
) -> tuple[Evidence, ...]:
    if (
        not isinstance(evidence_ids, Sequence)
        or isinstance(evidence_ids, (str, bytes, bytearray))
        or not evidence_ids
    ):
        raise CitationValidationError(f"{context} must cite supplied evidence")
    selected: list[Evidence] = []
    seen: set[str] = set()
    for value in evidence_ids:
        evidence_id = _normalized_text(value, f"{context} evidence id")
        if evidence_id in seen:
            raise CitationValidationError(
                f"{context} repeats evidence id: {evidence_id}"
            )
        if evidence_id not in index:
            raise CitationValidationError(
                f"{context} cites evidence outside supplied set: {evidence_id}"
            )
        seen.add(evidence_id)
        selected.append(index[evidence_id])
    return tuple(selected)


def validate_selection(
    selection: Mapping[str, Any],
    *,
    evidence_records: Sequence[Mapping[str, Any]],
) -> ValidatedSelection:
    """Fail closed unless every claim/refusal is semantically grounded.

    Exact proposition matching deliberately checks polarity, relation and every
    qualifier.  A refusal additionally requires declared evidence coverage for
    the requested proposition and is rejected if that proposition is in fact
    supported by any supplied evidence.
    """

    if not isinstance(selection, Mapping):
        raise CitationValidationError("model selection is not an object")
    validate_schema(selection, "claim-selection.schema.json")
    index = _evidence_index(evidence_records)
    answer_kind = selection["answer_kind"]
    if answer_kind == "answer":
        if "refusal" in selection:
            raise CitationValidationError("answer selection contains a refusal")
        claims = selection["claims"]
        claim_ids: set[str] = set()
        for claim in claims:
            claim_id = _normalized_text(claim["claim_id"], "claim_id")
            if claim_id in claim_ids:
                raise CitationValidationError(f"duplicate claim id: {claim_id}")
            claim_ids.add(claim_id)
            proposition = Proposition.from_mapping(claim)
            cited = _selected_evidence(
                claim["evidence_ids"],
                index=index,
                context=f"claim {claim_id}",
            )
            if not all(
                proposition in evidence.propositions for evidence in cited
            ):
                raise CitationValidationError(
                    f"claim {claim_id} is unsupported by its cited evidence"
                )
    else:
        if selection["claims"]:
            raise CitationValidationError("refusal selection contains claims")
        refusal = selection.get("refusal")
        if not isinstance(refusal, Mapping):
            raise CitationValidationError("refusal selection lacks refusal detail")
        requested = Proposition.from_mapping(refusal["requested_proposition"])
        cited = _selected_evidence(
            refusal["evidence_ids"],
            index=index,
            context="refusal",
        )
        if any(
            requested in evidence.propositions for evidence in index.values()
        ):
            raise CitationValidationError(
                "refusal contradicts a supported supplied proposition"
            )
        if not any(
            coverage.covers(requested)
            for evidence in cited
            for coverage in evidence.coverage
        ):
            raise CitationValidationError(
                "refusal lacks cited evidence coverage for requested proposition"
            )
    return ValidatedSelection(
        selection=json.loads(canonical_bytes(selection)),
        evidence_by_id=index,
    )


def _plain(value: str) -> str:
    return _SPACE.sub(" ", value.replace("[", "(").replace("]", ")").strip())


def _claim_sentence(claim: Mapping[str, Any]) -> str:
    polarity = claim["polarity"]
    if polarity == "affirmed":
        predicate = claim["relation"]
    elif polarity == "negated":
        predicate = f"does not {claim['relation']}"
    else:
        predicate = f"has unknown {claim['relation']}"
    sentence = f"{_plain(claim['subject'])} {_plain(predicate)} {_plain(claim['object'])}"
    qualifiers = claim["qualifiers"]
    if qualifiers:
        rendered = ", ".join(
            f"{_plain(key)}={_plain(value)}"
            for key, value in sorted(qualifiers.items())
        )
        sentence += f" ({rendered})"
    return sentence + "."


def render_selection(validated: ValidatedSelection) -> dict[str, Any]:
    """Render final prose and citation objects without another model call."""

    selection = validated.selection
    citations: list[dict[str, Any]] = []
    markers: dict[str, str] = {}

    def marker_for(evidence_id: str, claim_id: str | None) -> str:
        marker = markers.get(evidence_id)
        if marker is None:
            marker = f"[{len(markers) + 1}]"
            markers[evidence_id] = marker
            citations.append(
                {
                    "citation_id": f"citation-{len(citations) + 1:03d}",
                    "marker": marker,
                    "evidence_id": evidence_id,
                    "claim_ids": [],
                }
            )
        citation = next(item for item in citations if item["marker"] == marker)
        if claim_id is not None and claim_id not in citation["claim_ids"]:
            citation["claim_ids"].append(claim_id)
        return marker

    sentences: list[str] = []
    if selection["answer_kind"] == "answer":
        for claim in selection["claims"]:
            markers_for_claim = " ".join(
                marker_for(evidence_id, claim["claim_id"])
                for evidence_id in claim["evidence_ids"]
            )
            sentences.append(f"{_claim_sentence(claim)} {markers_for_claim}")
    else:
        refusal = selection["refusal"]
        markers_for_refusal = " ".join(
            marker_for(evidence_id, None)
            for evidence_id in refusal["evidence_ids"]
        )
        requested = refusal["requested_proposition"]
        sentences.append(
            "The supplied evidence does not establish that "
            f"{_plain(requested['subject'])} {_plain(requested['relation'])} "
            f"{_plain(requested['object'])}. {markers_for_refusal}"
        )

    result = {
        "schema": RENDERED_SCHEMA,
        "answer_kind": selection["answer_kind"],
        "answer": " ".join(sentences),
        "citations": citations,
        "selection_sha256": digest(selection),
    }
    validate_schema(result, "rendered-answer.schema.json")
    supplied_ids = set(validated.evidence_by_id)
    if any(item["evidence_id"] not in supplied_ids for item in citations):
        raise CitationValidationError("renderer emitted an unsupplied evidence id")
    return result


def validate_and_render(
    selection: Mapping[str, Any],
    *,
    evidence_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Convenience entry point for the production provider boundary."""

    return render_selection(
        validate_selection(selection, evidence_records=evidence_records)
    )
