"""Fail-closed contract checks for the clean-room evaluation track.

The module deliberately has no dependency on the corpus generator.  A
generator may emit any artefact formats it needs, but its evaluation export
must satisfy the versioned schemas and invariants in this package.
"""

from __future__ import annotations

import argparse
import base64
import copy
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import jsonschema
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


CLASSIFICATION = "CLEANROOM_SYNTHETIC"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_DIR = Path(__file__).with_name("schemas")
ASSET_DIR = Path(__file__).with_name("assets")
EPISODE_DIR = ASSET_DIR / "episodes"


class ContractError(ValueError):
    """Raised when an asset or result violates the public contract."""


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    status: str
    checks: tuple[str, ...]
    final_state_sha256: str


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load JSON asset {path}: {exc}") from exc


def _load_schema(name: str) -> dict[str, Any]:
    value = load_json(SCHEMA_DIR / name)
    if not isinstance(value, dict):
        raise ContractError(f"schema {name} is not an object")
    return value


def validate_schema(value: object, schema_name: str) -> None:
    try:
        jsonschema.Draft202012Validator(_load_schema(schema_name)).validate(value)
    except jsonschema.ValidationError as exc:
        location = "/".join(str(item) for item in exc.absolute_path) or "<root>"
        raise ContractError(
            f"{schema_name} validation failed at {location}: {exc.message}"
        ) from exc


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"invalid timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"timestamp lacks timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _unique(values: Iterable[str], what: str) -> None:
    items = list(values)
    duplicates = sorted(key for key, count in Counter(items).items() if count > 1)
    if duplicates:
        raise ContractError(f"duplicate {what}: {', '.join(duplicates)}")


def verify_taxonomy(taxonomy: Mapping[str, Any]) -> None:
    validate_schema(taxonomy, "taxonomy.schema.json")
    if taxonomy["classification"] != CLASSIFICATION:
        raise ContractError("taxonomy is not clean-room synthetic")
    competencies = taxonomy["competencies"]
    _unique((item["id"] for item in competencies), "competency ids")
    for item in competencies:
        if item["id"].lower() != item["id"]:
            raise ContractError(f"competency id is not stable lowercase: {item['id']}")
        if len(set(item["observable_outcomes"])) != len(item["observable_outcomes"]):
            raise ContractError(f"duplicate outcome in {item['id']}")


def _entity_index(episode: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entities = episode["entities"]
    _unique((item["id"] for item in entities), "entity ids")
    return {item["id"]: item for item in entities}


def _state_index(
    records: Sequence[Mapping[str, Any]], what: str
) -> dict[str, dict[str, Any]]:
    _unique((item["object_id"] for item in records), f"{what} object ids")
    return {
        item["object_id"]: {
            "state": item["state"],
            "version": item["version"],
            "facts": item.get("facts", {}),
        }
        for item in records
    }


def _replace_pointer(document: object, path: str, value: object) -> None:
    """Apply the deliberately small replace-only mutation contract."""

    parts = path.removeprefix("/").split("/")
    current = document
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ContractError(f"adversarial patch path does not resolve: {path}")
    leaf = parts[-1]
    if isinstance(current, list):
        current[int(leaf)] = value
    elif isinstance(current, dict) and leaf in current:
        current[leaf] = value
    else:
        raise ContractError(f"adversarial patch path does not resolve: {path}")

def evaluate_episode(
    episode: Mapping[str, Any],
    *,
    competency_ids: set[str],
    surface_ids: set[str],
    verify_adversarial: bool = True,
) -> EpisodeResult:
    """Replay expected mutations and enforce deterministic episode invariants."""

    validate_schema(episode, "episode.schema.json")
    if episode["classification"] != CLASSIFICATION:
        raise ContractError("episode is not clean-room synthetic")
    unknown_competencies = set(episode["competencies"]) - competency_ids
    if unknown_competencies:
        raise ContractError(
            f"unknown competencies in {episode['episode_id']}: "
            f"{sorted(unknown_competencies)}"
        )
    unknown_surfaces = set(episode["tool_surfaces"]) - surface_ids
    if unknown_surfaces:
        raise ContractError(
            f"unknown tool surfaces in {episode['episode_id']}: "
            f"{sorted(unknown_surfaces)}"
        )

    start = _parse_time(episode["time_window"]["start"])
    end = _parse_time(episode["time_window"]["end"])
    if end <= start or (end - start).days < 7:
        raise ContractError(
            f"{episode['episode_id']} is not a long-horizon episode"
        )

    entities = _entity_index(episode)
    states = _state_index(episode["initial_state"], "initial state")
    final_expected = _state_index(episode["final_state"], "final state")
    if set(states) - set(entities) or set(final_expected) != set(states):
        raise ContractError(
            f"state/entity coverage differs for {episode['episode_id']}"
        )
    events = episode["events"]
    _unique((event["event_id"] for event in events), "event ids")
    _unique(
        (
            event["expected_receipt"]
            for event in events
            if not event.get("duplicate_of")
        ),
        "primary event receipts",
    )

    episode_evidence = {
        evidence_id
        for event in events
        for evidence_id in event["evidence_refs"]
    }
    _unique(
        (item["challenge_id"] for item in episode["evidence_challenges"]),
        "evidence challenge ids",
    )
    for challenge in episode["evidence_challenges"]:
        required = set(challenge["required_evidence_refs"])
        supplied = set(challenge["supplied_evidence_refs"])
        disposition = challenge["expected_disposition"]
        if not supplied <= episode_evidence:
            raise ContractError(
                f"evidence challenge references unsupplied episode evidence: "
                f"{challenge['challenge_id']}"
            )
        if disposition == "SUPPORTED" and not required <= supplied:
            raise ContractError(
                f"supported challenge lacks required evidence: "
                f"{challenge['challenge_id']}"
            )
        if disposition == "REFUSE_UNSUPPORTED" and required <= supplied:
            raise ContractError(
                f"refusal challenge is actually supported: "
                f"{challenge['challenge_id']}"
            )

    previous_at = start
    previous_sequence = 0
    request_receipts: dict[str, str] = {}
    event_by_id: dict[str, Mapping[str, Any]] = {}
    checks: list[str] = []
    for event in events:
        at = _parse_time(event["at"])
        if (
            event["sequence"] != previous_sequence + 1
            or at < previous_at
            or at < start
            or at > end
        ):
            raise ContractError(
                f"non-monotonic sequence/time at {episode['episode_id']}/"
                f"{event['event_id']}"
            )
        previous_sequence = event["sequence"]
        previous_at = at
        event_by_id[event["event_id"]] = event

        if event["surface"] not in surface_ids:
            raise ContractError(f"unknown surface in event {event['event_id']}")
        actor = entities.get(event["actor_id"])
        if actor is None:
            raise ContractError(f"unknown actor in event {event['event_id']}")
        permitted_roles = episode["authorization"].get(event["action"], [])
        if actor.get("role") not in permitted_roles:
            raise ContractError(
                f"unauthorized action {event['action']} by {event['actor_id']}"
            )

        request_id = event["request_id"]
        duplicate_of = event.get("duplicate_of")
        if duplicate_of:
            primary = event_by_id.get(duplicate_of)
            if (
                primary is None
                or primary["request_id"] != request_id
                or primary["expected_receipt"] != event["expected_receipt"]
                or event["mutations"]
            ):
                raise ContractError(
                    f"idempotency violation at {episode['episode_id']}/"
                    f"{event['event_id']}"
                )
        elif request_id in request_receipts:
            raise ContractError(f"request id reused without duplicate_of: {request_id}")
        else:
            request_receipts[request_id] = event["expected_receipt"]

        currency_totals: defaultdict[str, int] = defaultdict(int)
        for entry in event.get("ledger_entries", []):
            if entry["account_id"] not in entities:
                raise ContractError(
                    f"ledger entry references unknown object {entry['account_id']}"
                )
            currency_totals[entry["currency"]] += entry["amount_minor"]
        if any(currency_totals.values()):
            raise ContractError(
                f"unbalanced ledger entries in {episode['episode_id']}/"
                f"{event['event_id']}"
            )

        for mutation in event["mutations"]:
            object_id = mutation["object_id"]
            current = states.get(object_id)
            if current is None:
                raise ContractError(f"mutation references unknown object {object_id}")
            if (
                current["state"] != mutation["from_state"]
                or current["version"] != mutation["from_version"]
                or mutation["to_version"] != mutation["from_version"] + 1
            ):
                raise ContractError(
                    f"invalid mutation precondition/version for {object_id}"
                )
            current["state"] = mutation["to_state"]
            current["version"] = mutation["to_version"]
        if not event["evidence_refs"]:
            raise ContractError(f"event lacks evidence refs: {event['event_id']}")

    checks.extend(
        (
            "chronology",
            "referential_integrity",
            "authorization",
            "version_monotonicity",
            "idempotency",
            "ledger_conservation",
            "evidence_presence",
        )
    )
    if states != final_expected:
        raise ContractError(
            f"replayed final state differs for {episode['episode_id']}: "
            f"expected={digest(final_expected)} actual={digest(states)}"
        )
    checks.append("final_state")
    checks.append("evidence_sufficiency")
    if verify_adversarial:
        _unique(
            (item["mutation_id"] for item in episode["adversarial_mutations"]),
            "adversarial mutation ids",
        )
        for mutation in episode["adversarial_mutations"]:
            mutated = copy.deepcopy(episode)
            patch = mutation["patch"]
            _replace_pointer(mutated, patch["path"], patch["value"])
            try:
                evaluate_episode(
                    mutated,
                    competency_ids=competency_ids,
                    surface_ids=surface_ids,
                    verify_adversarial=False,
                )
            except ContractError as exc:
                if mutation["expected_error"] not in str(exc):
                    raise ContractError(
                        f"adversarial mutation {mutation['mutation_id']} failed "
                        f"with unexpected error: {exc}"
                    ) from exc
            else:
                raise ContractError(
                    f"adversarial mutation was accepted: {mutation['mutation_id']}"
                )
        checks.append("adversarial_mutations")
    return EpisodeResult(
        episode_id=episode["episode_id"],
        status="PASS",
        checks=tuple(checks),
        final_state_sha256=digest(states),
    )


def _normalised_tokens(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="strict").lower()
    return re.findall(r"[a-z0-9]+", text)


def _shingles(path: Path, width: int) -> set[str]:
    tokens = _normalised_tokens(path)
    if len(tokens) < width:
        return {hashlib.sha256(" ".join(tokens).encode()).hexdigest()} if tokens else set()
    return {
        hashlib.sha256(" ".join(tokens[index : index + width]).encode()).hexdigest()
        for index in range(len(tokens) - width + 1)
    }


def verify_contamination(
    *,
    candidate_paths: Sequence[Path],
    sealed_paths: Sequence[Path],
    shingle_width: int = 13,
    maximum_overlap_ratio: float = 0.0,
) -> dict[str, Any]:
    """Compare candidate training material with the sealed evaluation assets."""

    if shingle_width < 5:
        raise ContractError("contamination shingle width must be at least 5")
    sealed: set[str] = set()
    for path in sealed_paths:
        sealed.update(_shingles(path, shingle_width))
    candidate: set[str] = set()
    for path in candidate_paths:
        candidate.update(_shingles(path, shingle_width))
    overlap = candidate & sealed
    denominator = max(1, len(sealed))
    ratio = len(overlap) / denominator
    status = "PASS" if ratio <= maximum_overlap_ratio else "FAIL"
    return {
        "schema": "cleanroom.contamination-report/v1",
        "status": status,
        "shingle_width": shingle_width,
        "sealed_shingles": len(sealed),
        "candidate_shingles": len(candidate),
        "overlap_shingles": len(overlap),
        "overlap_ratio": ratio,
        "maximum_overlap_ratio": maximum_overlap_ratio,
        "candidate_files": [
            {"path": path.name, "sha256": file_sha256(path)}
            for path in candidate_paths
        ],
        "sealed_commitment_sha256": digest(sorted(sealed)),
    }


def verify_experiment_run(
    run: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any],
    episode_ids: set[str],
) -> None:
    """Verify parity, transfer and reward-hacking gates for an experiment run."""

    validate_schema(run, "run.schema.json")
    if run["experiment_id"] != experiment["experiment_id"]:
        raise ContractError("run references the wrong experiment")
    required_arms = {item["id"] for item in experiment["arms"]}
    observed_arms = {item["arm_id"] for item in run["arms"]}
    if observed_arms != required_arms:
        raise ContractError(
            f"arm coverage differs: expected={sorted(required_arms)} "
            f"actual={sorted(observed_arms)}"
        )
    if set(run["episode_ids"]) != episode_ids:
        raise ContractError("run did not evaluate the entire sealed episode set")

    base_commitments = {item["base_model_sha256"] for item in run["arms"]}
    evaluator_commitments = {item["evaluator_sha256"] for item in run["arms"]}
    if len(base_commitments) != 1 or len(evaluator_commitments) != 1:
        raise ContractError("arms are not matched on base model and evaluator")

    by_id = {item["arm_id"]: item for item in run["arms"]}
    for pair in experiment["matched_pairs"]:
        treatment = by_id[pair["treatment"]]
        control = by_id[pair["control"]]
        for field in pair["equal_fields"]:
            if treatment[field] != control[field]:
                raise ContractError(
                    f"matched pair {pair['treatment']}/{pair['control']} "
                    f"differs on {field}"
                )

    seen_worlds = set(run["training_lineage"]["world_ids"])
    seen_templates = set(run["training_lineage"]["template_families"])
    if seen_worlds & set(run["evaluation_lineage"]["world_ids"]):
        raise ContractError("held-out transfer world leakage")
    if seen_templates & set(run["evaluation_lineage"]["template_families"]):
        raise ContractError("held-out transfer template-family leakage")

    gates = run["gates"]
    exact_zero = (
        "direct_hidden_state_reads",
        "unsupported_state_mutations",
        "canary_uses",
        "invariant_bypasses",
        "unsupported_claims",
    )
    failures = [name for name in exact_zero if gates.get(name) != 0]
    if failures:
        raise ContractError(f"reward-hacking gates failed: {', '.join(failures)}")
    if gates["contamination_status"] != "PASS":
        raise ContractError("contamination gate failed")


def build_release_manifest(
    *,
    root: Path,
    relative_paths: Sequence[str],
    release_id: str,
    repository_commit: str,
) -> dict[str, Any]:
    """Build a deterministic, path-safe hash enumeration of release assets."""

    if not re.fullmatch(r"[0-9a-f]{40}", repository_commit):
        raise ContractError("repository_commit must be a full SHA-1 revision")
    assets: list[dict[str, Any]] = []
    root = root.resolve()
    for relative in sorted(set(relative_paths)):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"asset escapes release root: {relative}") from exc
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"release asset is missing or symlinked: {relative}")
        assets.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "schema": "cleanroom.release-manifest/v1",
        "release_id": release_id,
        "classification": CLASSIFICATION,
        "repository_commit": repository_commit,
        "assets": assets,
        "asset_count": len(assets),
        "assets_sha256": digest(assets),
    }
    manifest["manifest_payload_sha256"] = digest(manifest)
    return manifest


def verify_release_manifest(manifest: Mapping[str, Any], *, root: Path) -> None:
    validate_schema(manifest, "release-manifest.schema.json")
    if manifest["classification"] != CLASSIFICATION:
        raise ContractError("release manifest is not clean-room synthetic")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    if manifest["manifest_payload_sha256"] != digest(unsigned):
        raise ContractError("release manifest payload digest differs")
    if manifest["asset_count"] != len(manifest["assets"]):
        raise ContractError("release asset count differs")
    if manifest["assets_sha256"] != digest(manifest["assets"]):
        raise ContractError("release asset-list digest differs")
    _unique((item["path"] for item in manifest["assets"]), "release asset paths")
    root = root.resolve()
    for asset in manifest["assets"]:
        path = (root / asset["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"release asset escapes root: {asset['path']}") from exc
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != asset["bytes"]
            or file_sha256(path) != asset["sha256"]
        ):
            raise ContractError(f"release asset differs: {asset['path']}")


def verify_sealed_set_manifest(manifest: Mapping[str, Any], *, root: Path) -> None:
    """Rehash the public preregistered fixture set against its frozen manifest."""

    validate_schema(manifest, "sealed-set.schema.json")
    _unique((item["path"] for item in manifest["assets"]), "sealed asset paths")
    lines: list[str] = []
    for asset in sorted(manifest["assets"], key=lambda item: item["path"]):
        path = (root / asset["path"]).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ContractError(f"sealed asset escapes root: {asset['path']}") from exc
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != asset["bytes"]
            or file_sha256(path) != asset["sha256"]
        ):
            raise ContractError(f"sealed asset differs: {asset['path']}")
        lines.append(
            f"{asset['sha256']}  cleanroom_eval/assets/{asset['path']}\n"
        )
    combined = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    if manifest["asset_count"] != len(manifest["assets"]):
        raise ContractError("sealed asset count differs")
    if manifest["asset_list_sha256"] != combined:
        raise ContractError("sealed asset-list commitment differs")


def verify_attestation(
    attestation: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    trusted_key_id: str,
    trusted_public_key_b64: str,
) -> None:
    """Verify an independent Ed25519 signature over the release commitment."""

    validate_schema(attestation, "attestation.schema.json")
    if attestation["signing_key_id"] != trusted_key_id:
        raise ContractError("attestation signer is not the independent trust anchor")
    if attestation["manifest_payload_sha256"] != manifest["manifest_payload_sha256"]:
        raise ContractError("attestation is not bound to this manifest")
    if (
        attestation["release_id"] != manifest["release_id"]
        or attestation["reviewed_asset_count"] != manifest["asset_count"]
    ):
        raise ContractError("attestation scope differs from the release manifest")
    signed = {key: value for key, value in attestation.items() if key != "signature"}
    try:
        public_key = base64.b64decode(trusted_public_key_b64, validate=True)
        signature = base64.b64decode(attestation["signature"], validate=True)
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature, canonical_bytes(signed)
        )
    except (ValueError, InvalidSignature) as exc:
        raise ContractError("independent attestation signature is invalid") from exc


def verify_scenario_partitions(
    partition: Mapping[str, Any],
    *,
    episodes: Sequence[Mapping[str, Any]],
) -> None:
    """Enforce scenario-level train/test disjointness across every locked field."""

    validate_schema(partition, "scenario-partitions.schema.json")
    fields = partition["disjoint_fields"]
    training = partition["training_lineage"]
    for field in fields:
        training_values = {item[field] for item in training}
        if field == "world_id":
            sealed_values = {item["world_id"] for item in episodes}
        elif field == "template_family":
            sealed_values = {item["template_family"] for item in episodes}
        else:
            sealed_values = {item["scenario_lineage"][field] for item in episodes}
        overlap = training_values & sealed_values
        if overlap:
            raise ContractError(
                f"scenario partition leakage in {field}: {sorted(overlap)}"
            )
    for episode in episodes:
        if episode["partition"] != partition["sealed_partition"]:
            raise ContractError(
                f"episode is not in the sealed partition: {episode['episode_id']}"
            )


def verify_bundle(root: Path = ASSET_DIR) -> dict[str, Any]:
    taxonomy = load_json(root / "competency-taxonomy.v1.json")
    experiment = load_json(root / "experiment.v1.json")
    sealed_set = load_json(root / "sealed-set.manifest.v1.json")
    preregistration = load_json(root / "preregistration.v1.json")
    partitions = load_json(root / "scenario-partitions.v1.json")
    verify_sealed_set_manifest(sealed_set, root=root)
    verify_taxonomy(taxonomy)
    validate_schema(preregistration, "preregistration.schema.json")
    if preregistration["episode_set_id"] != sealed_set["set_id"]:
        raise ContractError("preregistration references the wrong sealed set")
    validate_schema(experiment, "experiment.schema.json")
    if experiment["classification"] != CLASSIFICATION:
        raise ContractError("experiment is not clean-room synthetic")
    arm_ids = [item["id"] for item in experiment["arms"]]
    _unique(arm_ids, "experiment arm ids")
    if set(arm_ids) != {
        "BASE",
        "SFT",
        "SFT_MATCHED_CONTROL",
        "RL",
        "RL_MATCHED_CONTROL",
    }:
        raise ContractError("experiment arm set is incomplete")
    competency_ids = {item["id"] for item in taxonomy["competencies"]}
    surface_ids = {item["id"] for item in taxonomy["tool_surfaces"]}
    episode_paths = sorted((root / "episodes").glob("*.json"))
    if not episode_paths:
        raise ContractError("sealed episode set is empty")
    episodes = [load_json(path) for path in episode_paths]
    verify_scenario_partitions(partitions, episodes=episodes)
    _unique((item["episode_id"] for item in episodes), "episode ids")
    results = [
        evaluate_episode(
            episode,
            competency_ids=competency_ids,
            surface_ids=surface_ids,
        )
        for episode in episodes
    ]
    expected_ids = set(experiment["sealed_episode_ids"])
    observed_ids = {result.episode_id for result in results}
    if expected_ids != observed_ids:
        raise ContractError("experiment sealed episode set differs from assets")
    denominator = preregistration["primary_metric"]["denominator"]
    if denominator != f"all {len(results)} sealed test episodes":
        raise ContractError("preregistered denominator differs from sealed assets")
    return {
        "schema": "cleanroom.bundle-verification/v1",
        "status": "PASS",
        "taxonomy_version": taxonomy["version"],
        "experiment_id": experiment["experiment_id"],
        "sealed_set_sha256": file_sha256(root / "sealed-set.manifest.v1.json"),
        "episode_count": len(results),
        "episodes": [
            {
                "episode_id": result.episode_id,
                "status": result.status,
                "checks": list(result.checks),
                "final_state_sha256": result.final_state_sha256,
            }
            for result in results
        ],
    }


def set_variation(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure how much an episode set actually varies (#115 acceptance)."""

    def commitment(value: object) -> str:
        return digest(value)

    authorization = {commitment(e["authorization"]) for e in episodes}
    mutation_blocks = {
        commitment(sorted(m["expected_error"] for m in e["adversarial_mutations"]))
        for e in episodes
    }
    trap_blocks = {
        commitment(sorted(t["detector"] for t in e["reward_traps"]))
        for e in episodes
    }
    action_graphs = {
        commitment([(ev["surface"], ev["action"]) for ev in e["events"]]) for e in episodes
    }
    surface_counts = sorted({len(e["tool_surfaces"]) for e in episodes})
    step_counts = sorted({len(e["events"]) for e in episodes})
    role_counts = sorted({len({x["role"] for x in e["entities"] if x.get("role")}) for e in episodes})
    return {
        "episode_count": len(episodes),
        "distinct_authorization_maps": len(authorization),
        "distinct_adversarial_mutation_blocks": len(mutation_blocks),
        "distinct_reward_trap_blocks": len(trap_blocks),
        "distinct_action_graphs": len(action_graphs),
        "tool_surface_counts": surface_counts,
        "event_counts": step_counts,
        "role_counts": role_counts,
    }


def verify_episode_set(
    root: Path = ASSET_DIR,
    *,
    episode_dir: str = "episodes_v2",
    manifest_name: str = "sealed-set.manifest.v2.json",
) -> dict[str, Any]:
    """Verify a sealed episode set that is not bound to the v1 preregistration.

    Applies the sealed-set manifest rehash, taxonomy, scenario partitions and
    every per-episode deterministic check; reports the set's measured
    variation alongside the per-episode results.
    """

    taxonomy = load_json(root / "competency-taxonomy.v1.json")
    partitions = load_json(root / "scenario-partitions.v1.json")
    manifest = load_json(root / manifest_name)
    verify_sealed_set_manifest(manifest, root=root)
    verify_taxonomy(taxonomy)
    competency_ids = {item["id"] for item in taxonomy["competencies"]}
    surface_ids = {item["id"] for item in taxonomy["tool_surfaces"]}
    episode_paths = [
        path
        for path in sorted((root / episode_dir).glob("*.json"))
        if not path.name.endswith(".task.v1.json")
    ]
    if not episode_paths:
        raise ContractError(f"sealed episode set is empty: {episode_dir}")
    manifest_paths = {item["path"] for item in manifest["assets"]}
    for path in episode_paths:
        if path.relative_to(root).as_posix() not in manifest_paths:
            raise ContractError(f"episode is not enumerated by the manifest: {path.name}")
    episodes = [load_json(path) for path in episode_paths]
    verify_scenario_partitions(partitions, episodes=episodes)
    _unique((item["episode_id"] for item in episodes), "episode ids")
    results = [
        evaluate_episode(episode, competency_ids=competency_ids, surface_ids=surface_ids)
        for episode in episodes
    ]
    from .episode_contract import verify_contracts  # local import: avoids a cycle

    contracts = verify_contracts(root / episode_dir)
    return {
        "schema": "cleanroom.episode-set-verification/v1",
        "contracts": contracts,
        "status": "PASS",
        "set_id": manifest["set_id"],
        "sealed_set_sha256": file_sha256(root / manifest_name),
        "episode_count": len(results),
        "variation": set_variation(episodes),
        "episodes": [
            {
                "episode_id": result.episode_id,
                "status": result.status,
                "checks": list(result.checks),
                "final_state_sha256": result.final_state_sha256,
            }
            for result in results
        ],
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("verify-bundle", "verify-release", "verify-set"),
    )
    parser.add_argument("--root", type=Path, default=ASSET_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--episode-dir", default="episodes_v2")
    parser.add_argument("--set-manifest", default="sealed-set.manifest.v2.json")
    args = parser.parse_args()
    if args.command == "verify-bundle":
        result = verify_bundle(args.root)
    elif args.command == "verify-set":
        result = verify_episode_set(
            args.root, episode_dir=args.episode_dir, manifest_name=args.set_manifest
        )
    else:
        if args.manifest is None:
            parser.error("--manifest is required for verify-release")
        manifest = load_json(args.manifest)
        verify_release_manifest(manifest, root=args.root)
        result = {"status": "PASS", "manifest": str(args.manifest)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
