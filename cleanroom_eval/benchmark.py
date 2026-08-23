"""Deterministic matched-arm analysis for clean-room experiment receipts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .contract import CLASSIFICATION, ContractError, canonical_bytes, digest, load_json, validate_schema
from .runner import load_public_assets, verify_run_receipt


REPORT_SCHEMA = "cleanroom.matched-benchmark/v1"


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ContractError("cannot calculate a quantile from no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _episode_scores(receipt: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Calculate the preregistered primary metric at episode cluster grain."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in receipt["results"]["records"]:
        try:
            key = (str(record["arm_id"]), str(record["episode_id"]))
        except KeyError as exc:
            raise ContractError(f"result record lacks {exc.args[0]}") from exc
        grouped[key].append(record)

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    expected_episodes = set(receipt["episode_ids"])
    for arm in receipt["arms"]:
        arm_id = arm["arm_id"]
        observed = {
            episode_id for candidate_arm, episode_id in grouped
            if candidate_arm == arm_id
        }
        if observed != expected_episodes:
            raise ContractError(
                f"result coverage differs for {arm_id}: "
                f"expected={len(expected_episodes)} actual={len(observed)}"
            )
        for episode_id in sorted(expected_episodes):
            records = grouped[(arm_id, episode_id)]
            success = all(
                record.get("action_match") is True
                and record.get("citation_supported") is True
                and record.get("mutation_applied") is True
                for record in records
            )
            scores[arm_id][episode_id] = float(success)
    return dict(scores)


def _paired_interval(
    *,
    treatment: Mapping[str, float],
    control: Mapping[str, float],
    seed_material: str,
    samples: int,
) -> dict[str, Any]:
    if samples < 200:
        raise ContractError("cluster bootstrap requires at least 200 samples")
    clusters = sorted(set(treatment) & set(control))
    if set(treatment) != set(control) or not clusters:
        raise ContractError("paired contrast does not cover identical episodes")
    differences = [treatment[key] - control[key] for key in clusters]
    effect = sum(differences) / len(differences)
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [differences[generator.randrange(len(differences))] for _ in clusters]
        estimates.append(sum(draw) / len(draw))
    return {
        "effect": effect,
        "ci95_low": _quantile(estimates, 0.025),
        "ci95_high": _quantile(estimates, 0.975),
        "cluster_count": len(clusters),
        "bootstrap_samples": samples,
        "bootstrap_seed_sha256": hashlib.sha256(seed_material.encode()).hexdigest(),
    }


def _parse_contrast(value: str, arm_ids: set[str]) -> tuple[str, str]:
    for treatment in sorted(arm_ids, key=len, reverse=True):
        prefix = f"{treatment}-"
        if value.startswith(prefix) and value[len(prefix):] in arm_ids:
            return treatment, value[len(prefix):]
    raise ContractError(f"invalid preregistered contrast: {value}")


def build_benchmark_report(
    receipt: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any],
    episode_ids: set[str],
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Build a receipt-bound report without promoting mock results to evidence."""

    verify_run_receipt(receipt, experiment=experiment, episode_ids=episode_ids)
    scores = _episode_scores(receipt)
    arms = {
        item["arm_id"]: {
            "episode_success_rate": sum(scores[item["arm_id"]].values()) / len(episode_ids),
            "episodes_succeeded": int(sum(scores[item["arm_id"]].values())),
            "episodes_evaluated": len(episode_ids),
            "action_accuracy": item["metrics"]["action_accuracy"],
            "citation_support_rate": item["metrics"]["citation_support_rate"],
        }
        for item in receipt["arms"]
    }
    arm_ids = set(arms)
    contrasts = []
    for label in experiment["reporting"]["causal_contrasts"]:
        treatment, control = _parse_contrast(label, arm_ids)
        contrasts.append({
            "contrast": label,
            "treatment": treatment,
            "control": control,
            **_paired_interval(
                treatment=scores[treatment],
                control=scores[control],
                seed_material=f"{receipt['receipt_sha256']}:{label}",
                samples=bootstrap_samples,
            ),
        })

    gate_counts = {
        key: receipt["gates"][key]
        for key in (
            "direct_hidden_state_reads", "unsupported_state_mutations",
            "canary_uses", "invariant_bypasses", "unsupported_claims",
        )
    }
    status = (
        "MOCK_HARNESS_ONLY" if receipt["execution_mode"] == "MOCK_ONLY"
        else "REAL_PROVIDER_PENDING_INDEPENDENT_ATTESTATION"
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "classification": CLASSIFICATION,
        "benchmark_status": status,
        "experiment_id": experiment["experiment_id"],
        "source_run_id": receipt["run_id"],
        "source_receipt_sha256": receipt["receipt_sha256"],
        "primary_metric": {
            "name": "held_out_episode_success_under_all_deterministic_gates",
            "unit": "episode",
            "definition": (
                "An episode succeeds only when every event has the expected action, "
                "an allowed evidence citation, an authorized mutation, and the "
                "deterministic final-state replay succeeds."
            ),
            "preregistered": experiment["reporting"]["primary"]
            == "Held-out episode success under all deterministic gates",
        },
        "arms": [{"arm_id": arm_id, **arms[arm_id]} for arm_id in sorted(arms)],
        "causal_contrasts": contrasts,
        "held_out_transfer": {
            "status": "PASS",
            "training_world_count": len(receipt["training_lineage"]["world_ids"]),
            "evaluation_world_count": len(receipt["evaluation_lineage"]["world_ids"]),
            "training_template_family_count": len(receipt["training_lineage"]["template_families"]),
            "evaluation_template_family_count": len(receipt["evaluation_lineage"]["template_families"]),
        },
        "reward_hacking": {
            "status": "PASS" if not any(gate_counts.values()) else "FAIL",
            "gate_counts": gate_counts,
        },
        "contamination": {
            "status": receipt["gates"]["contamination_status"],
            "overlap_ratio": receipt["results"]["contamination"]["overlap_ratio"],
            "sealed_commitment_sha256": receipt["results"]["contamination"]["sealed_commitment_sha256"],
        },
        "interpretation_guard": (
            "Mock runs validate orchestration only and are not evidence of model value. "
            "Real-provider results remain non-releasable until independent attestation "
            "verifies the hash-enumerated package."
        ),
    }
    report["report_sha256"] = digest(report)
    validate_schema(report, "benchmark-report.schema.json")
    return report


def verify_benchmark_report(
    report: Mapping[str, Any], *, receipt: Mapping[str, Any],
    experiment: Mapping[str, Any], episode_ids: set[str],
) -> None:
    validate_schema(report, "benchmark-report.schema.json")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if report["report_sha256"] != digest(unsigned):
        raise ContractError("benchmark report digest differs")
    expected = build_benchmark_report(
        receipt, experiment=experiment, episode_ids=episode_ids,
        bootstrap_samples=report["causal_contrasts"][0]["bootstrap_samples"],
    )
    if canonical_bytes(report) != canonical_bytes(expected):
        raise ContractError("benchmark report differs from source receipt")


def render_benchmark_card(report: Mapping[str, Any]) -> str:
    """Render a compact deterministic Markdown card without inventing claims."""

    validate_schema(report, "benchmark-report.schema.json")
    lines = [
        "# Clean-room matched benchmark card", "",
        f"- Status: `{report['benchmark_status']}`",
        f"- Experiment: `{report['experiment_id']}`",
        f"- Source receipt: `{report['source_receipt_sha256']}`",
        f"- Primary metric: {report['primary_metric']['name']}", "",
        "## Arm results", "",
        "| Arm | Episodes passed | Success rate | Action accuracy | Citation support |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        lines.append(
            "| {arm_id} | {episodes_succeeded}/{episodes_evaluated} | "
            "{episode_success_rate:.3f} | {action_accuracy:.3f} | "
            "{citation_support_rate:.3f} |".format(**arm)
        )
    lines += ["", "## Preregistered contrasts", "",
              "| Contrast | Effect | 95% cluster-bootstrap interval | Clusters |",
              "|---|---:|---:|---:|"]
    for contrast in report["causal_contrasts"]:
        lines.append(
            "| {contrast} | {effect:+.3f} | [{ci95_low:+.3f}, "
            "{ci95_high:+.3f}] | {cluster_count} |".format(**contrast)
        )
    lines += ["", "## Integrity gates", "",
              f"- Held-out transfer: `{report['held_out_transfer']['status']}`",
              f"- Reward hacking: `{report['reward_hacking']['status']}`",
              f"- Contamination: `{report['contamination']['status']}`", "",
              report["interpretation_guard"], ""]
    return "\n".join(lines)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    _, experiment, episodes = load_public_assets()
    receipt = load_json(args.receipt)
    report = build_benchmark_report(
        receipt, experiment=experiment,
        episode_ids={item["episode_id"] for item in episodes},
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.card:
        args.card.write_text(render_benchmark_card(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
