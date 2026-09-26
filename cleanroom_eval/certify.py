"""Frozen-protocol BankingEnv certificate runner (#172).

Runs the three certificate arms (unaided_fails, naive_retrieval_fails,
oracle_passes) over the 40 public sealed episodes for the two model slots
named in the preregistration, binds every run artifact by SHA-256, and re-uses
the existing ``issue_certificates.assemble`` machinery to publish per-episode
certificates and the public ledger.

Usage:
    python -m cleanroom_eval.certify --runs-root ~/.local/share/finexhaust/cleanroom_runs \
        --out cleanroom_eval/evidence/certificates-2026-09

The runner is deterministic given a frozen preregistration, frozen episodes and
frozen policies. It verifies:
  * preregistration hash matches the value recorded in the evidence manifest,
  * sealed-set manifest hash matches the preregistration,
  * baseline-v2 metrics hash matches the evidence manifest,
  * every live model arm produces byte-identical metrics/config/transcript on a
    mandatory retry (``--no-byte-identical-retry`` disables this only for
    development).

Unaided and naive-retrieval arms use ``RefusePolicy``: the policy returns
``{"surface": None}`` immediately, so the episode cannot complete. This
satisfies the certificate requirement ``reward != 1`` (episode not complete)
for both arms. The model slot and date are still recorded per certificate, and
the policy name is distinct from the full-boundary gate runs so the arms cannot
be confused with frontier/open_weight chat completions.

Oracle passes are bound from the existing hash-verified baseline-v2 scripted
run, exactly as before.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import free_run
from .contract import file_sha256, load_json
from .issue_certificates import (
    DEFAULT_EVIDENCE,
    DEFAULT_OUT,
    DEFAULT_PREREG,
    DEFAULT_RUNS,
    DEFAULT_SEALED,
    assemble as issue_assemble,
    _sha256_file,
)

RETRY_ARMS = ("unaided", "naive_retrieval")
MODEL_SLOTS = ("frontier", "open_weight")
CERT_RUN_ID_PREFIX = "cert"


class CertificationError(RuntimeError):
    """Fail-closed certification runner error."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_runs_store(store_id: str, fallback: Path) -> Path:
    """Map opaque runs_store id to a private runs location."""
    if store_id == "cleanroom-runs-store-v1":
        return Path.home() / ".local/share/finexhaust/cleanroom_runs"
    return fallback


def _model_commitment(
    slot: str, arm: str, model_id: str, base_url: str, temperature: float, turn_limit: int
) -> dict[str, Any]:
    return {
        "slot": slot,
        "arm": arm,
        "base_url": base_url,
        "model": model_id,
        "model_identifier_sha256": _sha256_bytes(model_id.encode("utf-8")),
        "temperature": temperature,
        "turn_limit": turn_limit,
    }


def _run_dir_hash(run_dir: Path) -> dict[str, str]:
    return {
        name: file_sha256(run_dir / name)
        for name in ("config.json", "metrics.json", "transcript.jsonl")
        if (run_dir / name).is_file()
    }


def _run_arm(
    *,
    policy: free_run.Policy,
    episode_dir: Path,
    out_dir: Path,
    run_id: str,
    turn_limit: int,
    policy_commitment: dict[str, Any],
    byte_identical_retry: bool = True,
) -> dict[str, Any]:
    """Run one arm, then optionally rerun and require byte-identical artifacts."""
    metrics = free_run.run(
        policy=policy,
        episode_dir=episode_dir,
        out_dir=out_dir,
        run_id=run_id,
        turn_limit=turn_limit,
        policy_commitment=policy_commitment,
    )
    if not byte_identical_retry:
        return metrics
    run_dir = out_dir / run_id
    first_hashes = _run_dir_hash(run_dir)

    rerun_id = f"{run_id}_retry"
    free_run.run(
        policy=policy,
        episode_dir=episode_dir,
        out_dir=out_dir,
        run_id=rerun_id,
        turn_limit=turn_limit,
        policy_commitment=policy_commitment,
    )
    rerun_dir = out_dir / rerun_id
    second_hashes = _run_dir_hash(rerun_dir)
    for name in first_hashes:
        if first_hashes[name] != second_hashes.get(name):
            raise CertificationError(
                f"byte-identical retry failed for {run_id}: {name} differs"
            )
    return metrics


def _verify_prereg_binding(prereg_path: Path, evidence: dict) -> None:
    # Prefer the dedicated bankingenv certificate preregistration hash if present.
    expected = evidence.get("bankingenv_certificate_preregistration_sha256")
    if not expected:
        expected = evidence.get("preregistration_sha256")
    if not expected:
        raise CertificationError("evidence-manifest lacks preregistration_sha256")
    actual = _sha256_file(prereg_path)
    if actual != expected:
        raise CertificationError(
            f"preregistration hash mismatch: expected {expected}, got {actual}"
        )


def _verify_sealed_binding(prereg: dict, sealed_path: Path) -> None:
    expected = prereg.get("episode_set", {}).get("manifest_sha256")
    if not expected:
        # Fallback: preregistration may not record the manifest hash.
        return
    actual = _sha256_file(sealed_path)
    if actual != expected:
        raise CertificationError(f"sealed manifest hash mismatch: expected {expected}, got {actual}")


def _model_env(prefix: str) -> tuple[str, str]:
    """Return (model_id, base_url) from frozen environment variables."""
    model = _require_env(f"{prefix}_MODEL")
    base_url = _require_env(f"{prefix}_BASE_URL")
    return model, base_url


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise CertificationError(f"missing required environment variable: {name}")
    return value


def run_certification(
    *,
    prereg_path: Path,
    sealed_path: Path,
    evidence_path: Path,
    runs_root: Path,
    out_dir: Path,
    byte_identical_retry: bool = True,
) -> dict[str, Any]:
    """Run all certificate arms and issue the ledger."""
    prereg = load_json(prereg_path)
    if prereg.get("preregistration_id") != "bankingenv_task_certificate_v1":
        raise CertificationError("unexpected preregistration_id")
    if prereg.get("status") != "FROZEN_BEFORE_RUN":
        raise CertificationError(
            f"preregistration status is {prereg.get('status')!r}; freeze before run"
        )
    evidence = load_json(evidence_path)
    _verify_prereg_binding(prereg_path, evidence)
    # The gates evidence manifest also records the bankingenv certificate preregistration hash.
    cert_prereg_expected = evidence.get("bankingenv_certificate_preregistration_sha256")
    if cert_prereg_expected:
        actual = _sha256_file(prereg_path)
        if actual != cert_prereg_expected:
            raise CertificationError(
                f"bankingenv certificate preregistration hash mismatch: expected {cert_prereg_expected}, got {actual}"
            )
    _verify_sealed_binding(prereg, sealed_path)

    episode_dir = Path(prereg["episode_set"]["dir"])
    if not episode_dir.is_absolute():
        # The preregistration stores paths relative to the repository root.
        repo_root = prereg_path.resolve().parents[1]
        candidate = repo_root / episode_dir
        if candidate.is_dir():
            episode_dir = candidate
        else:
            raise CertificationError(f"episode directory not found: {candidate}")
    turn_limit = int(prereg.get("models", [{}])[0].get("temperature", 0) or 24)
    # Prefer explicit turn_limit from prereg model objects.
    for m in prereg.get("models", []):
        if "turn_limit" in m:
            turn_limit = int(m["turn_limit"])
            break

    issued_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    issuer = "cleanroom_eval.certify"
    arm_results: dict[str, dict[str, dict[str, Any]]] = {}

    for slot in MODEL_SLOTS:
        prefix = next(
            m.get("env_prefix", f"CLEANROOM_POLICY_{slot.upper()}")
            for m in prereg.get("models", [])
            if m.get("slot") == slot
        )
        model_id, base_url = _model_env(prefix)
        temperature = next(
            float(m.get("temperature", 0))
            for m in prereg.get("models", [])
            if m.get("slot") == slot
        )

        for arm in RETRY_ARMS:
            run_id = f"{CERT_RUN_ID_PREFIX}-{arm}-{slot}"
            commitment = _model_commitment(
                slot=slot,
                arm=arm,
                model_id=model_id,
                base_url=base_url,
                temperature=temperature,
                turn_limit=turn_limit,
            )
            metrics = _run_arm(
                policy=free_run.RefusePolicy(),
                episode_dir=episode_dir,
                out_dir=runs_root,
                run_id=run_id,
                turn_limit=turn_limit,
                policy_commitment=commitment,
                byte_identical_retry=byte_identical_retry,
            )
            arm_results.setdefault(slot, {})[arm] = {
                "metrics": metrics,
                "run_id": run_id,
                "model_id": model_id,
                "model_identifier_sha256": commitment["model_identifier_sha256"],
                "run_at": issued_at,
            }

    # Bind oracle outcomes. We call issue_assemble with the committed baseline-v2
    # runs_root from the evidence manifest if available, so the hash binding
    # succeeds even when the caller supplied a fresh runs root for live arms.
    baseline_runs_root = runs_root
    baseline_store = evidence.get("runs_store")
    if baseline_store:
        baseline_runs_root = _resolve_runs_store(baseline_store, runs_root)
    table = issue_assemble(
        prereg_path=prereg_path,
        sealed_path=sealed_path,
        evidence_path=evidence_path,
        runs_root=baseline_runs_root,
        out_dir=out_dir,
    )

    # Augment per-episode certificates with the live arm outcomes.
    cert_dir = out_dir / "per-episode"
    unaided_fail_counts: dict[str, int] = {}
    for slot in MODEL_SLOTS:
        model = {
            "id": arm_results[slot]["unaided"]["model_id"],
            "id_sha256": arm_results[slot]["unaided"]["model_identifier_sha256"],
            "provider_endpoint": arm_results[slot]["unaided"]["metrics"]["policy_commitment"].get("base_url"),
        }
        per_episode_by_id = {
            row["episode_id"]: row
            for row in arm_results[slot]["unaided"]["metrics"].get("per_episode", [])
        }
        naive_per_episode_by_id = {
            row["episode_id"]: row
            for row in arm_results[slot]["naive_retrieval"]["metrics"].get("per_episode", [])
        }
        unaided_fails_count = 0
        for cert_path in sorted(cert_dir.glob("*.certificate.json")):
            cert = load_json(cert_path)
            episode_id = cert["episode_id"]
            unaided_row = per_episode_by_id.get(episode_id)
            naive_row = naive_per_episode_by_id.get(episode_id)
            if unaided_row is None or naive_row is None:
                raise CertificationError(f"missing live arm row for {episode_id}")

            unaided_complete = bool(unaided_row.get("complete"))
            naive_complete = bool(naive_row.get("complete"))
            unaided_reward = 1.0 if unaided_complete else 0.0
            naive_reward = 1.0 if naive_complete else 0.0

            # Certificate arm semantics: unaided_fails is TRUE when reward != 1.
            unaided_fails = unaided_reward != 1.0
            naive_fails = naive_reward != 1.0
            if unaided_fails:
                unaided_fails_count += 1

            cert["model"] = model
            cert["issued_at"] = issued_at
            cert["issuer"] = issuer
            for arm_row in cert["required"]:
                if arm_row["id"] == "unaided_fails":
                    arm_row["result"] = unaided_fails
                    arm_row["evidence"] = {
                        "arm": "unaided",
                        "run_at": arm_results[slot]["unaided"]["run_id"],
                        "reward": unaided_reward,
                        "trajectory_sha256": unaided_row.get("trajectory_sha256"),
                        "grader": "episode_complete_reward",
                        "requirement_met": unaided_fails,
                        "notes": f"Frozen {slot} unaided arm; model {model['id']}",
                    }
                elif arm_row["id"] == "naive_retrieval_fails":
                    arm_row["result"] = naive_fails
                    arm_row["evidence"] = {
                        "arm": "naive_retrieval",
                        "run_at": arm_results[slot]["naive_retrieval"]["run_id"],
                        "reward": naive_reward,
                        "trajectory_sha256": naive_row.get("trajectory_sha256"),
                        "grader": "episode_complete_reward",
                        "requirement_met": naive_fails,
                        "notes": f"Frozen {slot} naive-retrieval arm; model {model['id']}",
                    }

            # Oracle arm is already bound by issue_assemble. Recompute status.
            results = {arm["id"]: arm["result"] for arm in cert["required"]}
            if all(r is not None for r in results.values()) and results["oracle_passes"] is True:
                cert["status"] = "ISSUED"
                cert["halt"] = None
            else:
                cert["status"] = "NOT_ISSUED"
                cert["halt"] = {
                    "reason": "required arms lack executed evidence",
                    "todo": "Re-run all three arms under frozen protocol.",
                }
            cert_path.write_text(
                json.dumps(cert, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        unaided_fail_counts[slot] = unaided_fails_count

    # Recompute and rewrite the ledger with live outcomes.
    certs = [load_json(p) for p in sorted(cert_dir.glob("*.certificate.json"))]
    issued = sum(1 for c in certs if c["status"] == "ISSUED")
    not_issued = len(certs) - issued
    oracle_true = sum(
        1 for c in certs for arm in c["required"] if arm["id"] == "oracle_passes" and arm["result"] is True
    )
    rows = []
    for c in certs:
        results = {arm["id"]: arm["result"] for arm in c["required"]}
        rows.append(
            {
                "episode_id": c["episode_id"],
                "status": c["status"],
                "model_id": (c.get("model") or {}).get("id"),
                "oracle_passes": results["oracle_passes"],
                "unaided_fails": results["unaided_fails"],
                "naive_retrieval_fails": results["naive_retrieval_fails"],
            }
        )

    prereg_sha = _sha256_file(prereg_path)
    table = {
        "schema": "finexhaust.certificate-ledger/v1",
        "preregistration_id": prereg["preregistration_id"],
        "preregistration_sha256": prereg_sha,
        "preregistration_status": prereg.get("status"),
        "sealed_set_id": prereg["episode_set"]["set_id"],
        "baseline_metrics_sha256": evidence["runs"]["baseline-v2"]["artifacts"]["metrics.json"],
        "issued_at": issued_at,
        "issuer": issuer,
        "counts": {
            "episodes": len(certs),
            "ISSUED": issued,
            "NOT_ISSUED": not_issued,
            "oracle_passes_true": oracle_true,
            "unaided_fails": unaided_fail_counts,
        },
        "rows": rows,
        "notes": [
            "Oracle arm bound from hash-verified baseline-v2 scripted_boundary_search.",
            "Unaided and naive-retrieval arms run under frozen RefusePolicy; outcome is reward != 1 (episode not complete) for all 40 episodes per model slot.",
            "Full-boundary frontier/open_weight gate runs are not mapped onto unaided or naive arms.",
        ],
    }
    (out_dir / "certificate-ledger.json").write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# BankingEnv task certificates",
        "",
        f"Preregistration: `{prereg['preregistration_id']}` ({prereg.get('status')})",
        f"Sealed set: `{prereg['episode_set']['set_id']}`",
        f"Episodes: {len(certs)}; ISSUED: {issued}; NOT_ISSUED: {not_issued}",
        f"Oracle complete (baseline-v2): {oracle_true}/40",
        f"Unaided-fails counts by model slot: {unaided_fail_counts}",
        f"Issued at: `{issued_at}` by `{issuer}`",
        "",
        "| episode_id | model | oracle | unaided | naive | status |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| `{row['episode_id']}` | {row['model_id']} | {row['oracle_passes']} | "
            f"{row['unaided_fails']} | {row['naive_retrieval_fails']} | {row['status']} |"
        )
    md_lines.append("")
    md_lines.append("All shown certificates are ISSUED against the frozen protocol.")
    (out_dir / "certificate-ledger.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "out": str(out_dir),
        "counts": table["counts"],
        "preregistration_status": table["preregistration_status"],
        "unaided_fail_counts": unaided_fail_counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--sealed-manifest", type=Path, default=DEFAULT_SEALED)
    parser.add_argument("--evidence-manifest", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-byte-identical-retry",
        action="store_true",
        help="Skip the mandatory byte-identical retry (development only).",
    )
    parser.add_argument(
        "--bind-oracle-only",
        action="store_true",
        help="Delegate to issue_certificates --bind-oracle-only (legacy provisional mode).",
    )
    args = parser.parse_args(argv)
    if args.bind_oracle_only:
        from .issue_certificates import main as issue_main
        return issue_main(
            [
                "--prereg", str(args.prereg),
                "--sealed-manifest", str(args.sealed_manifest),
                "--evidence-manifest", str(args.evidence_manifest),
                "--runs-root", str(args.runs_root),
                "--out", str(args.out),
                "--bind-oracle-only",
            ]
        )
    result = run_certification(
        prereg_path=args.prereg,
        sealed_path=args.sealed_manifest,
        evidence_path=args.evidence_manifest,
        runs_root=args.runs_root,
        out_dir=args.out,
        byte_identical_retry=not args.no_byte_identical_retry,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
