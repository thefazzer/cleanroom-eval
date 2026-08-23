"""Bind the clean-room gates evidence to immutable run bytes.

Generates the content-safe committed evidence pack for #145 receipt 1
(run-evidence). Every number the reports display must map to a hash in
this pack; changing any run byte invalidates the pack.

Usage:
    python -m cleanroom_eval.evidence.bind_gates_evidence \
        [--runs-root PATH] [--out PATH] [--verify]

--runs-root defaults to $CLEANROOM_RUNS_ROOT, else the private runs store.
--verify recomputes everything and fails on any divergence from the
committed pack instead of rewriting it.

Content-safety rules enforced here:
- transcripts, per-turn request/response bodies and provider credentials
  are NEVER copied: transcripts are bound by sha256 only, and the harness
  telemetry already stores request/response as hashes;
- the private runs location is recorded as an opaque store id, never a path;
- model identities come from each run's config.json commitment — never from
  planning documents or prose (an arm has been misnamed in prose twice).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cleanroom_eval.contract import digest, file_sha256  # noqa: E402
from cleanroom_eval.free_run import load_episodes  # noqa: E402

SCHEMA = "cleanroom.gates-evidence-manifest/v1"
STORE_ID = "cleanroom-runs-store-v1"  # opaque; the owner maps it to the private location
DEFAULT_OUT = REPO_ROOT / "cleanroom_eval" / "evidence" / "gates-2026-08"
RUN_IDS = ("baseline-v1", "baseline-v2", "frontier", "open_weight")
RUN_FILES = ("config.json", "transcript.jsonl", "metrics.json", "scores.json", "cost_cards.json")
ROOT_FILES = ("gates-preregistration.v1.json", "gates_report.json", "gates_report.md")
# The code whose behaviour the results depend on, hashed at the bound commit.
RUNNER_MODULES = (
    "cleanroom_eval/free_run.py",
    "cleanroom_eval/runner.py",
    "cleanroom_eval/contract.py",
    "cleanroom_eval/episode_contract.py",
    "cleanroom_eval/fire_gates.py",
    "cleanroom_eval/real_provider.py",
    "cleanroom_eval/cost_grader.py",
)
EPISODE_SETS = {
    "v1": "cleanroom_eval/assets/episodes",
    "v2": "cleanroom_eval/assets/episodes_v2",
}
FORBIDDEN_PATTERNS = (
    "api_key", "authorization", "bearer ", "sk-", "AKIA", "ghp_", "cfat_",
    "BEGIN RSA", "BEGIN OPENSSH", "BEGIN EC", "BEGIN PGP",
)


def _runs_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    env = os.environ.get("CLEANROOM_RUNS_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".local/share/finexhaust/cleanroom_runs"


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _episode_ids_sha256(episode_dir: Path) -> tuple[int, str]:
    """Reproduce free_run's config binding from the committed episode assets.

    Uses free_run.load_episodes itself so the binding can never drift from
    the loader that produced the run configs.
    """
    pairs = load_episodes(episode_dir)
    return len(pairs), digest([episode["episode_id"] for episode, _ in pairs])


def _content_safe(text: str, *, name: str) -> None:
    lowered = text.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in lowered:
            raise SystemExit(f"content-safety: {name} matches forbidden pattern {pattern!r}; refusing to emit")


def _endpoint_class(base_url: str) -> str:
    return base_url.split("//", 1)[-1].split("/", 1)[0]


def _hash_tree(root: Path, rel_to: Path) -> dict[str, str]:
    return {
        str(p.relative_to(rel_to)): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def build(runs_root: Path, out_dir: Path) -> dict[str, Any]:
    if not runs_root.is_dir():
        raise SystemExit(f"runs root not found: {runs_root}")

    # ---- 1. hash every immutable run artifact -------------------------------
    artifact_hashes: dict[str, str] = {}
    for name in ROOT_FILES:
        artifact_hashes[name] = file_sha256(runs_root / name)
    for run_id in RUN_IDS:
        run_dir = runs_root / run_id
        for name in RUN_FILES:
            artifact_hashes[f"{run_id}/{name}"] = file_sha256(run_dir / name)
        aiuc1 = run_dir / "aiuc1"
        if aiuc1.is_dir():
            artifact_hashes.update(
                {f"{run_id}/aiuc1/{rel}": sha for rel, sha in _hash_tree(aiuc1, aiuc1).items()}
            )

    # ---- 2. verify the report is bound to the preregistration ---------------
    report = json.loads((runs_root / "gates_report.json").read_text(encoding="utf-8"))
    prereg_sha = file_sha256(runs_root / "gates-preregistration.v1.json")
    if report["preregistration_sha256"] != prereg_sha:
        raise SystemExit(
            "gates_report.json is not bound to the preregistration on disk: "
            f"{report['preregistration_sha256']} != {prereg_sha}"
        )

    # ---- 3. bind episode sets: repo assets must equal what ran --------------
    episode_bindings: dict[str, Any] = {}
    config_by_run: dict[str, dict[str, Any]] = {}
    for run_id in RUN_IDS:
        config_by_run[run_id] = json.loads(
            (runs_root / run_id / "config.json").read_text(encoding="utf-8")
        )
    for set_id, rel in EPISODE_SETS.items():
        count, ids_sha = _episode_ids_sha256(REPO_ROOT / rel)
        users = [r for r in RUN_IDS if config_by_run[r]["episode_dir"].rstrip("/").endswith(Path(rel).name)]
        for run_id in users:
            recorded = config_by_run[run_id]["episode_ids_sha256"]
            if recorded != ids_sha:
                raise SystemExit(
                    f"episode set {set_id}: repo assets hash {ids_sha} but run "
                    f"{run_id} recorded {recorded} — repo episodes are not the ones that ran"
                )
        episode_bindings[set_id] = {
            "repo_path": rel,
            "episode_count": count,
            "episode_ids_sha256": ids_sha,
            "bound_runs": users,
        }

    # ---- 4. model identities: from run commitments only ---------------------
    identities = {}
    for run_id in RUN_IDS:
        config = config_by_run[run_id]
        commitment = config.get("policy_commitment") or {}
        identities[run_id] = {
            "policy": config["policy"],
            "model": commitment.get("model"),
            "model_identifier_sha256": commitment.get("model_identifier_sha256"),
            "endpoint_class": _endpoint_class(commitment["base_url"]) if commitment.get("base_url") else None,
            "temperature_commitment": commitment.get("temperature"),
            "turn_limit": config["turn_limit"],
            "episode_count": config["episode_count"],
            "episode_ids_sha256": config["episode_ids_sha256"],
            "identity_source": f"{run_id}/config.json@{artifact_hashes[f'{run_id}/config.json']}",
        }

    # ---- 5. telemetry summary (aggregates only; absent usage is UNKNOWN) ----
    telemetry = {}
    for run_id in RUN_IDS:
        metrics = json.loads((runs_root / run_id / "metrics.json").read_text(encoding="utf-8"))
        calls = metrics.get("policy_telemetry") or []
        statuses = Counter(str(c.get("status")) for c in calls)
        stop_reasons = Counter(
            str(e.get("stop_reason")) for e in metrics.get("per_episode", [])
        )
        elapsed = [c["elapsed_seconds"] for c in calls if isinstance(c.get("elapsed_seconds"), (int, float))]
        telemetry[run_id] = {
            "episodes": metrics["episodes"],
            "completed": metrics["completed"],
            "completion_rate": metrics["completion_rate"],
            "mean_turns": metrics["mean_turns"],
            "gates": metrics["gates"],
            "provider_calls": len(calls),
            "http_status_counts": dict(sorted(statuses.items())),
            "retry_count": statuses.get("429", 0),
            "transport_5xx": sum(v for k, v in statuses.items() if k.startswith("5")),
            "wall_clock_provider_seconds": round(sum(elapsed), 3) if elapsed else "UNKNOWN",
            "mean_call_latency_seconds": round(sum(elapsed) / len(elapsed), 3) if elapsed else "UNKNOWN",
            "termination_reasons": dict(sorted(stop_reasons.items())),
            # Harness >= v3 aggregates provider usage in metrics; earlier runs
            # never recorded it, so the fields stay UNKNOWN — never zero.
            "input_tokens": (metrics.get("provider_usage") or {}).get("input_tokens", "UNKNOWN"),
            "output_tokens": (metrics.get("provider_usage") or {}).get("output_tokens", "UNKNOWN"),
            "total_tokens": (metrics.get("provider_usage") or {}).get("total_tokens", "UNKNOWN"),
            "cached_tokens": (metrics.get("provider_usage") or {}).get("cached_tokens", "UNKNOWN"),
            "provider_inference_cost_usd": "UNKNOWN",
            "cost_note": (
                "harness v2 telemetry records status/latency/request+response hashes per call "
                "but not provider usage; token and inference-cost fields are UNKNOWN by policy, "
                "never zero. operational_expected_loss_usd lives in cost_cards.json and is "
                "never combined with inference cost."
            ),
        }

    # ---- 6. deviations from preregistration ---------------------------------
    prereg = json.loads((runs_root / "gates-preregistration.v1.json").read_text(encoding="utf-8"))
    prereg_models = {m["arm"]: m["selection"] for m in prereg.get("models", [])}
    deviations = [
        {
            "id": "substituted-frontier-arm",
            "preregistered": prereg_models.get("frontier"),
            "as_run": f"{identities['frontier']['model']} via {identities['frontier']['endpoint_class']}",
            "reason": "originally named arm unavailable at run time; substitution disclosed in the report",
        },
        {
            "id": "substituted-open-weight-arm",
            "preregistered": prereg_models.get("open_weight"),
            "as_run": f"{identities['open_weight']['model']} via {identities['open_weight']['endpoint_class']}",
            "reason": "originally named arm unavailable at run time; substitution disclosed in the report",
        },
        {
            "id": "harness-v1-to-v2-revision",
            "description": (
                "The first frontier run executed under harness v1, which treated provider 429 "
                "responses as fatal and rejected correct-but-more-specific version assertions. "
                "Both defects were fixed (retry/backoff and grading), the harness re-frozen as v2, "
                "and all reported model arms re-run. The v1 frontier artifact is retained in the "
                "archive store as superseded context and is not part of this evidence pack."
            ),
        },
        {
            "id": "archived-aborted-runs",
            "description": (
                "The archive store retains aborted attempts (a smoke run and two aborted "
                "open-weight attempts) by opaque id; none contribute to reported results."
            ),
        },
    ]

    # ---- 7. the binding manifest --------------------------------------------
    manifest = {
        "schema": SCHEMA,
        "preregistration_id": report["preregistration_id"],
        "preregistration_sha256": prereg_sha,
        "repository_commit": _git_head(),
        "runs_store": STORE_ID,
        "episode_sets": episode_bindings,
        "runner_code_sha256": {rel: file_sha256(REPO_ROOT / rel) for rel in RUNNER_MODULES},
        "runs": {
            run_id: {
                "status": next(r["status"] for r in report["runs"] if r["run_id"] == run_id),
                "artifacts": {
                    name: artifact_hashes[f"{run_id}/{name}"] for name in RUN_FILES
                },
                "report_row": next(r for r in report["runs"] if r["run_id"] == run_id),
            }
            for run_id in RUN_IDS
        },
        "mutation_sensitivity": report["mutation_sensitivity"],
        "verdicts": report["verdicts"],
        "verdict_qualification": (
            "H0-H1 supported on the frozen harness (v2). H2-H3 supported for the substituted "
            "arms recorded in model-identities.json; the originally named model arms were not run."
        ),
        "report_hashes": {name: artifact_hashes[name] for name in ROOT_FILES},
        "artifact_count": len(artifact_hashes),
    }

    # ---- 8. emit ------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    emitted: dict[str, str] = {}

    def emit(name: str, payload: Any) -> None:
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        _content_safe(text, name=name)
        (out_dir / name).write_text(text, encoding="utf-8")
        emitted[name] = file_sha256(out_dir / name)

    # verbatim copies of the safe run-level documents
    for src, dst in (
        ("gates-preregistration.v1.json", "gates-preregistration.v1.json"),
        ("gates_report.json", "gates-report.json"),
        ("gates_report.md", "gates-report.md"),
    ):
        text = (runs_root / src).read_text(encoding="utf-8")
        _content_safe(text, name=dst)
        (out_dir / dst).write_text(text, encoding="utf-8")
        emitted[dst] = file_sha256(out_dir / dst)

    emit("model-identities.json", identities)
    emit("telemetry-summary.json", telemetry)
    emit("deviations.json", deviations)
    emit("evidence-manifest.json", manifest)

    lines = [f"{sha}  {name}" for name, sha in sorted(artifact_hashes.items())]
    (out_dir / "artifact-manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    emitted["artifact-manifest.sha256"] = file_sha256(out_dir / "artifact-manifest.sha256")
    return {"out_dir": str(out_dir), "files": emitted, "artifacts_bound": len(artifact_hashes)}


def verify(runs_root: Path, out_dir: Path) -> None:
    """Re-derive the pack in a scratch location and fail on any divergence."""
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        fresh = build(runs_root, Path(scratch))
        for name in fresh["files"]:
            committed = out_dir / name
            if not committed.is_file():
                raise SystemExit(f"VERIFY FAIL: {name} missing from {out_dir}")
            if name == "evidence-manifest.json":
                # Two fields are commit-relative, not run-byte-relative:
                # repository_commit moves with every generation, and
                # runner_code_sha256 tracks the harness, which may evolve after
                # the pack is bound. Everything else must match the run bytes;
                # the recorded runner hashes must match the RECORDED commit.
                a = json.loads((Path(scratch) / name).read_text(encoding="utf-8"))
                b = json.loads(committed.read_text(encoding="utf-8"))
                recorded_commit = b.pop("repository_commit")
                a.pop("repository_commit")
                a.pop("runner_code_sha256")
                recorded_runner = b.pop("runner_code_sha256")
                if a != b:
                    raise SystemExit(f"VERIFY FAIL: {name} diverges from run bytes")
                for rel, recorded_sha in recorded_runner.items():
                    blob = subprocess.run(
                        ["git", "show", f"{recorded_commit}:{rel}"],
                        cwd=REPO_ROOT, capture_output=True, check=True,
                    ).stdout
                    if hashlib.sha256(blob).hexdigest() != recorded_sha:
                        raise SystemExit(
                            f"VERIFY FAIL: recorded runner hash for {rel} does not match "
                            f"the recorded commit {recorded_commit[:12]}"
                        )
            elif file_sha256(committed) != file_sha256(Path(scratch) / name):
                raise SystemExit(f"VERIFY FAIL: {name} diverges from run bytes")
    print("VERIFY PASS: committed evidence pack matches the run bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    runs_root = _runs_root(args.runs_root)
    out_dir = Path(args.out)
    if args.verify:
        verify(runs_root, out_dir)
        return
    result = build(runs_root, out_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
