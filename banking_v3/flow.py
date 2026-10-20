"""Prefect-only generation, validation and model execution entry points."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import secrets

from prefect import flow
from prefect.runtime import flow_run

from .contract import digest, load_json, load_package, write_json
from .environment import Environment
from .evaluate import reference_rollout, validate_dataset
from .generate import generate


def code_commitment():
    root = Path(__file__).parent
    return digest({p.name: digest(p.read_text()) for p in sorted(root.glob("*.py"))})


@flow(name="bankingenv-v3-regenerate", persist_result=False, log_prints=True)
def regenerate(output: str, development_seed: str = "bankingenv-v3-development-20261020", variants: int = 3):
    root = Path(output).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("generate outside the source checkout to keep evaluation material private")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(root.iterdir()):
        raise ValueError("output must be empty; historical runs are immutable")
    root.chmod(0o700)
    # This seed and all evaluation documents remain outside the source/public trees.
    answers = generate(root, development_seed, secrets.token_hex(32), variants)
    reports = {}
    for split in ("development", "evaluation"):
        subset = [a for a in answers if a["split"] == split]
        reports[split] = validate_dataset(root / split, subset)
        write_json(root / "evaluator" / f"{split}-validation.json", reports[split])
    signatures = {split: {a["mechanism_signature"] for a in answers if a["split"] == split}
                  for split in reports}
    if signatures["development"] & signatures["evaluation"]:
        raise AssertionError("mechanism split contamination")
    cases = {split: load_package(root / split)[1] for split in reports}
    for field in ("id", "world_id"):
        if {c[field] for c in cases["development"]} & {c[field] for c in cases["evaluation"]}:
            raise AssertionError(f"split overlap: {field}")
    dev_docs = {d["id"] for c in cases["development"] for d in c["documents"]}
    held_docs = {d["id"] for c in cases["evaluation"] for d in c["documents"]}
    if dev_docs & held_docs:
        raise AssertionError("document identity overlap")
    receipt = {"schema": "bankingenv.regeneration-receipt/v3", "prefect_flow_run_id": str(flow_run.id),
               "generated_at": datetime.now(timezone.utc).isoformat(), "code_sha256": code_commitment(),
               "dataset_version": "3.0", "validation_level": "MECHANICAL_ONLY",
               "splits": {split: {k: v for k, v in report.items() if k != "rows"} for split, report in reports.items()},
               "split_audit": {"case_id_overlap": 0, "document_id_overlap": 0, "world_id_overlap": 0,
                               "mechanism_signature_overlap": 0, "shared_workflow_primitives": True,
                               "shared_renderer": True, "claim": "compositional holdout, not unseen workflow families"},
               "restricted_source_inputs": 0, "external_source_inputs": 0,
               "human_certificates_issued": 0, "model_runs": [], "model_difficulty": "NOT_MEASURED"}
    write_json(root / "receipt.json", receipt)
    print({"cases": sum(r["cases"] for r in reports.values()), "flow_run_id": str(flow_run.id),
           "validation": "PASS", "adversarial_trajectories": sum(r["adversarial_trajectories"] for r in reports.values())})
    return receipt


@flow(name="bankingenv-v3-evaluate", persist_result=False, log_prints=True)
def evaluate(dataset: str, output: str, policy_name: str = "reference", turn_limit: int = 24, limit: int | None = None):
    if policy_name not in ("reference", "live") or not 1 <= turn_limit <= 64:
        raise ValueError("unsupported policy or turn budget")
    manifest, cases = load_package(dataset)
    if limit is not None:
        if not 1 <= limit <= len(cases):
            raise ValueError("invalid case limit")
        cases = cases[:limit]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(root.iterdir()):
        raise ValueError("run output must be empty")
    root.chmod(0o700)
    policy = None
    if policy_name == "live":
        from .provider import HTTPPolicy
        policy = HTTPPolicy()
    prereg = {"schema": "bankingenv.run-plan/v3", "case_index_sha256": manifest["case_index_sha256"],
              "case_ids": [c["id"] for c in cases], "code_sha256": code_commitment(), "turn_limit": turn_limit,
              "policy": policy_name, "model": policy.model if policy else None,
              "prefect_flow_run_id": str(flow_run.id), "missing_results_policy": "zero, retained in denominator",
              "scope": "full_set" if limit is None else "smoke_subset", "primary_metric": "all_eight_checks_pass_no_safety_violation"}
    write_json(root / "plan.json", prereg)
    results = []
    for case in cases:
        error = None
        if policy is None:
            grade, _, transcript = reference_rollout(case)
        else:
            env = Environment(case)
            for _ in range(turn_limit):
                try:
                    action = policy.act(env.observe(), env.transcript())
                except RuntimeError as exc:
                    error = str(exc)
                    break
                env.step(action)
                if env.observe()["state"]["closure"]:
                    break
            grade, transcript = env.grade(), env.transcript()
        grade["execution_error"] = error
        results.append(grade)
        write_json(root / "trajectories" / f"{case['id']}.json", transcript)
        write_json(root / "grades" / f"{case['id']}.json", grade)
    failures = Counter(k for row in results for k, passed in row["checks"].items() if not passed)
    report = {"schema": "bankingenv.run/v3", "plan_sha256": digest(prereg),
              "prefect_flow_run_id": str(flow_run.id), "cases": len(results),
              "complete": sum(r["complete"] for r in results), "failures": dict(failures),
              "execution_errors": sum(r["execution_error"] is not None for r in results),
              "results": results, "policy": policy_name,
              "evidence_class": "MODEL_RUN" if policy else "EVALUATOR_INFORMED_SOLVABILITY",
              "model": policy.model if policy else None, "telemetry": policy.telemetry if policy else [],
              "human_certification": "NOT_ISSUED"}
    write_json(root / "report.json", report)
    print({"complete": report["complete"], "cases": report["cases"], "execution_errors": report["execution_errors"]})
    return report


@flow(name="bankingenv-v3-grade-transcripts", persist_result=False, log_prints=True)
def grade_transcripts(dataset: str, transcripts: str, output: str):
    """Replay submitted actions, never trust submitted rewards or state snapshots."""
    from .evaluate import replay
    manifest, cases = load_package(dataset)
    source = Path(transcripts)
    expected_names = {c["id"] + ".json" for c in cases}
    if {p.name for p in source.glob("*.json")} - expected_names:
        raise ValueError("unknown case transcript")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(out.iterdir()):
        raise ValueError("grading output must be empty")
    rows = []
    for case in cases:
        path = source / (case["id"] + ".json")
        missing = not path.is_file()
        transcript = [] if missing else load_json(path)
        if not isinstance(transcript, list) or len(transcript) > 64:
            raise ValueError("transcript must contain at most 64 tool turns")
        # Accept runner transcripts or plain action lists; responses are ignored.
        actions = [row.get("request", row) if isinstance(row, dict) else row for row in transcript]
        row = replay(case, actions)
        row.update(missing=missing, submitted_actions_sha256=digest(actions))
        rows.append(row)
    report = {"schema": "bankingenv.regrade/v3", "prefect_flow_run_id": str(flow_run.id),
              "case_index_sha256": manifest["case_index_sha256"], "code_sha256": code_commitment(),
              "cases": len(cases), "complete": sum(r["complete"] for r in rows),
              "missing": sum(r["missing"] for r in rows), "results": rows,
              "model_identity": "NOT_ATTESTED", "certification": "NOT_ISSUED"}
    write_json(out / "report.json", report)
    return report
