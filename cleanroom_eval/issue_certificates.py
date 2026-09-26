"""Issue (or provisionally assemble) per-task BankingEnv certificates (#172).

Honest default: bind ``oracle_passes`` from the hash-bound baseline-v2 scripted
run and leave ``unaided_fails`` / ``naive_retrieval_fails`` null so status stays
``NOT_ISSUED``. Never invent arm outcomes. Never map full-boundary gate runs
onto unaided or naive arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
DEFAULT_PREREG = PACKAGE / "assets" / "certificate-preregistration.v1.json"
DEFAULT_SEALED = PACKAGE / "assets" / "sealed-set.manifest.v2.json"
DEFAULT_EVIDENCE = PACKAGE / "evidence" / "gates-2026-08" / "evidence-manifest.json"
DEFAULT_RUNS = Path.home() / ".local/share/finexhaust" / "cleanroom_runs"
DEFAULT_OUT = PACKAGE / "evidence" / "certificates-2026-09"

ARM_ORDER = ("unaided_fails", "naive_retrieval_fails", "oracle_passes")
ARM_DESCRIPTIONS = {
    "unaided_fails": (
        "Model plus instruction only; no environment or retrieval pack; "
        "certificate requires reward != 1."
    ),
    "naive_retrieval_fails": (
        "Same model with a lexical public-evidence pack and no boundary tools; "
        "certificate requires reward != 1."
    ),
    "oracle_passes": (
        "Sealed scripted policy through the boundary; certificate requires "
        "episode complete (reward == 1)."
    ),
}


class CertificateError(RuntimeError):
    """Fail-closed certificate assembly error."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def episode_id_to_asset_path(episode_id: str) -> str:
    """Map free_run episode_id to sealed-set relative path."""
    if not episode_id.startswith("episode_") or not episode_id.endswith("_v1"):
        raise CertificateError(f"unrecognised episode_id: {episode_id!r}")
    stem = episode_id[len("episode_") : -len("_v1")]
    return f"episodes_v2/{stem}.v1.json"


def sealed_episode_index(manifest: dict) -> dict[str, str]:
    """path -> sha256 for the 40 episode bodies (not task sidecars)."""
    out: dict[str, str] = {}
    for asset in manifest.get("assets") or []:
        path = str(asset.get("path") or "")
        if path.startswith("episodes_v2/") and path.endswith(".v1.json") and ".task." not in path:
            digest = str(asset.get("sha256") or "")
            if len(digest) != 64:
                raise CertificateError(f"bad sealed asset digest for {path}")
            out[path] = digest
    if len(out) != 40:
        raise CertificateError(f"expected 40 sealed episodes, found {len(out)}")
    return out


def verify_baseline_binding(evidence: dict, runs_root: Path) -> Path:
    """Return metrics path after checking evidence-manifest hash binding."""
    runs = evidence.get("runs") or {}
    row = runs.get("baseline-v2")
    if not isinstance(row, dict):
        raise CertificateError("evidence-manifest lacks baseline-v2 binding")
    artifacts = row.get("artifacts") or {}
    expected = artifacts.get("metrics.json")
    if not expected:
        raise CertificateError("baseline-v2 metrics.json digest missing from evidence-manifest")
    metrics_path = runs_root / "baseline-v2" / "metrics.json"
    if not metrics_path.is_file():
        raise CertificateError(f"baseline-v2 metrics not found: {metrics_path}")
    actual = _sha256_file(metrics_path)
    if actual != expected:
        raise CertificateError(
            f"baseline-v2 metrics hash mismatch: expected {expected}, got {actual}"
        )
    return metrics_path


def build_certificate(
    *,
    episode_id: str,
    sealed_sha256: str,
    oracle: dict | None,
    prereg_id: str,
    prereg_sha256: str,
    set_id: str,
) -> dict:
    required = []
    for arm_id in ARM_ORDER:
        if arm_id == "oracle_passes" and oracle is not None:
            complete = bool(oracle.get("complete"))
            required.append(
                {
                    "id": arm_id,
                    "description": ARM_DESCRIPTIONS[arm_id],
                    "result": complete,
                    "evidence": {
                        "arm": "oracle_script",
                        "run_at": str(oracle.get("run_at") or "baseline-v2"),
                        "reward": 1.0 if complete else 0.0,
                        "trajectory_sha256": oracle.get("trajectory_sha256"),
                        "grader": "scripted_boundary_search",
                        "requirement_met": complete,
                        "notes": "Bound from hash-verified baseline-v2 metrics; not a live model arm",
                    },
                }
            )
        else:
            required.append(
                {
                    "id": arm_id,
                    "description": ARM_DESCRIPTIONS[arm_id],
                    "result": None,
                    "evidence": None,
                }
            )

    missing = [row["id"] for row in required if row["result"] is None]
    status = "NOT_ISSUED"
    halt = {
        "reason": "required arms lack executed evidence",
        "todo": (
            "Run unaided_fails and naive_retrieval_fails under frozen policies "
            "for at least two models; then re-issue."
        ),
    }
    if missing:
        # stay NOT_ISSUED
        pass
    else:
        # only reachable once live arms land; keep fail-closed here
        if all(row["result"] is True for row in required if row["id"] != "oracle_passes") and any(
            row["id"] == "oracle_passes" and row["result"] is True for row in required
        ):
            status = "ISSUED"
            halt = None

    # Fail closed: never ISSUED while any result is null.
    if any(row["result"] is None for row in required):
        status = "NOT_ISSUED"
        halt = {
            "reason": "required arms lack executed evidence",
            "todo": (
                "Run unaided_fails and naive_retrieval_fails under frozen policies "
                "for at least two models; then re-issue."
            ),
        }

    return {
        "schema": "bankingenv.task-certificate/v1",
        "episode_id": episode_id,
        "sealed_episode_sha256": sealed_sha256,
        "status": status,
        "model": None,
        "required": required,
        "issued_at": None,
        "issuer": None,
        "protocol": {
            "preregistration_id": prereg_id,
            "preregistration_sha256": prereg_sha256,
            "set_id": set_id,
        },
        "halt": halt,
    }


def assemble(
    *,
    prereg_path: Path,
    sealed_path: Path,
    evidence_path: Path,
    runs_root: Path,
    out_dir: Path,
) -> dict:
    prereg = _load_json(prereg_path)
    if prereg.get("preregistration_id") != "bankingenv_task_certificate_v1":
        raise CertificateError("unexpected preregistration_id")
    sealed = _load_json(sealed_path)
    evidence = _load_json(evidence_path)
    metrics_path = verify_baseline_binding(evidence, runs_root)
    metrics = _load_json(metrics_path)
    if metrics.get("policy") != "scripted_boundary_search":
        raise CertificateError(
            f"baseline-v2 policy is {metrics.get('policy')!r}, not scripted_boundary_search"
        )
    if metrics.get("run_id") != "baseline-v2":
        raise CertificateError(f"unexpected baseline run_id: {metrics.get('run_id')!r}")

    index = sealed_episode_index(sealed)
    prereg_sha = _sha256_file(prereg_path)
    set_id = str(sealed.get("set_id") or prereg["episode_set"]["set_id"])

    certificates = []
    for row in metrics.get("per_episode") or []:
        episode_id = str(row["episode_id"])
        asset = episode_id_to_asset_path(episode_id)
        if asset not in index:
            raise CertificateError(f"no sealed asset for {episode_id} ({asset})")
        cert = build_certificate(
            episode_id=episode_id,
            sealed_sha256=index[asset],
            oracle={
                "complete": row.get("complete"),
                "trajectory_sha256": row.get("trajectory_sha256"),
                "run_at": "baseline-v2",
            },
            prereg_id=prereg["preregistration_id"],
            prereg_sha256=prereg_sha,
            set_id=set_id,
        )
        certificates.append(cert)

    if len(certificates) != 40:
        raise CertificateError(f"expected 40 certificates, built {len(certificates)}")

    issued = sum(1 for c in certificates if c["status"] == "ISSUED")
    not_issued = len(certificates) - issued
    oracle_true = sum(
        1
        for c in certificates
        for arm in c["required"]
        if arm["id"] == "oracle_passes" and arm["result"] is True
    )
    unaided_null = sum(
        1
        for c in certificates
        for arm in c["required"]
        if arm["id"] == "unaided_fails" and arm["result"] is None
    )

    table = {
        "schema": "finexhaust.certificate-ledger/v1",
        "preregistration_id": prereg["preregistration_id"],
        "preregistration_sha256": prereg_sha,
        "preregistration_status": prereg.get("status"),
        "sealed_set_id": set_id,
        "baseline_metrics_sha256": _sha256_file(metrics_path),
        "counts": {
            "episodes": len(certificates),
            "ISSUED": issued,
            "NOT_ISSUED": not_issued,
            "oracle_passes_true": oracle_true,
            "unaided_fails_null": unaided_null,
            "naive_retrieval_fails_null": unaided_null,
        },
        "rows": [
            {
                "episode_id": c["episode_id"],
                "status": c["status"],
                "oracle_passes": next(
                    arm["result"] for arm in c["required"] if arm["id"] == "oracle_passes"
                ),
                "unaided_fails": next(
                    arm["result"] for arm in c["required"] if arm["id"] == "unaided_fails"
                ),
                "naive_retrieval_fails": next(
                    arm["result"]
                    for arm in c["required"]
                    if arm["id"] == "naive_retrieval_fails"
                ),
            }
            for c in certificates
        ],
        "notes": [
            "Oracle arm bound from hash-verified baseline-v2 scripted_boundary_search only.",
            "Unaided and naive-retrieval arms are intentionally null; certificates remain NOT_ISSUED.",
            "Full-boundary frontier/open_weight gate runs must not be mapped onto unaided or naive arms.",
        ],
    }

    cert_dir = out_dir / "per-episode"
    for cert in certificates:
        _write_json(cert_dir / f"{cert['episode_id']}.certificate.json", cert)
    _write_json(out_dir / "certificate-ledger.json", table)

    md_lines = [
        "# BankingEnv task certificates (provisional)",
        "",
        f"Preregistration: `{prereg['preregistration_id']}` ({prereg.get('status')})",
        f"Sealed set: `{set_id}`",
        f"Episodes: {len(certificates)}; ISSUED: {issued}; NOT_ISSUED: {not_issued}",
        f"Oracle complete (baseline-v2): {oracle_true}/40",
        f"Unaided / naive arms executed: 0/40 (null; live run required)",
        "",
        "| episode_id | oracle | unaided | naive | status |",
        "|---|---|---|---|---|",
    ]
    for row in table["rows"]:
        md_lines.append(
            f"| `{row['episode_id']}` | {row['oracle_passes']} | {row['unaided_fails']} | "
            f"{row['naive_retrieval_fails']} | {row['status']} |"
        )
    md_lines.append("")
    md_lines.append(
        "No certificate is ISSUED. Do not treat this table as benchmark admission."
    )
    (out_dir / "certificate-ledger.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--sealed-manifest", type=Path, default=DEFAULT_SEALED)
    parser.add_argument("--evidence-manifest", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--bind-oracle-only",
        action="store_true",
        help="Assemble NOT_ISSUED certificates with oracle bound from baseline-v2.",
    )
    args = parser.parse_args(argv)
    if not args.bind_oracle_only:
        raise SystemExit("refusing to run without --bind-oracle-only (live arms not implemented)")
    table = assemble(
        prereg_path=args.prereg,
        sealed_path=args.sealed_manifest,
        evidence_path=args.evidence_manifest,
        runs_root=args.runs_root,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "counts": table["counts"],
                "preregistration_status": table["preregistration_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
