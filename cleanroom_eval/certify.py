"""Legacy certificate interface, disabled because scripted refusals are not model evidence.

Binding verification helpers remain available for historical audit.
Use banking_v3 for new measured runs; no certification is implied by a run.
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
from .contract import file_sha256, load_json, validate_schema
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
    """Fail closed unless the preregistration binds the sealed set it names.

    The frozen v1 preregistration records the sealed-set manifest's
    ``asset_list_sha256`` at top level as ``manifest_sha256``; a raw file
    digest under ``episode_set.manifest_sha256`` is also accepted.
    """
    raw_expected = prereg.get("episode_set", {}).get("manifest_sha256")
    asset_list_expected = prereg.get("manifest_sha256")
    if not raw_expected and not asset_list_expected:
        raise CertificationError("preregistration does not bind a sealed-set manifest")
    if raw_expected:
        actual = _sha256_file(sealed_path)
        if actual != raw_expected:
            raise CertificationError(f"sealed manifest hash mismatch: expected {raw_expected}, got {actual}")
    if asset_list_expected:
        actual = load_json(sealed_path).get("asset_list_sha256")
        if actual != asset_list_expected:
            raise CertificationError(
                f"sealed manifest asset_list_sha256 mismatch: expected {asset_list_expected}, got {actual}"
            )


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
    """Legacy certification is disabled: scripted refusals are not model evidence."""
    raise CertificationError(
        "Legacy certification disabled: RefusePolicy cannot establish model failure. "
        "Use banking_v3 for measured runs; certification requires separately reviewed evidence."
    )


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
