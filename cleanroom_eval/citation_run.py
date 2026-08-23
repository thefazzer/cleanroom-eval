"""Resumable exact-500 real-provider citation acceptance runner.

The runner checkpoints one immutable artefact per logical task.  A resumed run
never repeats an already checkpointed logical call, including a failed call.
Provider-internal HTTP retries remain visible in the call telemetry.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .citation_acceptance import (
    REQUIRED_ACCEPTANCE_CALLS,
    derive_acceptance_metrics_and_gates,
    verify_citation_acceptance,
)
from .citation_provider import CitationProviderConfig, SchemaConstrainedHTTPProvider
from .citation_tasks import (
    MANIFEST_PATH,
    load_provider_tasks,
    load_task_oracle,
    verify_citation_task_set,
)
from .citations import CitationValidationError, validate_and_render
from .contract import (
    ContractError,
    canonical_bytes,
    digest,
    file_sha256,
    load_json,
    validate_schema,
)

MAX_WORKERS = 16
MAX_REQUESTS_PER_SECOND = 100.0
RUN_SCHEMA = "cleanroom.citation-acceptance-execution/v1"
RECORD_SCHEMA = "cleanroom.citation-acceptance-call-record/v1"
CANONICAL_MANIFEST_SHA256 = "ae162bb9ec60c2234e0000ea8e5d9f328f0f4f48adb46dac589a5f63a1cbb80e"
ATTESTATION_PRIVATE_KEY_ENV = "CLEANROOM_CITATION_ATTESTATION_PRIVATE_KEY"
ATTESTATION_PUBLIC_KEY_ENV = "CLEANROOM_CITATION_ATTESTATION_PUBLIC_KEY"


def _required_key_path(value: Path | None, environment_name: str) -> Path:
    path = value or (
        Path(os.environ[environment_name]).expanduser()
        if os.environ.get(environment_name)
        else None
    )
    if path is None or not path.is_file():
        raise ContractError(f"required citation attestation key is absent: {environment_name}")
    return path


def _private_key(path: Path | None) -> Ed25519PrivateKey:
    key_path = _required_key_path(path, ATTESTATION_PRIVATE_KEY_ENV)
    try:
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as exc:
        raise ContractError("citation attestation private key is invalid") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ContractError("citation attestation key must be Ed25519")
    return key


def _public_key(path: Path | None) -> Ed25519PublicKey:
    key_path = _required_key_path(path, ATTESTATION_PUBLIC_KEY_ENV)
    try:
        key = serialization.load_pem_public_key(key_path.read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        raise ContractError("citation attestation public key is invalid") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ContractError("citation attestation key must be Ed25519")
    return key


def _public_bytes(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


class RateLimiter:
    """Thread-safe, no-burst minimum-interval limiter."""

    def __init__(self, requests_per_second: float) -> None:
        if not (0 < requests_per_second <= MAX_REQUESTS_PER_SECOND):
            raise ContractError("requests-per-second is outside strict bound")
        self._interval = 1.0 / requests_per_second
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self._interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


def _load_frozen_bundle(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    # Verify the bundle's committed bytes before enforcing its canonical
    # location. This keeps tamper failures specific and still rejects an
    # otherwise self-consistent copy at an alternate path below.
    summary = verify_citation_task_set(manifest_path)
    if manifest_path.resolve() != MANIFEST_PATH.resolve():
        raise ContractError("citation run must use the canonical frozen manifest")
    if file_sha256(manifest_path) != CANONICAL_MANIFEST_SHA256:
        raise ContractError("canonical frozen manifest digest differs")
    manifest = load_json(manifest_path)
    asset_root = manifest_path.parent
    tasks = load_provider_tasks(
        asset_root / "citation_tasks" / "citation-tasks.v1.jsonl"
    )
    oracles = load_task_oracle(
        asset_root / "citation_tasks" / "citation-task-oracle.v1.jsonl"
    )
    if len(tasks) != REQUIRED_ACCEPTANCE_CALLS or len(oracles) != len(tasks):
        raise ContractError("frozen citation bundle does not contain exactly 500 pairs")
    return tasks, oracles, {
        "set_id": summary["set_id"],
        "manifest_sha256": file_sha256(manifest_path),
        "task_list_sha256": summary["task_list_sha256"],
    }


def _selection_semantics(selection: Mapping[str, Any]) -> dict[str, Any]:
    if selection.get("answer_kind") == "refusal":
        return {
            "answer_kind": "refusal",
            "refusal": selection.get("refusal"),
        }
    claims = [
        {key: value for key, value in claim.items() if key != "claim_id"}
        for claim in selection.get("claims", [])
    ]
    claims.sort(key=canonical_bytes)
    return {"answer_kind": "answer", "claims": claims}


def _task_set_sha256(tasks: Sequence[Mapping[str, Any]]) -> str:
    return digest(list(tasks))


def _public_config(config: CitationProviderConfig) -> dict[str, Any]:
    """Return a commitment that cannot serialize the credential."""

    return {
        "provider": SchemaConstrainedHTTPProvider.name,
        "endpoint_sha256": hashlib.sha256(config.endpoint.encode("utf-8")).hexdigest(),
        "model_sha256": hashlib.sha256(config.model.encode("utf-8")).hexdigest(),
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "retry_backoff_seconds": config.retry_backoff_seconds,
        "max_output_tokens": config.max_output_tokens,
    }


def _error_code(error: Exception) -> str:
    message = str(error).casefold()
    if "outside supplied set" in message or "outside the task schema" in message:
        return "EVIDENCE_OUTSIDE_SET"
    if "refusal" in message:
        return "REFUSAL_REJECTED"
    if "unsupported" in message:
        return "UNSUPPORTED_ADDITION"
    if "schema" in message or "validation failed" in message:
        return "SCHEMA_INVALID"
    return "VALIDATION_REJECTED"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{threading.get_ident()}")
    data = canonical_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _record_path(run_dir: Path, sequence: int) -> Path:
    return run_dir / "records" / f"{sequence:04d}.json"


def _started_path(run_dir: Path, sequence: int) -> Path:
    return run_dir / "started" / f"{sequence:04d}.json"


def _execute_one(
    *,
    sequence: int,
    task: Mapping[str, Any],
    oracle: Mapping[str, Any],
    config: CitationProviderConfig,
    limiter: RateLimiter,
    record_path: Path,
    started_path: Path,
) -> dict[str, Any]:
    _atomic_json(started_path, {
        "schema": RUN_SCHEMA,
        "sequence": sequence,
        "task_id": task["task_id"],
        "task_sha256": digest(task),
    })
    limiter.wait()
    provider = SchemaConstrainedHTTPProvider(config)
    selection: Mapping[str, Any] | None = None
    rendered: Mapping[str, Any] | None = None
    error_code: str | None = None
    provider_completed = False
    try:
        selection_value = provider.select_claims(task)
        provider_completed = True
        if not isinstance(selection_value, Mapping):
            raise CitationValidationError("model selection is not an object")
        selection = json.loads(canonical_bytes(selection_value))
        validate_schema(selection, "claim-selection.schema.json")
        rendered = validate_and_render(selection, evidence_records=task["evidence"])
        validate_schema(rendered, "rendered-answer.schema.json")
        if _selection_semantics(selection_value) != _selection_semantics(
            oracle["expected_selection"]
        ):
            rendered = None
            error_code = "ORACLE_MISMATCH"
    except (ContractError, TypeError, ValueError) as exc:
        error_code = _error_code(exc)
    telemetry = provider.telemetry()
    if len(telemetry) != 1:
        raise ContractError("provider did not emit exactly one logical-call telemetry record")
    if telemetry[0].get("status") == "PASS" or telemetry[0].get("failure_class") == "RESPONSE":
        provider_completed = True
    record: dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "sequence": sequence,
        "task_id": task["task_id"],
        "task_sha256": digest(task),
        "oracle_sha256": digest(oracle),
        "provider_completed": provider_completed,
        "selection": selection,
        "rendered": rendered,
        "error_code": error_code,
        "provider_telemetry": telemetry[0],
    }
    record["record_sha256"] = digest(record)
    _atomic_json(record_path, record)
    started_path.unlink()
    return record


def _validate_record(
    record: Mapping[str, Any],
    *,
    sequence: int,
    task: Mapping[str, Any],
    oracle: Mapping[str, Any],
    max_retries: int,
) -> None:
    required = {
        "schema",
        "sequence",
        "task_id",
        "task_sha256",
        "oracle_sha256",
        "provider_completed",
        "selection",
        "rendered",
        "error_code",
        "provider_telemetry",
        "record_sha256",
    }
    if set(record) != required or record.get("schema") != RECORD_SCHEMA:
        raise ContractError(f"checkpoint {sequence} has invalid shape")
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    if record["record_sha256"] != digest(unsigned):
        raise ContractError(f"checkpoint {sequence} digest differs")
    if (
        record["sequence"] != sequence
        or record["task_id"] != task["task_id"]
        or record["task_sha256"] != digest(task)
        or record["oracle_sha256"] != digest(oracle)
    ):
        raise ContractError(f"checkpoint {sequence} is bound to another task or oracle")
    telemetry = record["provider_telemetry"]
    if not isinstance(telemetry, Mapping) or telemetry.get("task_id") != task["task_id"]:
        raise ContractError(f"checkpoint {sequence} telemetry is not task-bound")
    if telemetry.get("schema") != "cleanroom.citation-provider-call/v1":
        raise ContractError(f"checkpoint {sequence} telemetry schema differs")
    telemetry_keys = {
        "schema", "task_id", "request_sha256", "response_sha256",
        "provider_request_id", "attempts", "http_statuses", "latency_ms",
        "status", "failure_class",
    }
    if set(telemetry) != telemetry_keys:
        raise ContractError(f"checkpoint {sequence} telemetry shape differs")
    if not _is_sha256(telemetry["request_sha256"]):
        raise ContractError(f"checkpoint {sequence} request hash is invalid")
    if telemetry["response_sha256"] is not None and not _is_sha256(
        telemetry["response_sha256"]
    ):
        raise ContractError(f"checkpoint {sequence} response hash is invalid")
    if (
        isinstance(telemetry["latency_ms"], bool)
        or not isinstance(telemetry["latency_ms"], int)
        or telemetry["latency_ms"] < 0
    ):
        raise ContractError(f"checkpoint {sequence} latency is invalid")
    if telemetry["status"] not in {"PASS", "FAIL"}:
        raise ContractError(f"checkpoint {sequence} telemetry status is invalid")
    if telemetry["failure_class"] not in {None, "HTTP", "TRANSPORT", "RESPONSE"}:
        raise ContractError(f"checkpoint {sequence} failure class is invalid")
    if telemetry["status"] == "PASS" and (
        telemetry["failure_class"] is not None
        or not telemetry["provider_request_id"]
        or telemetry["response_sha256"] is None
    ):
        raise ContractError(f"checkpoint {sequence} success telemetry is incomplete")

    if not isinstance(telemetry.get("attempts"), int) or telemetry["attempts"] < 1:
        raise ContractError(f"checkpoint {sequence} telemetry attempts are invalid")
    if telemetry["attempts"] > max_retries + 1:
        raise ContractError(f"checkpoint {sequence} exceeds committed retry policy")
    if len(telemetry.get("http_statuses", [])) != telemetry["attempts"]:
        raise ContractError(f"checkpoint {sequence} telemetry attempt count differs")
    statuses = telemetry["http_statuses"]
    if any(
        status != "TRANSPORT" and not isinstance(status, int)
        for status in statuses
    ):
        raise ContractError(f"checkpoint {sequence} telemetry status history is invalid")
    if any(
        status != "TRANSPORT" and status not in {408, 425, 429, 500, 502, 503, 504}
        for status in statuses[:-1]
    ):
        raise ContractError(f"checkpoint {sequence} retried a non-retryable status")
    if telemetry["status"] == "PASS" and not (
        isinstance(statuses[-1], int) and 200 <= statuses[-1] < 300
    ):
        raise ContractError(f"checkpoint {sequence} final success status differs")
    if record["provider_completed"]:
        if telemetry.get("status") == "FAIL" and telemetry.get("failure_class") == "RESPONSE":
            if record["selection"] is not None or record["rendered"] is not None:
                raise ContractError(f"checkpoint {sequence} rejected response contains output")
            if record["error_code"] is None:
                raise ContractError(f"checkpoint {sequence} rejected response lacks error code")
            return
        if telemetry.get("status") != "PASS" or record["selection"] is None:
            raise ContractError(f"checkpoint {sequence} completion evidence differs")
        validate_schema(record["selection"], "claim-selection.schema.json")
        try:
            expected = validate_and_render(
                record["selection"], evidence_records=task["evidence"]
            )
        except (ContractError, TypeError, ValueError) as exc:
            if record["rendered"] is not None or record["error_code"] != _error_code(exc):
                raise ContractError(
                    f"checkpoint {sequence} validation rejection differs"
                ) from exc
        else:
            oracle_match = _selection_semantics(record["selection"]) == _selection_semantics(
                oracle["expected_selection"]
            )
            if not oracle_match:
                if record["rendered"] is not None or record["error_code"] != "ORACLE_MISMATCH":
                    raise ContractError(f"checkpoint {sequence} oracle rejection differs")
            else:
                if record["rendered"] != expected or record["error_code"] is not None:
                    raise ContractError(f"checkpoint {sequence} rendered output differs")
                validate_schema(record["rendered"], "rendered-answer.schema.json")
    else:
        if telemetry.get("status") != "FAIL":
            raise ContractError(f"checkpoint {sequence} failure telemetry differs")
        if record["selection"] is not None or record["rendered"] is not None:
            raise ContractError(f"checkpoint {sequence} failure contains output")
        if record["error_code"] is None:
            raise ContractError(f"checkpoint {sequence} failure lacks error code")


def _acceptance_receipt(
    records: Sequence[Mapping[str, Any]], *, task_set: Mapping[str, str]
) -> dict[str, Any]:
    summaries = []
    for record in records:
        selection = record["selection"]
        selection_valid = bool(record["provider_completed"] and selection is not None)
        semantic_valid = bool(selection_valid and record["rendered"] is not None)
        summaries.append({
            "sequence": record["sequence"],
            "task_id": record["task_id"],
            "task_sha256": record["task_sha256"],
            "oracle_sha256": record["oracle_sha256"],
            "provider_completed": record["provider_completed"],
            "selection_schema_valid": selection_valid,
            "semantic_valid": semantic_valid,
            "rendered_schema_valid": semantic_valid,
            "oracle_match": semantic_valid,
            "selection_sha256": digest(selection) if selection_valid else None,
            "rendered_sha256": digest(record["rendered"]) if semantic_valid else None,
            "error_code": record["error_code"],
        })
    metrics, gates = derive_acceptance_metrics_and_gates(
        summaries,
        execution_mode="REAL_PROVIDER",
        frozen_task_set_verified=True,
    )
    receipt: dict[str, Any] = {
        "schema": "cleanroom.citation-acceptance-run/v1",
        "provider": SchemaConstrainedHTTPProvider.name,
        "execution_mode": "REAL_PROVIDER",
        "task_set": dict(task_set),
        "acceptance_status": "PASS" if all(gates.values()) else "INCOMPLETE",
        "metrics": metrics,
        "gates": gates,
        "records": summaries,
    }
    receipt["receipt_sha256"] = digest(receipt)
    verify_citation_acceptance(receipt)
    return receipt


def _load_record(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"checkpoint {path} is not an object")
    return value


def verify_run(
    *,
    manifest_path: Path = MANIFEST_PATH,
    run_dir: Path,
    expected_public_key_path: Path | None = None,
) -> dict[str, Any]:
    expected_public_key = _public_key(expected_public_key_path)
    expected_public_bytes = _public_bytes(expected_public_key)
    expected_public_sha256 = hashlib.sha256(expected_public_bytes).hexdigest()
    tasks, oracles, task_set = _load_frozen_bundle(manifest_path)
    config_path = run_dir / "run-config.json"
    config_value = _load_record(config_path)
    if config_value.get("schema") != RUN_SCHEMA:
        raise ContractError("run configuration schema differs")
    config_keys = {
        "schema", "task_set", "task_set_sha256", "provider_config", "max_workers",
        "requests_per_second", "attestation_public_key_sha256", "config_sha256",
    }
    if set(config_value) != config_keys:
        raise ContractError("run configuration shape differs")
    unsigned_config = {
        key: value for key, value in config_value.items() if key != "config_sha256"
    }
    if config_value["config_sha256"] != digest(unsigned_config):
        raise ContractError("run configuration digest differs")
    if config_value["attestation_public_key_sha256"] != expected_public_sha256:
        raise ContractError("run attestation trust anchor differs")
    provider_config = config_value["provider_config"]
    provider_keys = {
        "provider", "endpoint_sha256", "model_sha256", "timeout_seconds",
        "max_retries", "retry_backoff_seconds", "max_output_tokens",
    }
    if not isinstance(provider_config, Mapping) or set(provider_config) != provider_keys:
        raise ContractError("public provider configuration shape differs")
    if (
        provider_config["provider"] != SchemaConstrainedHTTPProvider.name
        or not _is_sha256(provider_config["endpoint_sha256"])
        or not _is_sha256(provider_config["model_sha256"])
    ):
        raise ContractError("public provider commitment is invalid")
    if not (1 <= config_value["max_workers"] <= MAX_WORKERS):
        raise ContractError("recorded max-workers is outside strict bound")
    if not (
        0 < config_value["requests_per_second"] <= MAX_REQUESTS_PER_SECOND
    ):
        raise ContractError("recorded request rate is outside strict bound")

    if config_value.get("task_set_sha256") != _task_set_sha256(tasks):
        raise ContractError("run task-set commitment differs")
    if config_value.get("task_set") != task_set:
        raise ContractError("run frozen-manifest commitment differs")
    records: list[dict[str, Any]] = []
    for sequence, (task, oracle) in enumerate(zip(tasks, oracles, strict=True), start=1):
        record = _load_record(_record_path(run_dir, sequence))
        _validate_record(
            record,
            sequence=sequence,
            task=task,
            oracle=oracle,
            max_retries=provider_config["max_retries"],
        )
        records.append(record)
    expected_names = {f"{sequence:04d}.json" for sequence in range(1, 501)}
    actual_names = {path.name for path in (run_dir / "records").glob("*.json")}
    if actual_names != expected_names:
        raise ContractError("checkpoint set does not contain exactly sequences 1..500")
    expected_receipt = _acceptance_receipt(records, task_set=task_set)
    receipt = _load_record(run_dir / "receipt.json")
    verify_citation_acceptance(receipt)
    if receipt != expected_receipt:
        raise ContractError("citation acceptance receipt does not reproduce from checkpoints")
    manifest = _load_record(run_dir / "run-manifest.json")
    expected_manifest_payload = {
        "schema": RUN_SCHEMA,
        "task_set": task_set,
        "task_set_sha256": _task_set_sha256(tasks),
        "run_config_sha256": digest(config_value),
        "record_sha256s": [record["record_sha256"] for record in records],
        "receipt_sha256": receipt["receipt_sha256"],
    }
    manifest_sha256 = digest(expected_manifest_payload)
    if (
        {key: manifest.get(key) for key in expected_manifest_payload}
        != expected_manifest_payload
        or manifest.get("manifest_sha256") != manifest_sha256
    ):
        raise ContractError("run manifest does not enumerate exact verified artefacts")
    attestation = manifest.get("attestation")
    if not isinstance(attestation, Mapping) or set(attestation) != {
        "algorithm",
        "public_key_b64",
        "signed_manifest_sha256",
        "signature_b64",
    }:
        raise ContractError("run manifest attestation shape differs")
    if (
        attestation["algorithm"] != "Ed25519"
        or attestation["signed_manifest_sha256"] != manifest_sha256
    ):
        raise ContractError("run manifest attestation binding differs")
    try:
        public_bytes = base64.b64decode(attestation["public_key_b64"], validate=True)
        signature = base64.b64decode(attestation["signature_b64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ContractError("run manifest attestation encoding is invalid") from exc
    if public_bytes != expected_public_bytes:
        raise ContractError("run manifest attestation trust anchor differs")
    try:
        expected_public_key.verify(signature, bytes.fromhex(manifest_sha256))
    except (InvalidSignature, ValueError) as exc:
        raise ContractError("run manifest attestation signature is invalid") from exc
    if set(manifest) != set(expected_manifest_payload) | {
        "manifest_sha256",
        "attestation",
    }:
        raise ContractError("run manifest shape differs")
    return receipt


def run(
    *,
    manifest_path: Path = MANIFEST_PATH,
    run_dir: Path,
    config: CitationProviderConfig,
    max_workers: int,
    requests_per_second: float,
    attestation_private_key_path: Path | None = None,
    expected_public_key_path: Path | None = None,
) -> dict[str, Any]:
    if not (1 <= max_workers <= MAX_WORKERS):
        raise ContractError("max-workers is outside strict bound")
    tasks, oracles, task_set = _load_frozen_bundle(manifest_path)
    attestation_private_key = _private_key(attestation_private_key_path)
    expected_public_key = _public_key(expected_public_key_path)
    attestation_public_bytes = _public_bytes(attestation_private_key.public_key())
    if attestation_public_bytes != _public_bytes(expected_public_key):
        raise ContractError("citation attestation private/public keys differ")
    config_value = {
        "schema": RUN_SCHEMA,
        "task_set": task_set,
        "task_set_sha256": _task_set_sha256(tasks),
        "provider_config": _public_config(config),
        "max_workers": max_workers,
        "requests_per_second": requests_per_second,
        "attestation_public_key_sha256": hashlib.sha256(
            attestation_public_bytes
        ).hexdigest(),
    }
    config_value["config_sha256"] = digest(config_value)
    config_path = run_dir / "run-config.json"
    if config_path.exists():
        if _load_record(config_path) != config_value:
            raise ContractError("resume configuration differs from existing run")
    else:
        _atomic_json(config_path, config_value)

    pending: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
    records: dict[int, dict[str, Any]] = {}
    for sequence, (task, oracle) in enumerate(zip(tasks, oracles, strict=True), start=1):
        path = _record_path(run_dir, sequence)
        if path.exists():
            record = _load_record(path)
            _validate_record(
                record,
                sequence=sequence,
                task=task,
                oracle=oracle,
                max_retries=config.max_retries,
            )
            records[sequence] = record
            try:
                _started_path(run_dir, sequence).unlink()
            except FileNotFoundError:
                pass
        elif _started_path(run_dir, sequence).exists():
            raise ContractError(
                f"logical call {sequence} has indeterminate completion; "
                "do not resume this acceptance run"
            )
        else:
            pending.append((sequence, task, oracle))

    limiter = RateLimiter(requests_per_second)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _execute_one,
                sequence=sequence,
                task=task,
                oracle=oracle,
                config=config,
                limiter=limiter,
                record_path=_record_path(run_dir, sequence),
                started_path=_started_path(run_dir, sequence),
            ): (sequence, task)
            for sequence, task, oracle in pending
        }
        try:
            for future in as_completed(futures):
                sequence, _task = futures[future]
                record = future.result()
                records[sequence] = record
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    ordered = [records[sequence] for sequence in range(1, 501)]
    receipt = _acceptance_receipt(ordered, task_set=task_set)
    _atomic_json(run_dir / "receipt.json", receipt)
    manifest_payload: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "task_set": task_set,
        "task_set_sha256": _task_set_sha256(tasks),
        "run_config_sha256": digest(config_value),
        "record_sha256s": [record["record_sha256"] for record in ordered],
        "receipt_sha256": receipt["receipt_sha256"],
    }
    manifest_sha256 = digest(manifest_payload)
    manifest = {
        **manifest_payload,
        "manifest_sha256": manifest_sha256,
        "attestation": {
            "algorithm": "Ed25519",
            "public_key_b64": base64.b64encode(attestation_public_bytes).decode("ascii"),
            "signed_manifest_sha256": manifest_sha256,
            "signature_b64": base64.b64encode(
                attestation_private_key.sign(bytes.fromhex(manifest_sha256))
            ).decode("ascii"),
        },
    }
    _atomic_json(run_dir / "run-manifest.json", manifest)
    return verify_run(
        manifest_path=manifest_path,
        run_dir=run_dir,
        expected_public_key_path=expected_public_key_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run or resume exactly 500 calls")
    run_parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    run_parser.add_argument("--run-dir", required=True, type=Path)
    run_parser.add_argument("--attestation-private-key", type=Path)
    run_parser.add_argument("--attestation-public-key", type=Path)
    run_parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("CLEANROOM_CITATION_RUN_MAX_WORKERS", "4")),
    )
    run_parser.add_argument(
        "--requests-per-second",
        type=float,
        default=float(os.environ.get("CLEANROOM_CITATION_RUN_REQUESTS_PER_SECOND", "2")),
    )
    verify_parser = subparsers.add_parser("verify", help="verify a completed run")
    verify_parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    verify_parser.add_argument("--run-dir", required=True, type=Path)
    verify_parser.add_argument("--attestation-public-key", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            receipt = run(
                manifest_path=args.manifest,
                run_dir=args.run_dir,
                config=CitationProviderConfig.from_env(),
                max_workers=args.max_workers,
                requests_per_second=args.requests_per_second,
                attestation_private_key_path=args.attestation_private_key,
                expected_public_key_path=args.attestation_public_key,
            )
        else:
            receipt = verify_run(
                manifest_path=args.manifest,
                run_dir=args.run_dir,
                expected_public_key_path=args.attestation_public_key,
            )
    except ContractError as exc:
        print(f"citation acceptance: ERROR: {exc}")
        return 2
    print(
        f"citation acceptance: {receipt['acceptance_status']} "
        f"({receipt['metrics']['completed_calls']}/500 completed)"
    )
    return 0 if receipt["acceptance_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
