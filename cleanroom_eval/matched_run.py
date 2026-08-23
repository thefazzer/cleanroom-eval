"""Fail-closed orchestration for the real clean-room matched experiment.

This layer sits in front of :mod:`cleanroom_eval.runner`.  It does not train a
model and it never turns model names into evidence.  Instead it rehashes the
actual arm artifacts, verifies training receipts, freezes a balanced
arm/seed/episode allocation, and writes independently verifiable per-job
receipts.  Existing valid receipts are reused, making provider execution
resumable without silently accepting stale or foreign output.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import threading
from typing import Any, cast

from .contract import (
    ASSET_DIR,
    CLASSIFICATION,
    ContractError,
    canonical_bytes,
    digest,
    file_sha256,
    load_json,
    validate_schema,
    verify_contamination,
)
from .runner import ArmRunner, ProviderAdapter, load_provider, load_public_assets


ARM_KINDS = {
    "BASE": "FROZEN_BASE",
    "SFT": "SFT_DOMAIN",
    "SFT_MATCHED_CONTROL": "SFT_CONTROL",
    "RL": "RL_DOMAIN",
    "RL_MATCHED_CONTROL": "RL_CONTROL",
}
MATCHED_PAIRS = (
    ("SFT", "SFT_MATCHED_CONTROL"),
    ("RL", "RL_MATCHED_CONTROL"),
)
EQUAL_COMPUTE_FIELDS = (
    "training_tokens",
    "optimizer_steps",
    "compute_budget",
    "seed_schedule_sha256",
    "tool_policy_sha256",
)
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not parent_existed:
        _fsync_directory(path.parent.parent)
    temporary = path.with_suffix(
        path.suffix + f".tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _artifact(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        path = Path(value["path"]).expanduser()
        expected = value["sha256"]
    except (KeyError, TypeError) as exc:
        raise ContractError(f"{label} artifact reference is invalid") from exc
    if not path.is_file():
        raise ContractError(f"{label} artifact is missing: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ContractError(f"{label} artifact hash differs: {path}")
    return {
        "sha256": observed,
        "bytes": path.stat().st_size,
    }


def _training_receipt(
    reference: Mapping[str, Any],
    *,
    arm: Mapping[str, Any],
    source_checkpoint_sha256: str,
    checkpoint_sha256: str,
    training_sha256: str,
    seed_schedule_sha256: str,
) -> dict[str, Any]:
    committed = _artifact(reference, label=f"{arm['arm_id']} training receipt")
    receipt = load_json(Path(reference["path"]).expanduser())
    validate_schema(receipt, "matched-training-receipt.schema.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != digest(unsigned):
        raise ContractError(f"{arm['arm_id']} training receipt digest differs")
    expected = {
        "arm_id": arm["arm_id"],
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "output_checkpoint_sha256": checkpoint_sha256,
        "training_material_sha256": training_sha256,
        "seed_schedule_sha256": seed_schedule_sha256,
        "training_tokens": arm["training_tokens"],
        "optimizer_steps": arm["optimizer_steps"],
        "compute_budget": arm["compute_budget"],
    }
    for key, value in expected.items():
        if receipt[key] != value:
            raise ContractError(
                f"{arm['arm_id']} training receipt differs on {key}"
            )
    return {
        **committed,
        "claims": receipt,
        "claims_sha256": digest(receipt),
    }


def _seed_schedule(seeds: Sequence[Any]) -> tuple[list[int], str]:
    if (
        isinstance(seeds, (str, bytes))
        or len(seeds) < 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ContractError("evaluation seed schedule requires at least 3 unique integers")
    ordered = list(seeds)
    return ordered, digest({"evaluation_seeds": ordered})


def _inference_policy(reference: Mapping[str, Any]) -> dict[str, Any]:
    policy = load_json(Path(reference["path"]).expanduser())
    required = {"temperature", "timeout_seconds", "max_retries", "max_output_tokens"}
    if set(policy) != required:
        raise ContractError("inference configuration fields differ")
    if policy["temperature"] != 0:
        raise ContractError("matched inference temperature must be zero")
    if (
        isinstance(policy["max_retries"], bool)
        or not isinstance(policy["max_retries"], int)
        or not 0 <= policy["max_retries"] <= 2
    ):
        raise ContractError("matched inference retry policy is invalid")
    if (
        not isinstance(policy["timeout_seconds"], (int, float))
        or not 0 < policy["timeout_seconds"] <= 120
    ):
        raise ContractError("matched inference timeout is invalid")
    if (
        isinstance(policy["max_output_tokens"], bool)
        or not isinstance(policy["max_output_tokens"], int)
        or not 32 <= policy["max_output_tokens"] <= 4096
    ):
        raise ContractError("matched inference output-token limit is invalid")
    return dict(policy)


def _contamination_report(
    training_paths: Sequence[Path], experiment: Mapping[str, Any]
) -> dict[str, Any]:
    report = verify_contamination(
        candidate_paths=training_paths,
        sealed_paths=sorted((ASSET_DIR / "episodes").glob("*.json")),
        shingle_width=experiment["contamination_policy"]["shingle_width"],
        maximum_overlap_ratio=experiment["contamination_policy"][
            "maximum_overlap_ratio"
        ],
    )
    if report["status"] != "PASS":
        raise ContractError("matched-run contamination gate failed")
    return report


def build_run_plan(
    inputs: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any],
    episode_ids: Sequence[str],
) -> dict[str, Any]:
    """Rehash concrete artifacts and freeze the complete matched allocation."""

    validate_schema(inputs, "matched-run-inputs.schema.json")
    if inputs["classification"] != CLASSIFICATION:
        raise ContractError("matched-run inputs are not clean-room synthetic")
    if inputs["experiment_id"] != experiment["experiment_id"]:
        raise ContractError("matched-run inputs reference a different experiment")
    provider_identity = dict(inputs["provider_identity"])

    evaluator = _artifact(inputs["evaluator"], label="evaluator")
    tool_policy = _artifact(inputs["tool_policy"], label="tool policy")
    inference_config = _artifact(
        inputs["inference_config"], label="inference configuration"
    )
    inference_policy = _inference_policy(inputs["inference_config"])
    seeds, schedule_sha = _seed_schedule(inputs["evaluation_seeds"])
    arms_by_id = {arm["arm_id"]: arm for arm in inputs["arms"]}
    if set(arms_by_id) != set(ARM_KINDS):
        raise ContractError("matched-run inputs do not cover exactly the frozen arms")

    checkpoints = {
        arm_id: _artifact(arms_by_id[arm_id]["checkpoint"], label=f"{arm_id} checkpoint")
        for arm_id in ARM_KINDS
    }
    if len({item["sha256"] for item in checkpoints.values()}) != len(checkpoints):
        raise ContractError(
            "each frozen arm requires a distinct checkpoint artifact; "
            "one generic model cannot impersonate trained arms"
        )
    source_arms = {
        "SFT": "BASE",
        "SFT_MATCHED_CONTROL": "BASE",
        "RL": "SFT",
        "RL_MATCHED_CONTROL": "SFT_MATCHED_CONTROL",
    }
    arm_commitments: list[dict[str, Any]] = []
    for arm_id in ARM_KINDS:
        arm = arms_by_id[arm_id]
        if arm["checkpoint_kind"] != ARM_KINDS[arm_id]:
            raise ContractError(f"{arm_id} checkpoint kind is not admissible")
        checkpoint = checkpoints[arm_id]
        training_ref = arm.get("training_material")
        receipt_ref = arm.get("training_receipt")
        optimizer_ref = arm.get("optimizer_config")
        reward_ref = arm.get("reward_config")
        if arm_id == "BASE":
            if any(
                value is not None
                for value in (training_ref, receipt_ref, optimizer_ref, reward_ref)
            ):
                raise ContractError("BASE must not claim adaptation artifacts or configs")
            if any(arm[field] != 0 for field in ("training_tokens", "optimizer_steps")):
                raise ContractError("BASE must have zero training tokens and steps")
            training = None
            training_receipt = None
            optimizer_config = None
            reward_config = None
        else:
            if not isinstance(training_ref, Mapping) or not isinstance(
                receipt_ref, Mapping
            ):
                raise ContractError(f"{arm_id} lacks concrete training artifacts")
            if not isinstance(optimizer_ref, Mapping):
                raise ContractError(f"{arm_id} lacks a concrete optimizer configuration")
            if arm_id.startswith("RL") and not isinstance(reward_ref, Mapping):
                raise ContractError(f"{arm_id} lacks a concrete reward configuration")
            if arm_id.startswith("SFT") and reward_ref is not None:
                raise ContractError(f"{arm_id} must not claim an RL reward configuration")
            training = _artifact(
                training_ref, label=f"{arm_id} training material"
            )
            training_receipt = _training_receipt(
                receipt_ref,
                arm=arm,
                source_checkpoint_sha256=checkpoints[source_arms[arm_id]]["sha256"],
                checkpoint_sha256=checkpoint["sha256"],
                training_sha256=training["sha256"],
                seed_schedule_sha256=schedule_sha,
            )
            optimizer_config = _artifact(
                optimizer_ref, label=f"{arm_id} optimizer configuration"
            )
            reward_config = (
                _artifact(reward_ref, label=f"{arm_id} reward configuration")
                if isinstance(reward_ref, Mapping)
                else None
            )
        arm_commitments.append(
            {
                "arm_id": arm_id,
                "checkpoint_kind": arm["checkpoint_kind"],
                "model_identifier": arm["model_identifier"],
                "checkpoint": checkpoint,
                "training_material": training,
                "training_receipt": training_receipt,
                "optimizer_config": optimizer_config,
                "reward_config": reward_config,
                "training_tokens": arm["training_tokens"],
                "optimizer_steps": arm["optimizer_steps"],
                "compute_budget": arm["compute_budget"],
                "seed_schedule_sha256": schedule_sha,
                "tool_policy_sha256": tool_policy["sha256"],
            }
        )

    indexed = {item["arm_id"]: item for item in arm_commitments}
    for treatment, control in MATCHED_PAIRS:
        for field in EQUAL_COMPUTE_FIELDS:
            if indexed[treatment][field] != indexed[control][field]:
                raise ContractError(
                    f"matched pair {treatment}/{control} differs on {field}"
                )

    training_paths = [
        Path(arms_by_id[arm_id]["training_material"]["path"]).expanduser()
        for arm_id in ARM_KINDS
        if arm_id != "BASE"
    ]
    contamination = _contamination_report(training_paths, experiment)
    job_context_sha256 = digest(
        {
            "experiment_sha256": digest(experiment),
            "provider_identity": provider_identity,
            "evaluator": evaluator,
            "tool_policy": tool_policy,
            "inference_config": inference_config,
            "inference_policy": inference_policy,
            "evaluation_seeds": seeds,
            "arms": arm_commitments,
            "contamination": contamination,
            "episode_ids": sorted(episode_ids),
        }
    )

    jobs = []
    for arm_id in ARM_KINDS:
        arm = indexed[arm_id]
        for seed_index, seed in enumerate(seeds):
            for episode_id in sorted(episode_ids):
                job = {
                    "arm_id": arm_id,
                    "checkpoint_sha256": arm["checkpoint"]["sha256"],
                    "job_context_sha256": job_context_sha256,
                    "episode_id": episode_id,
                    "evaluation_seed": seed,
                    "seed_index": seed_index,
                }
                job["job_id"] = digest(job)
                jobs.append(job)
    # Hash-order prevents arm or episode ordering from becoming a provider signal.
    jobs.sort(key=lambda item: item["job_id"])
    plan: dict[str, Any] = {
        "schema": "cleanroom.matched-run-plan/v1",
        "classification": CLASSIFICATION,
        "experiment_id": experiment["experiment_id"],
        "experiment_sha256": digest(experiment),
        "provider_identity": provider_identity,
        "evaluator": evaluator,
        "tool_policy": tool_policy,
        "inference_config": inference_config,
        "inference_policy": inference_policy,
        "evaluation_seeds": seeds,
        "seed_schedule_sha256": schedule_sha,
        "arms": arm_commitments,
        "contamination": contamination,
        "job_context_sha256": job_context_sha256,
        "episode_ids": sorted(episode_ids),
        "jobs": jobs,
    }
    plan["plan_sha256"] = digest(plan)
    validate_schema(plan, "matched-run-plan.schema.json")
    verify_run_plan(plan, experiment=experiment, episode_ids=set(episode_ids))
    return plan


def verify_run_plan(
    plan: Mapping[str, Any],
    *,
    experiment: Mapping[str, Any],
    episode_ids: set[str],
) -> None:
    validate_schema(plan, "matched-run-plan.schema.json")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan["plan_sha256"] != digest(unsigned):
        raise ContractError("matched-run plan digest differs")
    if plan["experiment_sha256"] != digest(experiment):
        raise ContractError("matched-run plan experiment commitment differs")
    contamination = plan["contamination"]
    if contamination.get("status") != "PASS":
        raise ContractError("matched-run plan contamination gate failed")
    committed_training = {
        item["training_material"]["sha256"]
        for item in plan["arms"]
        if item["training_material"] is not None
    }
    if {
        item["sha256"] for item in contamination.get("candidate_files", [])
    } != committed_training:
        raise ContractError("matched-run contamination inputs differ from arm commitments")
    expected_job_context = digest(
        {
            "experiment_sha256": plan["experiment_sha256"],
            "provider_identity": plan["provider_identity"],
            "evaluator": plan["evaluator"],
            "tool_policy": plan["tool_policy"],
            "inference_config": plan["inference_config"],
            "inference_policy": plan["inference_policy"],
            "evaluation_seeds": plan["evaluation_seeds"],
            "arms": plan["arms"],
            "contamination": plan["contamination"],
            "episode_ids": plan["episode_ids"],
        }
    )
    if plan["job_context_sha256"] != expected_job_context:
        raise ContractError("matched-run job context commitment differs")
    policy = plan["inference_policy"]
    if (
        set(policy) != {"temperature", "timeout_seconds", "max_retries", "max_output_tokens"}
        or policy["temperature"] != 0
        or isinstance(policy["max_retries"], bool)
        or not isinstance(policy["max_retries"], int)
        or not 0 <= policy["max_retries"] <= 2
    ):
        raise ContractError("matched-run inference policy is invalid")
    if set(plan["episode_ids"]) != episode_ids:
        raise ContractError("matched-run plan episode allocation differs")
    arms = {item["arm_id"]: item for item in plan["arms"]}
    if set(arms) != set(ARM_KINDS):
        raise ContractError("matched-run plan arm set differs")
    for arm_id, arm in arms.items():
        if arm["checkpoint_kind"] != ARM_KINDS[arm_id]:
            raise ContractError(f"{arm_id} checkpoint kind is not admissible")
        adaptation = (
            arm["training_material"],
            arm["training_receipt"],
            arm["optimizer_config"],
        )
        if arm_id == "BASE":
            if any(value is not None for value in (*adaptation, arm["reward_config"])):
                raise ContractError("BASE must not claim adaptation artifacts or configs")
            if arm["training_tokens"] != 0 or arm["optimizer_steps"] != 0:
                raise ContractError("BASE must have zero training tokens and steps")
        else:
            if any(value is None for value in adaptation):
                raise ContractError(f"{arm_id} lacks committed adaptation artifacts")
            if arm["training_tokens"] <= 0 or arm["optimizer_steps"] <= 0:
                raise ContractError(f"{arm_id} adaptation compute is not positive")
            if arm_id.startswith("RL") and arm["reward_config"] is None:
                raise ContractError(f"{arm_id} lacks a committed reward configuration")
            if arm_id.startswith("SFT") and arm["reward_config"] is not None:
                raise ContractError(f"{arm_id} must not claim an RL reward configuration")
        if arm["seed_schedule_sha256"] != plan["seed_schedule_sha256"]:
            raise ContractError(f"{arm_id} seed schedule commitment differs")
        if arm["tool_policy_sha256"] != plan["tool_policy"]["sha256"]:
            raise ContractError(f"{arm_id} tool policy commitment differs")
    source_arms = {
        "SFT": "BASE",
        "SFT_MATCHED_CONTROL": "BASE",
        "RL": "SFT",
        "RL_MATCHED_CONTROL": "SFT_MATCHED_CONTROL",
    }
    for arm_id, source_arm in source_arms.items():
        arm = arms[arm_id]
        receipt = arm["training_receipt"]
        claims = receipt["claims"]
        validate_schema(claims, "matched-training-receipt.schema.json")
        if receipt["claims_sha256"] != digest(claims):
            raise ContractError(f"{arm_id} training receipt claims digest differs")
        unsigned = {
            key: value for key, value in claims.items() if key != "receipt_sha256"
        }
        if claims["receipt_sha256"] != digest(unsigned):
            raise ContractError(f"{arm_id} training receipt digest differs")
        expected_claims = {
            "arm_id": arm_id,
            "source_checkpoint_sha256": arms[source_arm]["checkpoint"]["sha256"],
            "output_checkpoint_sha256": arm["checkpoint"]["sha256"],
            "training_material_sha256": arm["training_material"]["sha256"],
            "seed_schedule_sha256": plan["seed_schedule_sha256"],
            "training_tokens": arm["training_tokens"],
            "optimizer_steps": arm["optimizer_steps"],
            "compute_budget": arm["compute_budget"],
        }
        for field, expected in expected_claims.items():
            if claims[field] != expected:
                raise ContractError(
                    f"{arm_id} training receipt differs on {field}"
                )
    if len({item["checkpoint"]["sha256"] for item in arms.values()}) != len(arms):
        raise ContractError("matched-run plan aliases checkpoints across arms")
    seeds = plan["evaluation_seeds"]
    if plan["seed_schedule_sha256"] != digest({"evaluation_seeds": seeds}):
        raise ContractError("matched-run seed schedule commitment differs")
    expected = {
        (arm_id, seed, episode_id)
        for arm_id in ARM_KINDS
        for seed in seeds
        for episode_id in episode_ids
    }
    observed = {
        (job["arm_id"], job["evaluation_seed"], job["episode_id"])
        for job in plan["jobs"]
    }
    if observed != expected or len(plan["jobs"]) != len(expected):
        raise ContractError("matched-run job allocation is not complete and unique")
    if len({job["job_id"] for job in plan["jobs"]}) != len(plan["jobs"]):
        raise ContractError("matched-run plan contains duplicate job ids")
    for job in plan["jobs"]:
        unsigned_job = {key: value for key, value in job.items() if key != "job_id"}
        if job["job_id"] != digest(unsigned_job):
            raise ContractError(f"matched-run job digest differs: {job['job_id']}")
        if job["job_context_sha256"] != plan["job_context_sha256"]:
            raise ContractError(f"matched-run job context differs: {job['job_id']}")
        if job["checkpoint_sha256"] != arms[job["arm_id"]]["checkpoint"]["sha256"]:
            raise ContractError(f"matched-run job checkpoint differs: {job['job_id']}")
    for treatment, control in MATCHED_PAIRS:
        for field in EQUAL_COMPUTE_FIELDS:
            if arms[treatment][field] != arms[control][field]:
                raise ContractError(
                    f"matched pair {treatment}/{control} differs on {field}"
                )


def _job_receipt(
    *,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    outcome: Any,
    provider: ProviderAdapter,
    telemetry_start: int,
) -> dict[str, Any]:
    telemetry = (
        list(provider.telemetry())[telemetry_start:]
        if callable(getattr(provider, "telemetry", None))
        else []
    )
    provider_metadata = dict(provider.metadata_for_arm({"id": job["arm_id"]}))
    receipt: dict[str, Any] = {
        "schema": "cleanroom.matched-job-receipt/v1",
        "classification": CLASSIFICATION,
        "plan_sha256": plan["plan_sha256"],
        "job_id": job["job_id"],
        "arm_id": job["arm_id"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "episode_id": job["episode_id"],
        "evaluation_seed": job["evaluation_seed"],
        "provider": provider.name,
        "provider_endpoint_sha256": provider_metadata["provider_endpoint_sha256"],
        "execution_mode": provider.execution_mode,
        "provider_lineage": dict(provider.training_lineage()),
        "metrics": outcome.metrics,
        "gate_counts": outcome.gate_counts,
        "records": outcome.records,
        "provider_calls": telemetry,
    }
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


class _ReceiptReplayProvider:
    """Replay retained provider outputs while checking request/output hashes."""

    name = "matched-receipt-replay"
    execution_mode = "REAL_PROVIDER"

    def __init__(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        calls: Sequence[Mapping[str, Any]],
        job: Mapping[str, Any],
        max_retries: int,
    ) -> None:
        self._records = records
        self._calls = calls
        self._job = job
        self._max_retries = max_retries
        self._index = 0

    def act(self, arm_id: str, request: Mapping[str, Any]) -> Any:
        if self._index >= len(self._records):
            raise ContractError("matched receipt replay has too few retained outputs")
        record = self._records[self._index]
        call = self._calls[self._index]
        self._index += 1
        if (
            arm_id != self._job["arm_id"]
            or record.get("arm_id") != arm_id
            or record.get("episode_id") != request.get("episode_id")
            or record.get("event_id") != request.get("event_id")
        ):
            raise ContractError("matched receipt replay record binding differs")
        response = record.get("provider_response")
        wire_request = call.get("wire_request")
        if not isinstance(wire_request, Mapping):
            raise ContractError("matched receipt wire request is absent")
        if call.get("evaluator_request_sha256") != digest(request):
            raise ContractError("matched receipt evaluator request hash differs")
        if call.get("wire_request_sha256") != digest(wire_request):
            raise ContractError("matched receipt wire request hash differs")
        if call.get("request_sha256") != call.get("wire_request_sha256"):
            raise ContractError("matched receipt legacy request hash domain differs")
        messages = wire_request.get("messages")
        if messages is None:
            if dict(wire_request) != dict(request):
                raise ContractError("matched receipt wire/evaluator request differs")
        else:
            try:
                provider_visible = json.loads(messages[1]["content"])
            except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ContractError(
                    "matched receipt wire request cannot be reconstructed"
                ) from exc
            if provider_visible != request:
                raise ContractError("matched receipt wire/evaluator request differs")
        self._validate_call_history(call, response_present=response is not None)
        if call.get("event_id") != request.get("event_id"):
            raise ContractError("matched receipt replay request/output hash differs")
        if response is None:
            if (
                call.get("status") != "FAIL"
                or call.get("response_sha256") is not None
                or not isinstance(record.get("provider_error"), str)
            ):
                raise ContractError("matched receipt failure evidence differs")
            raise ContractError(record["provider_error"])
        raw_b64 = call.get("raw_response_b64")
        if not isinstance(raw_b64, str):
            raise ContractError("matched receipt raw response is absent")
        try:
            raw = base64.b64decode(raw_b64, validate=True)
            envelope = json.loads(raw)
            parsed = (
                envelope["response"]
                if isinstance(envelope, Mapping) and "response" in envelope
                else json.loads(envelope["choices"][0]["message"]["content"])
            )
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ContractError(
                "matched receipt raw response cannot be reconstructed"
            ) from exc
        raw_sha = hashlib.sha256(raw).hexdigest()
        if (
            call.get("raw_response_sha256") != raw_sha
            or call.get("response_sha256") != raw_sha
            or call.get("parsed_response_sha256") != digest(response)
            or parsed != response
        ):
            raise ContractError("matched receipt replay request/output hash differs")
        return response

    def _validate_call_history(
        self, call: Mapping[str, Any], *, response_present: bool
    ) -> None:
        attempts = call.get("attempts")
        statuses = call.get("http_statuses")
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or not isinstance(statuses, list)
            or attempts != len(statuses)
            or not 1 <= attempts <= self._max_retries + 1
        ):
            raise ContractError("matched provider telemetry retry count differs")
        assert isinstance(attempts, int)
        assert isinstance(statuses, list)
        if any(
            status != "TRANSPORT" and status not in RETRYABLE_STATUS
            for status in statuses[:-1]
        ):
            raise ContractError("matched provider telemetry retried a terminal status")
        status = call.get("status")
        if status == "PASS":
            if (
                not response_present
                or not isinstance(statuses[-1], int)
                or not 200 <= statuses[-1] < 300
                or not call.get("provider_request_id")
                or call.get("failure_class") is not None
                or call.get("parsed_response_sha256") is None
                or call.get("raw_response_sha256") is None
            ):
                raise ContractError(
                    "matched provider telemetry success terminal state differs"
                )
        elif status == "FAIL":
            if response_present or call.get("failure_class") not in {
                "HTTP",
                "TRANSPORT",
                "RESPONSE",
            }:
                raise ContractError(
                    "matched provider telemetry failure terminal state differs"
                )
        else:
            raise ContractError("matched provider telemetry status differs")
        prompt = call.get("prompt_tokens")
        completion = call.get("completion_tokens")
        total = call.get("total_tokens")
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in (prompt, completion, total)
            )
            or total < prompt + completion
        ):
            raise ContractError("matched provider telemetry token totals differ")

    def assert_complete(self) -> None:
        if self._index != len(self._records):
            raise ContractError("matched receipt replay has extra retained outputs")


def verify_job_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    experiment: Mapping[str, Any],
    episode: Mapping[str, Any],
) -> None:
    validate_schema(receipt, "matched-job-receipt.schema.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != digest(unsigned):
        raise ContractError("matched job receipt digest differs")
    if receipt["plan_sha256"] != plan["plan_sha256"]:
        raise ContractError("matched job receipt plan commitment differs")
    jobs = {item["job_id"]: item for item in plan["jobs"]}
    job = jobs.get(receipt["job_id"])
    if job is None:
        raise ContractError("matched job receipt is not allocated by the run plan")
    for field in (
        "arm_id",
        "checkpoint_sha256",
        "episode_id",
        "evaluation_seed",
    ):
        if receipt[field] != job[field]:
            raise ContractError(f"matched job receipt differs on {field}")
    if receipt["execution_mode"] != "REAL_PROVIDER":
        raise ContractError("matched real-run receipt was not produced by a real provider")
    if (
        receipt["provider"] != plan["provider_identity"]["name"]
        or receipt["provider_endpoint_sha256"]
        != plan["provider_identity"]["endpoint_sha256"]
    ):
        raise ContractError("matched job receipt provider identity differs")
    lineage = receipt["provider_lineage"]
    if (
        episode.get("world_id") in set(lineage["world_ids"])
        or episode.get("template_family") in set(lineage["template_families"])
    ):
        raise ContractError(
            "matched job provider lineage overlaps sealed evaluation lineage"
        )
    if episode.get("episode_id") != job["episode_id"]:
        raise ContractError("matched job verifier received a different episode")
    records = receipt["records"]
    calls = receipt["provider_calls"]
    if len(calls) != len(records):
        raise ContractError("matched job provider-call coverage differs")
    for call in calls:
        if (
            call.get("job_id") != job["job_id"]
            or call.get("evaluation_seed") != job["evaluation_seed"]
            or call.get("arm_id") != job["arm_id"]
            or call.get("episode_id") != job["episode_id"]
            or call.get("status") not in {"PASS", "FAIL"}
        ):
            raise ContractError("matched job provider-call binding differs")
    metrics = receipt["metrics"]
    if (
        metrics.get("events") != len(records)
        or metrics.get("episodes") != 1
        or metrics.get("action_matches") != sum(record.get("action_match") is True for record in records)
        or metrics.get("supported_citations") != sum(record.get("citation_supported") is True for record in records)
    ):
        raise ContractError("matched job metrics differ from records")
    if not records or {
        (record["arm_id"], record["episode_id"]) for record in records
    } != {(job["arm_id"], job["episode_id"])}:
        raise ContractError("matched job receipt record coverage differs")
    experiment_arms = {item["id"]: item for item in experiment["arms"]}
    if job["arm_id"] not in experiment_arms:
        raise ContractError("matched job arm is absent from the frozen experiment")
    replay = _ReceiptReplayProvider(
        records=records,
        calls=calls,
        job=job,
        max_retries=plan["inference_policy"]["max_retries"],
    )
    for record, call in zip(records, calls, strict=True):
        replay._validate_call_history(
            call, response_present=record.get("provider_response") is not None
        )
    try:
        outcome = ArmRunner(
            provider=cast(ProviderAdapter, replay),
            taxonomy={},
            experiment=experiment,
            episodes=[episode],
            training_paths=[],
        )._run_arm(experiment_arms[job["arm_id"]])
    except ContractError as exc:
        raise ContractError(
            f"matched job receipt replay differs from retained outputs: {exc}"
        ) from exc
    replay.assert_complete()
    if (
        outcome.metrics != receipt["metrics"]
        or outcome.gate_counts != receipt["gate_counts"]
        or outcome.records != records
    ):
        raise ContractError(
            "matched job receipt replay differs from retained outputs "
            f"(metrics={outcome.metrics == receipt['metrics']}, "
            f"gates={outcome.gate_counts == receipt['gate_counts']}, "
            f"records={outcome.records == records})"
        )


def execute_pending_jobs(
    *,
    plan: Mapping[str, Any],
    provider: ProviderAdapter,
    taxonomy: Mapping[str, Any],
    experiment: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    training_paths: Sequence[Path],
    receipt_dir: Path,
    max_jobs: int | None = None,
) -> dict[str, int]:
    """Execute missing jobs and reuse only receipts that verify against the plan."""

    verify_run_plan(
        plan,
        experiment=experiment,
        episode_ids={item["episode_id"] for item in episodes},
    )
    if provider.execution_mode != "REAL_PROVIDER":
        raise ContractError("matched execution requires an actual REAL_PROVIDER adapter")
    if provider.name != plan["provider_identity"]["name"]:
        raise ContractError("active provider identity differs from frozen plan")
    begin_job = getattr(provider, "begin_job", None)
    if not callable(begin_job):
        raise ContractError("real provider lacks begin_job seed/checkpoint binding")
    plan_arms = {item["arm_id"]: item for item in plan["arms"]}
    experiment_arms = {item["id"]: item for item in experiment["arms"]}
    episode_index = {item["episode_id"]: item for item in episodes}
    observed_training = {file_sha256(path) for path in training_paths}
    committed_training = {
        arm["training_material"]["sha256"]
        for arm in plan_arms.values()
        if arm["training_material"] is not None
    }
    if observed_training != committed_training:
        raise ContractError("execution training files differ from frozen arm commitments")
    if _contamination_report(training_paths, experiment) != plan["contamination"]:
        raise ContractError("execution contamination report differs from frozen plan")
    lineage = provider.training_lineage()
    evaluation_worlds = {item["world_id"] for item in episodes}
    evaluation_templates = {item["template_family"] for item in episodes}
    if (
        set(lineage["world_ids"]) & evaluation_worlds
        or set(lineage["template_families"]) & evaluation_templates
    ):
        raise ContractError("provider training lineage overlaps sealed evaluation lineage")
    for arm_id, committed in plan_arms.items():
        metadata = dict(provider.metadata_for_arm(experiment_arms[arm_id]))
        if (
            metadata.get("provider_endpoint_sha256")
            != plan["provider_identity"]["endpoint_sha256"]
        ):
            raise ContractError(
                f"provider metadata differs from plan on {arm_id}/provider_endpoint_sha256"
            )
        for field, expected in (
            ("base_model_sha256", plan_arms["BASE"]["checkpoint"]["sha256"]),
            ("evaluator_sha256", plan["evaluator"]["sha256"]),
            ("inference_config_sha256", plan["inference_config"]["sha256"]),
            ("model_identifier_sha256", hashlib.sha256(committed["model_identifier"].encode("utf-8")).hexdigest()),
            ("checkpoint_sha256", committed["checkpoint"]["sha256"]),
            (
                "training_material_sha256",
                committed["training_material"]["sha256"]
                if committed["training_material"] is not None
                else "0" * 64,
            ),
            (
                "optimizer_config_sha256",
                committed["optimizer_config"]["sha256"]
                if committed["optimizer_config"] is not None
                else "0" * 64,
            ),
            (
                "reward_config_sha256",
                committed["reward_config"]["sha256"]
                if committed["reward_config"] is not None
                else "0" * 64,
            ),
            ("training_tokens", committed["training_tokens"]),
            ("optimizer_steps", committed["optimizer_steps"]),
            ("compute_budget", committed["compute_budget"]),
            ("seed_schedule_sha256", committed["seed_schedule_sha256"]),
            ("tool_policy_sha256", committed["tool_policy_sha256"]),
        ):
            if metadata.get(field) != expected:
                raise ContractError(f"provider metadata differs from plan on {arm_id}/{field}")

    receipt_dir_existed = receipt_dir.exists()
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if not receipt_dir_existed:
        _fsync_directory(receipt_dir.parent)
    completed = reused = 0
    for job in plan["jobs"]:
        target = receipt_dir / f"{job['job_id']}.json"
        started = receipt_dir / "started" / f"{job['job_id']}.json"
        if target.exists():
            verify_job_receipt(
                load_json(target),
                plan=plan,
                experiment=experiment,
                episode=episode_index[job["episode_id"]],
            )
            try:
                started.unlink()
                _fsync_directory(started.parent)
            except FileNotFoundError:
                pass
            reused += 1
            continue
        if started.exists():
            raise ContractError(
                f"matched job {job['job_id']} has indeterminate completion; "
                "do not resume this run"
            )
        if max_jobs is not None and completed >= max_jobs:
            break
        _atomic_json(
            started,
            {
                "schema": "cleanroom.matched-job-started/v1",
                "plan_sha256": plan["plan_sha256"],
                "job_id": job["job_id"],
            },
        )
        begin_job(dict(job))
        telemetry_start = (
            len(provider.telemetry())
            if callable(getattr(provider, "telemetry", None))
            else 0
        )
        runner = ArmRunner(
            provider=provider,
            taxonomy=taxonomy,
            experiment=experiment,
            episodes=[episode_index[job["episode_id"]]],
            training_paths=training_paths,
        )
        outcome = runner._run_arm(experiment_arms[job["arm_id"]])
        receipt = _job_receipt(
            plan=plan,
            job=job,
            outcome=outcome,
            provider=provider,
            telemetry_start=telemetry_start,
        )
        verify_job_receipt(
            receipt,
            plan=plan,
            experiment=experiment,
            episode=episode_index[job["episode_id"]],
        )
        _atomic_json(target, receipt)
        started.unlink()
        _fsync_directory(started.parent)
        completed += 1
    return {"executed": completed, "reused": reused}


def build_readiness_report(
    *,
    plan: Mapping[str, Any] | None,
    experiment: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    receipt_dir: Path | None = None,
    errors: Sequence[str] = (),
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    episode_index = {item["episode_id"]: item for item in episodes}
    episode_ids = set(episode_index)
    checks = {
        "concrete_artifacts_rehashed": False,
        "distinct_arm_checkpoints": False,
        "matched_compute_equal": False,
        "seed_schedule_frozen": False,
        "complete_episode_allocation": False,
        "no_indeterminate_started_jobs": False,
        "all_job_receipts_verified": False,
    }
    total_jobs = completed_jobs = invalid_jobs = indeterminate_jobs = 0
    discovered_errors = list(errors)
    if plan is not None:
        try:
            verify_run_plan(plan, experiment=experiment, episode_ids=episode_ids)
            if inputs is not None:
                rebuilt = build_run_plan(
                    inputs,
                    experiment=experiment,
                    episode_ids=sorted(episode_ids),
                )
                if canonical_bytes(rebuilt) != canonical_bytes(plan):
                    raise ContractError("run inputs no longer reproduce the frozen plan")
                checks["concrete_artifacts_rehashed"] = True
            checks.update(
                {
                    "distinct_arm_checkpoints": True,
                    "matched_compute_equal": True,
                    "seed_schedule_frozen": True,
                    "complete_episode_allocation": True,
                    "no_indeterminate_started_jobs": True,
                }
            )
            total_jobs = len(plan["jobs"])
            if receipt_dir is not None:
                for job in plan["jobs"]:
                    path = receipt_dir / f"{job['job_id']}.json"
                    started = receipt_dir / "started" / f"{job['job_id']}.json"
                    if started.exists() and not path.exists():
                        indeterminate_jobs += 1
                        checks["no_indeterminate_started_jobs"] = False
                        discovered_errors.append(
                            f"matched job {job['job_id']} has indeterminate completion"
                        )
                        continue
                    if not path.exists():
                        continue
                    try:
                        verify_job_receipt(
                            load_json(path),
                            plan=plan,
                            experiment=experiment,
                            episode=episode_index[job["episode_id"]],
                        )
                        completed_jobs += 1
                    except ContractError as exc:
                        invalid_jobs += 1
                        discovered_errors.append(str(exc))
                checks["all_job_receipts_verified"] = (
                    completed_jobs == total_jobs and invalid_jobs == 0
                )
        except ContractError as exc:
            discovered_errors.append(str(exc))
    missing = []
    if plan is None:
        missing += [
            "five concrete, distinct checkpoint artifacts",
            "four adaptation training-material artifacts",
            "four digest-valid training receipts",
            "evaluator, tool-policy and inference-configuration artifacts",
            "four optimizer and two RL reward configuration artifacts",
            "at least three frozen evaluation seeds",
        ]
    elif completed_jobs < total_jobs:
        missing.append(f"{total_jobs - completed_jobs} verified provider job receipts")
    if not checks["all_job_receipts_verified"]:
        missing.append("independent attestation of the completed receipt set")
    status = (
        "READY_FOR_INDEPENDENT_ATTESTATION"
        if all(checks.values()) and not discovered_errors
        else "READY_TO_EXECUTE"
        if (
            plan is not None
            and all(value for key, value in checks.items() if key != "all_job_receipts_verified")
            and not discovered_errors
        )
        else "BLOCKED_MISSING_OR_INVALID_INPUTS"
    )
    report: dict[str, Any] = {
        "schema": "cleanroom.matched-run-readiness/v1",
        "classification": CLASSIFICATION,
        "status": status,
        "plan_sha256": plan["plan_sha256"] if plan is not None else None,
        "checks": checks,
        "jobs": {
            "total": total_jobs,
            "verified": completed_jobs,
            "pending": max(0, total_jobs - completed_jobs - invalid_jobs),
            "invalid": invalid_jobs,
            "indeterminate": indeterminate_jobs,
        },
        "missing_external_artifacts": sorted(set(missing)),
        "errors": discovered_errors,
        "result_claim_allowed": False,
    }
    report["report_sha256"] = digest(report)
    validate_schema(report, "matched-run-readiness.schema.json")
    return report


def verify_readiness_report(
    report: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None,
    experiment: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    receipt_dir: Path | None = None,
    errors: Sequence[str] = (),
    inputs: Mapping[str, Any] | None = None,
) -> None:
    """Independently rederive every readiness field and reject stale claims."""

    validate_schema(report, "matched-run-readiness.schema.json")
    expected = build_readiness_report(
        plan=plan,
        experiment=experiment,
        episodes=episodes,
        receipt_dir=receipt_dir,
        errors=errors,
        inputs=inputs,
    )
    if canonical_bytes(report) != canonical_bytes(expected):
        raise ContractError("matched-run readiness report does not rederive")


def _write(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_json(path, value)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--inputs", required=True, type=Path)
    prepare.add_argument("--plan", required=True, type=Path)
    prepare.add_argument("--readiness", required=True, type=Path)
    execute = sub.add_parser("execute")
    execute.add_argument("--inputs", required=True, type=Path)
    execute.add_argument("--plan", required=True, type=Path)
    execute.add_argument("--provider", required=True)
    execute.add_argument("--training-file", action="append", type=Path, required=True)
    execute.add_argument("--receipt-dir", required=True, type=Path)
    execute.add_argument("--readiness", required=True, type=Path)
    execute.add_argument("--max-jobs", type=int)
    status = sub.add_parser("status")
    status.add_argument("--plan", required=True, type=Path)
    status.add_argument("--receipt-dir", required=True, type=Path)
    status.add_argument("--readiness", required=True, type=Path)
    args = parser.parse_args()

    taxonomy, experiment, episodes = load_public_assets()
    episode_ids = {item["episode_id"] for item in episodes}
    if args.command == "prepare":
        plan = build_run_plan(
            load_json(args.inputs),
            experiment=experiment,
            episode_ids=sorted(episode_ids),
        )
        _write(args.plan, plan)
        report = build_readiness_report(
            plan=plan,
            experiment=experiment,
            episodes=episodes,
            inputs=load_json(args.inputs),
        )
    else:
        plan = load_json(args.plan)
        if args.command == "execute":
            # Rebuild and byte-compare the plan so changed local artifacts cannot
            # be executed against a stale commitment.
            rebuilt = build_run_plan(
                load_json(args.inputs),
                experiment=experiment,
                episode_ids=sorted(episode_ids),
            )
            if canonical_bytes(rebuilt) != canonical_bytes(plan):
                raise ContractError("run inputs no longer reproduce the frozen plan")
            provider = load_provider(
                args.provider, experiment=experiment, taxonomy=taxonomy
            )
            execute_pending_jobs(
                plan=plan,
                provider=provider,
                taxonomy=taxonomy,
                experiment=experiment,
                episodes=episodes,
                training_paths=args.training_file,
                receipt_dir=args.receipt_dir,
                max_jobs=args.max_jobs,
            )
        report = build_readiness_report(
            plan=plan,
            experiment=experiment,
            episodes=episodes,
            receipt_dir=args.receipt_dir,
            inputs=load_json(args.inputs) if args.command == "execute" else None,
        )
    _write(args.readiness, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
