"""Strict schema-output HTTP provider for the clean-room citation pipeline.

Unlike the legacy citation-repair path, this provider never asks a model to
edit prose or insert citation markers.  Each request supplies a dynamic JSON
Schema whose evidence-id fields are enums of the task's evidence set.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Mapping
import jsonschema
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .contract import ContractError, canonical_bytes, load_json, validate_schema


ENV_PREFIX = "CLEANROOM_CITATION_PROVIDER_"
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_RETRIES = 2
MAX_TIMEOUT_SECONDS = 120.0
SYSTEM_INSTRUCTIONS = (
    "Select only semantic claims directly supported by the supplied clean-room "
    "evidence. Return the strict JSON Schema object. Do not write answer prose, "
    "citation markers, URLs, footnotes, or fields outside the schema. Preserve "
    "subject, relation, object, polarity, and qualifiers exactly. Refuse only "
    "when the cited evidence declares coverage of the requested proposition."
)


@dataclass(frozen=True, repr=False)
class CitationProviderConfig:
    endpoint: str
    auth_token: str = field(repr=False)
    model: str
    timeout_seconds: float = 20.0
    max_retries: int = 1
    retry_backoff_seconds: float = 0.25
    max_output_tokens: int = 1024

    def __repr__(self) -> str:
        return (
            "CitationProviderConfig("
            f"endpoint={self.endpoint!r}, auth_token=<redacted>, "
            f"model={self.model!r}, timeout_seconds={self.timeout_seconds!r}, "
            f"max_retries={self.max_retries!r})"
        )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "CitationProviderConfig":
        values = os.environ if environ is None else environ

        def required(name: str) -> str:
            value = values.get(ENV_PREFIX + name, "").strip()
            if not value:
                raise ContractError(
                    f"missing environment variable {ENV_PREFIX}{name}"
                )
            return value

        try:
            config = cls(
                endpoint=required("ENDPOINT"),
                auth_token=required("API_KEY"),
                model=required("MODEL"),
                timeout_seconds=float(
                    values.get(ENV_PREFIX + "TIMEOUT_SECONDS", "20")
                ),
                max_retries=int(values.get(ENV_PREFIX + "MAX_RETRIES", "1")),
                retry_backoff_seconds=float(
                    values.get(ENV_PREFIX + "RETRY_BACKOFF_SECONDS", "0.25")
                ),
                max_output_tokens=int(
                    values.get(ENV_PREFIX + "MAX_OUTPUT_TOKENS", "1024")
                ),
            )
        except ValueError as exc:
            raise ContractError(
                "citation provider numeric environment value is invalid"
            ) from exc
        config.validate()
        return config

    def validate(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ContractError("citation provider endpoint is not a safe HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise ContractError(
                "unencrypted citation provider HTTP is allowed only on loopback"
            )
        if not self.model.strip():
            raise ContractError("citation provider model is empty")
        if not (0 < self.timeout_seconds <= MAX_TIMEOUT_SECONDS):
            raise ContractError("citation provider timeout is outside strict bound")
        if not (0 <= self.max_retries <= MAX_RETRIES):
            raise ContractError("citation provider retry count is outside strict bound")
        if not (0 <= self.retry_backoff_seconds <= 2):
            raise ContractError("citation provider backoff is outside strict bound")
        if not (64 <= self.max_output_tokens <= 4096):
            raise ContractError(
                "citation provider output-token limit is outside strict bound"
            )


def constrained_selection_schema(evidence_ids: list[str]) -> dict[str, Any]:
    """Return the frozen output schema with task-local evidence-id enums."""

    if (
        not evidence_ids
        or len(set(evidence_ids)) != len(evidence_ids)
        or any(not isinstance(item, str) or not item for item in evidence_ids)
    ):
        raise ContractError("citation task evidence IDs are invalid")
    schema = copy.deepcopy(
        load_json(Path(__file__).with_name("schemas") / "claim-selection.schema.json")
    )
    enum = sorted(evidence_ids)
    schema["$defs"]["claim"]["properties"]["evidence_ids"]["items"] = {
        "type": "string",
        "enum": enum,
    }
    schema["$defs"]["refusal"]["properties"]["evidence_ids"]["items"] = {
        "type": "string",
        "enum": enum,
    }
    return schema


class SchemaConstrainedHTTPProvider:
    """OpenAI-compatible strict-json-schema selection provider."""

    name = "schema-constrained-citation-http-v1"
    execution_mode = "REAL_PROVIDER"

    def __init__(self, config: CitationProviderConfig) -> None:
        config.validate()
        self.config = config
        self._telemetry: list[dict[str, Any]] = []
        self._telemetry_lock = threading.Lock()

    def telemetry(self) -> list[dict[str, Any]]:
        with self._telemetry_lock:
            return [dict(item) for item in self._telemetry]

    def select_claims(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        validate_schema(task, "citation-task.schema.json")
        for evidence in task["evidence"]:
            validate_schema(evidence, "semantic-evidence.schema.json")
        evidence_ids = [item["evidence_id"] for item in task["evidence"]]
        schema = constrained_selection_schema(evidence_ids)
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": canonical_bytes(task).decode("utf-8"),
                },
            ],
            "temperature": 0,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "cleanroom_claim_selection_v1",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        request_body = canonical_bytes(payload)
        if len(request_body) > MAX_REQUEST_BYTES:
            raise ContractError("citation provider request exceeds byte limit")
        request_sha256 = hashlib.sha256(request_body).hexdigest()
        statuses: list[int | str] = []
        started = time.monotonic()
        response_bytes: bytes | None = None
        provider_request_id = ""
        for attempt in range(self.config.max_retries + 1):
            request = Request(
                self.config.endpoint,
                data=request_body,
                headers={
                    "Authorization": f"Bearer {self.config.auth_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "cleanroom-citation-provider/1",
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    statuses.append(int(response.status))
                    response_bytes = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(response_bytes) > MAX_RESPONSE_BYTES:
                        self._record(
                            task, request_sha256, started, statuses, response_bytes,
                            "FAIL", "RESPONSE",
                        )
                        raise ContractError(
                            "citation provider response exceeds byte limit"
                        )
                    provider_request_id = (
                        response.headers.get("x-request-id", "").strip()
                    )
                break
            except HTTPError as exc:
                statuses.append(int(exc.code))
                exc.close()
                if (
                    exc.code not in RETRYABLE_STATUS
                    or attempt >= self.config.max_retries
                ):
                    self._record(
                        task, request_sha256, started, statuses, None, "FAIL", "HTTP"
                    )
                    raise ContractError(
                        f"citation provider HTTP request failed with status {exc.code}"
                    ) from exc
            except (TimeoutError, socket.timeout, URLError) as exc:
                statuses.append("TRANSPORT")
                if attempt >= self.config.max_retries:
                    self._record(
                        task,
                        request_sha256,
                        started,
                        statuses,
                        None,
                        "FAIL",
                        "TRANSPORT",
                    )
                    raise ContractError(
                        "citation provider transport request failed"
                    ) from exc
            if self.config.retry_backoff_seconds:
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        if response_bytes is None:
            raise ContractError("citation provider returned no response")
        try:
            envelope = json.loads(response_bytes)
            provider_request_id = provider_request_id or str(
                envelope.get("id") or ""
            ).strip()
            content = envelope["choices"][0]["message"]["content"]
            selection = json.loads(content)
            if not provider_request_id:
                raise ContractError(
                    "citation provider response lacks request identifier"
                )
            validate_schema(selection, "claim-selection.schema.json")
            try:
                jsonschema.Draft202012Validator(schema).validate(selection)
            except jsonschema.ValidationError as exc:
                raise ContractError("citation provider selected an evidence ID outside the task schema") from exc
        except (
            ContractError,
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            self._record(
                task,
                request_sha256,
                started,
                statuses,
                response_bytes,
                "FAIL",
                "RESPONSE",
            )
            if isinstance(exc, ContractError):
                raise
            raise ContractError(
                "citation provider response envelope is invalid"
            ) from exc
        self._record(
            task,
            request_sha256,
            started,
            statuses,
            response_bytes,
            "PASS",
            None,
            provider_request_id=provider_request_id,
        )
        return selection

    def _record(
        self,
        task: Mapping[str, Any],
        request_sha256: str,
        started: float,
        statuses: list[int | str],
        response_bytes: bytes | None,
        status: str,
        failure_class: str | None,
        *,
        provider_request_id: str | None = None,
    ) -> None:
        self._telemetry.append(
            {
                "schema": "cleanroom.citation-provider-call/v1",
                "task_id": task["task_id"],
                "request_sha256": request_sha256,
                "response_sha256": (
                    hashlib.sha256(response_bytes).hexdigest()
                    if response_bytes is not None
                    else None
                ),
                "provider_request_id": provider_request_id,
                "attempts": len(statuses),
                "http_statuses": list(statuses),
                "latency_ms": max(0, int((time.monotonic() - started) * 1000)),
                "status": status,
                "failure_class": failure_class,
            }
        )


def create_citation_provider() -> SchemaConstrainedHTTPProvider:
    return SchemaConstrainedHTTPProvider(CitationProviderConfig.from_env())
