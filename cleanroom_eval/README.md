# Clean-room capital-markets evaluation contract

This directory defines the evaluation boundary independently of any corpus
generator or restricted research run. Every committed fixture is fictitious,
institution-neutral and marked `CLEANROOM_SYNTHETIC`.

## Frozen assets

- `assets/competency-taxonomy.v1.json` freezes twelve abstract operational
  competencies and six tool-surface contracts.
- `assets/episodes/*.json` contains forty hash-sealed, long-horizon episodes: five variants in each of lifecycle, booking/allocations, reconciliation, collateral/margin, settlement/exceptions, permissions/release, temporal causality and evidence sufficiency. Names and identifiers are invented.
- `assets/preregistration.v1.json` locks the all-40-episode primary metric, denominator and missing-result policy before execution.
- `assets/scenario-partitions.v1.json` fixes scenario-level train/test lineage fields and requires disjoint worlds, templates, namespaces, scenario families and render seeds.
- `assets/experiment.v1.json` preregisters BASE, SFT, SFT matched control, RL
  and RL matched control, including compute parity and causal contrasts.
- `assets/sealed-set.manifest.v1.json` enumerates exact fixture bytes and hashes.
- `assets/citation_tasks/citation-tasks.v1.jsonl` freezes exactly 500
  provider-visible semantic citation tasks derived from all forty episodes.
- `assets/citation_tasks/citation-task-oracle.v1.jsonl` keeps evaluator-only
  expected selections and adversarial labels in a physically separate asset;
  it is never part of a provider request.
- `assets/citation-tasks.manifest.v1.json` binds both citation assets to the
  sealed episode manifest, enumerates every task/oracle hash and freezes
  category, scenario-family and answer-kind coverage.

The episode replayer deterministically checks chronology, referential integrity,
authorization, monotonic object versions, idempotency, per-currency ledger
conservation, evidence presence, evidence sufficiency, committed adversarial-mutation rejection and final state. A failed check blocks success.

## Generator integration contract

A clean-room corpus generator must export:

1. A world manifest with `world_id`, `template_family`, `entity_namespace`,
   `render_seed_family`, generator revision and `CLEANROOM_SYNTHETIC`
   classification.
2. Episode definitions conforming to `schemas/episode.schema.json`. Each tool
   mutation carries the current object version, expected next version,
   idempotency request ID, immutable receipt ID and evidence references.
3. Six adapters named by the taxonomy: communications, case management, trade
   ledger, reference data, controls monitoring and evidence store. Adapters may
   expose observations and authorized mutations, never simulator hidden state.
4. A training lineage manifest enumerating every world/template/entity/render
   family used by SFT or RL. Evaluation families must be disjoint.
5. An experiment receipt conforming to `schemas/run.schema.json`, with one row
   for every preregistered arm and commitments for model, evaluator, seeds,
   tool policy, compute and training material.
6. Candidate training files for the 13-token hashed-shingle contamination
   check. The check emits commitments, not source text.

The generator owns rendering and simulator implementation. This package owns
the competency IDs, evaluation split, lifecycle invariants and release gates.

## Reward hacking and transfer

An aggregate result is invalid if any run directly reads hidden state, performs
an unsupported mutation, uses a canary, bypasses an invariant, makes an
unsupported material claim, or fails contamination screening. Transfer is
reported only on worlds, templates, entity namespaces and render-seed families
not used in training. SME scoring is limited to blinded residual communication
quality; deterministic state and value checks take precedence.

## Release and independent attestation

`build_release_manifest` creates a path-safe, byte-counted, SHA-256-enumerated
release manifest bound to a full repository revision. A release is not approved
by its generator. An independent reviewer must rehash the assets and return an
Ed25519 `cleanroom.independent-attestation/v1` envelope. `verify_attestation`
requires a separately supplied trusted public key and rejects payload, signer
or signature changes.

The checked-in sealed-set manifest freezes public fixtures; it is not an
independent release attestation.

## Verification

```bash
python3 -m cleanroom_eval.contract verify-bundle
python3 -m cleanroom_eval.build_citation_tasks
python3 -c 'from cleanroom_eval.citation_tasks import verify_citation_task_set; print(verify_citation_task_set())'
python3 -m cleanroom_eval.runner \
  --provider mock \
  --output /tmp/cleanroom-mock-run.json
python3 -m unittest tests.test_cleanroom_eval_contract -v
```

The mock command executes all 800 arm/event decisions deterministically. It
tests orchestration and deliberate treatment/control contrasts only. Its
receipt is permanently marked `MOCK_ONLY` and `BLOCKED_MOCK_PROVIDER`.

A real adapter is loaded as `--provider package.module:factory`. The factory is
called with the public experiment and taxonomy and must return an object with:

- `name` and `execution_mode = "REAL_PROVIDER"`;
- `metadata_for_arm(arm)`, returning model, evaluator, training, seed, tool
  policy and compute commitments;
- `training_lineage()`, returning complete `world_ids` and
  `template_families`;
- `act(arm_id, request)`, returning the provider-response schema.

The request schema intentionally excludes the gold action, mutations, final
state, expected receipt, reward traps and hidden canary. Real runs must name
their committed training files with `--training-file` so contamination can be
measured. Even a passing real-provider receipt remains
`BLOCKED_PENDING_INDEPENDENT_ATTESTATION`; the runner cannot self-approve a
release.

### OpenAI/DeepSeek-compatible adapter

`cleanroom_eval.real_provider:create_provider` is the concrete HTTP adapter.
It supports OpenAI-compatible and DeepSeek-compatible chat-completions
envelopes without importing a provider SDK. Configuration is exclusively via:

- `CLEANROOM_PROVIDER_KIND` (`openai` or `deepseek`)
- `CLEANROOM_PROVIDER_ENDPOINT` (HTTPS; loopback HTTP is test-only)
- `CLEANROOM_PROVIDER_API_KEY`
- `CLEANROOM_PROVIDER_TIMEOUT_SECONDS`, `MAX_RETRIES`,
  `RETRY_BACKOFF_SECONDS`, `MAX_OUTPUT_TOKENS`
- `CLEANROOM_PROVIDER_BASE_MODEL_SHA256`, `EVALUATOR_SHA256`,
  `INFERENCE_CONFIG_SHA256`, `SEED_SCHEDULE_SHA256`, `TOOL_POLICY_SHA256`
- `CLEANROOM_PROVIDER_ARM_METADATA_JSON`, with the declared model, training
  counts, compute and checkpoint/training-material/optimizer/reward
  commitments for all five arms (use the all-zero SHA-256 sentinel for
  configurations that do not apply to an arm)
- `CLEANROOM_PROVIDER_TRAINING_LINEAGE_JSON`, containing complete training
  world IDs and template families

The adapter never reads a dotenv file. API keys are used only in the
Authorization header and are excluded from representations, errors, receipts
and telemetry. Calls have bounded payloads, timeouts and retries. Run receipts
record request/response hashes, provider request ID, attempt statuses, latency
and token usage, but never prompts, responses or credentials.

### Real matched-run orchestration

`python3 -m cleanroom_eval.matched_run` is the admissibility layer for actual
BASE/SFT/RL and matched-control checkpoints. It rehashes concrete artifacts,
verifies training-receipt lineage and equal compute, freezes balanced seeds and
episode jobs, and resumes only from plan-bound provider receipts. See
`docs/cleanroom_real_matched_run.md`. A single generic model cannot be labelled
as the trained arms.

## Commercial-proof package

`commercial_proof/` wraps this contract in a mock-only API/container demo and a
hash-enumerated commercial package. Its manifest includes the frozen sealed-set
commitment, taxonomy, experiment, verifier, package plan, benchmark/security
documents and unexecuted legal/SOW templates. The package builder cannot
self-approve: legal, commercial, real-provider and independent-provenance gates
remain pending until separately reviewed evidence resolves them.
