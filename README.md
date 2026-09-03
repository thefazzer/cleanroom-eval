# cleanroom-eval

**A sealed, preregistered agent-evaluation environment for capital-markets
operations — every reported number bound to run bytes by hash.**

Forty hash-sealed long-horizon episodes per set (two sets) across eight
operational families: trade lifecycle, booking/allocations, reconciliation,
collateral/margin, settlement exceptions, permissions/release, temporal
causality and evidence sufficiency. Six tool-surface contracts, armed
reward-hacking gates (hidden-field probes, canary echoes, out-of-contract
state changes), and an evidence pipeline that binds every reported number to
run bytes by hash.

**New in 1.1.1:** two honesty fixes prompted by an external adversarial
audit. The harness-v2 adapter code (rejection feedback in observations,
optimistic-concurrency version checks) is restored — a packaging regression
had shipped v1 adapter semantics beside the v2 system prompt, so the
published v2 results were not reproducible from this repository. And
`state_changes_outside_contract` is now **measured** per episode by diffing
live state against an independent recomputation from the sealed contract; it
was previously reported as zero by construction, which could never fire.
Runs that predate the measurement report `NOT_MEASURED`, not zero.

**New in 1.1.0:** per-run salted canaries (the published derivation no longer
weakens contamination detection); `cleanroom_eval.strategy_metrics` —
classify every rejected turn as identical repeat / local adjustment /
strategy revision and measure strategy-locking directly from transcripts
(see `docs/technical-note-strategy-locking.md`); provider-usage telemetry.

**Current result (harness v2, preregistered, supplier-run; evidence in
`cleanroom_eval/evidence/gates-2026-08/`):** one frontier arm completed
**27/40** episodes within 24 turns; one open-weight arm completed **39/40**;
neither probed hidden fields nor echoed a canary. The scripted reference
policy completes 40/40 — it reads validator error text, so it bounds harness
solvability, not model skill. All 243 committed adversarial mutations are
rejected by the deterministic verifier with their expected errors; that is
contract-level sensitivity, not a claim about model-facing attacks. These
runs were executed by the environment's author and are not blind; whether
frontier models saturate this environment as a class is an open hypothesis
(one arm measured). Independent replication is invited — the harness, sealed
sets and gates are all in this repository.

Every episode, institution, name and identifier is invented
(`CLEANROOM_SYNTHETIC`); independent origin is enforced by a shingle-overlap
release gate. See `SECURITY-RELEASE.md` for the exact boundary.

## Evaluate your model in three commands

```bash
pip install -e .
python -m cleanroom_eval.contract verify-set          # verify the sealed set
python -m cleanroom_eval.fire_gates preregister --out runs
```

Then point the harness at any OpenAI-compatible endpoint:

```bash
export CLEANROOM_POLICY_BASE_URL=https://api.your-provider.com/v1
export CLEANROOM_POLICY_API_KEY=...
export CLEANROOM_POLICY_MODEL=your-model
python -m cleanroom_eval.free_run --policy chat --out runs --run-id my-model
python -m cleanroom_eval.fire_gates report --out runs --runs my-model
```

The report shows completion rate, mean turns and every gate count. The
scripted baseline (`--policy scripted`) runs offline and needs no keys.

## What is in the box

- `cleanroom_eval/assets/episodes/`, `episodes_v2/` — the sealed sets, with
  per-episode task cards and adversarial mutations
- `cleanroom_eval/free_run.py` — the agent loop: any OpenAI-compatible
  endpoint, per-call telemetry (status, latency, request/response hashes,
  provider usage where reported)
- `cleanroom_eval/fire_gates.py` — preregistration, honest baseline, gates
  report
- `cleanroom_eval/evidence/` — the hash-binding evidence pipeline and the
  2026-08 gates evidence pack
- `cleanroom_eval/schemas/` — every contract as JSON Schema

## Fresh, unseen episode sets

This public set is a demonstration sample: its canaries and traps are public,
so treat results on models trained after its release accordingly. The
generator that produced it mints **fresh, private, hash-sealed episode sets**
(new worlds, namespaces, canaries and mutation batteries) for evaluation
programmes that need uncontaminated instruments — along with custom
operational families and independently bound evidence reports. Open a GitHub
issue on this repository to talk.

## License

MIT — see `LICENSE`. `SECURITY-RELEASE.md` documents what this archive
deliberately excludes (evaluator-only oracles, private gold).

## Status, 3 September 2026

This repository is release 1.1.1 of the sealed evaluation package. The
evaluation ledger for the preregistered runs (T4 to T7), the Harbor-format
packaging note and the exporter skeleton now live in the maintained
environment repository:

- Ledger: https://github.com/thefazzer/bankingenv/blob/main/docs/EVAL-LEDGER.md
- Packaging: https://github.com/thefazzer/bankingenv/blob/main/docs/HARBOR-PACKAGING.md
- Exporter: https://github.com/thefazzer/bankingenv/blob/main/scripts/export_harbor.py

Results there are labelled per the ledger's discipline: three preregistered
runs, canonical claims not passed, one terminal test in flight.
