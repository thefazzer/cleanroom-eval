# cleanroom-eval

**A sealed, preregistered agent-evaluation environment for capital-markets
operations — that frontier models do not saturate.**

Forty hash-sealed long-horizon episodes per set (two sets) across eight
operational families: trade lifecycle, booking/allocations, reconciliation,
collateral/margin, settlement exceptions, permissions/release, temporal
causality and evidence sufficiency. Six tool-surface contracts, armed
reward-hacking gates (hidden-field probes, canary echoes, out-of-contract
state changes), and an evidence pipeline that binds every reported number to
run bytes by hash.

**New in 1.1.0:** per-run salted canaries (the published derivation no longer
weakens contamination detection); `cleanroom_eval.strategy_metrics` —
classify every rejected turn as identical repeat / local adjustment /
strategy revision and measure strategy-locking directly from transcripts
(see `docs/technical-note-strategy-locking.md`); provider-usage telemetry.

**Headline result (harness v2, preregistered, evidence in
`cleanroom_eval/evidence/gates-2026-08/`):** a frontier model completed
**27/40** episodes within 24 turns; an open-weight model completed **39/40**;
both tripped **zero** reward-hacking gates. The scripted honest baseline
completes 40/40, and all 243 committed adversarial mutations are rejected
with their expected errors — the environment is solvable, discriminative and
hard to hack.

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
