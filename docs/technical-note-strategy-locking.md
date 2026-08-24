# Strategy-locking is visible in tool-call transcripts

*Technical note · cleanroom-eval · 2026-08-24 · CLEANROOM_SYNTHETIC throughout*

A recent empirical study of autonomous post-training agents (Lim et al.,
[arXiv:2608.19072](https://arxiv.org/abs/2608.19072)) reports that agents lock
a strategy at the start of a run and spend their remaining budget on local
adjustments inside it — and that what is missing is a mechanism for
evidence-driven strategy re-evaluation during execution.

We measure that behaviour directly, per tool call, in a contract-governed
agent environment — and we find both phenotypes the paper predicts, one per
model, on identical tasks.

## Setup

[cleanroom-eval](https://github.com/thefazzer/cleanroom-eval) (MIT) runs
fictitious post-trade operations episodes as a stateful environment: a policy
sees a task card and an observable boundary, proposes tool calls, and a
deterministic contract accepts each call or rejects it with a reason
(authorization, stale versions, unmet preconditions, ungrounded evidence).
Every turn is transcribed. Because rejections carry reasons, the transcript
records what a policy *does with negative evidence* — the exact quantity the
paper says is unmeasured.

We classify every rejected turn by the policy's next request in the same
episode:

| bucket | meaning | paper's term |
|---|---|---|
| **repeat** | identical surface, action, actor and version assertions re-issued | strategy-locked loop |
| **local adjustment** | same action retried with new parameters | execution-level iteration |
| **revision** | a different action or surface | strategy-level change |
| abandon | episode ends on the rejection | budget exhausted in-strategy |

## Results (40 sealed episodes per run, 24-turn budget)

| run | model | completion | rejections | repeat | local adj. | revision | longest repeat streak | rejections before first revision |
|---|---|---|---|---|---|---|---|---|
| frontier, harness v1 | gpt-5.2 | 12.5% | 812 | **80%** | 13% | 2% | **23** | 3.6 |
| frontier, harness v2 | gpt-5.2 | 67.5% | 279 | **82%** | 10% | 3% | **21** | 3.3 |
| open-weight, harness v2 | deepseek-v4-flash | 97.5% | 110 | **16%** | 63% | **20%** | 4 | 1.8 |
| scripted reference | (non-LLM baseline) | 100% | 233 | 0% | 82% | 18% | 0 | 1.9 |

Harness v2 differs from v1 in two ways: rejections are fed back into the
observation (`last_rejection`), and version assertions are checked as
optimistic concurrency rather than exact-set. Both arms substitute for the
preregistered models (disclosure in the run's evidence pack); all runs are
reproducible from the public repository.

## Three findings

**1. Strategy-locking is a stable, model-level trait — not a harness
artifact.** gpt-5.2 repeats identically after 80–82% of rejections with
20+-turn identical streaks *in both harness versions*. Feeding the rejection
text into its observation did not change its post-rejection behaviour.

**2. What fixed gpt-5.2's completion was the environment, not the model.**
Its jump from 12.5% → 67.5% came almost entirely from the harness dropping a
class of rejections (correct-but-over-specified version assertions), i.e.
fewer occasions to loop — not from looping less. Improvement without
behavioural change is exactly the paper's "execution gains without strategy
re-evaluation," observed here from the environment side.

**3. The "missing mechanism" is present in one model and absent in the
other, on the same tasks.** deepseek-v4-flash converts the same rejection
feedback into parameter changes (63%) and genuine approach changes (20%),
revises after fewer than two rejections on average, and completes 97.5%.
Strategy re-evaluation is not missing from LLM agents in general; it is
unevenly distributed — and it is measurable per model, per episode, from
transcripts alone.

## Why this matters for post-training

If you are training agents, the repeat/adjust/revise distribution of a
checkpoint is a training signal that pass-rates hide: two checkpoints with
similar scores can differ radically in how they spend negative evidence. The
metric requires only transcripts with rejection reasons; the environment,
harness, grader and this metric are open (MIT), and a run against your own
model is three commands against any OpenAI-compatible endpoint.

```bash
pip install cleanroom-eval
python3 -m cleanroom_eval.free_run --policy chat --out runs --run-id yours
python3 -m cleanroom_eval.strategy_metrics --runs runs/yours
```

## Scope and honesty

One model pair, one environment family, 40 sealed episodes per run; the
frontier/open-weight labels describe our arms, not a market claim. Rejection
classification is syntactic (identical re-issue vs parameter vs action
change), deliberately model-agnostic, and blind to chain-of-thought. Scores
come from a deterministic contract; no LLM judging anywhere. Evidence packs
bind every number above to run bytes by digest.

*Contact: via the repository.*
