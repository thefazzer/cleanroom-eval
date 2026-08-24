"""Strategy-locking metrics over free-run transcripts.

    python3 -m cleanroom_eval.strategy_metrics --runs runs/frontier runs/open_weight

Motivated by arXiv:2608.19072 ("What is Missing from AI Post-Training"),
which finds agents lock a strategy early and spend their budget on local
adjustment inside it. A free-run transcript captures exactly that behaviour
at tool-call granularity, so it can be measured rather than asserted.

Every rejected turn is classified by what the policy does NEXT, inside the
same episode:

- ``repeat``            the next request is identical in surface, action,
                        actor and object-version assertions — pure loop;
- ``local_adjustment``  same surface and action, different parameters
                        (versions/evidence) — execution-level iteration
                        within the locked strategy;
- ``revision``          a different action or surface — a strategy-level
                        change of approach;
- ``abandon``           the episode ends (stop, error or turn budget) on
                        that rejection.

Reported per run: bucket fractions, the longest identical-repeat streak,
and revision latency (mean rejections endured before the first revision).
High ``repeat`` with long streaks is the paper's strategy-locking signature;
the harness-v1 frontier run is the canonical specimen.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

METRIC_SCHEMA = "cleanroom.strategy-metrics/v1"
BUCKETS = ("repeat", "local_adjustment", "revision", "abandon")


def _signature(row: Mapping[str, Any]) -> tuple | None:
    request = row.get("request")
    if not isinstance(request, dict):
        return None
    return (
        row.get("surface"),
        request.get("action"),
        request.get("actor_id"),
        tuple(sorted((request.get("object_versions") or {}).items())),
    )


def _approach(row: Mapping[str, Any]) -> tuple | None:
    request = row.get("request")
    if not isinstance(request, dict):
        return None
    return (row.get("surface"), request.get("action"))


def classify_episode(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify every rejection in one episode's ordered turns."""
    buckets = {name: 0 for name in BUCKETS}
    streak = longest_streak = 0
    rejections_before_first_revision: int | None = None
    rejections_seen = 0
    for index, row in enumerate(rows):
        if row.get("outcome") != "REJECTED":
            streak = 0
            continue
        rejections_seen += 1
        successor = rows[index + 1] if index + 1 < len(rows) else None
        if successor is None or successor.get("outcome") in ("STOP", "POLICY_ERROR"):
            buckets["abandon"] += 1
            streak = 0
            continue
        if _signature(successor) is not None and _signature(successor) == _signature(row):
            buckets["repeat"] += 1
            streak += 1
            longest_streak = max(longest_streak, streak)
            continue
        streak = 0
        if _approach(successor) is not None and _approach(successor) == _approach(row):
            buckets["local_adjustment"] += 1
        else:
            buckets["revision"] += 1
            if rejections_before_first_revision is None:
                rejections_before_first_revision = rejections_seen - 1
    return {
        "rejections": sum(buckets.values()),
        "buckets": buckets,
        "longest_repeat_streak": longest_streak,
        "rejections_before_first_revision": rejections_before_first_revision,
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    transcript = run_dir / "transcript.jsonl"
    if not transcript.is_file():
        raise FileNotFoundError(f"no transcript in {run_dir}")
    per_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in transcript.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            per_episode[row.get("episode_id", "?")].append(row)
    episodes = {
        episode_id: classify_episode(sorted(rows, key=lambda r: r.get("turn", 0)))
        for episode_id, rows in per_episode.items()
    }
    totals = {name: sum(e["buckets"][name] for e in episodes.values()) for name in BUCKETS}
    rejections = sum(totals.values())
    latencies = [
        e["rejections_before_first_revision"]
        for e in episodes.values()
        if e["rejections_before_first_revision"] is not None
    ]
    metrics_path = run_dir / "metrics.json"
    completion = None
    policy = None
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        completion = metrics.get("completion_rate")
        policy = metrics.get("policy")
        config = run_dir / "config.json"
        if config.is_file():
            policy = json.loads(config.read_text()).get("policy_commitment", {}).get("model") or policy
    return {
        "schema": METRIC_SCHEMA,
        "run_id": run_dir.name,
        "policy_or_model": policy,
        "completion_rate": completion,
        "rejections": rejections,
        "fractions": {name: (round(totals[name] / rejections, 4) if rejections else None) for name in BUCKETS},
        "totals": totals,
        "longest_repeat_streak": max((e["longest_repeat_streak"] for e in episodes.values()), default=0),
        "mean_rejections_before_first_revision": (
            round(sum(latencies) / len(latencies), 2) if latencies else None
        ),
        "episodes_with_any_revision": sum(
            1 for e in episodes.values() if e["buckets"]["revision"]
        ),
        "episodes": len(episodes),
    }


def compare(run_dirs: Iterable[Path]) -> dict[str, Any]:
    return {
        "schema": METRIC_SCHEMA,
        "interpretation": (
            "repeat = identical request re-issued after rejection (strategy-locked loop); "
            "local_adjustment = same action retried with new parameters (execution-level); "
            "revision = action or surface changed (strategy-level). "
            "Framing: arXiv:2608.19072."
        ),
        "runs": [analyze_run(Path(d)) for d in run_dirs],
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare(args.runs)
    body = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(body + "\n", encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
