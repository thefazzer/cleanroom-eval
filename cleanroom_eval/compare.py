"""Side-by-side comparison of completed free runs.

    python3 -m cleanroom_eval.compare --runs runs/checkpoint-a runs/checkpoint-b \
        [--out compare.json] [--md compare.md]

The buyer-facing question is "did the checkpoint get better, and *how*":
pass-rates alone hide whether an improvement came from behaviour or from the
environment (see docs/technical-note-strategy-locking.md, finding 2). Each
run row therefore joins four independent measurements, all recomputed from
run bytes; nothing is trusted from prose:

- contract outcomes: completion, mean turns, safety gates (forbidden output
  keys, canary echoes), rejection counts by category;
- strategy metrics: repeat / local-adjustment / revision fractions, longest
  identical-repeat streak, revision latency (from the transcript, via
  cleanroom_eval.strategy_metrics);
- cost-denominated grade: expected operational loss vs the null policy
  (from scores.json);
- provider usage: token aggregates when the harness recorded them (v3+),
  UNKNOWN otherwise, never zero.

Operational expected loss and provider inference cost are NEVER combined;
they answer different questions and stay in separate fields.

With exactly two runs a ``deltas`` section reports second-minus-first
differences for the headline numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .strategy_metrics import analyze_run

COMPARE_SCHEMA = "cleanroom.run-comparison/v1"
UNKNOWN = "UNKNOWN"


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_row(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metrics = _load(run_dir / "metrics.json")
    if metrics is None:
        raise FileNotFoundError(f"no metrics.json in {run_dir}")
    strategy = analyze_run(run_dir)
    scores = _load(run_dir / "scores.json")
    usage = metrics.get("provider_usage") or {}
    gates = metrics.get("gates") or {}
    return {
        "run_id": run_dir.name,
        "policy_or_model": strategy["policy_or_model"] or metrics.get("policy"),
        "episodes": metrics.get("episodes"),
        "completion_rate": metrics.get("completion_rate"),
        "mean_turns": metrics.get("mean_turns"),
        "gates": {
            "forbidden_output_keys": gates.get("forbidden_output_keys"),
            "canary_echoes": gates.get("canary_echoes"),
            "rejected_requests": gates.get("rejected_requests"),
            "rejections_by_category": gates.get("rejections_by_category"),
        },
        "strategy": {
            "fractions": strategy["fractions"],
            "longest_repeat_streak": strategy["longest_repeat_streak"],
            "mean_rejections_before_first_revision": strategy["mean_rejections_before_first_revision"],
            "episodes_with_any_revision": strategy["episodes_with_any_revision"],
        },
        "operational_loss": {
            "expected_loss_usd": (scores or {}).get("expected_loss_usd", UNKNOWN),
            "null_policy_loss_usd": (scores or {}).get("null_policy_loss_usd", UNKNOWN),
            "loss_avoided_fraction": (scores or {}).get("loss_avoided_fraction", UNKNOWN),
        },
        "provider_usage": {
            "provider_calls": usage.get("provider_calls", UNKNOWN),
            "input_tokens": usage.get("input_tokens", UNKNOWN),
            "output_tokens": usage.get("output_tokens", UNKNOWN),
            "total_tokens": usage.get("total_tokens", UNKNOWN),
        },
    }


def _delta(after: Any, before: Any) -> Any:
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        return round(after - before, 4)
    return UNKNOWN


def compare_runs(run_dirs: list[Path]) -> dict[str, Any]:
    rows = [run_row(d) for d in run_dirs]
    report: dict[str, Any] = {
        "schema": COMPARE_SCHEMA,
        "interpretation": (
            "completion measures outcome; strategy fractions measure what the policy does "
            "with rejections (repeat = strategy-locked loop, revision = approach change); "
            "operational loss is graded in dollars against the null policy. A completion "
            "gain without a strategy-fraction change came from the environment, not the "
            "model — see docs/technical-note-strategy-locking.md."
        ),
        "runs": rows,
    }
    if len(rows) == 2:
        before, after = rows
        report["deltas"] = {
            "order": f"{after['run_id']} minus {before['run_id']}",
            "completion_rate": _delta(after["completion_rate"], before["completion_rate"]),
            "mean_turns": _delta(after["mean_turns"], before["mean_turns"]),
            "repeat_fraction": _delta(
                after["strategy"]["fractions"]["repeat"], before["strategy"]["fractions"]["repeat"]),
            "revision_fraction": _delta(
                after["strategy"]["fractions"]["revision"], before["strategy"]["fractions"]["revision"]),
            "longest_repeat_streak": _delta(
                after["strategy"]["longest_repeat_streak"], before["strategy"]["longest_repeat_streak"]),
            "expected_loss_usd": _delta(
                after["operational_loss"]["expected_loss_usd"],
                before["operational_loss"]["expected_loss_usd"]),
            "total_tokens": _delta(
                after["provider_usage"]["total_tokens"], before["provider_usage"]["total_tokens"]),
        }
    return report


def _cell(value: Any) -> str:
    if value is None or value == UNKNOWN:
        return "?"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def to_markdown(report: dict[str, Any]) -> str:
    header = (
        "| run | model | completion | rejections | repeat | local adj. | revision "
        "| longest streak | expected loss (USD) | total tokens |"
    )
    lines = ["# Run comparison", "", header, "|" + "---|" * 10]
    for row in report["runs"]:
        fractions = row["strategy"]["fractions"]
        pct = lambda v: "?" if v is None else f"{v * 100:.0f}%"  # noqa: E731
        completion = row["completion_rate"]
        lines.append(
            f"| {row['run_id']} | {_cell(row['policy_or_model'])} "
            f"| {'?' if completion is None else f'{completion * 100:.1f}%'} "
            f"| {_cell(row['gates']['rejected_requests'])} "
            f"| {pct(fractions['repeat'])} | {pct(fractions['local_adjustment'])} "
            f"| {pct(fractions['revision'])} "
            f"| {_cell(row['strategy']['longest_repeat_streak'])} "
            f"| {_cell(row['operational_loss']['expected_loss_usd'])} "
            f"| {_cell(row['provider_usage']['total_tokens'])} |"
        )
    if "deltas" in report:
        deltas = report["deltas"]
        lines += ["", f"**Deltas** ({deltas['order']}): " + ", ".join(
            f"{key} {_cell(value)}" for key, value in deltas.items() if key != "order")]
    lines += ["", report["interpretation"], ""]
    return "\n".join(lines)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args()
    report = compare_runs(args.runs)
    body = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(body + "\n", encoding="utf-8")
    if args.md:
        args.md.write_text(to_markdown(report), encoding="utf-8")
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
