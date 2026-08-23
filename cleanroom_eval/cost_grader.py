"""Cost-denominated outcome grading for clean-room episodes (#118).

Ports the *method* of the record-lane cost card — every priced line carries
a basis token and a confidence, nothing is invented, and where a term is not
estimable the card says so and names the one input that would change that —
onto clean-room episodes using only their own sealed quantities. No record
text, no client names, no rate table beyond the public sourced basis.

Per episode, a card prices two scenarios:

  resolution_cost_usd   what a competent desk spends closing the episode the
                        way the sealed script does: engaged minutes per event
                        x sourced seat rate, plus capacity consumed while the
                        case is open over the MEASURED script elapsed time.
  unresolved_cost_usd   what it costs to never close it inside the window:
                        the same capacity term over the full time window.

Money at risk is reported separately and is NOT in either headline, exactly
as in the record lane: the ledger amounts the episode states are evidence of
an exposure class, not of a loss.

A run's ``scores.json`` then denominates outcomes: a completed episode costs
its resolution; an incomplete one costs its unresolved figure. A
refuse-everything null policy therefore costs ``sum(unresolved)``, and the
loss a policy avoids relative to that null is the outcome score.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract import digest

CARD_SCHEMA = "cleanroom.episode-cost-card/v1"
SCORES_SCHEMA = "cleanroom.free-run-scores/v1"

# Sourced London all-in USD/hour by corporate band (Morgan McKinley 2026 base
# salary survey plus stated on-cost and overhead assumptions). Bands:
# 1 Analyst, 2 Associate, 3 Vice President, 4 Director, 5 Managing Director.
RATE_USD_PER_HOUR = {1: 70, 2: 89, 3: 130, 4: 191, 5: 191}
RATE_BASIS = "sourced_base_salary_morgan_mckinley_2026_plus_stated_oncost_assumptions"
ENGAGED_MINUTES_PER_EVENT = 20
CAPACITY_WHILE_OPEN = 0.05
BUSINESS_FRACTION = (5 / 7) * (8 / 24)

# Role -> band. An ASSUMED mapping: the clean-room roles are synthetic seats.
BAND_BY_ROLE = {
    "operations_analyst": 1,
    "operations_supervisor": 3,
    "product_controller": 3,
    "middle_office_controller": 3,
    "reconciliation_controller": 3,
    "collateral_manager": 4,
    "reference_data_steward": 2,
    "release_manager": 4,
}
DEFAULT_BAND = 2

MINOR_UNITS = 100  # amount_minor -> major currency units


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_card(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Price one sealed episode from its own quantities. Never raises on a thin episode."""

    events = [e for e in episode["events"] if not e.get("duplicate_of")]
    roles = {e["id"]: e.get("role") for e in episode["entities"] if e.get("role")}
    card_id = f"cost_{episode['episode_id']}"
    if len(events) < 2:
        return {
            "schema": CARD_SCHEMA,
            "card_id": card_id,
            "episode_id": episode["episode_id"],
            "resolution_cost_usd": None,
            "unresolved_cost_usd": None,
            "status": "not_estimable",
            "needs": "at least two timestamped events to measure elapsed time",
            "rate_basis": RATE_BASIS,
        }

    t0 = _parse(events[0]["at"])
    t1 = _parse(events[-1]["at"])
    window_h = (_parse(episode["time_window"]["end"]) - _parse(episode["time_window"]["start"])).total_seconds() / 3600
    script_h = max((t1 - t0).total_seconds() / 3600, 0.0)

    people = sorted({e["actor_id"] for e in events})
    band_hours: dict[int, float] = {}
    unmapped_roles = set()
    for event in events:
        role = roles.get(event["actor_id"])
        band = BAND_BY_ROLE.get(role or "", None)
        if band is None:
            unmapped_roles.add(role or "unrolled")
            band = DEFAULT_BAND
        band_hours[band] = band_hours.get(band, 0.0) + ENGAGED_MINUTES_PER_EVENT / 60
    engaged_h = sum(band_hours.values())
    engaged_usd = sum(h * RATE_USD_PER_HOUR[b] for b, h in band_hours.items())
    mean_rate = engaged_usd / engaged_h if engaged_h else RATE_USD_PER_HOUR[DEFAULT_BAND]

    def delay(hours: float) -> float:
        return CAPACITY_WHILE_OPEN * len(people) * hours * mean_rate * BUSINESS_FRACTION

    resolution = engaged_usd + delay(script_h)
    unresolved = delay(window_h)

    # Money at risk: stated ledger amounts are the only in-episode money.
    amounts = []
    for event in episode["events"]:
        for entry in event.get("ledger_entries", []):
            if entry["amount_minor"] > 0:
                amounts.append({"currency": entry["currency"], "amount": entry["amount_minor"] / MINOR_UNITS})
    severity = next(
        (row["facts"].get("severity") for row in episode["initial_state"] if row.get("facts", {}).get("severity")),
        None,
    )
    if amounts:
        tier, status = "A", "stated_in_episode"
        note = "ledger amounts are stated by the episode; exposure follows from settlement/funding arithmetic on them"
        needs = None
    elif severity in ("material", "critical"):
        tier, status = "C", "signal_only"
        note = f"case severity '{severity}' is stated; no amount or count"
        needs = "owner: does this exception class carry client, regulatory or P&L impact, and roughly what?"
    else:
        tier, status = "D", "not_estimable_from_episode"
        note = "no ledger amount, count or severity signal in the episode"
        needs = "owner: does this incident carry money at risk at all?"

    grade = "A" if tier == "A" else "C" if tier == "C" else "D"
    return {
        "schema": CARD_SCHEMA,
        "card_id": card_id,
        "episode_id": episode["episode_id"],
        "grade": grade,
        "grade_meaning": {
            "A": "money stated in-episode; delay x people measured from sealed timestamps",
            "C": "severity signal only; delay x people is the only priced term",
            "D": "delay x people only; nothing in the episode prices money at risk",
        }[grade],
        "resolution_cost_usd": round(resolution),
        "unresolved_cost_usd": round(unresolved),
        "status": "priced",
        "delay_people": {
            "engaged_usd": round(engaged_usd),
            "delay_usd_script": round(delay(script_h)),
            "delay_usd_window": round(delay(window_h)),
            "script_elapsed_hours": round(script_h, 1),
            "window_hours": round(window_h, 1),
            "people": len(people),
            "events": len(events),
            "engaged_hours": round(engaged_h, 2),
            "basis": {
                "elapsed": "measured · sealed event timestamps (script) and time_window (unresolved)",
                "people": "measured · distinct actor ids on primary events",
                "grade": (
                    "assumed · synthetic seats mapped role->band"
                    + (f"; unmapped roles priced as Associate: {sorted(unmapped_roles)}" if unmapped_roles else "")
                ),
                "rate": RATE_BASIS,
                "engaged_minutes": f"assumed · {ENGAGED_MINUTES_PER_EVENT} min per primary event",
                "capacity_while_open": f"assumed · {int(CAPACITY_WHILE_OPEN * 100)}% of each participant's capacity, business hours",
            },
            "confidence": "high on elapsed and people; low on grade mix (synthetic seats); the capacity fraction is the swing input",
            "sensitivity": {
                "resolution_capacity_2.5pct": round(engaged_usd + delay(script_h) / 2),
                "resolution_capacity_10pct": round(engaged_usd + delay(script_h) * 2),
                "unresolved_capacity_2.5pct": round(delay(window_h) / 2),
                "unresolved_capacity_10pct": round(delay(window_h) * 2),
            },
        },
        "money_at_risk": {
            "tier": tier,
            "status": status,
            "note": note,
            "needs": needs,
            "evidence": {"amounts": amounts, "severity": severity},
            "basis": "the episode's own ledger entries and case facts; no external data",
            "confidence": {"A": "medium", "C": "low", "D": "none"}[tier],
            "included_in_headline": False,
        },
        "rate_basis": RATE_BASIS,
        "provenance": {
            "module": "cleanroom_eval.cost_grader",
            "constants": ["RATE_USD_PER_HOUR", "BAND_BY_ROLE", "ENGAGED_MINUTES_PER_EVENT", "CAPACITY_WHILE_OPEN"],
            "method_origin": "record-lane cost card method (delay x people; basis tokens; null where not estimable)",
        },
    }


def build_cards(episodes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {e["episode_id"]: build_card(e) for e in episodes}


def score_run(metrics: Mapping[str, Any], cards: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Denominate a free-run's outcomes in measured cost against a refuse-everything null."""

    rows = []
    total_loss = 0.0
    null_loss = 0.0
    priced = 0
    for outcome in metrics["per_episode"]:
        card = cards.get(outcome["episode_id"])
        if card is None or card.get("status") != "priced":
            rows.append({
                "episode_id": outcome["episode_id"],
                "complete": outcome["complete"],
                "expected_loss_usd": None,
                "null_policy_loss_usd": None,
                "cost_card_id": card["card_id"] if card else None,
                "status": "not_evidenced" if card is None else card["status"],
                "needs": card.get("needs") if card else "no cost card for this episode",
            })
            continue
        priced += 1
        loss = card["resolution_cost_usd"] if outcome["complete"] else card["unresolved_cost_usd"]
        total_loss += loss
        null_loss += card["unresolved_cost_usd"]
        rows.append({
            "episode_id": outcome["episode_id"],
            "complete": outcome["complete"],
            "expected_loss_usd": loss,
            "null_policy_loss_usd": card["unresolved_cost_usd"],
            "loss_avoided_usd": card["unresolved_cost_usd"] - loss,
            "cost_card_id": card["card_id"],
            "grade": card["grade"],
            "money_at_risk_tier": card["money_at_risk"]["tier"],
            "status": "priced",
        })
    return {
        "schema": SCORES_SCHEMA,
        "run_id": metrics["run_id"],
        "policy": metrics["policy"],
        "all_pass": metrics["all_pass"],
        "rate_basis": RATE_BASIS,
        "episodes": len(rows),
        "priced_episodes": priced,
        "unpriced_episodes": len(rows) - priced,
        "expected_loss_usd": round(total_loss),
        "null_policy_loss_usd": round(null_loss),
        "loss_avoided_usd": round(null_loss - total_loss),
        "loss_avoided_fraction": round((null_loss - total_loss) / null_loss, 4) if null_loss else None,
        "headline_basis": "delay x people only; money at risk reported per card, not summed",
        "cards_sha256": digest(cards),
        "per_episode": rows,
    }


def write_scores(run_dir: Path, metrics: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cards = build_cards(episodes)
    scores = score_run(metrics, cards)
    (run_dir / "cost_cards.json").write_text(json.dumps(cards, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    return scores
