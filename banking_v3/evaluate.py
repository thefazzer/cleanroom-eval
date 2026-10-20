"""Executable reference rollouts, transcript replay and adversarial validation."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from .contract import CRITERIA, digest, load_package
from .environment import Environment
from .rules import resolve


def reference_rollout(case):
    """Evaluator-informed solvability witness, explicitly NOT a model baseline."""
    env = Environment(case)
    gold = resolve(case)
    docs = {d["id"]: d for d in case["documents"]}
    requests = []
    def send(tool, **args):
        if tool not in ("read_document", "inspect"):
            args.update(version=env.observe()["state"]["version"], request_id=f"step-{len(requests)}")
        request = {"tool": tool, "args": args}
        requests.append(request)
        response = env.step(request)
        if not response["ok"]:
            raise AssertionError(f"reference rejected: {response['error']}")
        return response
    for document_id in gold["evidence"]:
        send("read_document", document_id=document_id)
    send("assess", policy_id=gold["policy"], value=gold["target"], diagnosis=gold["diagnosis"], route=gold["route"],
         citations=[{"id": i, "sha256": docs[i]["sha256"]} for i in gold["evidence"]])
    if gold["route"] == "approve":
        send("request_approval")
    if gold["route"] == "escalate":
        send("escalate", owner=gold["owner"], reason="missing_evidence")
    else:
        send("execute", operation=gold["operation"])
    receipt = send("inspect")["state"]["receipt"]
    send("close", receipt_id=receipt["id"], status="handed_off" if gold["route"] == "escalate" else "resolved")
    return env.grade(), requests, env.transcript()


def replay(case, requests):
    env = Environment(case)
    for request in requests:
        env.step(request)
    return env.grade()


def attacks(requests):
    """Single-fault mutations target arithmetic, evidence, routing and control gates."""
    scenarios = {}
    def altered(name, tool, key, value):
        copy = deepcopy(requests)
        action = next(r for r in copy if r["tool"] == tool)
        action["args"][key] = value(action["args"][key]) if callable(value) else value
        scenarios[name] = copy
    altered("wrong_calculation", "assess", "value", lambda v: v + 1)
    altered("wrong_diagnosis", "assess", "diagnosis", lambda v: "missing_evidence" if v != "missing_evidence" else "within_authority")
    altered("wrong_route", "assess", "route", lambda v: "execute" if v != "execute" else "escalate")
    altered("wrong_policy", "assess", "policy_id", "nonexistent")
    altered("missing_citations", "assess", "citations", [])
    altered("fabricated_citation", "assess", "citations", [{"id": "nonexistent", "sha256": "0" * 64}])
    altered("wrong_closure", "close", "status", lambda v: "resolved" if v != "resolved" else "handed_off")
    altered("fabricated_receipt", "close", "receipt_id", "fabricated")
    altered("stale_version", "assess", "version", -1)
    for tool in ("read_document", "inspect", "close", "request_approval"):
        if any(r["tool"] == tool for r in requests):
            scenarios[f"omit_{tool}"] = [deepcopy(r) for r in requests if r["tool"] != tool]
    if any(r["tool"] == "escalate" for r in requests):
        altered("wrong_escalation_owner", "escalate", "owner", "wrong_queue")
    else:
        altered("wrong_operation", "execute", "operation", "unsupported")
    return scenarios


def validate_dataset(root, answers):
    manifest, cases = load_package(root)
    expected = {a["case_id"]: a for a in answers}
    if set(expected) != {c["id"] for c in cases}:
        raise ValueError("evaluator index is not the exact case set")
    rows, failures, total_attacks = [], [], 0
    routes, family_counts, attack_counts = Counter(), Counter(), Counter()
    for case in cases:
        answer = expected[case["id"]]
        if answer["case_sha256"] != digest(case) or answer["expected"] != resolve(case):
            raise ValueError("answer binding mismatch")
        grade, requests, transcript = reference_rollout(case)
        if not grade["complete"]:
            raise AssertionError("unsolvable case")
        failed_attacks = []
        for name, altered in attacks(requests).items():
            total_attacks += 1
            attack_counts[name] += 1
            if replay(case, altered)["complete"]:
                failed_attacks.append(name)
                failures.append({"case_id": case["id"], "attack": name})
        routes[answer["expected"]["route"]] += 1
        family_counts[case["family"]] += 1
        rows.append({"case_id": case["id"], "case_sha256": digest(case),
                     "solvability": "PASS", "arithmetic_crosscheck": answer["arithmetic_crosscheck"],
                     "reference_turns": grade["turns"], "trajectory_sha256": digest(transcript),
                     "attack_failures": failed_attacks, "model_difficulty": "NOT_MEASURED",
                     "human_certification": "NOT_ISSUED"})
    if failures:
        raise AssertionError(f"adversarial completions: {failures}")
    return {"schema": "bankingenv.validation/v3", "split": manifest["split"],
            "case_index_sha256": manifest["case_index_sha256"], "cases": len(cases),
            "documents": sum(len(c["documents"]) for c in cases),
            "scored_criteria": len(cases) * len(CRITERIA), "criteria_per_case": len(CRITERIA),
            "criteria_interpretation": "Eight rubric checks; not eight statistically independent decisions.",
            "reference_complete": len(rows), "adversarial_trajectories": total_attacks,
            "adversarial_completions": 0, "attack_counts": dict(attack_counts),
            "routes": dict(routes), "families": dict(family_counts), "rows": rows,
            "validation_level": "MECHANICAL_ONLY", "independent_review": "NOT_RUN"}
