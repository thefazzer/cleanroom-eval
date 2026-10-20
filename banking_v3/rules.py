"""Evaluator-side operational semantics. Not imported by the provider adapter.

Rules are fictional house policies, not regulatory or legal assertions.
The documents are the source of every instance-specific input.
"""
from __future__ import annotations

OPERATIONS = {"settlement": "settle", "allocations": "allocate", "reconciliation": "post_adjustment",
              "margin": "issue_call", "permissions": "grant_scope", "lifecycle": "amend",
              "evidence": "attest", "temporal": "advance_sequence"}
FORMULAS = {
    "settlement": "instruction total minus sum of unique completed payment amounts",
    "allocations": "block quantity minus sum of unique allocated quantities",
    "reconciliation": "statement balance minus opening ledger balance minus sum of unique postings",
    "margin": "max(0, exposure minus collateral minus threshold); zero if below minimum transfer",
    "permissions": "requested scope bitmask AND approved scope bitmask AND complement of revoked scope bitmask",
    "lifecycle": "latest effective amendment not later than the case cutoff, ordered by effective minute then revision",
    "evidence": "count unique valid proof records from approved independent issuers",
    "temporal": "highest contiguous acknowledged sequence after the opening sequence; ignore gaps and duplicates",
}


def authoritative(case, kind):
    docs = [d for d in case["documents"] if d["kind"] == kind and d["status"] == "CURRENT"
            and d["effective_minute"] <= case["cutoff_minute"]]
    if len(docs) != 1:
        raise ValueError(f"case must contain exactly one current {kind}")
    return docs[0]


def target_value(family, facts):
    rows = list({r["event_id"]: r for r in facts["rows"]}.values())
    if family == "settlement":
        return facts["instruction_total"] - sum(r["amount"] for r in rows if r["completed"])
    if family == "allocations":
        return facts["block_quantity"] - sum(r["quantity"] for r in rows)
    if family == "reconciliation":
        return facts["statement_balance"] - facts["opening_balance"] - sum(r["amount"] for r in rows)
    if family == "margin":
        call = max(0, facts["exposure"] - facts["collateral"] - facts["threshold"])
        return call if call >= facts["minimum_transfer"] else 0
    if family == "permissions":
        return facts["requested_mask"] & facts["approved_mask"] & ~facts["revoked_mask"]
    if family == "lifecycle":
        eligible = [r for r in rows if r["effective_minute"] <= facts["cutoff_minute"]]
        return max(eligible, key=lambda r: (r["effective_minute"], r["revision"]))["notional"]
    if family == "evidence":
        return len({r["issuer"] for r in rows if r["valid"] and r["issuer"] in facts["approved_issuers"]})
    if family == "temporal":
        sequence = facts["opening_sequence"]
        acks = {r["sequence"] for r in rows if r["acknowledged"]}
        while sequence + 1 in acks:
            sequence += 1
        return sequence
    raise ValueError("unknown family")


def resolve(case):
    policy = authoritative(case, "policy")
    record = authoritative(case, "record")
    control = authoritative(case, "control")
    target = target_value(case["family"], record["data"])
    missing = not control["data"]["supporting_evidence_verified"]
    material = record["data"]["risk_amount"] > policy["data"]["approval_limit"]
    return {
        "target": target,
        "diagnosis": "missing_evidence" if missing else "approval_required" if material else "within_authority",
        "route": "escalate" if missing else "approve" if material else "execute",
        "operation": OPERATIONS[case["family"]],
        "evidence": [d["id"] for d in (policy, record, control)],
        "policy": policy["id"],
        "owner": policy["data"]["escalation_owner"],
    }
