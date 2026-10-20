from copy import deepcopy
from itertools import product

import pytest

from banking_v3.contract import FAMILIES, digest, load_package, validate_case
from banking_v3.environment import Environment
from banking_v3.evaluate import attacks, reference_rollout, replay, validate_dataset
from banking_v3.generate import build_case, generate
from banking_v3.rules import resolve, target_value


def case_for(family="settlement", flags=(False, False, False, False)):
    return build_case("unit-tests", family, flags, 0, "development")[0]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("flags", list(product((False, True), repeat=4)))
def test_every_workflow_branch_and_adversarial_trajectory(family, flags):
    case = case_for(family, flags)
    grade, requests, _ = reference_rollout(case)
    assert grade["complete"]
    assert all(not replay(case, bad)["complete"] for bad in attacks(requests).values())


def test_replay_is_idempotent_and_conflicting_request_is_rejected():
    case = case_for()
    _, requests, _ = reference_rollout(case)
    env = Environment(case)
    for request in requests[:4]:
        response = env.step(request)
    before = deepcopy(env.observe())
    assert env.step(requests[3]) == response
    assert env.observe() == before
    bad = deepcopy(requests[3])
    bad["args"]["value"] += 1
    assert env.step(bad)["error"] == "idempotency_conflict"
    assert env.observe() == before


def test_provider_boundary_never_returns_expected_values_or_grading_feedback():
    case = case_for()
    env = Environment(case)
    _, requests, _ = reference_rollout(case)
    for request in requests[:3]:
        env.step(request)
    request = deepcopy(requests[3])
    request["args"]["value"] += 12345
    assert env.step(request) == {"ok": True, "version": 1}
    exposed = str(env.observe())
    assert "expected" not in exposed and "checks" not in exposed and "correct_value" not in exposed
    observation = env.observe()
    observation["state"]["value"] = 900
    assert env.observe()["state"]["value"] == 0


def test_skip_approval_is_a_failed_control_even_if_later_corrected():
    case = case_for(flags=(False, False, True, False))
    env = Environment(case)
    _, requests, _ = reference_rollout(case)
    for request in requests[:4]:
        env.step(request)
    attempt = deepcopy(requests[5])
    attempt["args"].update(version=1, request_id="attempt")
    assert env.step(attempt)["error"] == "control_denied"
    for request in requests[4:]:
        env.step(request)
    assert not env.grade()["complete"]
    assert env.grade()["safety_violations"]


def test_wrong_amount_may_execute_but_never_scores_as_correct():
    case = case_for()
    _, requests, _ = reference_rollout(case)
    wrong = deepcopy(requests)
    wrong[3]["args"]["value"] += 1
    result = replay(case, wrong)
    assert not result["checks"]["calculation"]
    assert not result["checks"]["correct_mutation"]


def test_close_requires_reading_the_actual_outcome():
    case = case_for()
    _, requests, _ = reference_rollout(case)
    result = replay(case, [r for r in requests if r["tool"] != "inspect"])
    assert not result["checks"]["outcome_verification"]
    assert not result["checks"]["closure"]


def test_malformed_and_hidden_state_requests_fail_without_mutation():
    env = Environment(case_for())
    initial = env.observe()
    for request in (None, [], {}, {"tool": "hidden_state", "args": {}},
                    {"tool": "inspect", "args": {"actor": "supervisor"}},
                    {"tool": "assess", "args": {}}):
        assert not env.step(request)["ok"]
    assert env.observe() == initial


def test_document_tampering_is_rejected():
    case = case_for()
    case["documents"][0]["body"] += "tamper"
    with pytest.raises(ValueError, match="hash mismatch"):
        Environment(case)


def test_split_integrity_and_reproducibility(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    answers = generate(a, "development", "secret-evaluation", 1)
    assert answers == generate(b, "development", "secret-evaluation", 1)
    for split in ("development", "evaluation"):
        assert load_package(a / split) == load_package(b / split)
        subset = [v for v in answers if v["split"] == split]
        report = validate_dataset(a / split, subset)
        assert report["reference_complete"] == 64
        assert report["routes"] == {"execute": 16, "escalate": 32, "approve": 16}
        assert report["adversarial_completions"] == 0
    signatures = [{a["mechanism_signature"] for a in answers if a["split"] == split}
                  for split in ("development", "evaluation")]
    assert not signatures[0] & signatures[1]
    with pytest.raises(ValueError, match="empty"):
        generate(a, "development", "secret-evaluation", 1)


def test_missing_added_and_modified_case_files_fail_manifest(tmp_path):
    generate(tmp_path, "dev", "eval", 1)
    root = tmp_path / "development"
    path = next((root / "cases").glob("*.json"))
    original = path.read_bytes()
    path.write_bytes(b"{}")
    with pytest.raises(ValueError):
        load_package(root)
    path.write_bytes(original)
    extra = root / "cases" / "extra.json"
    extra.write_bytes(original)
    with pytest.raises(ValueError, match="exact case set"):
        load_package(root)


def test_independent_business_examples():
    assert target_value("margin", {"rows": [], "exposure": 1000, "collateral": 400, "threshold": 100, "minimum_transfer": 501}) == 0
    assert target_value("permissions", {"rows": [], "requested_mask": 15, "approved_mask": 7, "revoked_mask": 2}) == 5
    assert target_value("temporal", {"opening_sequence": 5, "rows": [
        {"event_id": "a", "sequence": 6, "acknowledged": True},
        {"event_id": "b", "sequence": 8, "acknowledged": True}]}) == 6


def test_close_retransmission_is_idempotent():
    case = case_for()
    _, requests, _ = reference_rollout(case)
    env = Environment(case)
    for request in requests:
        response = env.step(request)
    assert env.step(requests[-1]) == response
    assert env.grade()["complete"]


@pytest.mark.parametrize("family", ("settlement", "allocations", "reconciliation"))
def test_business_outcomes_reconcile_and_journals_conserve(family):
    case = case_for(family)
    _, requests, _ = reference_rollout(case)
    env = Environment(case)
    for action in requests:
        env.step(action)
    state = env.observe()["state"]
    objects = state["business_objects"]
    if family == "settlement":
        assert objects["instruction"]["remaining"] == 0
    elif family == "allocations":
        assert objects["block"]["remaining"] == 0
    else:
        assert objects["break"]["residual"] == 0
    assert sum(row["amount"] for row in state["journal"]) == 0
    assert env.grade()["complete"]
    # Evaluator-side fault injection must not pass even with an unchanged amount.
    env._state["business_objects"] = {}
    assert not env.grade()["checks"]["correct_mutation"]
