"""Portable scientific acceptance (darwin#18).

Darwin consumes the setups-owned ``darwin-wrf-setups.acceptance-assessment/1``
and ``study-requirements/1`` contracts through copied producer fixtures
(tests/data/setups), runs checks into ``darwin.assessment-evidence/1``
receipts and records decisions only from an authority the exact
requirements authorize.
"""

import json
from pathlib import Path

import pytest

from darwin import acceptance

SETUPS = Path(__file__).parent / "data" / "setups" / "fixtures" / "acceptance-assessment-v1"
CASES = json.loads((SETUPS / "manifest.json").read_text())["cases"]


def _inputs(case):
    return {
        "assessment": (SETUPS / case["assessment"]).read_bytes(),
        "requirements": (SETUPS / case["requirements"]).read_bytes(),
        "dataset": case["dataset"],
        "evidence": {k: (SETUPS / p).read_bytes() for k, p in case["evidence"].items()},
        "prior": (SETUPS / case["prior"]).read_bytes() if case["prior"] else None,
        "allow_synthetic": case["test_mode"],
    }


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_darwin_agrees_with_every_setups_producer_case(case):
    expect = case["expect"]
    if "error" in expect:
        with pytest.raises(acceptance.AcceptanceError) as caught:
            acceptance.validate_assessment(**_inputs(case))
        assert caught.value.code == expect["error"]
        return
    result = acceptance.validate_assessment(**_inputs(case))
    assert result["status"] == expect["status"]
    assert result["assessment"]["sha256"] == expect["assessment_sha256"]
    assert result["requirements"]["sha256"] == expect["requirements_sha256"]
    assert result["authority"] == expect["authority"]
    assert result["evidence"] == expect["evidence"]
    assert result["supersedes"] == expect["supersedes"]


REQUIREMENTS = (SETUPS / "requirements.json").read_bytes()
DATASET = {
    "id": "synthetic:dataset",
    "version": "1",
    "sha256": "7136b2416b4f8b59171a7aaa5cb6f799bb81f84c9f03a1fc3fd56e4e66bf3774",
}
REVIEWER = {
    "kind": "reviewer",
    "id": "synthetic:reviewer",
    "authorization": {
        "id": "synthetic:reviewer-authorization",
        "version": "1",
        "sha256": "44beb9be6083709be51762c04df8921acb1dad31dd947a99c6ec8db664cfe5ec",
    },
}


def _receipts(outcomes=None):
    return acceptance.run_checks(
        DATASET,
        REQUIREMENTS,
        inputs={"synthetic:outcomes": outcomes or {}},
        allow_synthetic=True,
    )


def test_checks_run_every_criterion_of_the_exact_requirements_revision():
    receipts = _receipts({"synthetic:criterion-b": "fail"})
    assert [r["criterion"] for r in receipts] == [
        "synthetic:criterion-a",
        "synthetic:criterion-b",
    ]
    assert [r["outcome"] for r in receipts] == ["pass", "fail"]
    for receipt in receipts:
        assert receipt["contract"] == "darwin.assessment-evidence/1"
        assert receipt["dataset"] == DATASET
        assert receipt["requirements"] == {
            "id": "synthetic:requirements",
            "revision": "1",
            "sha256": "b5a7d3ea218154aacf3fb64ceafecc46ae171733038dd17d94becf39b47c1015",
        }
        assert receipt["evidence_type"] == "synthetic:test-receipt"
        # Evidence is not a decision: it names no authority or status.
        assert "authority" not in receipt
        assert "status" not in receipt


def test_checks_refuse_unimplemented_check_names():
    document = json.loads(REQUIREMENTS)
    document["acceptance"]["criteria"][0]["check"] = "darwin.unimplemented"
    with pytest.raises(acceptance.AcceptanceError) as caught:
        acceptance.run_checks(
            DATASET, json.dumps(document).encode(), inputs={}, allow_synthetic=True
        )
    assert caught.value.code == "unsupported_check"


def test_checks_refuse_synthetic_requirements_outside_test_mode():
    with pytest.raises(acceptance.AcceptanceError) as caught:
        acceptance.run_checks(DATASET, REQUIREMENTS, inputs={})
    assert caught.value.code == "synthetic_not_trusted"


def test_checks_refuse_a_dynamic_dataset_version():
    with pytest.raises(acceptance.AcceptanceError) as caught:
        acceptance.run_checks(
            {**DATASET, "version": "latest"}, REQUIREMENTS, inputs={}, allow_synthetic=True
        )
    assert caught.value.code == "invalid_dataset"


def _record(status, receipts, authority, **kwargs):
    return acceptance.record_assessment(
        assessment_id=kwargs.pop("assessment_id", f"synthetic:{status}"),
        dataset=DATASET,
        requirements=REQUIREMENTS,
        status=status,
        evidence=[acceptance.canonical_bytes(r) for r in receipts],
        authority=authority,
        allow_synthetic=True,
        **kwargs,
    )


def _evidence_bytes(*receipt_lists):
    return {r["id"]: acceptance.canonical_bytes(r) for receipts in receipt_lists for r in receipts}


def test_reviewer_acceptance_binds_dataset_requirements_evidence_and_authority():
    receipts = _receipts()
    record = _record("accepted", receipts, REVIEWER)
    state = acceptance.validate_assessment(
        record,
        requirements=REQUIREMENTS,
        dataset=DATASET,
        evidence=_evidence_bytes(receipts),
        allow_synthetic=True,
    )
    assert state["status"] == "accepted"
    assert state["authority"] == REVIEWER
    assert state["dataset"] == DATASET
    cited = {e["artifact"]["id"]: e["artifact"]["sha256"] for e in state["evidence"]}
    assert cited == {
        r["id"]: acceptance.sha256_bytes(acceptance.canonical_bytes(r)) for r in receipts
    }


def test_a_decision_needs_an_authorized_reviewer_or_policy():
    with pytest.raises(acceptance.AcceptanceError) as missing:
        _record("accepted", _receipts(), None)
    assert missing.value.code == "missing_authority"
    stranger = {**REVIEWER, "id": "synthetic:someone-else"}
    with pytest.raises(acceptance.AcceptanceError) as unlisted:
        _record("rejected", _receipts(), stranger)
    assert unlisted.value.code == "unauthorized_authority"


def test_unassessed_records_carry_no_authority():
    record = _record("unassessed", [], None)
    assert json.loads(record)["authority"] is None
    with pytest.raises(acceptance.AcceptanceError) as caught:
        _record("unassessed", [], REVIEWER)
    assert caught.value.code == "invalid_assessment"


def test_evidence_for_another_dataset_is_refused():
    receipts = acceptance.run_checks(
        {**DATASET, "version": "2"}, REQUIREMENTS, inputs={}, allow_synthetic=True
    )
    with pytest.raises(acceptance.AcceptanceError) as caught:
        _record("accepted", receipts, REVIEWER)
    assert caught.value.code == "dataset_mismatch"


def test_reassessment_supersedes_by_exact_bytes_and_keeps_prior_evidence():
    failing = _receipts({"synthetic:criterion-a": "fail"})
    rejected = _record("rejected", failing[:1], REVIEWER, assessment_id="synthetic:first")
    rejected_copy = bytes(rejected)
    later = _receipts()
    record = _record(
        "accepted",
        later,
        REVIEWER,
        assessment_id="synthetic:second",
        prior=rejected,
        prior_evidence=_evidence_bytes(failing[:1]),
    )
    assert rejected == rejected_copy
    document = json.loads(record)
    assert document["supersedes"] == {
        "id": "synthetic:first",
        "sha256": acceptance.sha256_bytes(rejected),
    }
    prior_entries = json.loads(rejected)["evidence"]
    assert all(entry in document["evidence"] for entry in prior_entries)
    state = acceptance.validate_assessment(
        record,
        requirements=REQUIREMENTS,
        dataset=DATASET,
        evidence=_evidence_bytes(failing[:1], later),
        prior=rejected,
        allow_synthetic=True,
    )
    assert state["status"] == "accepted"
