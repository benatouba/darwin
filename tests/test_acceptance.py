"""Portable scientific acceptance (darwin#18).

Darwin runs checks named by exact setups requirements into
``darwin.assessment-evidence/1`` receipts and records
``darwin-wrf-setups.acceptance-assessment/1`` decisions only from an
authority those requirements authorize. Conformance of the record validator
with the copied setups producer fixtures is tested in test_study_contracts.
"""

import json
from pathlib import Path

import pytest
from decision_signing import public_key, sign

from darwin import acceptance

SETUPS = (
    Path(__file__).parent
    / "fixtures"
    / "setups-contracts"
    / "fixtures"
    / "acceptance-assessment-v1"
)
REQUIREMENTS = (SETUPS / "requirements.json").read_bytes()
DATASET = {
    "id": "synthetic:dataset",
    "version": "1",
    "sha256": "7136b2416b4f8b59171a7aaa5cb6f799bb81f84c9f03a1fc3fd56e4e66bf3774",
}
# The requirements entry, with its pinned signing_keys; records cite the authority
# without them.
REVIEWER = json.loads(REQUIREMENTS)["acceptance"]["authorized_authorities"][0]
AUTHORITY = {key: REVIEWER[key] for key in ("kind", "id", "authorization")}
REVIEWER_AUTHORIZATION = (SETUPS / "authorizations" / "reviewer.json").read_bytes()
AUTHORIZATIONS = {
    (REVIEWER["authorization"]["id"], REVIEWER["authorization"]["version"]): REVIEWER_AUTHORIZATION
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
            "sha256": acceptance.sha256_bytes(REQUIREMENTS),
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
        authorization=kwargs.pop(
            "authorization", REVIEWER_AUTHORIZATION if authority is not None else None
        ),
        allow_synthetic=True,
        **kwargs,
    )


def _evidence_bytes(*receipt_lists):
    return {
        (r["id"], "1"): acceptance.canonical_bytes(r)
        for receipts in receipt_lists
        for r in receipts
    }


def test_reviewer_acceptance_binds_dataset_requirements_evidence_and_authority():
    receipts = _receipts()
    record = _record("accepted", receipts, REVIEWER)
    state = acceptance.validate_assessment(
        record,
        requirements=REQUIREMENTS,
        dataset=DATASET,
        evidence=_evidence_bytes(receipts),
        authorizations=AUTHORIZATIONS,
        signature=sign(record),
        allow_synthetic=True,
    )
    assert state["status"] == "accepted"
    assert state["authority"] == AUTHORITY
    assert state["signer"] == public_key("reviewer")
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
        authorizations=AUTHORIZATIONS,
        prior=rejected,
        signature=sign(record),
        allow_synthetic=True,
    )
    assert state["status"] == "accepted"


def test_a_decision_needs_the_exact_authorization_bytes_the_requirements_pin():
    # Copying the authority entry out of the requirements is not enough.
    with pytest.raises(acceptance.AcceptanceError) as missing:
        _record("accepted", _receipts(), REVIEWER, authorization=None)
    assert missing.value.code == "authorization_unavailable"
    with pytest.raises(acceptance.AcceptanceError) as forged:
        _record("accepted", _receipts(), REVIEWER, authorization=b"{}")
    assert forged.value.code == "authorization_mismatch"


def _hand_written(status, receipts):
    """A record built without record_assessment, citing the given receipts."""
    document = json.loads(_record("unassessed", [], None))
    document.update(
        status=status,
        authority=AUTHORITY,
        evidence=[acceptance.evidence_reference(acceptance.canonical_bytes(r)) for r in receipts],
    )
    return acceptance.canonical_bytes(document)


def test_hand_written_records_citing_another_datasets_receipts_are_refused():
    other = acceptance.run_checks(
        {**DATASET, "version": "2"}, REQUIREMENTS, inputs={}, allow_synthetic=True
    )
    record = _hand_written("accepted", other)
    with pytest.raises(acceptance.AcceptanceError) as caught:
        acceptance.validate_assessment(
            record,
            requirements=REQUIREMENTS,
            dataset=DATASET,
            evidence=_evidence_bytes(other),
            authorizations=AUTHORIZATIONS,
            signature=sign(record),
            allow_synthetic=True,
        )
    assert caught.value.code == "evidence_dataset_mismatch"


def test_acceptance_resting_only_on_failed_checks_is_refused():
    failing = _receipts({"synthetic:criterion-a": "fail"})
    record = _hand_written("accepted", failing)
    with pytest.raises(acceptance.AcceptanceError) as caught:
        acceptance.validate_assessment(
            record,
            requirements=REQUIREMENTS,
            dataset=DATASET,
            evidence=_evidence_bytes(failing),
            authorizations=AUTHORIZATIONS,
            signature=sign(record),
            allow_synthetic=True,
        )
    assert caught.value.code == "failing_evidence"
    # The same failing evidence can back a rejection.
    rejected = _record("rejected", failing, REVIEWER)
    assert json.loads(rejected)["status"] == "rejected"


def test_malformed_receipt_bytes_are_a_structured_error():
    with pytest.raises(acceptance.AcceptanceError) as caught:
        acceptance.record_assessment(
            assessment_id="synthetic:malformed",
            dataset=DATASET,
            requirements=REQUIREMENTS,
            status="rejected",
            evidence=[b"not json"],
            authority=REVIEWER,
            authorization=REVIEWER_AUTHORIZATION,
            allow_synthetic=True,
        )
    assert caught.value.code == "invalid_evidence"


def _validate(record, receipts, **kwargs):
    return acceptance.validate_assessment(
        record,
        requirements=REQUIREMENTS,
        dataset=DATASET,
        evidence=_evidence_bytes(receipts),
        authorizations=AUTHORIZATIONS,
        allow_synthetic=True,
        **kwargs,
    )


def test_a_recorded_decision_counts_only_once_its_authority_signs_the_exact_bytes():
    receipts = _receipts()
    record = _record("accepted", receipts, REVIEWER)
    with pytest.raises(acceptance.AcceptanceError) as unsigned:
        _validate(record, receipts)
    assert unsigned.value.code == "signature_unavailable"
    for key, namespace, code in (
        ("unpinned", "darwin-acceptance", "signer_not_authorized"),
        ("policy", "darwin-acceptance", "signer_not_authorized"),
        ("reviewer", "file", "signature_namespace_mismatch"),
    ):
        with pytest.raises(acceptance.AcceptanceError) as caught:
            _validate(record, receipts, signature=sign(record, key, namespace))
        assert caught.value.code == code
    reformatted = json.dumps(json.loads(record)).encode()
    with pytest.raises(acceptance.AcceptanceError) as altered:
        _validate(reformatted, receipts, signature=sign(record))
    assert altered.value.code == "signature_mismatch"
    assert _validate(record, receipts, signature=sign(record))["signer"] == public_key()


def test_an_unassessed_record_is_never_signed():
    record = _record("unassessed", [], None)
    assert _validate(record, [])["signer"] is None
    with pytest.raises(acceptance.AcceptanceError) as caught:
        _validate(record, [], signature=sign(record))
    assert caught.value.code == "unexpected_signature"


def test_the_test_signer_reproduces_the_producers_ssh_keygen_signature():
    record = (SETUPS / "accepted.json").read_bytes()
    assert sign(record) == (SETUPS / "accepted.json.sig").read_bytes()


def _write_json(path, document):
    path.write_text(json.dumps(document))
    return str(path)


def test_cli_runs_checks_and_records_the_same_bytes_as_python(tmp_path, capsys):
    dataset = _write_json(tmp_path / "dataset.json", DATASET)
    requirements = tmp_path / "requirements.json"
    requirements.write_bytes(REQUIREMENTS)
    acceptance.main(
        [
            "run-checks",
            "--dataset",
            dataset,
            "--requirements",
            str(requirements),
            "--out-dir",
            str(tmp_path / "evidence"),
            "--test-mode",
        ]
    )
    written = json.loads(capsys.readouterr().out)["receipts"]
    receipts = _receipts()
    expected = [acceptance.canonical_bytes(r) for r in receipts]
    assert [Path(p).read_bytes() for p in written] == expected
    authorization = tmp_path / "reviewer.json"
    authorization.write_bytes(REVIEWER_AUTHORIZATION)
    acceptance.main(
        [
            "record",
            "--assessment-id",
            "synthetic:accepted",
            "--dataset",
            dataset,
            "--requirements",
            str(requirements),
            "--status",
            "accepted",
            "--evidence",
            *written,
            "--authority",
            _write_json(tmp_path / "authority.json", REVIEWER),
            "--authorization",
            str(authorization),
            "--output",
            str(tmp_path / "record.json"),
            "--test-mode",
        ]
    )
    record = (tmp_path / "record.json").read_bytes()
    assert record == _record("accepted", receipts, REVIEWER)
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["signature"] == str(tmp_path / "record.json.sig")
    assert "ssh-keygen -Y sign" in recorded["sign"] and "-n darwin-acceptance" in recorded["sign"]

    # The reviewer signs; validation of the bundle needs that signature.
    bundle = {
        "assessment": "record.json",
        "requirements": "requirements.json",
        "evidence": [
            {"id": json.loads(Path(p).read_bytes())["id"], "version": "1", "path": p}
            for p in written
        ],
        "authorizations": [
            {"id": "synthetic:reviewer-authorization", "version": "1", "path": "reviewer.json"}
        ],
    }
    unsigned = _write_json(tmp_path / "bundle.json", bundle)
    with pytest.raises(SystemExit):
        acceptance.main(["validate", "--bundle", unsigned, "--dataset", dataset, "--test-mode"])
    assert json.loads(capsys.readouterr().err)["code"] == "signature_unavailable"
    (tmp_path / "record.json.sig").write_bytes(sign(record))
    signed = _write_json(tmp_path / "bundle.json", {**bundle, "signature": "record.json.sig"})
    acceptance.main(["validate", "--bundle", signed, "--dataset", dataset, "--test-mode"])
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "accepted"
    assert report["signer"] == public_key()


def test_cli_reports_structured_errors(tmp_path, capsys):
    requirements = tmp_path / "requirements.json"
    requirements.write_bytes(REQUIREMENTS)
    receipts = []
    for index, receipt in enumerate(_receipts()):
        path = tmp_path / f"receipt-{index}.json"
        path.write_bytes(acceptance.canonical_bytes(receipt))
        receipts.append(str(path))
    with pytest.raises(SystemExit):
        acceptance.main(
            [
                "record",
                "--assessment-id",
                "synthetic:x",
                "--dataset",
                _write_json(tmp_path / "dataset.json", DATASET),
                "--requirements",
                str(requirements),
                "--status",
                "accepted",
                "--evidence",
                *receipts,
                "--authority",
                _write_json(tmp_path / "authority.json", REVIEWER),
                "--output",
                str(tmp_path / "record.json"),
                "--test-mode",
            ]
        )
    assert json.loads(capsys.readouterr().err)["code"] == "authorization_unavailable"
    assert not (tmp_path / "record.json").exists()
