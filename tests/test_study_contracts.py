"""Darwin reads setups-owned requirements/assessments from copied producer fixtures."""

import json
from importlib.resources import files
from pathlib import Path

import pytest

from darwin.study_contracts import (
    StudyContractError,
    inspect_requirements,
    validate_assessment,
)

ROOT = Path(__file__).parent / "fixtures" / "setups-contracts"
CASES = ROOT / "fixtures" / "acceptance-assessment-v1"
MANIFEST = json.loads((CASES / "manifest.json").read_text())


@pytest.mark.parametrize("name", ["study-requirements", "acceptance-assessment"])
def test_vendored_runtime_schemas_equal_copied_producer_schemas(name):
    vendored = files("darwin").joinpath(f"setups-{name}-v1.schema.json").read_bytes()
    assert vendored == (ROOT / f"{name}-v1.schema.json").read_bytes()


def _kwargs(case):
    return {
        "requirements": (CASES / case["requirements"]).read_bytes(),
        "dataset": case["dataset"],
        "evidence": {key: (CASES / path).read_bytes() for key, path in case["evidence"].items()},
        "prior": (CASES / case["prior"]).read_bytes() if case["prior"] else None,
        "allow_synthetic": case["test_mode"],
    }


@pytest.mark.parametrize("case", MANIFEST["cases"], ids=lambda case: case["name"])
def test_assessment_cases_match_producer_expectations(case):
    assessment = (CASES / case["assessment"]).read_bytes()
    expect = case["expect"]
    if "error" in expect:
        with pytest.raises(StudyContractError) as caught:
            validate_assessment(assessment, **_kwargs(case))
        assert caught.value.code == expect["error"]
        return
    result = validate_assessment(assessment, **_kwargs(case))
    assert result["status"] == expect["status"]
    assert result["assessment"]["sha256"] == expect["assessment_sha256"]
    assert result["requirements"]["sha256"] == expect["requirements_sha256"]
    assert result["authority"] == expect["authority"]
    assert result["evidence"] == expect["evidence"]
    assert result["supersedes"] == expect["supersedes"]


def test_draft_requirements_fixture_is_identity_only_never_ready():
    data = (ROOT / "study-requirements-v1.fixture.json").read_bytes()
    report = inspect_requirements(data)
    assert report["review_status"] == "draft"
    assert report["ready"] is False
    assert report["requirements"]["revision"] == "1"


def test_synthetic_authorities_are_refused_outside_test_mode():
    data = (CASES / "requirements.json").read_bytes()
    with pytest.raises(StudyContractError) as caught:
        inspect_requirements(data)
    assert caught.value.code == "synthetic_not_trusted"
    assert inspect_requirements(data, allow_synthetic=True)["ready"] is True


def test_unsupported_contract_is_structured():
    with pytest.raises(StudyContractError) as caught:
        inspect_requirements(b'{"contract": "other/1"}')
    assert caught.value.code == "unsupported_contract"
