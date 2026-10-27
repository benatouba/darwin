"""Producer fixtures for darwin.product-handoff/2 consumers (dashboard#18).

The committed fixtures under darwin/contracts/fixtures/product-handoff-v2 are
what consumers copy. Every case must keep its documented outcome, and the
acceptance records a handoff cites must validate against the exact dataset
identity of the product they are attached to.
"""

import json
from importlib.resources import files
from pathlib import Path

import pytest

from darwin import acceptance, handoff

ROOT = Path(str(files("darwin").joinpath("contracts", "fixtures", "product-handoff-v2")))
CASES = json.loads((ROOT / "cases.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fixture_cases_keep_their_outcome(case):
    data = (ROOT / case["handoff"]).read_bytes()
    kwargs = {
        "artifact_root": ROOT / case["artifact_root"] if case.get("artifact_root") else None,
        "allow_synthetic": case["test_mode"],
    }
    if "error" in case["expect"]:
        with pytest.raises(handoff.HandoffError) as caught:
            handoff.validate_handoff(data, **kwargs)
        assert caught.value.code == case["expect"]["error"]
        return
    document = handoff.validate_handoff(data, **kwargs)
    statuses = {p["product_id"]: p["scientific_status"] for p in document["products"]}
    assert statuses == case["expect"]["scientific_status"]
    completeness = {p["product_id"]: p["execution_completeness"] for p in document["products"]}
    assert completeness == case["expect"]["execution_completeness"]


def test_cited_acceptance_records_bind_the_exact_product_dataset():
    document = json.loads((ROOT / "handoff-assessed.json").read_bytes())
    requirements = (ROOT / "acceptance/requirements.json").read_bytes()
    evidence = {
        json.loads(p.read_bytes())["id"]: p.read_bytes()
        for p in sorted((ROOT / "acceptance/evidence").iterdir())
    }
    records = {
        json.loads(p.read_bytes())["id"]: p.read_bytes()
        for p in sorted((ROOT / "acceptance").glob("assessment-*.json"))
    }
    checked = 0
    for product in document["products"]:
        for reference in product["acceptance"]:
            record = records[reference["id"]]
            assert acceptance.sha256_bytes(record) == reference["sha256"]
            prior = records[reference["supersedes"]["id"]] if reference["supersedes"] else None
            state = acceptance.validate_assessment(
                record,
                requirements=requirements,
                dataset=product["dataset"],
                evidence=evidence,
                prior=prior,
                allow_synthetic=True,
            )
            assert state["status"] == reference["status"]
            checked += 1
    assert checked == 2


def test_v2_fixture_products_carry_every_legacy_v1_field():
    legacy = json.loads((ROOT / "legacy-v1.json").read_text())
    current = json.loads((ROOT / "handoff-unassessed.json").read_text())
    for old, new in zip(legacy["products"], current["products"], strict=True):
        assert {k: v for k, v in old.items() if k != "status"}.items() <= new.items()
