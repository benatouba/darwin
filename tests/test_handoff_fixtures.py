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
from darwin.study_contracts import load_assessment_bundle

ROOT = Path(str(files("darwin").joinpath("contracts", "fixtures", "product-handoff-v2")))
CASES = json.loads((ROOT / "cases.json").read_text())["cases"]


def _bundles(spec: dict | None) -> dict | None:
    """Exact acceptance record bytes a case names, as validate_handoff takes them."""
    if spec is None:
        return None
    return {
        product: [load_assessment_bundle(ROOT / path) for path in paths]
        for product, paths in spec.items()
    }


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fixture_cases_keep_their_outcome(case):
    data = (ROOT / case["handoff"]).read_bytes()
    kwargs = {
        "artifact_root": ROOT / case["artifact_root"] if case.get("artifact_root") else None,
        "assessments": _bundles(case.get("acceptance")),
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
    bundles = {
        json.loads(bundle["assessment"])["id"]: bundle
        for bundle in map(load_assessment_bundle, sorted((ROOT / "acceptance").glob("bundle-*")))
    }
    checked = 0
    for product in document["products"]:
        for reference in product["acceptance"]:
            bundle = bundles[reference["id"]]
            assert acceptance.sha256_bytes(bundle["assessment"]) == reference["sha256"]
            state = acceptance.validate_assessment(
                bundle["assessment"],
                requirements=bundle["requirements"],
                dataset=product["dataset"],
                evidence=bundle["evidence"],
                authorizations=bundle["authorizations"],
                prior=bundle["prior"],
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


def _cli_assessments():
    return [
        arg
        for path in ("acceptance/bundle-1-rejected.json", "acceptance/bundle-2-accepted.json")
        for arg in ("--assessment", f"synthetic:prcp-slice={ROOT / path}")
    ]


def test_cli_prepares_the_same_assessed_handoff_as_python(tmp_path):
    artifacts = tmp_path / "artifacts.json"
    artifacts.write_text(
        json.dumps(
            {
                "synthetic:prcp-slice": [str(ROOT / "artifacts/prcp-slice.json")],
                "synthetic:landmask": [str(ROOT / "artifacts/landmask.json")],
            }
        )
    )
    execution = tmp_path / "execution.json"
    run = {"id": "synthetic:execution-report", "version": "1", "sha256": "c9e73d7c056ce843a739e08d7ff04e357b7eef67536467c5eb1ebf17f245b0d0"}
    execution.write_text(
        json.dumps({"synthetic:prcp-slice": {"completeness": "partial", "evidence": run}})
    )
    handoff.main(
        [
            "prepare",
            str(ROOT / "manifest.yaml"),
            str(tmp_path / "handoff.json"),
            "--artifacts",
            str(artifacts),
            "--execution",
            str(execution),
            *_cli_assessments(),
            "--creator",
            "synthetic:darwin-fixture-builder",
            "--created-at",
            "2026-10-27T00:00:00Z",
            "--test-mode",
        ]
    )
    assert (tmp_path / "handoff.json").read_bytes() == (ROOT / "handoff-assessed.json").read_bytes()


def test_cli_validation_verifies_acceptance_like_python(capsys):
    target = str(ROOT / "handoff-assessed.json")
    handoff.main(["validate", target, "--test-mode", *_cli_assessments()])
    assert json.loads(capsys.readouterr().out)["products"] == 2
    with pytest.raises(SystemExit):
        handoff.main(["validate", target, "--test-mode"])
    assert json.loads(capsys.readouterr().err)["code"] == "acceptance_unverified"


def test_committed_fixtures_are_what_the_builder_writes(tmp_path):
    # Receipts pin the check implementation hash, so a changed acceptance.py
    # must come with regenerated consumer fixtures.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_handoff_fixtures", Path(__file__).with_name("build_handoff_fixtures.py")
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.build(tmp_path / "fixtures")
    built = {p.relative_to(tmp_path / "fixtures"): p.read_bytes()
             for p in (tmp_path / "fixtures").rglob("*") if p.is_file()}
    committed = {p.relative_to(ROOT): p.read_bytes() for p in ROOT.rglob("*") if p.is_file()}
    assert built.keys() == committed.keys()
    assert [name for name in built if built[name] != committed[name]] == []
