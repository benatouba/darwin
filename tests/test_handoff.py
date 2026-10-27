"""Product/publication handoff v2 (darwin#18).

Darwin prepares immutable products and a ``darwin.product-handoff/2``
document binding exact artifacts, sources, recipe, execution completeness,
scientific status and acceptance references. Preparation runs without any
catalogue: it never publishes, moves a current pointer or grants access.
"""

import json
from pathlib import Path

import pytest

from darwin import acceptance, handoff
from darwin.publisher.immutable import staged_artifact
from darwin.publisher.manifest import load_manifest
from darwin.publisher.smoke import build_smoke_release

SETUPS = Path(__file__).parent / "fixtures" / "setups-contracts" / "fixtures" / "acceptance-assessment-v1"
REQUIREMENTS = (SETUPS / "requirements.json").read_bytes()
REVIEWER = json.loads(REQUIREMENTS)["acceptance"]["authorized_authorities"][0]
AUTHORIZATION = (SETUPS / "authorizations" / "reviewer.json").read_bytes()
AUTHORIZATIONS = {
    (REVIEWER["authorization"]["id"], REVIEWER["authorization"]["version"]): AUTHORIZATION
}
SMOKE = Path(__file__).resolve().parents[1] / "darwin/publisher/manifests/smoke-jan2020-prcp.yaml"
STAMP = "2026-10-27T00:00:00Z"


def _artifact(root: Path, name: str, text: str) -> Path:
    path = root / name
    provenance = {
        "builder": "tests.test_handoff",
        "sources": [{"id": "fixture", "sha256": "0" * 64}],
    }
    with staged_artifact(path, provenance=provenance) as staged:
        staged.write_text(text)
    return path


@pytest.fixture
def smoke_artifacts(tmp_path):
    manifest = load_manifest(SMOKE)
    root = tmp_path / "artifacts"
    return {
        spec.product_id: [_artifact(root, f"{spec.product_id}.bin", spec.product_id)]
        for spec in manifest.products
    }


def _prepare(tmp_path, artifacts, name="handoff.json", **kwargs):
    return handoff.prepare_handoff(
        SMOKE, tmp_path / name, artifacts=artifacts, created_at=STAMP, **kwargs
    )


def test_preparation_binds_exact_artifacts_sources_and_recipe(tmp_path, smoke_artifacts):
    document = _prepare(tmp_path, smoke_artifacts)
    manifest = load_manifest(SMOKE)
    assert document["contract"] == "darwin.product-handoff/2"
    assert json.loads((tmp_path / "handoff.json").read_text()) == document
    landmask = next(
        p for p in document["products"] if p["product_id"] == "GAR_d02km_static_landmask"
    )
    [artifact] = landmask["artifacts"]
    assert artifact["name"] == "GAR_d02km_static_landmask.bin"
    # sha256 of the exact single-file bytes in the darwin.artifact-identity/1 line form
    record = json.loads(
        (tmp_path / "artifacts/GAR_d02km_static_landmask.bin.identity.json").read_text()
    )
    assert artifact["sha256"] == record["artifact"]["sha256"]
    assert artifact["identity_record_sha256"] == acceptance.sha256_bytes(
        (tmp_path / "artifacts/GAR_d02km_static_landmask.bin.identity.json").read_bytes()
    )
    assert landmask["sources"] == [
        {"id": entry.path, "sha256": entry.sha256} for entry in manifest.sources
    ]
    assert landmask["recipe"] == {
        "id": "release-manifest:smoke-jan2020-prcp",
        "version": manifest.revision,
        "sha256": manifest.revision,
    }
    assert landmask["dataset"]["id"] == "GAR_d02km_static_landmask"
    assert landmask["dataset"]["version"] == landmask["version"]
    handoff.validate_handoff(
        (tmp_path / "handoff.json").read_bytes(), artifact_root=tmp_path / "artifacts"
    )


def test_preparation_neither_publishes_nor_claims_visibility(tmp_path, smoke_artifacts):
    document = _prepare(tmp_path, smoke_artifacts)
    for product in document["products"]:
        assert product["lifecycle"] == "prepared"
        assert product["scientific_status"] == "unassessed"
        assert product["acceptance"] == []
        assert product["execution_completeness"] is None
        for claim in ("status", "access", "current_version", "published"):
            assert claim not in product


def test_v2_products_stay_readable_by_the_legacy_v1_importer(tmp_path, smoke_artifacts):
    document = _prepare(tmp_path, smoke_artifacts)
    legacy = build_smoke_release(SMOKE, tmp_path / "legacy.json", created_at=STAMP)
    assert document["manifest_revision"] == legacy["manifest_revision"]
    assert document["release"] == legacy["release"]
    for old, new in zip(legacy["products"], document["products"], strict=True):
        for key, value in old.items():
            if key != "status":
                assert new[key] == value, key


def test_unbound_or_tampered_artifacts_are_refused(tmp_path, smoke_artifacts):
    loose = tmp_path / "loose.bin"
    loose.write_text("no identity record")
    with pytest.raises(handoff.HandoffError) as unbound:
        _prepare(tmp_path, {**smoke_artifacts, "GAR_d02km_static_landmask": [loose]})
    assert unbound.value.code == "unbound_artifact"
    smoke_artifacts["GAR_d02km_static_landmask"][0].chmod(0o644)
    smoke_artifacts["GAR_d02km_static_landmask"][0].write_text("changed")
    with pytest.raises(handoff.HandoffError) as tampered:
        _prepare(tmp_path, smoke_artifacts)
    assert tampered.value.code == "artifact_mismatch"


def test_every_product_needs_exact_artifacts(tmp_path, smoke_artifacts):
    del smoke_artifacts["GAR_d02km_static_landmask"]
    with pytest.raises(handoff.HandoffError) as caught:
        _prepare(tmp_path, smoke_artifacts)
    assert caught.value.code == "missing_artifacts"


def _assessments_for(dataset):
    receipts = acceptance.run_checks(
        dataset,
        REQUIREMENTS,
        inputs={"synthetic:outcomes": {"synthetic:criterion-a": "fail"}},
        allow_synthetic=True,
    )
    failing = {(r["id"], "1"): acceptance.canonical_bytes(r) for r in receipts}
    rejected = acceptance.record_assessment(
        assessment_id="synthetic:first",
        dataset=dataset,
        requirements=REQUIREMENTS,
        status="rejected",
        evidence=list(failing.values()),
        authority=REVIEWER,
        authorization=AUTHORIZATION,
        allow_synthetic=True,
    )
    rerun = acceptance.run_checks(dataset, REQUIREMENTS, inputs={}, allow_synthetic=True)
    passing = {(r["id"], "1"): acceptance.canonical_bytes(r) for r in rerun}
    accepted = acceptance.record_assessment(
        assessment_id="synthetic:second",
        dataset=dataset,
        requirements=REQUIREMENTS,
        status="accepted",
        evidence=list(passing.values()),
        authority=REVIEWER,
        authorization=AUTHORIZATION,
        prior=rejected,
        prior_evidence=failing,
        allow_synthetic=True,
    )
    base = {
        "requirements": REQUIREMENTS,
        "evidence": {**failing, **passing},
        "authorizations": AUTHORIZATIONS,
    }
    return [{**base, "assessment": rejected}, {**base, "assessment": accepted, "prior": rejected}]


def test_status_and_completeness_are_independent_and_history_is_kept(tmp_path, smoke_artifacts):
    first = _prepare(tmp_path, smoke_artifacts)
    landmask = next(p for p in first["products"] if p["product_id"] == "GAR_d02km_static_landmask")
    chain = _assessments_for(landmask["dataset"])
    execution = {
        "completeness": "partial",
        "evidence": {"id": "run", "version": "1", "sha256": "1" * 64},
    }
    document = _prepare(
        tmp_path,
        smoke_artifacts,
        name="assessed.json",
        assessments={"GAR_d02km_static_landmask": chain},
        execution={"GAR_d02km_static_landmask": execution},
        allow_synthetic=True,
    )
    product = next(
        p for p in document["products"] if p["product_id"] == "GAR_d02km_static_landmask"
    )
    assert product["scientific_status"] == "accepted"
    assert product["execution_completeness"] == "partial"
    assert [ref["status"] for ref in product["acceptance"]] == ["rejected", "accepted"]
    assert product["acceptance"][1]["supersedes"] == {
        "id": "synthetic:first",
        "sha256": product["acceptance"][0]["sha256"],
    }
    assert product["lifecycle"] == "prepared"
    others = [p for p in document["products"] if p is not product]
    assert {p["scientific_status"] for p in others} == {"unassessed"}
    assert document["synthetic"] is True


def test_acceptance_for_another_dataset_or_a_broken_chain_is_refused(tmp_path, smoke_artifacts):
    first = _prepare(tmp_path, smoke_artifacts)
    landmask = next(p for p in first["products"] if p["product_id"] == "GAR_d02km_static_landmask")
    chain = _assessments_for(landmask["dataset"])
    with pytest.raises(acceptance.AcceptanceError) as other:
        _prepare(
            tmp_path,
            smoke_artifacts,
            name="other.json",
            assessments={"GAR_d02km_m_2d_prcp_2020-01": chain},
            allow_synthetic=True,
        )
    assert other.value.code == "dataset_mismatch"
    with pytest.raises(handoff.HandoffError) as broken:
        _prepare(
            tmp_path,
            smoke_artifacts,
            name="broken.json",
            assessments={"GAR_d02km_static_landmask": chain[1:]},
            allow_synthetic=True,
        )
    assert broken.value.code == "broken_acceptance_chain"


def _mutated(tmp_path, smoke_artifacts, change):
    document = _prepare(tmp_path, smoke_artifacts)
    change(document)
    return json.dumps(document).encode()


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda d: d.update(contract="darwin.product-handoff/9"), "unsupported_contract"),
        (lambda d: d["products"][0].update(status="published"), "invalid_handoff"),
        (lambda d: d["products"][0].update(scientific_status="accepted"), "status_mismatch"),
        (lambda d: d["products"][0]["artifacts"][0].update(sha256="f" * 64), "dataset_mismatch"),
        (lambda d: d["products"][0]["dataset"].update(version="latest"), "invalid_handoff"),
    ],
    ids=[
        "unknown-contract",
        "publication-claim",
        "unbacked-status",
        "tampered-artifact",
        "dynamic",
    ],
)
def test_consumers_refuse_tampered_or_mismatched_handoffs(tmp_path, smoke_artifacts, change, code):
    with pytest.raises(handoff.HandoffError) as caught:
        handoff.validate_handoff(_mutated(tmp_path, smoke_artifacts, change))
    assert caught.value.code == code


def test_consumers_detect_artifact_bytes_that_drifted(tmp_path, smoke_artifacts):
    _prepare(tmp_path, smoke_artifacts)
    path = smoke_artifacts["GAR_d02km_static_landmask"][0]
    path.chmod(0o644)
    path.write_text("drifted")
    with pytest.raises(handoff.HandoffError) as caught:
        handoff.validate_handoff(
            (tmp_path / "handoff.json").read_bytes(), artifact_root=tmp_path / "artifacts"
        )
    assert caught.value.code == "artifact_mismatch"


def test_preparation_never_overwrites_a_handoff(tmp_path, smoke_artifacts):
    _prepare(tmp_path, smoke_artifacts)
    with pytest.raises(FileExistsError):
        _prepare(tmp_path, smoke_artifacts)


def _assessed(tmp_path, smoke_artifacts):
    first = _prepare(tmp_path, smoke_artifacts, name="first.json")
    landmask = next(p for p in first["products"] if p["product_id"] == "GAR_d02km_static_landmask")
    chain = _assessments_for(landmask["dataset"])
    _prepare(
        tmp_path,
        smoke_artifacts,
        assessments={"GAR_d02km_static_landmask": chain},
        allow_synthetic=True,
    )
    return json.loads((tmp_path / "handoff.json").read_bytes()), chain


def _forge(document):
    forged = json.loads(json.dumps(document))
    product = next(
        p for p in forged["products"] if p["product_id"] == "GAR_d02km_static_landmask"
    )
    product["acceptance"] = [
        {
            **product["acceptance"][-1],
            "id": "forged-assessment",
            "sha256": "0" * 64,
            "supersedes": None,
        }
    ]
    return json.dumps(forged).encode()


def test_acceptance_claims_are_verified_against_the_exact_record_bytes(tmp_path, smoke_artifacts):
    document, chain = _assessed(tmp_path, smoke_artifacts)
    bundles = {"GAR_d02km_static_landmask": chain}
    data = json.dumps(document).encode()
    checked = handoff.validate_handoff(data, assessments=bundles, allow_synthetic=True)
    assert checked["products"] == document["products"]
    with pytest.raises(handoff.HandoffError) as forged:
        handoff.validate_handoff(_forge(document), assessments=bundles, allow_synthetic=True)
    assert forged.value.code == "acceptance_mismatch"


def test_acceptance_claims_without_record_bytes_are_refused_by_default(tmp_path, smoke_artifacts):
    document, _ = _assessed(tmp_path, smoke_artifacts)
    with pytest.raises(handoff.HandoffError) as caught:
        handoff.validate_handoff(json.dumps(document).encode(), allow_synthetic=True)
    assert caught.value.code == "acceptance_unverified"


def test_unverified_reading_still_refuses_an_inconsistent_chain(tmp_path, smoke_artifacts):
    document, _ = _assessed(tmp_path, smoke_artifacts)
    product = next(
        p for p in document["products"] if p["product_id"] == "GAR_d02km_static_landmask"
    )
    product["acceptance"][1]["supersedes"] = None
    with pytest.raises(handoff.HandoffError) as caught:
        handoff.validate_handoff(
            json.dumps(document).encode(), verify_acceptance=False, allow_synthetic=True
        )
    assert caught.value.code == "broken_acceptance_chain"
