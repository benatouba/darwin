"""Regenerate darwin/contracts/fixtures/product-handoff-v2 (synthetic only).

Usage: ``python tests/build_handoff_fixtures.py <empty output directory>``,
then review the diff and replace the committed fixture directory. The output
is synthetic interoperability data: no scientific acceptance, no reviewed
requirements and no real reviewer or policy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from darwin import acceptance, handoff
from darwin.publisher.immutable import staged_artifact
from darwin.publisher.smoke import build_smoke_release

SETUPS = Path(__file__).parent / "fixtures" / "setups-contracts" / "fixtures" / "acceptance-assessment-v1"
STAMP = "2026-10-27T00:00:00Z"
AUTHORIZATION_ID = ("synthetic:reviewer-authorization", "1")
CREATOR = "synthetic:darwin-fixture-builder"
SOURCE_SHA = "c9e73d7c056ce843a739e08d7ff04e357b7eef67536467c5eb1ebf17f245b0d0"
MANIFEST = f"""\
# Synthetic darwin.product-handoff/2 fixture manifest: not a reviewed release.
name: synthetic-handoff-v2
release: synthetic1
sources:
  - path: synthetic:source
    sha256: {SOURCE_SHA}
products:
  - product_id: "synthetic:prcp-slice"
    kind: slice
    citation: "Synthetic fixture; not a scientific product"
    variable: prcp
    domain: d02km
    format: zarr
    units: mm h-1
    palette: viridis
    transform: linear
    color_range: [0.0, 2.0]
    uris:
      analysis: "synthetic:prcp-slice.zarr"
    aggregations:
      - method: sum
        interval: daily
        missing_data: require-complete
  - product_id: "synthetic:landmask"
    kind: reference
    citation: "Synthetic fixture; not a scientific product"
    variable: landmask
    domain: d02km
    format: zarr
    units: "1"
    palette: greys
    transform: linear
    color_range: [0.0, 1.0]
    uris:
      analysis: "synthetic:landmask.zarr"
"""
NOTICE = (
    "Synthetic interoperability fixtures for darwin.product-handoff/2 consumers. "
    "Not scientific acceptance, not reviewed requirements, not a real reviewer or "
    "policy, and not a publication: consumers refuse them outside explicit test mode."
)


def _artifact(root: Path, name: str, payload: dict) -> Path:
    path = root / "artifacts" / name
    provenance = {
        "builder": "tests.build_handoff_fixtures",
        "sources": [{"id": "synthetic:source", "sha256": SOURCE_SHA}],
    }
    with staged_artifact(path, provenance=provenance) as staged:
        staged.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _bundle(root: Path, name: str, record: str, prior: str | None, evidence: dict) -> None:
    """A load_assessment_bundle file citing exact bytes by relative path."""
    bundle = {
        "assessment": record,
        "requirements": "requirements.json",
        "evidence": [
            {"id": key[0], "version": key[1], "path": f"evidence/{acceptance.sha256_bytes(data)}.json"}
            for key, data in evidence.items()
        ],
        "authorizations": [{"id": AUTHORIZATION_ID[0], "version": AUTHORIZATION_ID[1],
                            "path": "authorizations/reviewer.json"}],
        "prior": prior,
    }
    _write(root / "acceptance" / name, acceptance.canonical_bytes(bundle))


def _assessments(root: Path, dataset: dict) -> list[dict]:
    requirements = (SETUPS / "requirements.json").read_bytes()
    _write(root / "acceptance/requirements.json", requirements)
    authorization = (SETUPS / "authorizations/reviewer.json").read_bytes()
    _write(root / "acceptance/authorizations/reviewer.json", authorization)
    reviewer = json.loads(requirements)["acceptance"]["authorized_authorities"][0]
    assert (reviewer["authorization"]["id"], reviewer["authorization"]["version"]) == AUTHORIZATION_ID
    failing = acceptance.run_checks(
        dataset,
        requirements,
        inputs={"synthetic:outcomes": {"synthetic:criterion-a": "fail"}},
        allow_synthetic=True,
    )[:1]
    passing = acceptance.run_checks(dataset, requirements, inputs={}, allow_synthetic=True)
    evidence = {}
    for receipt in failing + passing:
        data = acceptance.canonical_bytes(receipt)
        evidence[(receipt["id"], acceptance.RECEIPT_VERSION)] = data
        _write(root / "acceptance/evidence" / f"{acceptance.sha256_bytes(data)}.json", data)
    rejected = acceptance.record_assessment(
        assessment_id="synthetic:prcp-slice-assessment-1",
        dataset=dataset,
        requirements=requirements,
        status="rejected",
        evidence=[acceptance.canonical_bytes(r) for r in failing],
        authority=reviewer,
        authorization=authorization,
        allow_synthetic=True,
    )
    accepted = acceptance.record_assessment(
        assessment_id="synthetic:prcp-slice-assessment-2",
        dataset=dataset,
        requirements=requirements,
        status="accepted",
        evidence=[acceptance.canonical_bytes(r) for r in passing],
        authority=reviewer,
        authorization=authorization,
        prior=rejected,
        prior_evidence=evidence,
        allow_synthetic=True,
    )
    _write(root / "acceptance/assessment-1-rejected.json", rejected)
    _write(root / "acceptance/assessment-2-accepted.json", accepted)
    _bundle(root, "bundle-1-rejected.json", "assessment-1-rejected.json", None,
            dict(list(evidence.items())[:1]))
    _bundle(root, "bundle-2-accepted.json", "assessment-2-accepted.json",
            "assessment-1-rejected.json", evidence)
    base = {"requirements": requirements, "evidence": evidence,
            "authorizations": {AUTHORIZATION_ID: authorization}}
    return [{**base, "assessment": rejected}, {**base, "assessment": accepted, "prior": rejected}]


def _negative(root: Path, name: str, document: dict, change) -> None:
    mutated = json.loads(json.dumps(document))
    change(mutated)
    _write(root / "negative" / name, acceptance.canonical_bytes(mutated))


def build(root: Path) -> None:
    """Write every fixture file below ``root`` (must not exist yet)."""
    root.mkdir(parents=True)
    manifest = root / "manifest.yaml"
    manifest.write_text(MANIFEST)
    artifacts = {
        "synthetic:prcp-slice": [_artifact(root, "prcp-slice.json", {"prcp": [0.0, 1.5]})],
        "synthetic:landmask": [_artifact(root, "landmask.json", {"landmask": [0, 1]})],
    }
    build_smoke_release(manifest, root / "legacy-v1.json", creator=CREATOR, created_at=STAMP)
    (root / "legacy-v1.json.identity.json").unlink()
    run = {"id": "synthetic:execution-report", "version": "1", "sha256": SOURCE_SHA}
    unassessed = handoff.prepare_handoff(
        manifest,
        root / "handoff-unassessed.json",
        artifacts=artifacts,
        execution={"synthetic:prcp-slice": {"completeness": "complete", "evidence": run}},
        creator=CREATOR,
        created_at=STAMP,
        allow_synthetic=True,
    )
    dataset = unassessed["products"][0]["dataset"]
    handoff.prepare_handoff(
        manifest,
        root / "handoff-assessed.json",
        artifacts=artifacts,
        execution={"synthetic:prcp-slice": {"completeness": "partial", "evidence": run}},
        assessments={"synthetic:prcp-slice": _assessments(root, dataset)},
        creator=CREATOR,
        created_at=STAMP,
        allow_synthetic=True,
    )
    for name in ("handoff-unassessed.json", "handoff-assessed.json"):
        (root / f"{name}.identity.json").unlink()
    _negative(root, "unknown-contract.json", unassessed,
              lambda d: d.update(contract="darwin.product-handoff/99"))
    _negative(root, "publication-claim.json", unassessed,
              lambda d: d["products"][0].update(status="published"))
    _negative(root, "unbacked-acceptance.json", unassessed,
              lambda d: d["products"][0].update(scientific_status="accepted"))
    _negative(root, "tampered-artifact.json", unassessed,
              lambda d: d["products"][0]["artifacts"][0].update(sha256="f" * 64))
    _negative(root, "dynamic-version.json", unassessed,
              lambda d: d["products"][0]["dataset"].update(version="latest"))
    assessed = json.loads((root / "handoff-assessed.json").read_bytes())

    def forge(document):
        product = next(p for p in document["products"] if p["acceptance"])
        product["acceptance"] = [{**product["acceptance"][-1], "id": "synthetic:forged-assessment",
                                  "sha256": "0" * 64, "supersedes": None}]

    _negative(root, "forged-reference.json", assessed, forge)
    bundle = {"synthetic:prcp-slice": ["acceptance/bundle-1-rejected.json",
                                       "acceptance/bundle-2-accepted.json"]}
    both = {"synthetic:prcp-slice": None, "synthetic:landmask": None}
    ok = lambda status, completeness: {  # noqa: E731
        "scientific_status": {**dict.fromkeys(both, "unassessed"), **status},
        "execution_completeness": {**both, **completeness},
    }
    cases = [
        {"name": "unassessed", "handoff": "handoff-unassessed.json", "artifact_root": "artifacts",
         "test_mode": True,
         "expect": ok({}, {"synthetic:prcp-slice": "complete"})},
        {"name": "rejected-then-accepted", "handoff": "handoff-assessed.json",
         "artifact_root": "artifacts", "acceptance": bundle, "test_mode": True,
         "expect": ok({"synthetic:prcp-slice": "accepted"}, {"synthetic:prcp-slice": "partial"})},
        {"name": "acceptance-without-record-bytes", "handoff": "handoff-assessed.json",
         "test_mode": True, "expect": {"error": "acceptance_unverified"}},
        {"name": "forged-reference", "handoff": "negative/forged-reference.json",
         "acceptance": bundle, "test_mode": True, "expect": {"error": "acceptance_mismatch"}},
        {"name": "synthetic-outside-test-mode", "handoff": "handoff-assessed.json",
         "test_mode": False, "expect": {"error": "synthetic_not_trusted"}},
        *(
            {"name": name.removesuffix(".json"), "handoff": f"negative/{name}", "test_mode": True,
             "expect": {"error": code}}
            for name, code in (
                ("unknown-contract.json", "unsupported_contract"),
                ("publication-claim.json", "invalid_handoff"),
                ("unbacked-acceptance.json", "status_mismatch"),
                ("tampered-artifact.json", "dataset_mismatch"),
                ("dynamic-version.json", "invalid_handoff"),
            )
        ),
    ]
    _write(root / "cases.json", acceptance.canonical_bytes(
        {"fixture_set": "darwin.product-handoff-fixtures/2", "synthetic": True,
         "notice": NOTICE, "cases": cases}
    ))


if __name__ == "__main__":
    build(Path(sys.argv[1]))
