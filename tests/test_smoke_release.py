"""Smoke T1 (darwin#23): reviewed manifest in, published smoke release out.

RED tests for the Jan-2020 precipitation smoke release: the
version-controlled manifest pins every source by sha256, the builder
registers and publishes one candidate per requested product, and every
released product carries full provenance (manifest revision, creator,
creation time, citation, plus normal period/method where applicable).
The handoff JSON is what the dashboard catalogue ticket ingests.
"""

import hashlib
import json
from pathlib import Path

from darwin.publisher.manifest import load_manifest
from darwin.publisher.smoke import build_smoke_release

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "darwin" / "publisher" / "manifests" / "smoke-jan2020-prcp.yaml"

EXPECTED_PRODUCTS = (
    "GAR_d02km_static_landmask",
    "GAR_d02km_m_2d_prcp_2020-01",
    "GAR_d02km_prcp_sum_1991-2020_01",
    "AWS-PCerroCrocker_2020-01",
    "CMP_prcp_CerroCrocker_2020-01",
)


def test_smoke_manifest_is_pinned_and_reviewed():
    manifest = load_manifest(MANIFEST_PATH)
    assert manifest.release == "smoke1"
    # Revision is the sha256 of the exact reviewed bytes (recomputed here
    # from an independent read, not via the loader).
    assert manifest.revision == hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert [spec.product_id for spec in manifest.products] == list(EXPECTED_PRODUCTS)
    assert manifest.sources, "smoke manifest must pin its source data"
    for entry in manifest.sources:
        assert entry.sha256, f"source {entry.path} has no pinned checksum"
        # Smoke sources are repo-local so review can re-verify the bytes.
        actual = (REPO_ROOT / entry.path).read_bytes()
        assert hashlib.sha256(actual).hexdigest() == entry.sha256


def test_build_publishes_every_product_with_provenance(tmp_path):
    handoff = build_smoke_release(
        MANIFEST_PATH,
        tmp_path / "handoff.json",
        creator="reviewer: smoke",
        created_at="2026-09-22T00:00:00Z",
    )
    manifest = load_manifest(MANIFEST_PATH)
    assert handoff["manifest_revision"] == manifest.revision
    assert handoff["creator"] == "reviewer: smoke"
    assert handoff["created_at"] == "2026-09-22T00:00:00Z"
    by_id = {item["product_id"]: item for item in handoff["products"]}
    assert sorted(by_id) == sorted(EXPECTED_PRODUCTS)
    for product_id, item in by_id.items():
        assert item["status"] == "published", product_id
        assert item["manifest_revision"] == manifest.revision, product_id
        assert item["creator"] == "reviewer: smoke", product_id
        assert item["created_at"] == "2026-09-22T00:00:00Z", product_id
        assert item["citation"], f"{product_id} has no citation"
    normal_product = by_id["GAR_d02km_prcp_sum_1991-2020_01"]
    assert normal_product["normal_period"] == "1991-2020"
    assert normal_product["method"], "Normal product must record its computation method"
    aws = by_id["AWS-PCerroCrocker_2020-01"]
    assert aws["station_id"] == "11_AWS-PCerroCrocker"


def test_comparison_spec_carries_machine_readable_pairing(tmp_path):
    handoff = build_smoke_release(MANIFEST_PATH, tmp_path / "handoff.json")
    by_id = {item["product_id"]: item for item in handoff["products"]}
    comparison = by_id["CMP_prcp_CerroCrocker_2020-01"]
    assert comparison["model_product"] == "GAR_d02km_m_2d_prcp_2020-01"
    assert comparison["model_variable"] == "prcp"
    assert comparison["observation_product"] == "AWS-PCerroCrocker_2020-01"
    assert comparison["observation_field"] == "PCP_tot_bucket"
    # Both ends of the pairing resolve to products in the same release.
    assert comparison["model_product"] in by_id
    assert comparison["observation_product"] in by_id


def test_manifest_revision_stable_across_rebuilds(tmp_path):
    # created_at stamps build time, so rebuilds diverge in stamp only:
    # the revision still binds the exact reviewed manifest bytes.
    first = build_smoke_release(
        MANIFEST_PATH, tmp_path / "a.json", created_at="2026-09-22T00:00:00Z"
    )
    second = build_smoke_release(
        MANIFEST_PATH, tmp_path / "b.json", created_at="2026-09-23T00:00:00Z"
    )
    assert first["manifest_revision"] == second["manifest_revision"]
    assert first["created_at"] != second["created_at"]


def test_smoke_products_carry_publisher_declared_presentation(tmp_path):
    handoff = build_smoke_release(MANIFEST_PATH, tmp_path / "handoff.json")
    by_id = {item["product_id"]: item for item in handoff["products"]}
    for product_id, item in by_id.items():
        assert item["palette"], f"{product_id} declares no palette"
        assert item["transform"], f"{product_id} declares no transform"
        lo, hi = item["color_range"]
        assert lo < hi, f"{product_id} has no fixed colour range"
        assert item["uris"].get("analysis"), f"{product_id} declares no analysis store"
    # The January slice and its Climate normal share one fixed prcp range
    # across the slice's temporal coverage.
    assert (
        by_id["GAR_d02km_m_2d_prcp_2020-01"]["color_range"]
        == by_id["GAR_d02km_prcp_sum_1991-2020_01"]["color_range"]
    )


def test_smoke_products_carry_catalogue_identity(tmp_path):
    handoff = build_smoke_release(MANIFEST_PATH, tmp_path / "handoff.json")
    by_id = {item["product_id"]: item for item in handoff["products"]}
    for product_id, item in by_id.items():
        assert item["variable"], f"{product_id} declares no variable"
        assert item["domain"], f"{product_id} declares no domain"
        assert item["format"], f"{product_id} declares no format"
    assert by_id["GAR_d02km_m_2d_prcp_2020-01"]["variable"] == "prcp"
    assert by_id["GAR_d02km_prcp_sum_1991-2020_01"]["variable"] == "prcp_sum"


def test_build_writes_handoff_file(tmp_path):
    out_path = tmp_path / "handoff.json"
    handoff = build_smoke_release(MANIFEST_PATH, out_path)
    assert out_path.exists()
    assert json.loads(out_path.read_text()) == handoff
