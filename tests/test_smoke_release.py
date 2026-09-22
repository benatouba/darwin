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
    normal = by_id["GAR_d02km_prcp_sum_1991-2020_01"]
    assert normal["normal_period"] == "1991-2020"
    assert normal["method"], "normal layer must record its computation method"
    aws = by_id["AWS-PCerroCrocker_2020-01"]
    assert aws["station_id"] == "11_AWS-PCerroCrocker"


def test_build_writes_handoff_file(tmp_path):
    out_path = tmp_path / "handoff.json"
    handoff = build_smoke_release(MANIFEST_PATH, out_path)
    assert out_path.exists()
    assert json.loads(out_path.read_text()) == handoff
