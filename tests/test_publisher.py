"""D1 (darwin#3): publisher manifest-in / candidate-out seam.

RED tests: reviewed version-controlled YAML manifest run creates an
immutable candidate with recorded manifest revision; lifecycle
candidate/published/superseded/failed with current-version pointer;
source paths/checksums come from the background manifest, never from
scanning source data on run.
"""

import dataclasses
from pathlib import Path

import pytest

from darwin.publisher import lifecycle as _lifecycle  # noqa: F401
from darwin.publisher.candidate import run_manifest
from darwin.publisher.catalogue import Catalogue
from darwin.publisher.manifest import AggregationSpec, load_manifest

MANIFEST_TEXT = """\
name: d02km-normals-1991-2020
release: v1
sources:
  - path: /archive/GAR/v1/d02km/h/2d/GAR_d02km_h_2d_t2_1991.nc
    sha256: abc123
  - path: /archive/GAR/v1/d02km/m/2d/GAR_d02km_m_2d_prcp_1991.nc
    sha256: def456
products:
  - product_id: GAR_d02km_t2_max_1991-2020
    kind: normal
"""


@pytest.fixture()
def manifest_path(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text(MANIFEST_TEXT)
    return path


def test_manifest_load_records_content_revision(manifest_path):
    manifest = load_manifest(manifest_path)
    assert manifest.name == "d02km-normals-1991-2020"
    assert len(manifest.revision) == 64  # sha256 of manifest bytes
    assert len(manifest.sources) == 2
    assert manifest.sources[0].sha256 == "abc123"


def test_run_creates_immutable_candidate_with_manifest_revision(manifest_path):
    manifest = load_manifest(manifest_path)
    catalogue = Catalogue()
    candidate = run_manifest(manifest, catalogue)
    assert candidate.status == "candidate"
    assert candidate.manifest_revision == manifest.revision
    assert candidate.product_id == "GAR_d02km_t2_max_1991-2020"
    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.status = "published"  # type: ignore[misc]


def test_run_never_scans_source_data(manifest_path):
    """Source paths in the manifest need not exist: checksums come from
    the background manifest, the publisher never scans the archive."""
    manifest = load_manifest(manifest_path)
    assert not Path(manifest.sources[0].path).exists()
    candidate = run_manifest(manifest, Catalogue())
    assert candidate.status == "candidate"


def test_publish_moves_current_pointer_and_supersedes(manifest_path):
    manifest = load_manifest(manifest_path)
    catalogue = Catalogue()
    first = run_manifest(manifest, catalogue)
    catalogue.publish(first.candidate_id)
    assert catalogue.current(first.product_id).candidate_id == first.candidate_id

    second = run_manifest(manifest, catalogue)
    catalogue.publish(second.candidate_id)
    assert catalogue.current(first.product_id).candidate_id == second.candidate_id
    # Old published version stays addressable, marked superseded.
    assert catalogue.get(first.product_id, first.version).status == "superseded"
    assert catalogue.get(first.product_id, second.version).status == "published"


def test_failed_candidate_never_becomes_current(manifest_path):
    manifest = load_manifest(manifest_path)
    catalogue = Catalogue()
    candidate = run_manifest(manifest, catalogue)
    catalogue.fail(candidate.candidate_id, "checksum mismatch")
    failed = catalogue.get(candidate.product_id, candidate.version)
    assert failed.status == "failed"
    with pytest.raises(LookupError):
        catalogue.current(candidate.product_id)


def test_invalid_transitions_raise(manifest_path):
    manifest = load_manifest(manifest_path)
    catalogue = Catalogue()
    candidate = run_manifest(manifest, catalogue)
    catalogue.fail(candidate.candidate_id, "bad input")
    with pytest.raises(ValueError):
        catalogue.publish(candidate.candidate_id)
    with pytest.raises(ValueError):
        catalogue.fail(candidate.candidate_id, "again")


AGGREGATION_MANIFEST_TEXT = """\
name: d02km-aggregation-contract
release: v1
products:
  - product_id: GAR_d02km_m_2d_prcp_2020-01
    kind: slice
    aggregations:
      - method: sum
        interval: daily
        missing_data: require-complete
      - method: mean
        interval: monthly
        missing_data: exclude-missing
  - product_id: GAR_d02km_static_landmask
    kind: reference
"""


def test_manifest_parses_per_product_aggregations(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text(AGGREGATION_MANIFEST_TEXT)
    manifest = load_manifest(path)
    by_id = {spec.product_id: spec for spec in manifest.products}
    assert by_id["GAR_d02km_m_2d_prcp_2020-01"].aggregations == (
        AggregationSpec(method="sum", interval="daily", missing_data="require-complete"),
        AggregationSpec(method="mean", interval="monthly", missing_data="exclude-missing"),
    )
    assert by_id["GAR_d02km_static_landmask"].aggregations == ()


@pytest.mark.parametrize(
    "aggregation_yaml",
    [
        "      - method: median\n        interval: daily\n        missing_data: require-complete",
        "      - method: sum\n        interval: hourly\n        missing_data: require-complete",
        "      - method: sum\n        interval: daily\n        missing_data: ignore-gaps",
        "      - method: sum\n        interval: daily",
        "      - {method: sum, interval: daily, missing_data: require-complete, extra: 1}",
        "      - not-a-mapping",
    ],
)
def test_manifest_rejects_invalid_aggregation_declarations(tmp_path, aggregation_yaml):
    path = tmp_path / "manifest.yaml"
    path.write_text(
        "name: broken-aggregations\n"
        "release: v1\n"
        "products:\n"
        "  - product_id: GAR_d02km_m_2d_prcp_2020-01\n"
        "    kind: slice\n"
        "    aggregations:\n" + aggregation_yaml + "\n"
    )
    with pytest.raises(ValueError, match="GAR_d02km_m_2d_prcp_2020-01"):
        load_manifest(path)


@pytest.mark.parametrize("declaration", ["5", "{method: sum}", ""])
def test_manifest_rejects_non_list_aggregations(tmp_path, declaration):
    path = tmp_path / "manifest.yaml"
    path.write_text(
        "name: broken-aggregations\n"
        "release: v1\n"
        "products:\n"
        "  - product_id: GAR_d02km_m_2d_prcp_2020-01\n"
        "    kind: slice\n"
        f"    aggregations: {declaration}\n"
    )
    with pytest.raises(ValueError, match="GAR_d02km_m_2d_prcp_2020-01"):
        load_manifest(path)
