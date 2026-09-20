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
from darwin.publisher.manifest import load_manifest

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
