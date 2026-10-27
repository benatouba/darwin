"""Publication is invisible until complete and cannot replace existing bytes."""

import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from darwin.publisher.immutable import (
    ArtifactConflict,
    ArtifactIntegrityError,
    identity_path,
    staged_artifact,
    verify_artifact,
)


@pytest.mark.parametrize("directory", [False, True])
def test_visibility_and_conflicting_retry(tmp_path, directory):
    destination = tmp_path / "product"
    with staged_artifact(destination) as staged:
        if directory:
            staged.mkdir()
            (staged / "data").write_bytes(b"original")
        else:
            staged.write_bytes(b"original")
        assert not destination.exists()
    data = destination / "data" if directory else destination
    with pytest.raises(FileExistsError), staged_artifact(destination):
        pytest.fail("existing artifacts must be rejected before writing")
    assert data.read_bytes() == b"original"


def test_failure_cleans_only_own_staging(tmp_path):
    destination = tmp_path / "product"
    other = tmp_path / ".product.staging-other"
    other.mkdir()
    with pytest.raises(RuntimeError), staged_artifact(destination) as staged:
        staged.mkdir()
        (staged / "data").write_bytes(b"partial")
        raise RuntimeError("interrupted writer")
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == [other]
    with staged_artifact(destination) as staged:
        staged.write_bytes(b"retry")
    assert destination.read_bytes() == b"retry"


@pytest.mark.parametrize("directory", [False, True])
def test_concurrent_writers_have_exactly_one_winner(tmp_path, directory):
    destination = tmp_path / "product"
    ready = Barrier(2)

    def writer(value):
        try:
            with staged_artifact(destination) as staged:
                if directory:
                    staged.mkdir()
                    (staged / "data").write_bytes(value)
                else:
                    staged.write_bytes(value)
                ready.wait(timeout=10)
            return value
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(writer, [b"one", b"two"]))
    winners = [value for value in results if value is not None]
    assert len(winners) == 1
    assert (destination / "data" if directory else destination).read_bytes() == winners[0]
    assert list(tmp_path.iterdir()) == [destination]


def test_existing_dangling_symlink_is_not_replaced(tmp_path):
    destination = tmp_path / "product"
    destination.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError), staged_artifact(destination):
        pytest.fail("symlinks must not be followed or replaced")
    assert destination.is_symlink()


SOURCE = {"id": "GAR_d02km_m_2d_prcp_2020.nc", "sha256": "a" * 64}
PROVENANCE = {"builder": "tests.writer/1", "sources": [SOURCE], "parameters": {"step": 1}}


def _write_tree(staged, payload=b"original"):
    staged.mkdir()
    (staged / "zarr.json").write_bytes(b"{}")
    (staged / "c").mkdir()
    (staged / "c" / "0").write_bytes(payload)


def test_published_artifact_is_bound_to_exact_bytes_and_provenance(tmp_path):
    destination = tmp_path / "product.zarr"
    with staged_artifact(destination, provenance=PROVENANCE) as staged:
        _write_tree(staged)
    record = verify_artifact(destination)
    assert record["contract"] == "darwin.artifact-identity/1"
    assert record["artifact"]["kind"] == "directory"
    assert [entry["path"] for entry in record["artifact"]["files"]] == ["c/0", "zarr.json"]
    # Independent worked value: sha256(b"original").
    assert record["artifact"]["files"][0]["sha256"] == (
        "0682c5f2076f099c34cfdd15a9e063849ed437a49677e6fcc5b4198c76575be5"
    )
    assert record["provenance"]["sources"] == [SOURCE]
    assert record["provenance"]["artifact_class"] == "derived"
    assert record["scientific_status"] == "unassessed"
    assert identity_path(destination).name == "product.zarr.identity.json"

    (destination / "c" / "0").write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact(destination)


def test_unbound_artifacts_do_not_verify(tmp_path):
    destination = tmp_path / "legacy.zarr"
    with staged_artifact(destination) as staged:
        _write_tree(staged)
    with pytest.raises(ArtifactIntegrityError, match="identity"):
        verify_artifact(destination)


@pytest.mark.parametrize(
    "provenance",
    [
        {"builder": "x", "sources": []},
        {"builder": "x", "sources": [{"id": "s", "sha256": "not-a-digest"}]},
        {"sources": [SOURCE]},
    ],
)
def test_provenance_requires_builder_and_pinned_sources(tmp_path, provenance):
    destination = tmp_path / "product"
    with pytest.raises(ValueError), staged_artifact(destination, provenance=provenance):
        pytest.fail("invalid provenance must be refused before writing")
    assert list(tmp_path.iterdir()) == []


def test_derived_artifact_cannot_be_written_over_or_inside_a_source(tmp_path):
    source = tmp_path / "original.nc"
    source.write_bytes(b"simulation output")
    provenance = {"builder": "x", "sources": [{"id": "o", "path": str(source), "sha256": "b" * 64}]}
    for destination in (source, tmp_path / "original.nc" / "inner"):
        with pytest.raises((ValueError, FileExistsError)), staged_artifact(
            destination, provenance=provenance
        ):
            pytest.fail("derived products must stay distinct from original outputs")
    assert source.read_bytes() == b"simulation output"


def test_identical_retry_is_idempotent_only_when_requested(tmp_path):
    destination = tmp_path / "product.zarr"
    with staged_artifact(destination, provenance=PROVENANCE) as staged:
        _write_tree(staged)
    record = identity_path(destination).read_bytes()
    with pytest.raises(ArtifactConflict), staged_artifact(destination, provenance=PROVENANCE):
        pytest.fail("the default policy refuses before writing")
    with staged_artifact(destination, provenance=PROVENANCE, retry="verify-identical") as staged:
        _write_tree(staged)
    assert identity_path(destination).read_bytes() == record
    assert verify_artifact(destination)["provenance"]["parameters"] == {"step": 1}
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "product.zarr",
        "product.zarr.identity.json",
    ]


@pytest.mark.parametrize(
    ("payload", "provenance"),
    [
        (b"changed", PROVENANCE),
        (b"original", {**PROVENANCE, "parameters": {"step": 2}}),
    ],
)
def test_conflicting_retry_never_changes_existing_bytes(tmp_path, payload, provenance):
    destination = tmp_path / "product.zarr"
    with staged_artifact(destination, provenance=PROVENANCE) as staged:
        _write_tree(staged)
    record = identity_path(destination).read_bytes()
    with pytest.raises(ArtifactConflict), staged_artifact(
        destination, provenance=provenance, retry="verify-identical"
    ) as staged:
        _write_tree(staged, payload)
    assert (destination / "c" / "0").read_bytes() == b"original"
    assert identity_path(destination).read_bytes() == record
    verify_artifact(destination)


def test_an_unbound_artifact_is_never_bound_afterwards(tmp_path):
    # Equal bytes alone never prove provenance: whoever wrote them made no claim.
    destination = tmp_path / "product.zarr"
    with staged_artifact(destination) as staged:
        _write_tree(staged)
    with pytest.raises(ArtifactConflict), staged_artifact(
        destination, provenance=PROVENANCE, retry="verify-identical"
    ) as staged:
        _write_tree(staged)
    assert not identity_path(destination).exists()


def test_only_the_same_provenance_completes_an_interrupted_publication(tmp_path):
    destination = tmp_path / "product.zarr"
    with staged_artifact(destination, provenance=PROVENANCE) as staged:
        _write_tree(staged)
    record = identity_path(destination).read_bytes()
    shutil.rmtree(destination)  # the state a crash before the artifact rename leaves
    with pytest.raises(ArtifactConflict), staged_artifact(
        destination, provenance={**PROVENANCE, "builder": "other"}, retry="verify-identical"
    ) as staged:
        _write_tree(staged)
    assert not destination.exists()
    with staged_artifact(destination, provenance=PROVENANCE, retry="verify-identical") as staged:
        _write_tree(staged)
    assert identity_path(destination).read_bytes() == record
    assert verify_artifact(destination)["artifact"]["kind"] == "directory"


def test_concurrent_identical_writers_both_succeed_with_one_copy(tmp_path):
    destination = tmp_path / "product.zarr"
    ready = Barrier(2)

    def writer(_):
        with staged_artifact(
            destination, provenance=PROVENANCE, retry="verify-identical"
        ) as staged:
            _write_tree(staged)
            ready.wait(timeout=10)
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(writer, range(2))) == [True, True]
    verify_artifact(destination)
    assert len(list(tmp_path.iterdir())) == 2


def test_historical_versions_remain_independently_verifiable(tmp_path):
    for version, payload in (("v1", b"original"), ("v2", b"corrected")):
        with staged_artifact(
            tmp_path / version / "product.zarr",
            provenance={**PROVENANCE, "version": version},
        ) as staged:
            _write_tree(staged, payload)
    first = verify_artifact(tmp_path / "v1" / "product.zarr")
    second = verify_artifact(tmp_path / "v2" / "product.zarr")
    assert first["artifact"]["sha256"] != second["artifact"]["sha256"]
    assert (tmp_path / "v1" / "product.zarr" / "c" / "0").read_bytes() == b"original"


def test_source_overlap_is_detected_through_symlinked_parents(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    source = real / "original.zarr"
    source.mkdir()
    (tmp_path / "alias").symlink_to(real)
    provenance = {"builder": "x", "sources": [{"id": "o", "path": str(source), "sha256": "b" * 64}]}
    with pytest.raises(ValueError, match="overlaps"), staged_artifact(
        tmp_path / "alias" / "original.zarr" / "derived", provenance=provenance
    ):
        pytest.fail("a derived product must not be written inside its source")
    assert list(source.iterdir()) == []
