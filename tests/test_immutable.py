"""Publication is invisible until complete and cannot replace existing bytes."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from darwin.publisher.immutable import staged_artifact


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
