"""S10 #31: canonical store payload hash (tamper-evidence for handoffs)."""

import numpy as np
import zarr

from darwin.publisher.multiyear_store import hash_store


def _tiny_store(path):
    group = zarr.open_group(str(path), mode="w")
    group.create_array("m", data=np.array([[1.0, 2.0], [3.0, 4.0]]))
    group.create_array("a", data=np.array([1, 2, 3], dtype=np.int64))
    return path


def test_hash_is_deterministic(tmp_path):
    store = _tiny_store(tmp_path / "s.zarr")
    assert hash_store(store) == hash_store(store)


def test_hash_detects_tampered_value(tmp_path):
    store = _tiny_store(tmp_path / "s.zarr")
    before = hash_store(store)
    arr = zarr.open_group(str(store), mode="a")["m"]
    arr[0, 0] = 99.0
    assert hash_store(store) != before


def test_hash_matches_canonical_golden(tmp_path):
    store = _tiny_store(tmp_path / "s.zarr")
    assert hash_store(store) == "79e96b63803d71b53e3f0ea34c0c76c06b7583a6bd5e53b9b120a6cc12b51638"
