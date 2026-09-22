"""Smoke analysis stores (T4): real bundled Zarrs built from pinned NetCDFs.

The dashboard serves Explore reads from these seed stores, so they must
carry verbatim source values: shapes, native coordinates, the UTC time
axis and the static land mask, all asserted against independently
measured literals (ncdump-equivalent, not the builder's own output).
"""

from pathlib import Path

REPO_ROOT = Path("/home/ben/projects/darwin")


def test_slice_store_matches_source_bytes(tmp_path):
    """The slice store carries the real 12-month prcp field on native coords."""
    from darwin.publisher.analysis_stores import build_smoke_analysis_stores

    out = build_smoke_analysis_stores(REPO_ROOT, tmp_path)
    import zarr

    group = zarr.open_group(str(out["slice"]), mode="r")
    array = group["prcp"]
    assert array.shape == (12, 230, 245)
    assert float(group["x"][0]) == -244000.0
    assert float(group["x"][-1]) == 244000.0
    assert float(group["y"][0]) == -274001.375
    assert group.attrs["time_units"] == "days since 2020-01-01 00:00:00"
    assert [int(v) for v in group["time"][:]] == [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    assert abs(float(array[:].max()) - 1.6485823392868042) < 1e-9
    assert abs(float(array[:].mean()) - 0.047143787145614624) < 1e-9


def test_mask_store_matches_static_landmask(tmp_path):
    """The mask store carries the real binary land mask on the same grid."""
    from darwin.publisher.analysis_stores import build_smoke_analysis_stores

    out = build_smoke_analysis_stores(REPO_ROOT, tmp_path)
    import zarr

    group = zarr.open_group(str(out["mask"]), mode="r")
    mask = group["landmask"][:]
    assert mask.shape == (230, 245)
    assert set(int(v) for v in mask.flat) <= {0, 1}
    assert abs(float(mask.mean()) - 0.03472936898469925) < 1e-9
    assert float(group["x"][0]) == -244000.0
