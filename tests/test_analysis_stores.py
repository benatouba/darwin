"""Smoke analysis stores (T4): real bundled Zarrs built from pinned NetCDFs.

The dashboard serves Explore reads from these seed stores, so they must
carry verbatim source values: shapes, native coordinates, the UTC time
axis and the static land mask, all asserted against independently
measured literals (ncdump-equivalent, not the builder's own output).
"""

def test_slice_store_matches_source_bytes(tmp_path, analysis_sources):
    """The slice store carries the real 12-month prcp field on native coords."""
    from darwin.publisher.analysis_stores import build_smoke_analysis_stores

    out = build_smoke_analysis_stores(analysis_sources, tmp_path)
    import zarr

    group = zarr.open_group(str(out["slice"]), mode="r")
    array = group["prcp"]
    assert array.shape == (12, 230, 245)
    assert float(group["x"][0]) == -244000.0
    assert float(group["x"][-1]) == 244000.0
    assert float(group["y"][0]) == -274001.375
    assert group.attrs["time_units"] == "hours since 1970-01-01 00:00:00"
    assert [int(v) for v in group["time"][:]] == [
        438288,
        439032,
        439728,
        440472,
        441192,
        441936,
        442656,
        443400,
        444144,
        444864,
        445608,
        446328,
    ]
    assert abs(float(array[:].max()) - 1.6485823392868042) < 1e-9
    assert abs(float(array[:].mean()) - 0.047143787145614624) < 1e-9
    assert group.attrs["crs"] == (
        "+k_0=1.0 +units=m +lat_0=2.0 +lon_0=-90.31006622 +x_0=0.0 +y_0=0.0"
        " +ellps=WGS84 +datum=WGS84 +name=WRFMercator +proj=merc +no_defs"
    )


def test_mask_store_matches_static_landmask(tmp_path, analysis_sources):
    """The mask store carries the real binary land mask on the same grid."""
    from darwin.publisher.analysis_stores import build_smoke_analysis_stores

    out = build_smoke_analysis_stores(analysis_sources, tmp_path)
    import zarr

    group = zarr.open_group(str(out["mask"]), mode="r")
    mask = group["landmask"][:]
    assert mask.shape == (230, 245)
    assert set(int(v) for v in mask.flat) <= {0, 1}
    assert abs(float(mask.mean()) - 0.03472936898469925) < 1e-9
    assert float(group["x"][0]) == -244000.0
    assert group.attrs["crs"] == (
        "+k_0=1.0 +units=m +lat_0=2.0 +lon_0=-90.31006622 +x_0=0.0 +y_0=0.0"
        " +ellps=WGS84 +datum=WGS84 +name=WRFMercator +proj=merc +no_defs"
    )
