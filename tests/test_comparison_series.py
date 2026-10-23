"""Comparison series stores (T6): published GAR/AWS January series.

The dashboard serves the real prcp Comparison result from these seed
stores, so they must carry true values: the GAR January-2020 monthly
rate sampled nearest the Station location record, uniformly
disaggregated to constant hourly/daily depth series, and the AWS
January hourly/daily depths with real completeness exclusions, all
asserted against independently measured literals (CSV + NetCDF probes,
not the builder's own stores).
"""

def test_gar_sampled_store_carries_jan2020_nearest_series(tmp_path, comparison_sources):
    """GAR store: 12 monthly nearest rates; hourly/daily depths (rate x 1h/24h)."""
    from darwin.publisher.comparison_series import build_comparison_series

    out = build_comparison_series(comparison_sources, tmp_path)
    import zarr

    group = zarr.open_group(str(out["gar_sampled"]), mode="r")
    monthly = group["monthly"][:]
    assert monthly.shape == (12,)
    # float32 source precision: the probe literal is the float64 rounding.
    assert abs(float(monthly[0]) - 0.15625376) < 1e-7
    assert group["hourly"].shape == (744,)
    assert group["daily"].shape == (31,)
    assert bool((group["hourly"][:] == float(monthly[0])).all())
    assert bool((group["daily"][:] == float(monthly[0]) * 24.0).all())
    assert group["monthly"].attrs["units"] == "mm h-1"
    # Both disaggregated series are depths so they pair unit-clean with
    # the AWS tip-sum depths (hourly depth = rate x 1h, numerically the rate).
    assert group["hourly"].attrs["units"] == "mm"
    assert group["daily"].attrs["units"] == "mm"
    assert group.attrs["constant_series"] is True
    assert int(group.attrs["grid_y"]) == 101
    assert int(group.attrs["grid_x"]) == 121
    # Native GAR projection: verbatim source pyproj_srs (WRF Mercator).
    assert group.attrs["crs"] == (
        "+k_0=1.0 +units=m +lat_0=2.0 +lon_0=-90.31006622 +x_0=0.0 "
        "+y_0=0.0 +ellps=WGS84 +datum=WGS84 +name=WRFMercator +proj=merc +no_defs"
    )


def test_aws_january_store_has_real_exclusions(tmp_path, comparison_sources):
    """AWS store: Jan-2025 hourly/daily tip sums with 8 excluded hours.

    The engine threshold (HOURLY_READINGS_REQUIRED = 5) keeps the two
    5-reading hours (Jan 12 11:00, Jan 30 14:00); the second one flips
    Jan 12 to 22/24 hourly coverage, so all 31 days stay valid.
    """
    from darwin.publisher.comparison_series import build_comparison_series

    out = build_comparison_series(comparison_sources, tmp_path)
    import zarr

    group = zarr.open_group(str(out["aws_january"]), mode="r")
    hourly = group["hourly"][:]
    hourly_valid = group["hourly_valid"][:]
    assert hourly.shape == (744,)
    assert int(hourly_valid.sum()) == 736
    assert 744 - int(hourly_valid.sum()) == 8
    daily = group["daily"][:]
    daily_valid = group["daily_valid"][:]
    assert daily.shape == (31,)
    assert int(daily_valid.sum()) == 31
    assert abs(float(hourly[hourly_valid].sum()) - 135.8) < 1e-6
    assert float(hourly[0]) == 0.0
    assert group.attrs["units"] == "mm"
    assert group["hourly"].attrs["units"] == "mm"
    assert group["daily"].attrs["units"] == "mm"
