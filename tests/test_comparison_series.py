"""Legacy smoke inputs retain values but cannot establish paired model skill.

UTC below is an explicit test declaration, not an approval of real-source timezone.
"""


def test_gar_sampled_store_carries_jan2020_nearest_series(tmp_path, comparison_sources):
    """GAR store: 12 authoritative monthly rates, no fabricated fine cadences."""
    from darwin.publisher.comparison_series import build_comparison_series

    out = build_comparison_series(comparison_sources, tmp_path, obs_timezone="UTC")
    import zarr

    group = zarr.open_group(str(out["gar_sampled"]), mode="r")
    monthly = group["monthly"][:]
    assert monthly.shape == (12,)
    # float32 source precision: the probe literal is the float64 rounding.
    assert abs(float(monthly[0]) - 0.15625376) < 1e-7
    assert "hourly" not in group and "daily" not in group
    assert group["monthly"].attrs["units"] == "mm h-1"
    assert group.attrs["cadences"] == ["monthly"]
    assert group.attrs["source_coverage"]["start"].startswith("2020-01-01")
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

    out = build_comparison_series(comparison_sources, tmp_path, obs_timezone="UTC")
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
    assert group["hourly_time"].shape == (744,)
    assert group["hourly_time_bounds"].shape == (744, 2)
    assert group.attrs["source_coverage"]["first_timestamp"].startswith("2025-01-01")


def test_legacy_sources_are_not_repaired_by_guessing_a_timezone(tmp_path, comparison_sources):
    import pytest

    from darwin.publisher.comparison_series import build_comparison_series
    from darwin.publisher.series import TimeValidationError

    with pytest.raises(TimeValidationError, match="explicit timezone"):
        build_comparison_series(comparison_sources, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_actual_legacy_model_year_cannot_be_paired_with_2025(tmp_path, comparison_sources):
    import pytest

    from darwin.publisher.comparison import compare
    from darwin.publisher.comparison_series import build_comparison_series
    from darwin.publisher.series import TimeSeries, TimeValidationError, load_series

    paths = build_comparison_series(comparison_sources, tmp_path, obs_timezone="UTC")
    model = load_series(paths["gar_sampled"], "monthly")
    # Even an explicitly prepared monthly observation cannot overlap the wrong year.
    observations = TimeSeries([135.8], ["2025-01-01T00:00:00+00:00"], "monthly")
    with pytest.raises(TimeValidationError, match="no overlapping"):
        compare("prcp", "PCP_tot_bucket", model, observations)
