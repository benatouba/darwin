"""Comparison series stores (T6): published GAR/AWS January series.

The dashboard serves the real prcp Comparison result from these seed
stores, so they must carry true values: the GAR January-2020 monthly
rate sampled nearest the CerroCrocker station uniformly disaggregated
to constant hourly/daily series, and the AWS January hourly/daily
depths with real completeness exclusions, all asserted against
independently measured literals (CSV + NetCDF probes, not the
builder's own stores).
"""

from pathlib import Path

REPO_ROOT = Path("/home/ben/projects/darwin")


def test_gar_sampled_store_carries_jan2020_nearest_series(tmp_path):
    """GAR store: 12 monthly nearest rates, Jan rate constant over 744h/31d."""
    from darwin.publisher.comparison_series import build_comparison_series

    out = build_comparison_series(REPO_ROOT, tmp_path)
    import zarr

    group = zarr.open_group(str(out["gar_sampled"]), mode="r")
    monthly = group["monthly"][:]
    assert monthly.shape == (12,)
    # float32 source precision: the probe literal is the float64 rounding.
    assert abs(float(monthly[0]) - 0.15625376) < 1e-7
    assert group["hourly"].shape == (744,)
    assert group["daily"].shape == (31,)
    assert bool((group["hourly"][:] == float(monthly[0])).all())
    assert bool((group["daily"][:] == float(monthly[0])).all())
    assert group.attrs["units"] == "mm h-1"
    assert int(group.attrs["grid_y"]) == 101
    assert int(group.attrs["grid_x"]) == 121


def test_aws_january_store_has_real_exclusions(tmp_path):
    """AWS store: Jan-2025 hourly/daily tip sums with 8 excluded hours.

    The engine threshold (HOURLY_READINGS_REQUIRED = 5) keeps the two
    5-reading hours (Jan 12 11:00, Jan 30 14:00); the second one flips
    Jan 12 to 22/24 hourly coverage, so all 31 days stay valid.
    """
    from darwin.publisher.comparison_series import build_comparison_series

    out = build_comparison_series(REPO_ROOT, tmp_path)
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
