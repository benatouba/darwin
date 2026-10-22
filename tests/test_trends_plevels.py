from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from darwin import accessors

xr.set_options(keep_attrs=True)
# pytest data directory
datadir = Path(__file__).resolve().parent / "data"
data_y_2d_prcp = datadir / "MM_d02km_y_2d_prcp.nc"
data_m_2d_t2 = datadir / "MM_d02km_m_2d_t2.nc"
data_m_2d_prcp = datadir / "MM_d02km_m_2d_prcp.nc"
data_m_2d_et = datadir / "MM_d02km_m_2d_et.nc"
data_landmask = datadir / "MM_d02km_static_landmask.nc"
data_hgt = datadir / "MM_d02km_static_hgt.nc"


@pytest.fixture(autouse=True)
def plot_outputs(tmp_path, monkeypatch):
    """Keep generated plots out of the checkout and close each test's figures."""
    monkeypatch.chdir(tmp_path)
    yield
    plt.close("all")


def test_xarray_accessor() -> None:
    ds = xr.open_dataset(data_y_2d_prcp)
    assert isinstance(ds, xr.Dataset)
    assert hasattr(ds, "darwin")


def test_seasons() -> None:
    ds = xr.open_dataset(data_y_2d_prcp)
    ds.darwin.add_seasons()
    assert isinstance(ds.seasons, xr.DataArray)
    assert ds.seasons.min() == 1
    assert ds.seasons.max() == 1


def test_filter_season() -> None:
    ds = xr.open_dataset(data_y_2d_prcp)
    ds = ds.darwin.assign_seasons_as_coords()
    assert ds.coords["season"].min() == 1
    assert ds.coords["season"].max() == 1


def test_plot_facet_from_coords() -> None:
    from matplotlib import pyplot as plt

    ds = xr.open_dataset(data_m_2d_t2)
    ds.darwin.convert_units()
    landmask = xr.open_dataset(data_landmask)
    hgt = xr.open_dataset(data_hgt)
    ds = ds.darwin.mask_land(landmask, hgt)
    ds = ds.mean(dim=("south_north", "west_east"))
    ds.darwin.assign_seasons_as_coords()
    _, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    ds.darwin.plot_facet_from_coords(ax=ax)
    plt.savefig("test_plot_facet_from_coords_t2.png")


def test_plot_facet_from_coords_gls() -> None:
    from matplotlib import pyplot as plt

    ds = xr.open_dataset(data_m_2d_t2)
    ds.darwin.convert_units()
    landmask = xr.open_dataset(data_landmask)
    hgt = xr.open_dataset(data_hgt)
    ds = ds.darwin.mask_land(landmask, hgt)
    ds = ds.mean(dim=("south_north", "west_east"))
    ds.darwin.assign_seasons_as_coords()
    _, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    ds.darwin.plot_facet_from_coords(ax=ax, model_type="gls")
    plt.savefig("test_plot_facet_from_coords_t2_gls.png")


def test_plot_facet_from_coords_prcp() -> None:
    from matplotlib import pyplot as plt

    ds = xr.open_dataset(data_m_2d_prcp)
    ds.darwin.convert_units()
    landmask = xr.open_dataset(data_landmask)
    hgt = xr.open_dataset(data_hgt)
    ds = ds.darwin.mask_land(landmask, hgt)
    ds = ds.mean(dim=("south_north", "west_east"))
    ds.darwin.assign_seasons_as_coords()
    _, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    ds.darwin.plot_facet_from_coords(ax=ax)
    plt.savefig("test_plot_facet_from_coords_prcp.png")


def test_plot_facet_from_coords_prcp_rls() -> None:
    from matplotlib import pyplot as plt

    ds = xr.open_dataset(data_m_2d_prcp)
    ds.darwin.convert_units()
    landmask = xr.open_dataset(data_landmask)
    hgt = xr.open_dataset(data_hgt)
    ds = ds.darwin.mask_land(landmask, hgt)
    ds = ds.mean(dim=("south_north", "west_east"))
    ds.darwin.assign_seasons_as_coords()
    _, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    ds.darwin.plot_facet_from_coords(ax=ax, model_type="rls")
    plt.savefig("test_plot_facet_from_coords_prcp_rls.png")


def test_plot_facet_from_coords_gls_ar1_prcp() -> None:
    from matplotlib import pyplot as plt

    ds = xr.open_dataset(data_m_2d_prcp)
    ds.darwin.convert_units()
    landmask = xr.open_dataset(data_landmask)
    hgt = xr.open_dataset(data_hgt)
    ds = ds.darwin.mask_land(landmask, hgt)
    ds = ds.mean(dim=("south_north", "west_east"))
    ds.darwin.assign_seasons_as_coords()
    _, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    ds.darwin.plot_facet_from_coords(ax=ax, model_type="gls-ar1")
    plt.savefig("test_plot_facet_from_coords_prcp_gls-ar1.png")


def test_plot_facet_from_coords_gls_hs_prcp() -> None:
    from matplotlib import pyplot as plt

    ds = xr.open_dataset(data_m_2d_prcp)
    ds.darwin.convert_units()
    landmask = xr.open_dataset(data_landmask)
    hgt = xr.open_dataset(data_hgt)
    ds = ds.darwin.mask_land(landmask, hgt)
    ds = ds.mean(dim=("south_north", "west_east"))
    ds.darwin.assign_seasons_as_coords()
    _, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    ds.darwin.plot_facet_from_coords(ax=ax, model_type="gls-hs")
    plt.savefig("test_plot_facet_from_coords_prcp_gls-hs.png")


def test_plot_facet_from_coords_dry_wet() -> None:
    from matplotlib import pyplot as plt

    ds = xr.open_dataset(data_m_2d_t2)
    ds.darwin.convert_units()
    landmask = xr.open_dataset(data_landmask)
    hgt = xr.open_dataset(data_hgt)
    ds = ds.darwin.mask_land(landmask, hgt)
    ds = ds.mean(dim=("south_north", "west_east"))
    ds.darwin.assign_seasons_as_coords(dry_wet=True)
    _, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    ds.darwin.plot_facet_from_coords(ax=ax)
    plt.savefig("test_plot_facet_from_coords_dry-wet.png")


def test_plot_facet_from_coords_prcp_dry_wet() -> None:
    from matplotlib import pyplot as plt

    ds = xr.open_dataset(data_m_2d_prcp)
    ds.darwin.convert_units()
    landmask = xr.open_dataset(data_landmask)
    hgt = xr.open_dataset(data_hgt)
    ds = ds.darwin.mask_land(landmask, hgt)
    ds = ds.mean(dim=("south_north", "west_east"))
    ds.darwin.assign_seasons_as_coords(dry_wet=True)
    _, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    ds.darwin.plot_facet_from_coords(ax=ax)
    plt.savefig("test_plot_facet_from_coords_prcp_dry-wet.png")


def test_plot_facet_from_coords_prcp_monthly() -> None:
    from matplotlib import pyplot as plt

    ds = xr.open_dataset(data_m_2d_prcp)
    ds.darwin.convert_units()
    landmask = xr.open_dataset(data_landmask)
    hgt = xr.open_dataset(data_hgt)
    ds = ds.darwin.mask_land(landmask, hgt)
    ds = ds.mean(dim=("south_north", "west_east"))
    ds = ds.darwin.assign_seasons_as_coords()
    _, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    ds.darwin.plot_facet_from_coords(ax=ax)
    plt.savefig("test_plot_facet_from_coords_prcp_monthly.png")


def test_plot_facet_from_coords_prcp_monthly_regression() -> None:
    from matplotlib import pyplot as plt

    ds = xr.merge(
        (xr.open_dataset(data_m_2d_prcp), xr.open_dataset(data_m_2d_et)),
        compat="no_conflicts",
    )
    ds.darwin.convert_units()
    landmask = xr.open_dataset(data_landmask)
    hgt = xr.open_dataset(data_hgt)
    ds = ds.darwin.mask_land(landmask, hgt)
    ds = ds.mean(dim=("south_north", "west_east"))
    ds = ds.darwin.assign_seasons_as_coords()
    _, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    ds.darwin.plot_facet_from_coords(ax=ax, xvar="et", yvar="prcp")
    plt.savefig("test_plot_facet_from_coords_prcp-et_monthly_regression.png")


def _monthly_series():
    times = np.arange("2000-01", "2010-01", dtype="datetime64[M]")
    return xr.Dataset(
        {"t2": ("time", 280.0 + np.arange(len(times)) * 0.01, {"units": "K"})},
        coords={"time": times},
        attrs={"VARNAME": "t2"},
    )


@pytest.mark.parametrize("dry_wet", [False, True])
def test_season_assignment_is_idempotent_and_records_grouping(dry_wet):
    ds = _monthly_series().darwin.assign_seasons_as_coords(dry_wet=dry_wet)
    repeated = ds.darwin.assign_seasons_as_coords(dry_wet=dry_wet)
    xr.testing.assert_identical(repeated, ds)
    assert ds.season.attrs["grouping"] == ("dry-wet" if dry_wet else "quarterly")


def test_dry_wet_facets_use_actual_months_and_days(monkeypatch):
    ds = _monthly_series().rename({"t2": "prcp"})
    ds.attrs["VARNAME"] = "prcp"
    ds.prcp.attrs["units"] = "mm/day"
    ds = ds.darwin.assign_seasons_as_coords(dry_wet=True)
    calls = []
    original = accessors.DatasetAccessor.plot_linear_regression

    def capture(self, ax, xdata, ydata, nino_class, *args, **kwargs):
        calls.append((np.array(xdata), np.array(ydata), np.array(nino_class)))
        return original(self, ax, xdata, ydata, nino_class, *args, **kwargs)

    monkeypatch.setattr(accessors.DatasetAccessor, "plot_linear_regression", capture)
    _, axes = plt.subplots(2, 1)
    ds.darwin.plot_facet_from_coords(ax=axes)
    assert len(calls) == 2
    for index, months in enumerate(((12, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11))):
        selected = ds.where(ds.time.dt.month.isin(months), drop=True)
        expected = (
            selected.prcp.groupby("time.year").mean()
            * selected.time.dt.days_in_month.groupby("time.year").sum()
        )
        np.testing.assert_allclose(calls[index][1], expected.values)
        anomaly = accessors.nino_anomaly
        anomaly = anomaly.loc[
            anomaly.index.month.isin(months) & anomaly.index.year.isin(calls[index][0])
        ]
        grouped = anomaly.groupby(anomaly.index.year)["value"]
        classes = np.where(grouped.max() >= 1.13, 1, np.where(grouped.min() <= -1.13, -1, 0))
        np.testing.assert_array_equal(calls[index][2], classes)
    assert [axis.get_title() for axis in axes] == ["Wet", "Dry"]


@pytest.mark.parametrize("model_type", ["ols", "gls", "gls-ar1", "gls-hs", "rls"])
@pytest.mark.parametrize("fill_between", [False, True])
def test_regression_models_plot_known_linear_trend(model_type, fill_between):
    accessor = _monthly_series().darwin
    xdata = np.arange(20, dtype=float)
    ydata = 4.0 + 2.0 * xdata + 0.1 * np.sin(xdata)
    _, ax = plt.subplots()
    result = accessor.plot_linear_regression(
        ax, xdata, ydata, np.zeros(20), model_type=model_type,
        fill_between=fill_between,
    )
    assert result.params[1] == pytest.approx(2.0, abs=0.02)
    assert len(ax.lines) == 1
    assert len(ax.collections) == (2 if fill_between else 1)


def test_unknown_regression_model_is_rejected():
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="unknown regression model"):
        _monthly_series().darwin.plot_linear_regression(
            ax, np.arange(5), np.arange(5), np.zeros(5), model_type="unknown"
        )


def test_switching_season_grouping_updates_existing_coordinate():
    quarterly = _monthly_series().darwin.assign_seasons_as_coords()
    dry_wet = quarterly.darwin.assign_seasons_as_coords(dry_wet=True)
    assert set(dry_wet.season.values) == {1, 2}
    assert dry_wet.season.attrs["grouping"] == "dry-wet"
    quarterly_again = dry_wet.darwin.assign_seasons_as_coords()
    assert set(quarterly_again.season.values) == {1, 2, 3, 4}


@pytest.mark.parametrize("model_type", ["gls-ar1", "gls-hs"])
def test_gls_receives_covariance_not_inverse_weights(model_type):
    xdata = np.arange(20, dtype=float)
    ydata = 4.0 + 2.0 * xdata + (1 + xdata / 10) * np.sin(xdata)
    design = accessors.sm.add_constant(xdata)
    residuals = accessors.sm.OLS(ydata, design).fit().resid
    if model_type == "gls-ar1":
        rho = accessors.sm.tsa.acf(residuals, fft=False, nlags=1)[1]
        expected = rho ** np.abs(np.subtract.outer(xdata, xdata))
    else:
        floor = max(float(np.mean(residuals**2)) * 1e-12, np.finfo(float).eps)
        log_variance = np.log(np.maximum(residuals**2, floor))
        coefficients = np.linalg.lstsq(design, log_variance, rcond=None)[0]
        expected = np.diag(np.exp(design @ coefficients))
    _, ax = plt.subplots()
    result = _monthly_series().darwin.plot_linear_regression(
        ax, xdata, ydata, np.zeros(20), model_type=model_type,
    )
    np.testing.assert_allclose(result.model.sigma, expected)
    assert np.all(np.linalg.eigvalsh(result.model.sigma) > 0)


def test_robust_confidence_band_uses_parameter_covariance():
    xdata = np.arange(20, dtype=float)
    ydata = 4.0 + 2.0 * xdata + np.sin(xdata)
    _, ax = plt.subplots()
    result = _monthly_series().darwin.plot_linear_regression(
        ax, xdata, ydata, np.zeros(20), model_type="rls",
    )
    design = accessors.sm.add_constant(xdata)
    variance = np.diag(design @ result.cov_params() @ design.T)
    margin = 1.959963984540054 * np.sqrt(variance)
    mean = result.predict(design)
    vertices = ax.collections[-1].get_paths()[0].vertices
    assert vertices[:, 1].min() == pytest.approx((mean - margin).min())
    assert vertices[:, 1].max() == pytest.approx((mean + margin).max())
