from pathlib import Path

from darwin import trends_plevels
import xarray as xr

xr.set_options(keep_attrs=True)
# pytest data directory
datadir = Path.cwd() / "tests" / "data"
data_y_2d_prcp = datadir / "MM_d02km_y_2d_prcp.nc"
data_m_2d_t2 = datadir / "MM_d02km_m_2d_t2.nc"
data_m_2d_prcp = datadir / "MM_d02km_m_2d_prcp.nc"
data_m_2d_et = datadir / "MM_d02km_m_2d_et.nc"
data_landmask = datadir / "MM_d02km_static_landmask.nc"
data_hgt = datadir / "MM_d02km_static_hgt.nc"


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
    ds.darwin.plot_facet_from_coords(ax=ax, model_type="gls-ar1")
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

    ds = xr.merge((xr.open_dataset(data_m_2d_prcp), xr.open_dataset(data_m_2d_et)))
    ds.darwin.convert_units()
    landmask = xr.open_dataset(data_landmask)
    hgt = xr.open_dataset(data_hgt)
    ds = ds.darwin.mask_land(landmask, hgt)
    ds = ds.mean(dim=("south_north", "west_east"))
    ds = ds.darwin.assign_seasons_as_coords()
    _, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    ds.darwin.plot_facet_from_coords(ax=ax, xvar="et", yvar="prcp")
    plt.savefig("test_plot_facet_from_coords_prcp-et_monthly_regression.png")
