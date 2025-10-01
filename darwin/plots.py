"""Plot trends and mean values for MM data. May convert to core of the package.

CLI Usage:
    trends_plevels.py --var t2c --type mean --start 2001 --end 2010 --frequency y
    trends_plevels.py --var t2c --type trend --start 2001 --end 2010 --frequency y
    trends_plevels.py --var t2c --type timeseries --start 2001 --end 2010 --frequency y
    trends_plevels.py -v t2c -t timeseries -l 100 -s 2001 -e 2010 -f y
"""

import logging
import sys
from datetime import date, datetime

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import salem
import statsmodels.api as sm
import typer
import xarray as xr
from metpy.units import units as mpunits
from rich import print  # noqa: A004

import darwin.accessors  # noqa: F401
from darwin.constants import axis_labels
from darwin.core import Experiment
from darwin.defaults import (
    bar_vars,
    default_end,
    default_start,
    mm_path,
    nlevels,
    pvalue_threshold,
)
from darwin.utils import days_in_slot

pd.options.mode.copy_on_write = True
mpl.rcParams["axes.axisbelow"] = True
p_greater = r"$\it{p}$ >= 0.05"
p_less = r"$\it{p}$ < 0.05"

logging.getLogger("matplotlib.font_manager").disabled = True
xr.set_options(keep_attrs=True)

app = typer.Typer(pretty_exceptions_show_locals=False)


def plot_regression_residuals(
    xdata: npt.ArrayLike,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    title: str,
) -> None:
    """Plot regression residuals.

    Args:
        xdata: The independent variable.
        model: The regression model results object.
        title: The title of the plot.
    """
    _, ax_residuals = plt.subplots(1, 1)
    ax_residuals.scatter(xdata, model.resid, color="black")
    ax_residuals.grid(zorder=0)
    ax_residuals.set_ylabel("residuals")
    ax_residuals.set_xlabel("year")
    plt.savefig(f"plots/residuals/{title}.png")


def trend_1d(
    da: xr.DataArray,
) -> sm.regression.linear_model.OLSResults:
    """Calculate the 1-dimensional temporal trend for a variable.

    Returns:
        The model object.
    """
    xdata = da.time.dt.year.to_numpy() if "time" in da.dims else da.year.to_numpy()
    return sm.OLS(
        da.to_numpy(),
        sm.add_constant(xdata),
    ).fit()
    # return model.predict(sm.add_constant(da.time.dt.year.to_numpy()))


def get_filestem(
    filestem: str,
    level: list[int] | None,
    highest_alt: float,
    lowest_alt: float,
    var: list[str],
    start: date,
    end: date,
    *,
    mask: bool,
    mask_sea: bool,
) -> str:
    """Create the full filestem from variables.

    Args:
        filestem: The existing (partial) filestem.
        level: The (pressure) levels included.
        highest_alt: The highest (ground) altitude included.
        lowest_alt: The lowest (ground) altitude included.
        var: The variables included.
        start: The start date.
        end: The end date.
        mask: Whether the data was masked (for land).
        mask_sea: Whether the data was masked for sea.

    Returns:
        The full filestem.
    """
    if level:
        filestem += f"_p-{'-'.join(str(lev) for lev in level)}"
    else:
        filestem += f"_hgt-{int(lowest_alt)}-{int(highest_alt)}"
    var_part = "-".join(v for v in var)
    filestem += f"_v-{var_part}_y-{str(start)[:4]}-{str(end)[:4]}"
    if not mask:
        filestem += "_nomask"
    if mask_sea:
        filestem += "_masksea"
    return filestem


def create_full_year_plot(
    ds: xr.Dataset,
    var: list[str],
    nino_anomaly: pd.DataFrame,
    output: str,
    *,
    create_plot: bool,
) -> None:
    """Create a full year plot."""
    ds = ds.mean(dim=("south_north", "west_east")).squeeze()
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), sharex=True, sharey=True, layout="tight")
    data = ds.groupby("time.year").mean("time")
    years: list[int] = np.unique(ds.time.dt.year.to_numpy())
    if var[0] in bar_vars:
        ndays_list = [days_in_slot(1, 1, 12, 31, y) for y in years]
        data = data * mpunits.Quantity(ndays_list, "d")
    # rolling mean 5-month
    nino_select = nino_anomaly.rolling(5).mean()
    nino_select["year"] = nino_select.index.year
    nino_select_max = nino_select.groupby("year").max("year")
    nino_select_min = nino_select.groupby("year").min("year")
    nino_select = nino_select.groupby("year").mean("year")
    nino_select["class"] = 0
    nino_select.loc[nino_select_min["value"] <= -1.13, "class"] = -1
    nino_select.loc[nino_select_max["value"] >= 1.13, "class"] = 1
    nino_data = (data[var[0]] * np.where(nino_select["class"] == 1, 1, np.nan)).to_numpy()
    nina_data = (data[var[0]] * np.where(nino_select["class"] == -1, 1, np.nan)).to_numpy()
    no_nino_data = (data[var[0]] * np.where(nino_select["class"] != 1, 1, np.nan)).to_numpy()
    no_nina_data = (data[var[0]] * np.where(nino_select["class"] != -1, 1, np.nan)).to_numpy()
    print(f"Nino y mean: {np.nanmean(nino_data)}")
    print(f"Nino y anomaly: {np.nanmean(nino_data) - np.nanmean(data[var[0]])}")
    print(f"Nina y mean: {np.nanmean(nina_data)}")
    print(f"Nina y anomaly: {np.nanmean(nina_data) - np.nanmean(data[var[0]])}")
    print(f"No Nino y mean: {np.nanmean(no_nino_data)}")
    print(f"No Nino y anomaly: {np.nanmean(no_nino_data) - np.nanmean(data[var[0]])}")
    print(f"No Nina y mean: {np.nanmean(no_nina_data)}")
    print(f"No Nina y anomaly: {np.nanmean(no_nina_data) - np.nanmean(data[var[0]])}")
    reg_results = data.darwin.plot_linear_regression(
        ax,
        xdata=data.year.to_numpy(),
        ydata=data[var[0]].to_numpy(),
        nino_class=nino_select["class"].to_numpy(),
    )

    print(reg_results.t_test("x1 = 0").pvalue)
    pval_label = p_less if reg_results.t_test("x1 = 0").pvalue < pvalue_threshold else p_greater
    units = data[data.darwin.varname].metpy.units
    split_units = f"{units:~P}".split("/")
    if len(split_units) == 2:
        if split_units[1] not in ("y", "a", "h", "d"):
            ann_units = f"{split_units[0]}$\\,${split_units[1]}$^{{-1}}$"
        else:
            ann_units = f"{split_units[0]}$\\,$a$^{{-1}}$"
    else:
        ann_units = f"{units:~P}$\\,$a$^{{-1}}$"
    ann_units = ann_units.replace("°C", "K")
    mean = data[data.darwin.varname].mean()
    labels = [
        f"$\\overline{{y}}$ = {mean.metpy.magnitude:.3f} {mean.metpy.units:~P}",
        f"$\\beta$ = {reg_results.params[1]:.3f} {ann_units}",
        f"$R^2$ = {reg_results.rsquared:.2f}",
        pval_label,
    ]
    labels = [" $|$ ".join(labels)]
    xlabel, ylabel = data.darwin.get_timeseries_axis_labels(0, is_2d=True)
    xlabel = "year"
    data.darwin._add_axis_annotations_for_regression(
        ax,
        labels=labels,
        title=None,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    plt.tight_layout()
    if not create_plot:
        sys.exit()
    fig.savefig(output)
    print(f"Output: {output}")
    sys.exit()


def create_timeseries_plot(
    ds: xr.Dataset,
    output: str,
    *,
    create_plot: bool,
) -> None:
    """Create a timeseries plot."""
    ds = ds.mean(dim=("south_north", "west_east")).squeeze()
    print(ds.time.dt.season)
    ds.darwin.assign_seasons_as_coords()
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, layout="tight")
    ds.darwin.plot_facet_from_coords(ax=ax)
    if not create_plot:
        sys.exit()
    fig.savefig(output)
    print(f"Output: {output}")
    sys.exit()


def create_regression_plot(
    ds: xr.Dataset,
    var: list[str],
    output: str,
    *,
    create_plot: bool,
) -> None:
    """Create a regression plot."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False, layout="tight")
    ds = ds.mean(dim=("south_north", "west_east")).squeeze()
    ds.darwin.assign_seasons_as_coords()
    ds.darwin.plot_facet_from_coords(
        ax=ax,
        xvar=var[0],
        yvar=var[1],
        xlog=False,
        ylog=False,
        fill_between=False,
    )
    if not create_plot:
        sys.exit()
    fig.savefig(output)
    print(f"Output: {output}")
    sys.exit()


def create_mean_plot(
    ds: xr.Dataset,
    var: list[str],
    output: str,
    *,
    create_plot: bool,
) -> None:
    """Create a mean plot."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False, layout="tight")
    ds = ds.mean(dim=("south_north", "west_east")).squeeze()
    ds.darwin.assign_seasons_as_coords()
    ds.darwin.plot_facet_from_coords(ax=ax, xvar=var[0], yvar=var[1])
    if not create_plot:
        sys.exit()
    fig.savefig(output)
    print(f"Output: {output}")
    sys.exit()


def create_trend_map(
    ds: xr.Dataset,
    ax: plt.Axes,
    output: str,
) -> None:
    """Create a trend plot."""
    if len(ds.dims) != 3:
        msg = "trend map only implemented for 3-dimensional data."
        raise NotImplementedError(msg)
    dummy_da = ds[ds.VARNAME].mean(axis=0).squeeze().metpy.dequantify()
    ds["trend"] = dummy_da.copy()
    ds["trend_pvalue"] = dummy_da.copy()
    for i in np.arange(ds.sizes["south_north"]):
        for j in np.arange(ds.sizes["west_east"]):
            ydata = ds[ds.VARNAME][:, i, j]
            if not np.all(np.isnan(ydata)):
                print(
                    f"{i * ds.sizes['south_north'] + j} / {ds.sizes['south_north'] * ds.sizes['west_east']}"
                )
                reg_results = trend_1d(ydata)
                print(reg_results.params[1])
                ds["trend"][i, j] = reg_results.params[1]
                ds["trend_pvalue"][i, j] = reg_results.pvalues[1]
    # print(ds["trend_pvalue"].to_numpy().max())
    # p_fdr_poscorr = fdrcorrection(
    #     ds["trend_pvalue"].to_numpy().flatten(order="F"),
    # )[1]
    # p_fdr_t = fdrcorrection(
    #     ds["trend_pvalue"].to_numpy().flatten(),
    # )[0]
    # print(p_fdr_t.max())
    # ds["trend_pvalue"].values = p_fdr_poscorr.reshape(ds.sizes["south_north"], ds.sizes["west_east"])

    ds.darwin.plot_facet_map(
        output=output,
        varname="trend",
        overlay_dim="trend_pvalue",
        cmap="RdBu_r",
        ax=ax,
    )


@app.command()
def main(
    var: list[str],
    level: list[int] | None = None,
    colorbarlabel: str = "",
    plot_type: str = "mean",
    start: datetime = default_start,
    end: datetime = default_end,
    frequency: str = "y",
    lowest_alt: float = 0.1,
    highest_alt: float = 2000.0,
    *,
    mask: bool = True,
    plot: bool = True,
    mask_sea: bool = False,
) -> None:
    """Plot GAR time series."""
    dim = "2d" if not level else "3d_press"
    data_dict = {}

    filestem = get_filestem(
        plot_type,
        level,
        highest_alt,
        lowest_alt,
        var,
        start,
        end,
        mask=mask,
        mask_sea=mask_sea,
    )
    output = f"plots/{filestem}.png"
    landmask = salem.open_xr_dataset(mm_path / "MM_d02km_static_landmask.nc")
    hgt = salem.open_xr_dataset(mm_path / "MM_d02km_static_hgt.nc")
    landmask = landmask.darwin.crop_margins(mask=mask).isel(time=0)
    hgt = hgt.darwin.crop_margins(mask=mask).isel(time=0)

    is_level_comparison = level is not None and len(level) == nlevels["2"]
    fig = plt.figure(
        figsize=(10, 4) if plot_type == "timeseries" else (10, 8),
        frameon=False,
        layout="tight",
        facecolor="white",
        edgecolor="white",
    )
    ax = fig.add_subplot(111)
    # prepare data
    for v in var:
        ds = salem.open_xr_dataset(mm_path / f"MM_d02km_{frequency}_{dim}_{v}.nc")
        ds.attrs["level"] = "-".join(str(lev) for lev in level) if level else ""
        if level and not is_level_comparison:
            ds = ds.sel(pressure=level[0])
        elif level and is_level_comparison:
            ds1 = ds.sel(pressure=level[0])
            ds2 = ds.sel(pressure=level[1])
            ds = ds2 - ds1

        ds = ds.sel(time=slice(start, end))
        ds = ds.darwin.crop_margins(mask=mask)
        if mask:
            ds = ds.darwin.mask_height_zone(landmask, hgt, lowest_alt, highest_alt)
        elif mask_sea:
            ds = ds.darwin.mask_sea(landmask)

        # select and clean data
        ds.darwin.convert_units()
        data_dict[v] = ds

    ds: xr.Dataset = data_dict[var[0]] if len(var) == 1 else xr.merge([data_dict[v] for v in var])
    print("Data mean:")
    print(ds[var[0]].mean().as_numpy())
    print(ds[var[0]].metpy.units)

    if plot_type == "full_year":
        create_full_year_plot(ds, var, nino_anomaly, output, create_plot=plot)
    if plot_type == "timeseries":
        create_timeseries_plot(ds, output, create_plot=plot)
    elif plot_type == "regression":
        create_regression_plot(ds, var, output, create_plot=plot)
    elif plot_type == "mean":
        create_mean_plot(ds, var, nino_anomaly, output, create_plot=plot)
    elif plot_type == "trend":
        ds = ds.groupby("time.year").mean("time")
        print(ds)
        create_trend_map(ds, ax, output)
    elif plot_type == "map":
        exp = Experiment(ds.mean("time"))
        print(exp)
        mod_var = var[0]
        cmap = "YlGnBu" if mod_var in ["et", "rh2", "sh2", "q2", "prcp"] else "YlOrRd"
        cmap = "gray" if mod_var == "swdown" else cmap

        unit = f"mean {axis_labels[var]}"
        annotation_years = {f"{start} - {end}": (0.1, 0.952)}
        exp.darwin.plot_map(
            unit=unit,
            cmap=cmap,
            frequency=frequency[0],
            markercolors=means[var]
            if pd.to_datetime(start) >= pd.to_datetime("2022-04-01")
            else None,
            annotations=annotation_years,
        )
    else:
        msg = f"Unknown type: {plot_type}"
        raise ValueError(msg)
    units = ds[var[0]].attrs["units"]
    ds = xr.Dataset(
        data_vars={
            var[0]: (
                ["south_north", "west_east"],
                ds,
                {"units": units},
            )
        },
        coords={
            "lat": landmask.lat,
            "lon": landmask.lon,
        },
    )

    # plot mean
    if not plot:
        sys.exit()
    _, ax = plt.subplots(1, 1)
    base_map = landmask.salem.get_map()
    base_map.set_data(ds[var])
    base_map.set_cmap(ds.darwin.cmap)
    base_map.set_scale_bar()
    base_map.plot(ax=ax)
    base_map.append_colorbar(ax=ax, label=colorbarlabel)
    plt.savefig(f"plots/{filestem}.png")
    print(f"Output: {output}")


if __name__ == "__main__":
    app()
