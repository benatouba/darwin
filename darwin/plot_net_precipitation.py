import logging
import sys

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import salem
import xarray as xr
from metpy.units import units as mpunits
from rich import print  # noqa: A004

from darwin.defaults import (
    bar_vars,
    mm_path,
    ndims,
)
from darwin.plots import default_end, default_start, get_filestem
from darwin.utils import days_in_slot, time_slots

pd.options.mode.copy_on_write = True
mpl.rcParams["axes.axisbelow"] = True
p_greater = r"$\it{p}$ >= 0.05"
p_less = r"$\it{p}$ < 0.05"

logging.getLogger("matplotlib.font_manager").disabled = True
xr.set_options(keep_attrs=True)


def main():
    dim = "2d"
    data_dict = {}

    var = ["et", "prcp"]
    filestem = get_filestem(
        "mean",
        level=None,
        highest_alt=2000.0,
        lowest_alt=0.1,
        var=[var[0]],
        start=default_start,
        end=default_end,
        mask=True,
        mask_sea=False,
    )
    output = f"plots/{filestem}.png"
    landmask = salem.open_xr_dataset(mm_path / "MM_d02km_static_landmask.nc")
    hgt = salem.open_xr_dataset(mm_path / "MM_d02km_static_hgt.nc")
    landmask = landmask.darwin.crop_margins(mask=True).isel(time=0)
    hgt = hgt.darwin.crop_margins(mask=True).isel(time=0)

    is_level_comparison = False
    fig = plt.figure(
        figsize=(10, 8),
        frameon=False,
        layout="tight",
        facecolor="white",
        edgecolor="white",
    )
    frequency = "m"
    level = None
    start = default_start
    end = default_end
    highest_alt = 2000.0
    lowest_alt = 0.1
    mask = True
    mask_sea = False
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
    ds = ds.mean(dim=("south_north", "west_east")).squeeze()
    print(ds.time.dt.season)
    ds.darwin.assign_seasons_as_coords()
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, layout="tight")
    varname = var[1]
    axes = ax
    for i, time_val in enumerate(
        # np.unique(self._obj.coords[self._time_varname].to_numpy())
        np.arange(1, 5)
    ):
        axi = axes[i // 2, i % 2] if axes.ndim > 1 else axes[i]
        years = ds.darwin._obj.loc[{"season": time_val}].time.dt.year.to_numpy()
        ndays_list = [days_in_slot(*time_slots[i], year) for year in np.unique(years)]
        data = (
            ds.darwin._obj.loc[{"season": time_val}]
            .assign_coords(year=("season", years))
            .swap_dims({"season": "year"})
        )
        season_to_month_nums = {
            1: (1, 2, 3),
            2: (4, 5, 6),
            3: (7, 8, 9),
            4: (10, 11, 12),
        }
        months = season_to_month_nums[time_val]
        # select the months of nino_anomaly for this time val
        xvar = None
        yvar = None
        xdata = (
            np.unique(years)
            if xvar is None
            else (data[xvar].groupby("year").mean("year")).to_numpy()
        )
        data = (
            data[varname].groupby("year").mean("year").to_numpy()
            if yvar is None
            else data[yvar].groupby("year").mean("year").to_numpy()
        )
        if varname in bar_vars:
            data *= mpunits.Quantity(ndays_list, "d")
            data = data.magnitude
        if xvar is not None and varname in bar_vars:
            xdata *= mpunits.Quantity(ndays_list, "d")
            xdata = xdata.magnitude
        print(f"Data mean: {np.nanmean(data)}")
        # plot_regression_residuals(
        #     xdata, reg_results, f"residuals_{self.varname}_{time_val}_{model_type}"
        # )
        # print(reg_results.summary())
        is_2d = axes.ndim == ndims["2d"]
        labels = {
            "t2": "air temperature in °C",
            "t2max": "max. air temperature in °C",
            "theta": "potential air temperature in °C",
            "geopotential": "layer thickness in m",
            "prcp": "precipitation in mm",
            "et": "actual evapotranspiration in mm",
            "wateravailability": "net precipitation in mm",
            "net precipitation": "net precipitation in mm",
            "q": "mixing ratio in g kg$^{-1}$",
            "q2": "mixing ratio in g kg$^{-1}$",
            "pblh": "planetary boundary layer height in m",
        }
        # xlabel = labels[xvar] if i in (2, 3) else ""
        # ylabel = labels[yvar] if i in (0, 2) else ""
        axi.bar(xdata, data, color="white", edgecolor="black", hatch="///")
        # axi.set_yscale("log")
    ds.darwin.plot_facet_from_coords(ax=ax)

    fig.savefig(output)
    print(f"Output: {output}")
    sys.exit()


if __name__ == "__main__":
    main()
