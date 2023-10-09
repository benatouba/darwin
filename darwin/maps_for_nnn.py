"""Create maps for the Niño-Niña-Neutral paper."""
#!/usr/bin/env python

from typing import Optional

from cdo import Cdo
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.dataset import DataVariables


#
# # constants
# basepath = Path("/home/ben/data/GAR/rc_trop_ls_MM/")
# variable = "prcp"
# enso_event = "nino"
# cdo = Cdo(
#     returnNoneOnError=True,
# )
# combinations = {
#     "nina": ["2010", "2011"],
#     "nino": ["2015", "2016"],
#     "neutral": ["2016", "2017"],
# }
#
# # naming rc_trop_ls_MM_d02km_m_2d_v10_2022.nc
#
#
# def make_path(folder: Path, year: Union[str, int], variable: str) -> Path:
#     """Make a path to a file.
#
#     Args:
#         folder: Folder where all files are stored.
#         year: Year of the file or "static" for static files.
#         variable: Variable of the file.
#
#     Returns:
#         Path to the file.
#     """
#     if year == "static":
#         return folder / f"rc_trop_ls_MM_d02km_static_{variable}.nc"
#     return folder / f"rc_trop_ls_MM_d02km_d_2d_{variable}_{year}.nc"
#
#
# def hydro_year(ifile1: Path, ifile2: Path, ofile=False) -> xr.Dataset:
#     """Concatenate two files into one hydrological year.
#
#     Args:
#         ofile (bool): If True, save the file (default: False).
#         ifile1: Input file 1.
#         ifile2: Input file 2.
#
#     Returns:
#         Dataset with the two files concatenated.
#     """
#     if "static" in ifile1.stem or "static" in ifile2.stem:
#         raise ValueError("Cannot concatenate static files.")
#     variable = ifile1.stem.split("_")[-2]
#     year1 = ifile1.stem.split("_")[-1]
#     year2 = ifile2.stem.split("_")[-1]
#     cdo = Cdo()
#     return cdo.seldate(
#         f"{year1}-11-01,{year2}-01-31",
#         input=f"-cat {ifile1} {ifile2}",
#         ofile=ofile,
#         returnXDataset=variable,
#     )
#
#
# # In[4]:
#
#
# nina = hydro_year(
#     make_path(basepath, combinations["nina"][0], variable),
#     make_path(basepath, combinations["nina"][1], variable),
# )
# nino = hydro_year(
#     make_path(basepath, combinations["nino"][0], variable),
#     make_path(basepath, combinations["nino"][1], variable),
# )
# neutral = hydro_year(
#     make_path(basepath, combinations["neutral"][0], variable),
#     make_path(basepath, combinations["neutral"][1], variable),
# )
# # nino = nino.drop([np.datetime64('2016-02-29')], dim='time')
#
#
# # In[5]:
#
#
# oceans = sa.read_shapefile(sa.get_demo_file("ne_50m_ocean.shp"), cached=True)
# nina_land = nina.salem.roi(shape=oceans, all_touched=False)
# neutral_land = neutral.salem.roi(shape=oceans, all_touched=False)
# nino_land = nino.salem.roi(shape=oceans, all_touched=False)
#
#
# # In[14]:
#
#
# def plot_map(ds, name: str, remove_boundaries: Optional[int] = False, **kwargs) -> None:
#     """Plot a map.
#
#     Args:
#         ds (darwin.Dataset): Dataset to plot.
#         name: Name of the plot used for the final part of the filename.
#         remove_boundaries: If an integer is given, remove this amount of grid cells from the
#         boundaries.
#         **kwargs: Keyword arguments passed to ds.plot_map.
#     """
#     fig, ax = plt.subplots(
#         figsize=(16, 14),
#         sharex=True,
#         sharey=True,
#     )
#     if remove_boundaries:
#         ds = ds.remove_boundaries(40)
#     ds.plot_map(ax=ax, **kwargs)
#     plt.savefig(f"rc_trop_ls_MM_{variable}_sum_map_{name}.png")
#     plt.show()
#
#
# plot_map_args = {
#     "aggregation": "sum",
#     "save": False,
#     "stations": True,
#     "cbar": True,
#     "unit": "mm",
#     "cmap": "YlGnBu",
# }
#
# diff = nina.copy()
# diff[variable].data = nino[variable].data - nina[variable].data
# diff.attrs = neutral.attrs
# ds = darwin.Experiment(diff)
# plot_map(ds, name=enso_event, **plot_map_args)
# plt.savefig(f"rc_trop_ls_MM_{variable}_sum_map_nino-nina.png")
#
#
# # In[8]:
#
#
# ds = darwin.Experiment(globals()[enso_event])
# plot_map(ds, name=enso_event, **plot_map_args)
# plt.savefig(f"rc_trop_ls_MM_{variable}_sum_map_{enso_event}.png")
#
#
# # In[9]:
#
#
# fig = plt.figure(figsize=(12, 6))
# plt.plot(
#     nina_land.time,
#     (24 * nina_land.mean(dim=("west_east", "south_north")).cumsum()[variable]),
#     label="La Nina - 2010/2011",
# )
# plt.plot(
#     nina_land.time,
#     (24 * nino_land.mean(dim=("west_east", "south_north")).cumsum()[variable]),
#     label="El Nino - 2015/2016",
# )
# plt.plot(
#     nina_land.time,
#     (24 * neutral_land.mean(dim=("west_east", "south_north")).cumsum()[variable]),
#     label="Neutral - 2016/2017",
# )
# locs, labels = plt.xticks()
# plt.xticks(locs, ["11-01", "11-15", "12-01", "12-15", "01-01", "01-15", "02-01"])
# plt.grid()
# plt.legend()
# # plt.tight_layout()
# plt.ylabel(f"cumulative {variable} over land in mm")
# plt.savefig(f"rc_trop_ls_MM_{variable}_cumsum_line_nino-nina-neutral.png")
# plt.show()
#
# # zonal classification by altitude
# path = basepath / "rc_trop_ls_MM_d02km_static_hgt.nc"
# landmask_path = basepath / "rc_trop_ls_MM_d02km_static_landmask.nc"
# hgt = darwin.open_dataset(from_path=path)
# landmask = darwin.open_dataset(from_path=landmask_path)
# hgt_land = hgt.copy()
# hgt_land["hgt"].values = np.where(landmask.landmask, hgt.hgt, np.nan)
# hgt_land["hgt"].values = np.where(hgt.hgt < 900, 2, hgt_land.hgt)
# hgt_land["hgt"].values = np.where(hgt.hgt < 100, 1, hgt_land.hgt)
# hgt_land["hgt"].values = np.where(hgt.hgt >= 900, 3, hgt_land.hgt)
# hgt_land["hgt"].values = np.where(landmask.landmask, hgt_land.hgt, np.nan)
# hgt_land.salem.quick_map("hgt")
# plt.show()
#
#
# zone = 1  # lowlands z < 100 m
# nina_copy = nina.copy()
# mask = xr.where(hgt_land.hgt == 1, 1, np.nan, keep_attrs=True)
# nina_zone1 = nina_copy * mask.values
#
# nina_zone1.mean(dim=("south_north", "west_east"))
#
# ds = darwin.Experiment(nina_zone1)
# plot_map(ds, name=f"{enso_event}_zone_{zone}", remove_boundaries=False)
#
#
# # In[93]:
#
#
# mask[0].mean(dim=("south_north", "west_east"))


# use hgt_land to mask nina
def mask(ds: xr.Dataset, mask: xr.DataArray, var: str) -> xr.DataArray:
    """Mask a dataset with another dataset.

    Args:
        ds: Dataset to mask.
        mask: Mask dataset.
        var: Variable to mask.

    Returns:
        Masked dataset.
    """
    return ds[var] * mask.values


def define_hgt_zone(hgt: xr.Dataset, altitudes: tuple) -> xr.Dataset:
    """Define an altitude/height zone.

    Args:
        hgt: Dataset with height information.
        zone: Altitude zone to mask.
        altitudes: Altitude range to mask.

    Returns:
        Mask with 1 for the altitude zone and NaN for the rest.
    """
    return xr.where(
        (hgt >= altitudes[0]) & (hgt < altitudes[1]),
        1,
        np.nan,
        keep_attrs=True,
    )


def mask_by_hgt_zone(
    ds: xr.Dataset,
    hgt: xr.Dataset,
    altitudes: tuple,
    variables: Optional[DataVariables] = None,
) -> xr.Dataset:
    """Mask a dataset with a height zone.

    Args:
        ds: Dataset to mask.
        hgt: Dataset with height information.
        zone: Altitude zone to mask.
        altitudes: Altitude range to mask.

    Returns:
        Masked dataset.
    """
    ds = ds.copy()
    mask_ = define_hgt_zone(hgt, altitudes)
    if not variables:
        variables = ds.data_vars
    for var in variables.keys():
        ds[var] = mask(ds, mask_.hgt, var)
        print(ds[var].mean(dim=("south_north", "west_east")))
    return ds


def get_mean_timeseries(ds: xr.Dataset) -> xr.Dataset:
    """Get the mean timeseries of a dataset.

    Args:
        ds: Dataset to get the mean timeseries from.

    Returns:
        Mean timeseries.
    """
    return ds.mean(dim=("south_north", "west_east"))


def plot_timeseries(
    ds: xr.Dataset,
    label: str,
    plot_type: Optional[str] = "line",
    xticks: Optional[list] = None,
    yticks: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """Plot a timeseries.

    Args:
        ds: Dataset to plot.
        label: Label for the plot.
        xticks: X ticks for the plot.
        yticks: Y ticks for the plot.
        **kwargs: Keyword arguments for the plot.
    """
    if ax:
        if plot_type == "bar":
            ax.bar(ds.time, ds, label=label, **kwargs)
        elif plot_type == "scatter":
            ax.scatter(ds.time, ds, label=label, **kwargs)
        else:
            ax.plot(ds.time, ds, label=label, **kwargs)
        if xticks:
            locs, labels = plt.xticks()
            ax.xticks(locs, xticks)
        if yticks:
            locs, labels = plt.yticks()
            ax.yticks(yticks)
        ax.grid()
        ax.legend()
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(base=1))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    elif plot_type == "bar":
        plt.bar(ds.time, ds, label=label, **kwargs)
    elif plot_type == "scatter":
        plt.scatter(ds.time, ds, label=label, **kwargs)
    else:
        plt.plot(ds.time, ds, label=label, **kwargs)
    if xticks:
        locs, labels = plt.xticks()
        plt.xticks(locs, xticks)
    if yticks:
        locs, labels = plt.yticks()
        plt.yticks(yticks)
    plt.grid()
    plt.legend()


def concatenate(files: list, ofile: Optional[str] = None) -> xr.Dataset:
    """Concatenate a list of files via cdo concatenate.

    Args:
        files: List of files to concatenate.
        ofile: Path to output file if yu want to save.

    Returns:
        Concatenated dataset.
    """
    cdo = Cdo()
    return cdo.mergetime(input=files, returnXDataset=True)


def get_nino_index(index: str) -> xr.Dataset:
    """Get the nino index.

    The data source is the NOAA Working group on Surface Pressure (
        https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino12/
    )

    Args:
        index: Index to get.

    Returns:
        Nino index.
    """
    df = pd.read_csv("nino12index.csv")
    df_long = pd.melt(df, id_vars=["year"], var_name="month", value_name="value")
    df_long["time"] = pd.to_datetime(df_long["year"].astype(str) + "-" + df_long["month"].astype(str))
    return xr.Dataset.from_dataframe(df_long[["value"]].set_index(df_long["time"])).sortby("time")