from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import salem
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point

coordinates = {
    "minasrojas": (-0.618625, -90.3673),
    "militar": (-0.489962, -90.2808),
    "puertoayora": (-0.743708, -90.3027),
    "puertovillamil": (-0.946400, -90.9741),
    "puertobacceriomoreno": (-0.89515, -89.6068),
    "eljunco": (-0.893768, -89.4804),
    "lagalapaguera": (-0.91197, -89.4387),
    "cuevadesucre": (-0.843216, -91.0284),
    "negra": (-0.848344, -91.1312),
    "crocker": (-0.642398, -90.326),
    "rosa": (-0.65453, -90.4035),
    # "met-e_bellavista": (-0.692384, -90.3282),
    # "met-e_puerto_ayora": (-0.743708, -90.3027)
}

color_map = LinearSegmentedColormap.from_list(
    "temperature",
    ["white", "steelblue", "c", "khaki", "orange", "orangered", "r", "darkred"],
)


def remove_nonalphanumerics(string: str) -> str:
    return "".join(ch for ch in string if ch.isalnum())


def open_dataset(
    experiment,
    variable,
    year,
    domain="d02",
    dimensions="2d",
    frequency="d",
    base_folder="/home/ben/data/GAR/",
    **kwargs,
):
    file = glob(
        f"{base_folder}/{experiment}/products/{domain}/{frequency}/{dimensions}/{experiment}*{variable}*{year}.nc*"
    )[0]
    ds = xr.open_dataset(file, decode_cf=False, **kwargs)
    ds = ds.isel(west_east=slice(10, -10), south_north=slice(10, -10))
    split = file.split("/")[-1].split("_")
    var = split[-2]
    if var == "lu":
        var = split[-2] + "_" + split[-1].split(".")[0]
    # TODO: Implement other projections
    projection = {
        "lat_0": float(ds.attrs["PROJ_ENVI_STRING"].split(",")[3]),
        "lon_0": float(ds.attrs["PROJ_ENVI_STRING"].split(",")[4]),
        "x_0": float(ds.attrs["PROJ_ENVI_STRING"].split(",")[5]),
        "y_0": float(ds.attrs["PROJ_ENVI_STRING"].split(",")[5]),
        "ellps": remove_nonalphanumerics(ds.attrs["PROJ_ENVI_STRING"].split(",")[7]),
        "name": remove_nonalphanumerics(
            str(ds.attrs["PROJ_ENVI_STRING"].split(",")[8])
        ),
    }
    if projection["name"].lower() == "wrfmercator":
        proj = "merc"
    pyproj_srs = (
        f"+proj={proj} +lat_0={str(projection['lat_0'])} +lon_0={str(projection['lon_0'])} +k=1 "
        f"+x_0={str(projection['x_0'])} +y_0={str(projection['y_0'])} +ellps={projection['ellps']} "
        f"+datum={projection['ellps']} +units=m +no_defs"
    )
    ds[var].attrs["pyproj_srs"] = pyproj_srs
    ds[var].attrs["projection_info"] = projection
    ds.attrs["experiment"] = experiment
    return ds


def open_experiment(**kwargs):
    """
    kwargs = arguments passed to open_dataset
    """
    return Experiment(open_dataset(**kwargs))


class Experiment:
    def __init__(self, ds):
        self.wrf_product = ds
        self.variable = self.__rename_wrf_to_measured()
        self.__add_measurements()

    def __rename_wrf_to_measured(self):
        if self.wrf_product.VARNAME == "prcp":
            return "PCP"
        elif self.wrf_product.VARNAME == "t2":
            return "T"
        elif self.wrf_product.VARNAME == "q2":
            return "RH"
        elif self.wrf_product.VARNAME == "ws10":
            return "WS"
        else:
            raise "Please add variable name translation"

    def __add_measurements(self):
        self.measurements_files = glob_measurements()
        self.measurements = {}
        for f in self.measurements_files:
            ds = load_measurements(f, self.variable)
            self.measurements[ds.attrs["name"]] = ds

    def plot_station(self, station, sample_rate="D", **kwargs):
        timeseries = {}
        data = self.measurements[station]
        # Check for multiple measurement types of precipitation
        if self.variable == "PCP":
            variables = ["PCP_diff_radar", "PCP_diffmin_radar", "PCP_acoustic"]
        else:
            variables = [self.variable]
        fig, ax = plt.subplots(figsize=(12, 10))
        for var in data.columns:
            if var in variables and self.variable == "PCP":
                timeseries[var] = data[var].resample(sample_rate).sum()
                timeseries[var].cumsum().plot(ax=ax, **kwargs)
            elif var in variables:
                timeseries[var] = data[var].resample(sample_rate).mean()
                timeseries[var].plot(ax=ax, **kwargs)
        closest_gridpoint = self.__get_closest_gridpoint_values(station)
        if var in variables and self.variable == "PCP":
            closest_gridpoint = closest_gridpoint * 24
            closest_gridpoint.cumsum().plot(ax=ax)
        elif var in variables:
            closest_gridpoint.plot(ax=ax)
        if var in variables:
            plt.legend()
            plt.title(station)
            plt.xlabel("")

    def plot_stations(self, **kwargs):
        for station in self.measurements.keys():
            self.plot_station(station, sample_rate="D", **kwargs)

    def __find_closest_gridpoint(self, station):
        ds = self.wrf_product
        # for var in self.measurements.keys():
        #     df = self.measurements[var]
        #     break
        abslat = np.abs(ds.lat - coordinates[station][0])
        abslon = np.abs(ds.lon - coordinates[station][1])
        c = np.maximum(abslon, abslat)
        return np.where(c == np.min(c))

    def __get_closest_gridpoint_values(self, station):
        ([xlon], [xlat]) = self.__find_closest_gridpoint(station)
        selected = self.wrf_product.isel(west_east=xlon, south_north=xlat)
        selected = pd.Series(selected[self.wrf_product.VARNAME])
        selected.index = pd.DatetimeIndex(pd.date_range("2022-04-01", "2022-06-30"))
        selected.name = self.wrf_product.attrs["experiment"]
        return selected

    def plot_map(self, ax=None, aggregation="mean", **kwargs):
        """
        kwargs: args for  salem's quick_map
        """
        if "cmap" not in locals() or "cmap" not in globals():
            cmap = color_map
        if aggregation == "mean":
            data = self.wrf_product[self.wrf_product.VARNAME].mean(
                dim="time", skipna=True, keep_attrs=True
            )
        elif aggregation == "sum":
            data = self.wrf_product[self.wrf_product.VARNAME].sum(
                dim="time", skipna=True, keep_attrs=True
            )
        else:
            raise "aggregation not implemented"
        base_map = data.salem.get_map(**kwargs)
        base_map.set_data(data)
        lats = []
        lons = []
        stations = []
        points = []
        for key, value in coordinates.items():
            (lat, lon) = value
            lats.append(lat)
            lons.append(lon)
            stations.append(key)
            points.append(Point(lon, lat))
            base_map.set_points(
                lon,
                lat,
                text=key,
            )
        base_map.set_cmap(cmap)
        if ax:
            base_map.plot(ax=ax)
        else:
            base_map.visualize()
        return base_map


def plot_pcp(data, title=None, sample_rate="D", **kwargs):
    timeseries = {}
    variables = ["PCP_diff_radar", "PCP_diffmin_radar", "PCP_acoustic"]
    fig, ax = plt.subplots()
    for var in data.columns:
        if var in variables:
            timeseries[var] = data[var].resample(sample_rate).sum()
            timeseries[var].cumsum().plot(ax=ax)
    if var in variables:
        plt.legend()
        if title:
            plt.title(title)
        plt.xlabel("")


def glob_measurements(path="/home/ben/data", ds_number="??"):
    files = glob(f"{path}/darwin_measured/{ds_number}_AWS*/*[!xlsx_complete]")
    return [file for file in files if not file.count("xlsx") and not file.count("_-_")]


def load_measurements(path, variable):
    ds = pd.read_csv(path, parse_dates=["datetime"], index_col=["datetime"])
    ds = ds.loc["2022-04-01":"2022-06-30"]
    if variable == "PCP":
        filter_col = [col for col in ds if col.startswith(variable)]
    ds.attrs["name"] = path.split("/")[-1].split("_")[-3]
    ds.attrs["coordinates"] = [
        value
        for key, value in coordinates.items()
        if key.split("_")[-1] in ds.attrs["name"].lower()
    ]
    return ds[filter_col]


# plt.savefig(f'pcp_{title}.png')
