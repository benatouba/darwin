from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import salem
import xarray as xr
from metpy.calc import relative_humidity_from_mixing_ratio
from metpy.units import units
from pyproj import Proj
from windrose import WindroseAxes

from constants import basepath as gar_basepath
from constants import color_map, coordinates
from utils import glob_files, transform_k_to_c


class FilePath(type(Path())):
    """FilePath."""

    def __init__(self, *args):
        """Initiate a FilePath instance from a str."""
        super().__init__()
        self.a = args[0] if args else ""
        if self.suffix.lower() in [".nc", ".nc4", ".netcdf"]:
            self.__assign_file_infos(self.stem.split("_"))

    def __assign_file_infos(self, file_infos: list):
        if "static" not in self.stem:
            self.year = file_infos.pop()
        self.var = file_infos.pop()
        if "3d" not in file_infos:
            self.dimensionality = file_infos.pop()
        else:
            self.dimensionality = f"{file_infos.pop()}_{file_infos.pop()}"
        self.frequency = file_infos.pop()
        self.project = "_".join(file_infos)


gar_basepath = FilePath(gar_basepath)


def open_dataset(
    experiment=None,
    variable=None,
    year=None,
    domain="d02",
    dimensions="2d",
    frequency="d",
    from_path=None,
    basepath=gar_basepath,
    engine="salem",
    **kwargs,
):
    """Open a Dataset."""
    if engine == "xarray":
        if from_path:
            file = from_path
            gar_ds = xr.open_dataset(from_path, **kwargs)
        else:
            file = glob_files(
                f"{basepath}/{experiment}/products/{domain}/{frequency}/{dimensions}/"
                f"{experiment}*{variable}*{year}.nc*"
            )[0]
            gar_ds = xr.open_dataset(file, **kwargs)
        gar_ds.attrs["experiment"] = str(experiment)
        gar_ds.attrs["year"] = str(year)
    elif engine == "salem":
        if from_path:
            file = from_path
            gar_ds = salem.open_xr_dataset(from_path, **kwargs)
        else:
            file = glob_files(
                f"{basepath}/{experiment}/products/{domain}/{frequency}/{dimensions}/"
                f"{experiment}*{variable}*{year}.nc*"
            )[0]
            gar_ds = salem.open_xr_dataset(file, **kwargs)
        gar_ds.attrs["experiment"] = str(experiment)
        gar_ds.attrs["year"] = str(year)
    else:
        raise ValueError("Engine type not supported.")

    split = file.split("/")[-1].split("_")
    var = split[-2]
    if var == "lu":
        var = split[-2] + "_" + split[-1].split(".")[0]
    # projection = {
    #     "lat_0": float(ds.attrs["PROJ_ENVI_STRING"].split(",")[3]),
    #     "lon_0": float(ds.attrs["PROJ_ENVI_STRING"].split(",")[4]),
    #     "x_0": float(ds.attrs["PROJ_ENVI_STRING"].split(",")[5]),
    #     "y_0": float(ds.attrs["PROJ_ENVI_STRING"].split(",")[5]),
    #     "ellps": remove_nonalphanumerics(ds.attrs["PROJ_ENVI_STRING"].split(",")[7]),
    #     "name": remove_nonalphanumerics(
    #         str(ds.attrs["PROJ_ENVI_STRING"].split(",")[8])
    #     ),
    # }
    #
    # proj = "merc" if projection["name"].lower() == "wrfmercator" else "lcc"
    # pyproj_srs = (
    #     f"+proj={proj} +lat_0={str(projection['lat_0'])} +lon_0={str(projection['lon_0'])} +k=1 "
    #     f"+x_0={str(projection['x_0'])} +y_0={str(projection['y_0'])} "
    #     f"+ellps={projection['ellps']} +datum={projection['ellps']} +units=m +no_defs"
    # )
    #
    # ds[var].attrs["pyproj_srs"] = pyproj_srs
    # ds[var].attrs["projection_info"] = projection
    return gar_ds


def open_experiment(**kwargs):
    """
    Open an Experiment for GAR.

    kwargs = arguments passed to open_dataset.
    """
    return Experiment(open_dataset(**kwargs))


class Experiment:
    """Class to hold all data associated with one variable for one run."""

    def __init__(self, gar_ds):
        """Initiate an instance of class Experiment."""
        self.wrf_product = gar_ds
        self.varname = gar_ds.VARNAME
        self.varname_translations = self.__translate_varname(gar_ds.VARNAME)
        try:
            self.experiment = gar_ds.attrs["experiment"]
            self.year = gar_ds.attrs["year"]
        except KeyError:
            print("WARNING: Could not add experiment and/or year.")
        self.measurements = {}
        try:
            self.__add_measurements()
        except KeyError:
            Warning("Measurements could not be found")

    def remove_boundaries(self, gar_ds):
        """Crop the outer 10 rows/columns of the model data."""
        self.wrf_product[gar_ds.VARNAME] = gar_ds.isel(
            west_east=slice(10, -10), south_north=slice(10, -10)
        )

    def add_extracted_simulated_points_from_file(self, file):
        """Load measurements from file."""
        extracted = pd.read_csv(file, parse_dates=["datetime"], index_col=["datetime"])
        path = file.rsplit("/")[-1].split(".")[0].split("_")
        if self.__translate_varname(self.variable) not in path:
            raise ValueError(
                f"""
                The name of the selected file does not seem to contain the variable.
                {path}
                Please name it in style of *_variable_*
                """
            )

        extracted.attrs["name"] = self.variable
        self.extracted = extracted

    def add_product(self, variable=None, wrf_ds=None):
        """Add another variable to the Experiment."""
        if wrf_ds:
            self.wrf_product[wrf_ds.VARNAME] = wrf_ds
            return
        wrf_ds = open_dataset(self.experiment, variable, self.year)
        if wrf_ds.VARNAME not in self.wrf_product.keys():
            self.wrf_product[wrf_ds.VARNAME] = wrf_ds
            self.__added_variables.append(wrf_ds.VARNAME)

    def __translate_varname(self, varname):
        varname = varname.lower()
        variable_names = [
            ["prcp", "PCP"],
            ["et"],
            ["t2", "T"],
            ["q2", "q"],
            ["rh", "RH"],
            ["ws10", "ws"],
            ["press", "P"],
        ]
        for combination in variable_names:
            if varname in combination:
                return combination
        raise NameError("Please add variable name translation")

    def __add_measurements(self, variable=None):
        if not variable:
            variable = self.variable
        self.measurements_files = glob_measurements()
        for file in self.measurements_files:
            measured = load_measurements(file, variable)
            self.measurements[measured.attrs["name"]] = measured

    def compute_rh(self):
        """Compute relative humidity from mixing ratio."""
        if self.variable.lower() not in ["q", "q2", "qv", "rh"]:
            raise NameError("Can only be calculated from any of 'q', 'qv', 'q2', 'RH'")
        if self.variable.lower() in ["rh", "q2"] and "t2" not in self.__added_variables:
            self.add_product("t2")
        if (
            self.variable.lower() in ["rh", "q2"]
            and "psfc" not in self.__added_variables
        ):
            self.add_product("psfc")
        rel_humidity = self.wrf_product["q2"].copy()
        psfc = self.wrf_product["psfc"].psfc * units("mbar")
        temperature = (self.wrf_product["t2"].temperature + 273.15) * units("K")
        mixing_ratio = self.wrf_product["q2"].mixing_ratio * units("kg/kg")
        rel_humidity["rh"] = relative_humidity_from_mixing_ratio(
            pressure=psfc,
            temperature=temperature,
            mixing_ratio=mixing_ratio,
        )
        return xr.Dataset(
            rel_humidity,
            coords=self.wrf_product["q2"].coords,
            attrs=self.wrf_product["q2"].attrs,
        )

    def plot_station(self, station, sample_rate="D", save=False, **kwargs):
        """Plot timeseries data of a specific station of the darwin network."""
        timeseries = {}
        data = self.measurements[station]
        # Check for multiple measurement types of precipitation
        if self.variable == "PCP":
            variables = [
                "PCP_diff_radar",
                "PCP_tot_bucket",
                "PCP_diffmin_radar",
                "PCP_acoustic",
            ]
        else:
            variables = [self.variable]
        _, ax = plt.subplots(figsize=(12, 10))
        # if isinstance(data, pd.DataFrame):
        #     names = data.columns
        if isinstance(data, pd.Series):
            data = data.to_frame(name=data.name)
        var = ""
        for var in data.columns:
            if var in variables and self.variable == "PCP":
                timeseries[var] = data[var].resample(sample_rate).sum()
                timeseries[var].cumsum().plot(ax=ax, **kwargs)
            elif var in variables:
                timeseries[var] = data[var].resample(sample_rate).mean()
                timeseries[var].plot(ax=ax, **kwargs)
        if hasattr(self, "extracted"):
            extracted = pd.Series(self.extracted[station])
            extracted.index = pd.DatetimeIndex(
                pd.date_range("2022-04-01", "2022-06-30")
            )
            extracted.name = self.experiment
        else:
            extracted = self.__get_closest_gridpoint_values(station)
        if var in variables and self.variable == "PCP":
            extracted.cumsum().plot(ax=ax, c="k", **kwargs)
        elif var in variables:
            extracted.plot(ax=ax, c="k", **kwargs)
        if var in variables:
            plt.legend()
            plt.title(station)
            plt.xlabel("")
        if save:
            plt.savefig(f"{self.variable.lower()}_{station.lower()}.png")

    def plot_stations(self, **kwargs):
        """Plot data of multiple stations using the 'plot_station'-method."""
        for station in self.measurements.keys():
            self.plot_station(station, sample_rate="D", **kwargs)

    def __find_closest_gridpoint(self, station):
        wrf_ds = self.wrf_product[self.variable[0]]
        # for var in self.measurements.keys():
        #     df = self.measurements[var]
        #     break
        abslat = np.abs(wrf_ds.lat - coordinates[station][0])
        abslon = np.abs(wrf_ds.lon - coordinates[station][1])
        coords = np.maximum(abslon, abslat)
        return np.where(coords == np.min(coords))

    def __get_closest_gridpoint_values(self, station: str) -> pd.Series:
        ([xlon], [xlat]) = self.__find_closest_gridpoint(station)
        selected = self.wrf_product[self.variable[0]].isel(
            west_east=xlon, south_north=xlat
        )
        selected = pd.Series(selected[self.wrf_product[self.variable].VARNAME])
        selected.index = pd.DatetimeIndex(pd.date_range("2022-04-01", "2022-06-30"))
        selected.name = self.wrf_product[self.variable[0]].attrs["experiment"]
        return selected

    def plot_map(
        self, variable=None, ax=None, aggregation="mean", save=False, **kwargs
    ):
        """
        Plot a map of Experiment data.

        kwargs: args for  salem's quick_map.
        """
        variable = variable or self.__translate_varname(self.variable)
        cmap = color_map if "cmap" not in locals() or "cmap" not in globals() else None
        if aggregation == "mean":
            data = self.wrf_product[variable][self.wrf_product[variable].VARNAME].mean(
                dim="time", skipna=True, keep_attrs=True
            )

        elif aggregation == "sum":
            data = self.wrf_product[variable][self.wrf_product[variable].VARNAME].sum(
                dim="time", skipna=True, keep_attrs=True
            )
        else:
            raise NameError("aggregation not implemented")
        # try:
        #     base_map = data.salem.get_map(**kwargs)
        # except RuntimeError:
        #     grid = salem.Grid(Proj(self.wrf_product[variable].attrs['PROJ_ENVI_STRING']))
        #     print(grid)
        grid = salem.mercator_grid(
            (
                self.wrf_product[variable].attrs["LON_0"],
                self.wrf_product[variable].attrs["LAT_0"],
            )
        )
        base_map = data.salem.get_map(grid=grid, **kwargs)
        base_map.set_data(data)
        for key, value in coordinates.items():
            lat, lon = value
            base_map.set_points(lon, lat, text=key)
        base_map.set_cmap(cmap)
        base_map.set_scale_bar()
        if ax:
            base_map.plot(ax=ax)
            base_map.append_colorbar(ax=ax)
        else:
            base_map.visualize(add_cbar=True)
        if save:
            plt.savefig(f"{self.variable.lower()}_{aggregation.lower()}_map.png")
        return base_map


def plot_pcp(data, title=None, sample_rate="D", **kwargs):
    """Plot precipitation timeseries from the darwin measurement network."""
    timeseries = {}
    variables = ["PCP_diff_radar", "PCP_diffmin_radar", "PCP_acoustic"]
    var = ""
    for var in data.columns:
        if var in variables:
            timeseries[var] = data[var].resample(sample_rate).sum()
            timeseries[var].cumsum().plot(**kwargs)
    if var in variables:
        plt.legend()
        if title:
            plt.title(title)
        plt.xlabel("")


def glob_measurements(
    path="/home/ben/data/darwin_measured", ds_number="??", type_of_data="AWS-P"
):
    """Glob for measurement data of the darwin measurement network."""
    files = glob(f"{path}/{ds_number}_{type_of_data}*/*[!xlsx_complete]")
    return [file for file in files if not file.count("xlsx") and not file.count("_-_")]


def load_measurements(path, variable):
    """
    Load measured data from darwin measurement network.

    Parameters
    ----------
    path : str
        Path to a csv-file containing measurements
    variable : {str, list}
        name(s) of the variable to load

    Returns
    ---------
    pandas.Series
        a pandas Series of the data in specified csv-file
    """
    measured = pd.read_csv(path, parse_dates=["datetime"], index_col=["datetime"])
    measured = measured.loc["2022-04-01":"2022-06-30"]
    if variable == "PCP":
        filter_col = [col for col in measured if col in variable]
    else:
        filter_col = variable
    measured.attrs["name"] = path.split("/")[-1].split("_")[-3]
    measured.attrs["coordinates"] = [
        value
        for key, value in coordinates.items()
        if key.rsplit("_", maxsplit=1)[-1] in measured.attrs["name"].lower()
    ]
    return measured[filter_col]
