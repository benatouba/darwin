import copy
import datetime
from   glob import glob
from   pathlib import Path

from   matplotlib import pyplot as plt
from   metpy.calc import relative_humidity_from_mixing_ratio
from   metpy.units import units
import numpy as np
import pandas as pd
from   pytz import timezone
import salem
import xarray as xr

from   constants import basepath as gar_basepath, coordinates
from   utils import glob_files

# from pyproj import Proj
# from windrose import WindroseAxes


class FilePath(type(Path())):
    """FilePath."""

    def __init__(self, *args):
        """Initiate a FilePath instance from a str."""
        super().__init__()
        self.a = args[0] if args else ""
        if self.suffix in [".nc", ".nc4", ".netcdf"]:
            self.__assign_wrf_infos(self.stem.split("_"))

    def __assign_wrf_infos(self, file_infos: list):
        if "static" not in self.stem:
            self.year = file_infos.pop()
        if "flux" not in self.stem:
            self.var = file_infos.pop()
        else:
            self.var = file_infos.pop() + "_" + file_infos.pop()
        if "3d" not in file_infos and "static" not in self.stem:
            self.dimensionality = file_infos.pop()
        elif "static" not in self.stem:
            self.dimensionality = f"{file_infos.pop()}_{file_infos.pop()}"
        self.frequency = file_infos.pop()
        self.domain = file_infos.pop()
        self.experiment = "_".join(file_infos)


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
                f"{basepath}/{experiment}/{experiment}*_{frequency}_*{variable}*{year}.nc*"
            )[0]
            gar_ds = xr.open_dataset(file, **kwargs)
    # NOTE: salem does something weird and adds pyproj_srs during load.
    # It is recommended not to use it.
    elif engine == "salem":
        if from_path:
            file = str(from_path)
            gar_ds = salem.open_xr_dataset(from_path, **kwargs)
        else:
            path = f"{basepath}/{experiment}/{experiment}*_{frequency}_*{variable}*{year}.nc*"
            file = glob_files(path)[0]
            if not file:
                raise FileNotFoundError(
                    f"No file found for {path}. Did you mean to use the 'from_path' argument?"
                )
            gar_ds = salem.open_xr_dataset(file, **kwargs)
        gar_ds.attrs["experiment"] = str(experiment)
        gar_ds.attrs["year"] = str(year)
    else:
        raise ValueError("Engine type not supported.")

    split = file.split("/")[-1].split("_")
    var = split[-2]
    if var == "lu":
        var = split[-2] + "_" + split[-1].split(".")[0]
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

    def __getitem__(self, key):
        """Get a item.

            self (): The object.
            key (): Object key.

        Returns:
            The object with name key.
        """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Set an item.

        self (): The object
        key (): Name under which to store the item.
        value (): The item to store.
        """
        setattr(self, key, value)

    def copy(self):
        return copy.copy(self)

    def remove_boundaries(self, grid_points):
        """Crop the outer rows/columns of the model data."""
        copy = self.copy()
        copy.wrf_product = self.wrf_product.isel(
            west_east=slice(grid_points, -grid_points),
            south_north=slice(grid_points, -grid_points),
        )
        return copy

    def add_extracted_simulated_points_from_file(self, file):
        """Load measurements from file."""
        extracted = pd.read_csv(file, parse_dates=["datetime"], index_col=["datetime"])
        path = file.rsplit("/")[-1].split(".")[0].split("_")
        if self.__translate_varname(self.varname) not in path:
            raise ValueError(
                f"""
                The name of the selected file does not seem to contain the variable.
                {path}
                Please name it in style of *_variable_*
                """
            )

        extracted.attrs["name"] = self.varname
        self.extracted = extracted

    def add_product(self, variable=None, wrf_ds=None):
        """Add another variable to the Experiment."""
        if wrf_ds:
            self.__added_variables.append(wrf_ds)
            return
        wrf_ds = open_dataset(self.experiment, variable, self.year)
        if wrf_ds.VARNAME != self.varname:
            self.__added_variables.append(wrf_ds.VARNAME)

    def __translate_varname(self, varname):
        print("Getting variable name translations")
        varname = varname.lower()
        variable_names = [
            ["prcp", "PCP"],
            ["hgt"],
            ["et"],
            ["potevap"],
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
        print("Adding measurements to dataset")
        if not variable:
            variable = self.varname
        self.measurements_files = glob_measurements()
        for file in self.measurements_files:
            measured = load_measurements(file, variable)
            self.measurements[measured.attrs["name"]] = measured

    def plot_station(self, station, sample_rate="D", save=False, **kwargs):
        """Plot timeseries data of a specific station of the darwin network."""
        timeseries = {}
        data = self.measurements[station]
        # Check for multiple measurement types of precipitation
        if self.varname == "PCP":
            variables = [
                "PCP_diff_radar",
                "PCP_tot_bucket",
                "PCP_diffmin_radar",
                "PCP_acoustic",
            ]
        else:
            variables = [self.varname]
        _, ax = plt.subplots(figsize=(12, 10))
        # if isinstance(data, pd.DataFrame):
        #     names = data.columns
        if isinstance(data, pd.Series):
            data = data.to_frame(name=data.name)
        var = ""
        for var in data.columns:
            if var in variables and self.varname == "PCP":
                timeseries[var] = data[var].resample(sample_rate).sum()
                timeseries[var].cumsum().plot(ax=ax, **kwargs)
            elif var in variables:
                timeseries[var] = data[var].resample(sample_rate).mean()
                timeseries[var].plot(ax=ax, **kwargs)
        if hasattr(self, "extracted"):
            extracted = pd.Series(self.extracted[station])
            extracted.index = pd.DatetimeIndex(
                pd.date_range("2022-04-01", "2022-09-30")
            )
            extracted.name = self.experiment
        else:
            extracted = self.__get_closest_gridpoint_values(station)
        if var in variables and self.varname == "PCP":
            extracted.cumsum().plot(ax=ax, c="k", **kwargs)
        elif var in variables:
            extracted.plot(ax=ax, c="k", **kwargs)
        if var in variables:
            plt.legend()
            plt.title(station)
            plt.xlabel("")
        if save:
            plt.savefig(f"{self.varname.lower()}_{station.lower()}.png")

    def plot_stations(self, **kwargs):
        """Plot data of multiple stations using the 'plot_station'-method."""
        for station in self.measurements.keys():
            self.plot_station(station, sample_rate="D", **kwargs)

    def __find_closest_gridpoint(self, station):
        wrf_ds = self.wrf_product
        # for var in self.measurements.keys():
        #     df = self.measurements[var]
        #     break
        abslat = np.abs(wrf_ds.lat - coordinates[station][0])
        abslon = np.abs(wrf_ds.lon - coordinates[station][1])
        coords = np.maximum(abslon, abslat)
        return np.where(coords == np.min(coords))

    def __get_closest_gridpoint_values(self, station: str) -> pd.Series:
        ([xlon], [xlat]) = self.__find_closest_gridpoint(station)
        selected = self.wrf_product.isel(west_east=xlon, south_north=xlat)
        selected = pd.Series(selected[self.varname])
        selected.index = pd.DatetimeIndex(pd.date_range("2022-04-01", "2022-09-30"))
        selected.name = self.wrf_product.attrs["experiment"]
        return selected

    def plot_map(
        self,
        varname=None,
        ax=None,
        aggregation="mean",
        save=False,
        stations=True,
        cbar=True,
        unit="",
        cmap=None,
        **kwargs,
    ):
        """
        Plot a map of Experiment data.

        kwargs: args for  salem's quick_map.
        """
        varname = varname or self.__translate_varname(self.varname)
        if aggregation == "mean":
            data = self.wrf_product.mean(dim="time", skipna=True, keep_attrs=True)
        elif aggregation == "sum":
            data = self.wrf_product.sum(dim="time", skipna=True, keep_attrs=True)
        else:
            raise NameError("Aggregation not implemented")
        base_map = data.salem.get_map(**kwargs)
        if self.varname in ["prcp", "et", "potevap"]:
            data[self.varname] = data[self.varname] * 24
        base_map.set_data(data[self.varname])
        if stations:
            for key, value in coordinates.items():
                lat, lon = value
                base_map.set_points(lon, lat, text=key)
        base_map.set_cmap(cmap)
        base_map.set_scale_bar()
        if ax:
            base_map.plot(ax=ax)
            if cbar:
                base_map.append_colorbar(ax=ax, label=unit)
        else:
            base_map.visualize(add_cbar=True)
        if save:
            plt.savefig(
                f"{self.experiment}_{self.varname.lower()}_{aggregation.lower()}_map.png"
            )
        return base_map

    def set_units(self) -> xr.DataArray:
        """Add units to xarray data."""
        ds = self.wrf_product
        var = self.varname
        ds[var] = ds[var] * units(ds[var].attrs["units"])
        return self.wrf_product


def compute_rh(
    mixing_ratio: xr.Dataset, temperature: xr.Dataset, pressure: xr.Dataset
) -> xr.Dataset:
    """Compute relative humidity from mixing ratio."""
    psfc = pressure.psfc * units("mbar")
    temperature = temperature.t2 * units("K")
    mixing_ratio = mixing_ratio.q2 * units("kg/kg")
    rel_humidity = relative_humidity_from_mixing_ratio(
        pressure=psfc,
        temperature=temperature,
        mixing_ratio=mixing_ratio,
    )
    return xr.Dataset(
        rel_humidity,
        coords=temperature.coords,
        attrs=temperature.attrs,
    )


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
    return glob(f"{path}/{ds_number}_{type_of_data}*.csv")


def open_measurements(path):
    df = pd.read_csv(path)
    df.index = pd.DatetimeIndex(
        df.datetime, tz=timezone("Pacific/Galapagos")
    ).tz_convert(datetime.timezone.utc)
    return MeasurementFrame(df)


class MeasurementFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(MeasurementFrame, self).__init__(*args, **kwargs)

    def join_wrfdata(self, wrfdata, inplace=False, join='inner'):
        return pd.concat([self.data, wrfdata], axis=1, join=join)


def load_measurements(path: [str, Path], variable: str = "all"):
    """
    Load measured data from darwin measurement network.

    Parameters
    ----------
    path : {str, Path-like object}
        Path to a csv-file containing measurements
    variable : {str, list}
        name(s) of the variable to load

    Returns
    ---------
    pandas.Series
        a pandas Series of the data in specified csv-file
    """
    measured = pd.read_csv(path, parse_dates=["datetime"], index_col=["datetime"])
    if variable == "all":
        return measured
    if variable == "prcp":
        filter_col = [
            col
            for col in measured
            if col in ["PCP_diff_radar", "PCP_acoustic", "PCP_tot_bucket", "PCP"]
        ]
    else:
        filter_col = variable
    measured.attrs["name"] = path.split("/")[-1].split("_")[-3]
    measured.attrs["coordinates"] = [
        value
        for key, value in coordinates.items()
        if key.rsplit("_", maxsplit=1)[-1] in measured.attrs["name"].lower()
    ]
    return measured[filter_col]
