from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import salem
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from metpy.calc import relative_humidity_from_mixing_ratio
from metpy.units import units
from windrose import WindroseAxes
from utils import glob_files, remove_nonalphanumerics, transform_k_to_c

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


class FilePath(type(Path())):
    def __init__(self, *args):
        super().__init__()
        self.a = args[0] if args else ''
        if self.suffix.lower() in [".nc", ".nc4", ".netcdf"]:
            self.__assign_file_infos(self.stem.split("_"))

    def __assign_file_infos(self, file_infos: list):
        if "static" not in self.stem:
            self.year = self.file_infos.pop()
        self.var = self.file_infos.pop()
        if "3d" not in self.file_infos:
            self.dimensionality = self.file_infos.pop()
        else:
            self.dimensionality = f"{self.file_infos.pop()}_{self.file_infos.pop()}"
        self.frequency = self.file_infos.pop()
        self.project = "_".join(self.file_infos, "_")


basepath = FilePath("/home/ben/data/GAR/")


def open_dataset(
    experiment=None,
    variable=None,
    year=None,
    domain="d02",
    dimensions="2d",
    frequency="d",
    from_path=None,
    basepath=basepath,
    engine="salem",
    **kwargs,
):
    if engine == "xarray":
        if from_path:
            file = from_path
            ds = xr.open_dataset(from_path, decode_cf=False, **kwargs)
        else:
            file = glob_files(
                f"{basepath}/{experiment}/products/{domain}/{frequency}/{dimensions}/"
                f"{experiment}*{variable}*{year}.nc*"
            )[0]
            ds = xr.open_dataset(file, decode_cf=False, **kwargs)
    elif engine == "salem":
        if from_path:
            file = from_path
            ds = salem.open_xr_dataset(from_path, **kwargs)
        else:
            file = glob_files(
                f"{basepath}/{experiment}/products/{domain}/{frequency}/{dimensions}/"
                f"{experiment}*{variable}*{year}.nc*"
            )[0]
            ds = salem.open_xr_dataset(file, **kwargs)
        ds.attrs["experiment"] = experiment
        ds.attrs["year"] = year
    else:
        raise ValueError("Engine type not supported.")
    ds = ds.isel(west_east=slice(10, -10), south_north=slice(10, -10))
    if ds.VARNAME.lower() in ["t2", "t"]:
        ds[ds.VARNAME].data = transform_k_to_c(ds)

    split = file.split("/")[-1].split("_")
    print(split)
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
    #     f"+x_0={str(projection['x_0'])} +y_0={str(projection['y_0'])} +ellps={projection['ellps']} "
    #     f"+datum={projection['ellps']} +units=m +no_defs"
    # )
    #
    # ds[var].attrs["pyproj_srs"] = pyproj_srs
    # ds[var].attrs["projection_info"] = projection
    return ds


def open_experiment(**kwargs):
    """
    kwargs = arguments passed to open_dataset
    """
    return Experiment(open_dataset(**kwargs))


class Experiment:
    def __init__(self, ds):
        self.wrf_product = {ds.VARNAME: ds}
        self.variable = self.__translate_varname(ds.VARNAME)
        self.__added_variables = []
        self.experiment = ds.attrs["experiment"]
        self.year = ds.attrs["year"]
        self.__add_measurements()

    def add_product(self, variable=None, ds=None):
        if ds:
            self.wrf_product[ds.VARNAME] = ds
            return
        ds = open_dataset(self.experiment, variable, self.year)
        if ds.VARNAME not in self.wrf_product.keys():
            self.wrf_product[ds.VARNAME] = ds
            self.__added_variables.append(ds.VARNAME)

    def __translate_varname(self, varname):
        varname = varname.lower()
        if varname == "prcp":
            return "PCP"
        elif varname.startswith("pcp"):
            return "prcp"
        elif varname in ["t2", "t"]:
            return "T"
        elif varname == "q2":
            return "RH"
        elif varname == "rh":
            return "q2"
        elif varname == "ws10":
            return "WS"
        elif varname == "ws":
            return "ws10"
        elif varname == "press":
            return "P"
        else:
            raise NameError("Please add variable name translation")

    def __add_measurements(self, variable=None):
        if not variable:
            variable = self.variable
        self.measurements_files = glob_measurements()
        self.measurements = {}
        for f in self.measurements_files:
            ds = load_measurements(f, variable)
            self.measurements[ds.attrs["name"]] = ds

    def compute_rh(self):
        if self.variable.lower() not in ["q", "q2", "qv", "rh"]:
            raise NameError("Can only be calculated from any of 'q', 'qv', 'q2', 'RH'")
        if self.variable.lower() in ["rh", "q2"] and "t2" not in self.__added_variables:
            self.add_product("t2")
        if (
            self.variable.lower() in ["rh", "q2"]
            and "psfc" not in self.__added_variables
        ):
            self.add_product("psfc")
        rh = self.wrf_product["q2"].copy()
        psfc = self.wrf_product["psfc"].psfc * units("mbar")
        t2 = (self.wrf_product["t2"].t2 + 273.15) * units("K")
        q2 = self.wrf_product["q2"].q2 * units("kg/kg")
        rh["rh"] = relative_humidity_from_mixing_ratio(
            pressure=psfc,
            temperature=t2,
            mixing_ratio=q2,
        )
        return xr.Dataset(
            rh, coords=self.wrf_product["q2"].coords, attrs=self.wrf_product["q2"].attrs
        )

    def plot_station(self, station, sample_rate="D", save=False, **kwargs):
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
        if save:
            plt.savefig(f"{self.variable.lower()}_{station.lower()}.png")

    def plot_stations(self, **kwargs):
        for station in self.measurements.keys():
            self.plot_station(station, sample_rate="D", **kwargs)

    def __find_closest_gridpoint(self, station):
        ds = self.wrf_product[self.variable[0]]
        # for var in self.measurements.keys():
        #     df = self.measurements[var]
        #     break
        abslat = np.abs(ds.lat - coordinates[station][0])
        abslon = np.abs(ds.lon - coordinates[station][1])
        c = np.maximum(abslon, abslat)
        return np.where(c == np.min(c))

    def __get_closest_gridpoint_values(self, station):
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
        kwargs: args for  salem's quick_map
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
        base_map = data.salem.get_map(**kwargs)
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
    path="/home/ben/data/darwin_measured", ds_number="??", type="AWS-P"
):
    files = glob(f"{path}/{ds_number}_{type}*/*[!xlsx_complete]")
    return [file for file in files if not file.count("xlsx") and not file.count("_-_")]


def load_measurements(path, variable):
    ds = pd.read_csv(path, parse_dates=["datetime"], index_col=["datetime"])
    ds = ds.loc["2022-04-01":"2022-06-30"]
    if variable == "PCP":
        filter_col = [col for col in ds if col.startswith(variable)]
    else:
        filter_col = variable
    ds.attrs["name"] = path.split("/")[-1].split("_")[-3]
    ds.attrs["coordinates"] = [
        value
        for key, value in coordinates.items()
        if key.split("_")[-1] in ds.attrs["name"].lower()
    ]
    return ds[filter_col]
