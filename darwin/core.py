"""Base module for the GAR project."""

from __future__ import annotations

import copy
import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pint_xarray  # noqa: F401
import salem
import xarray as xr
from icecream import ic
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from metpy.calc import relative_humidity_from_mixing_ratio
from metpy.units import units
from pytz import timezone

from darwin.defaults import coordinates, gar_path, measured_path

if TYPE_CHECKING:
    import pint

xr.set_options(keep_attrs=True)
gar_path = Path(gar_path)


def open_dataset_from_file(file: Path) -> xr.Dataset:
    """Open a Dataset from a path.

    Args:
        file: file path
    """
    return salem.open_wrf_dataset(file)


def get_path_from_elements(
    experiment: str = "MM",
    variable: str = "t2",
    year: str = "2022",
    domain: str = "d02km",
    dimensions: str = "2d",
    frequency: str = "d",
    base: Path = gar_path,
) -> Path:
    """Get the path to a dataset from its elements."""
    if variable in ["hgt", "landmask", "lai", "lu_index"]:
        return base / f"{experiment}/{experiment}_{domain}_static_{variable}.nc"
    return (
        base / f"{experiment}/{experiment}_{domain}_{frequency}_{dimensions}_{variable}_{year}.nc"
    )


def open_dataset(
    experiment: str = "MM",
    variable: str = "t2",
    year: str = "2022",
    domain: str = "d02km",
    dimensions: str = "2d",
    frequency: str = "d",
    basepath: Path = gar_path,
    **kwargs,
) -> xr.Dataset:
    """Open a Dataset."""
    path = get_path_from_elements(
        experiment, variable, year, domain, dimensions, frequency, basepath
    )
    return open_dataset_from_file(path)


def open_experiment_from_file(file: Path, **kwargs: Path) -> Experiment:
    """Open an Experiment for GAR.

    kwargs = arguments passed to open_dataset.
    """
    hgt = kwargs.pop("hgt") if "hgt" in kwargs else None
    landmask = kwargs.pop("landmask") if "landmask" in kwargs else None
    if hgt:
        hgt = file.parent / "MM" / "MM_d02km_static_hgt.nc"
    if landmask:
        landmask = file.parent / "MM" / "MM_d02km_static_landmask.nc"
    return Experiment(
        open_dataset_from_file(file),
        hgt=hgt,
        landmask=landmask,
    )


def open_experiment(
    experiment: str = "MM",
    variable: str = "t2",
    year: str = "2022",
    domain: str = "d02",
    dimensions: str = "2d",
    frequency: str = "d",
    basepath: Path = gar_path,
    **kwargs: Path,
) -> Experiment:
    """Open an Experiment for GAR.

    kwargs = arguments passed to open_dataset.
    """
    filepath = get_path_from_elements(
        experiment, variable, year, domain, dimensions, frequency, basepath
    )
    hgt = kwargs.pop("hgt") if "hgt" in kwargs else None
    landmask = kwargs.pop("landmask") if "landmask" in kwargs else None
    if hgt:
        hgt = get_path_from_elements(
            experiment, variable, year, domain, dimensions, frequency, basepath
        )
    return Experiment(
        open_dataset_from_file(filepath),
        hgt=hgt,
        landmask=landmask,
    )


class Experiment:
    """Class to hold all data associated with one variable for one run."""

    def __init__(
        self: Experiment,
        gar_ds: xr.Dataset,
        *,
        with_measurements: bool = False,
        hgt: Path | None = None,
        landmask: Path | None = None,
    ) -> None:
        """Initiate an instance of class Experiment."""
        self.varname: str = gar_ds.VARNAME
        self.units: str = gar_ds[gar_ds.VARNAME].attrs["units"]
        if self.units == "k":
            self.units = "K"
        elif self.units == "w m-2":
            self.units = "W/m**2"
        elif self.units == "pa":
            self.units = "Pa"
        self.wrf_product: xr.Dataset = gar_ds.pint.quantify({self.varname: self.units})
        self.varname_translations: list[str] = self.__translate_varname(gar_ds.VARNAME)
        try:
            self.experiment = gar_ds.attrs["experiment"]
            self.year = gar_ds.attrs["year"]
        except KeyError:
            _ = ic("WARNING: Could not add experiment and/or year.")
        self.measurements = {}
        self.__added_variables: list[str] = []
        self.extracted: pd.DataFrame | None = None
        if hgt:
            self.hgt = xr.open_dataset(hgt)
        if landmask:
            self.landmask = xr.open_dataset(landmask)
        if with_measurements:
            self.__add_measurements()

    def __getitem__(self: Experiment, key: str) -> str:
        """Get a item.

            self (): The object.
            key (): Object key.

        Returns:
            The object with name key.
        """
        return getattr(self, key)

    def __setitem__(self: Experiment, key: str, value: str | float | bool) -> None:
        """Set an item.

        self (): The object key (): Name under which to store the item.
        value (): The item to store.
        """
        setattr(self, key, value)

    def copy(self: Experiment) -> Experiment:
        """Copy the object.

        Returns:
            A copy of the object.
        """
        return copy.copy(self)

    def remove_boundaries(self: Experiment, grid_points: int = 1) -> Experiment:
        """Crop the outer rows/columns of the model data."""
        copy = self.copy()
        copy.wrf_product = self.wrf_product.isel(
            west_east=slice(grid_points, -grid_points),
            south_north=slice(grid_points, -grid_points),
        )
        return copy

    def add_extracted_simulated_points_from_file(self: Experiment, file: str) -> None:
        """Load measurements from file."""
        extracted = pd.read_csv(file, parse_dates=["datetime"], index_col=["datetime"])
        path = file.rsplit("/")[-1].split(".")[0].split("_")
        if self.__translate_varname(self.varname) not in path:
            msg = f"""
            The name of the selected file does not seem to contain the variable. {path}
            Please name it in style of *_variable_*
            """
            raise ValueError(msg)

        extracted.attrs["name"] = self.varname
        self.extracted = extracted

    def add_product(
        self: Experiment, variable: str, wrf_ds: xr.Dataset | None = None
    ) -> xr.Dataset:
        """Add another variable to the Experiment."""
        if wrf_ds is not None:
            self.__added_variables.append(wrf_ds.darwin.VARNAME)
            return wrf_ds
        ds: xr.Dataset = open_dataset(experiment=self.experiment, variable=variable, year=self.year)
        if self.varname != ds.darwin.VARNAME:
            self.__added_variables.append(ds.darwin.VARNAME)
        return ds

    def __translate_varname(self: Experiment, varname: str) -> list[str]:
        varname = varname.lower()
        variable_names = [
            ["prcp", "PCP"],
            ["hgt"],
            ["et"],
            ["cape"],
            ["cin"],
            ["potevap"],
            ["t2", "T"],
            ["t2c", "T"],
            ["q2", "q"],
            ["rh", "RH"],
            ["rh2", "RH"],
            ["sh2", "SH"],
            ["ws10", "ws"],
            ["wd10", "WDIR"],
            ["swdown", "SLR"],
            ["psfc", "Pabs"],
        ]
        for combination in variable_names:
            if varname in combination:
                return combination
        msg = "Please add variable name translation"
        raise NameError(msg)

    def __add_measurements(self: Experiment, variable: str | None = None) -> None:
        ic("Adding measurements to dataset")
        if not variable:
            variable = self.varname
        self.measurements_files = glob_measurements()
        for file in self.measurements_files:
            measured = load_measurements(Path(file), variable)
            self.measurements[measured.attrs["name"]] = measured

    def plot_station(  # noqa: C901
        self: Experiment,
        station: str,
        sample_rate: str = "D",
        *,
        write: str | Path | bool = False,
        **kwargs: dict[str, str | int | bool],
    ) -> None:
        """Plot timeseries data of a specific station of the darwin network."""
        timeseries = {}
        data: pd.DataFrame = self.measurements[station]
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
            extracted.index = pd.DatetimeIndex(pd.date_range("2022-04-01", "2022-09-30"))
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
        if isinstance(write, str):
            plt.savefig(write)
        elif write:
            plt.savefig(f"{self.varname.lower()}_{station.lower()}.png")

    def plot_stations(self: Experiment, **kwargs: dict[str | Any]) -> None:
        """Plot data of multiple stations using the 'plot_station'-method."""
        for station in self.measurements:
            self.plot_station(station, sample_rate="D", **kwargs)

    def __find_closest_gridpoint(self: Experiment, station: str) -> tuple[float, float]:
        wrf_ds = self.wrf_product
        # for var in self.measurements.keys():
        #     df = self.measurements[var]
        #     break
        abslat = np.abs(wrf_ds.lat - coordinates[station][0])
        abslon = np.abs(wrf_ds.lon - coordinates[station][1])
        coords = np.maximum(abslon, abslat)
        return np.where(coords == np.min(coords))

    def __get_closest_gridpoint_values(self: Experiment, station: str) -> pd.Series:
        ([xlon], [xlat]) = self.__find_closest_gridpoint(station)
        selected = self.wrf_product.isel(west_east=xlon, south_north=xlat)
        selected = pd.Series(selected[self.varname])
        selected.index = pd.DatetimeIndex(pd.date_range("2022-04-01", "2022-09-30"))
        selected.name = self.wrf_product.attrs["experiment"]
        return selected

    def plot_map(  # noqa: C901, PLR0913, PLR0912
        self: Experiment,
        varname: str | None = None,
        ax: plt.Axes | None = None,
        aggregation: str = "mean",
        unit: str | pint.Unit = "",
        cmap: str | None = None,
        markercolors: list[str] | None = None,
        frequency: str = "d",
        annotations: dict[str, tuple[float, float]] | None = None,
        *,
        write: bool | str | Path = False,
        stations: bool = True,
        cbar: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        extend: str = "neither",
        **kwargs: Path | str | int | bool,
    ) -> salem.Map:
        """Plot a map of Experiment data.

        kwargs: args for  salem's quick_map.
        """
        u = open_dataset_from_file(
            gar_path / "MM" / f"MM_d02km_{frequency}_2d_u10.nc"
        ).sel(time=self.wrf_product.time)["u10"]
        v = open_dataset_from_file(
            gar_path / "MM" / f"MM_d02km_{frequency}_2d_v10.nc"
        ).sel(time=self.wrf_product.time)["v10"]

        self.wrf_product["u10"] = u
        self.wrf_product["v10"] = v

        varname = varname or self.__translate_varname(self.varname)
        base_map = self.wrf_product.salem.get_map(**kwargs)

        if "time" in self.wrf_product.dims:
            if self.varname in ["prcp", "et", "potevap"]:
                data = self.wrf_product.mean(dim="time", skipna=True, keep_attrs=True)
                data[self.varname] = data[self.varname].pint.to("mm/d")
            else:
                data = self.wrf_product.mean(dim="time", skipna=True, keep_attrs=True)
                if self.varname == "t2":
                    data[self.varname] = data[self.varname].pint.to("degC")
                if self.varname == "psfc":
                    data[self.varname] = data[self.varname].pint.to("hPa")
        else:
            data = self.wrf_product
        base_map.set_data(data[self.varname])
        base_map.set_vmin(vmin)
        base_map.set_vmax(vmax)
        base_map.set_extend(extend)
        if cmap:
            cm = ScalarMappable(
                cmap=cmap,
                norm=Normalize(data[self.varname].min(), data[self.varname].max()),
            )
            cm.set_clim(
                # vmin=data[self.varname].min().to_numpy(),
                # vmax=data[self.varname].max().to_numpy(),
                vmin=vmin,
                vmax=vmax,
            )
        # if self.varname == "prcp":
        #     base_map.set_vmin(0)
        #     base_map.set_vmax(10)
        #     base_map.set_extend("max")
        # elif self.varname == "t2":
        #     base_map.set_vmin(15)
        #     base_map.set_vmax(30)
        #     base_map.set_extend("both")
        base_map.set_cmap(cmap)
        halign = {
            "Bellavista": "left",
            "Cerro Crocker": "left",
            "Cueva de Sucre": "left",
            "El Junco": "center",
            "La Galapaguera": "left",
            "Militar": "left",
            "Minas Rojas": "center",
            "Puerto Ayora": "left",
            "Puerto Baquerizo Moreno": "right",
            "Puerto Villamil": "left",
            "Sierra Negra": "right",
            "Santa Rosa": "right",
        }
        valign = {
            "Bellavista": "center",
            "Cerro Crocker": "center",
            "Cueva de Sucre": "center",
            "El Junco": "center",
            "La Galapaguera": "top",
            "Militar": "center",
            "Minas Rojas": "bottom",
            "Puerto Ayora": "top",
            "Puerto Baquerizo Moreno": "center",
            "Puerto Villamil": "center",
            "Sierra Negra": "center",
            "Santa Rosa": "center",
        }
        coordinates_sorted = dict(sorted(coordinates.items(), key=lambda item: item[0]))
        if stations:
            for i, (key, value) in enumerate(coordinates_sorted.items()):
                if markercolors is None:
                    continue
                if markercolors[i] == 0 or not markercolors[i] or np.isnan(markercolors[i]):
                    continue
                lat, lon = value
                c_rgba = cm.to_rgba(markercolors[i]) if markercolors is not None else "k"
                base_map.set_points(
                    lon,
                    lat,
                    text=key if cbar else None,
                    facecolor=c_rgba,
                    text_kwargs={"ha": halign[key], "va": valign[key]},
                )
        base_map.set_scale_bar()
        # if not ax:
        #     fig = plt.figure(
        #         figsize=(10, 8),
        #         frameon=False,
        #         layout="tight",
        #         facecolor="white",
        #         edgecolor="white",
        #     )
        #     ax = fig.add_subplot(111)
        base_map.plot(ax=ax)
        u = u[0, 4::7, 4::7]
        v = v[0, 4::7, 4::7]
        xx, yy = base_map.grid.transform(
            u.west_east.values, u.south_north.values, crs=u.salem.grid.proj
        )
        xx, yy = np.meshgrid(xx, yy)
        qu = ax.quiver(xx, yy, u.values, v.values)
        plt.quiverkey(qu, 0.08, 0.55, 5, "5 m s$^{-1}$", labelpos="E", coordinates="figure")
        if annotations:
            for key, value in annotations.items():
                ax.annotate(
                    key,
                    xy=value,
                    xycoords="axes fraction",
                    fontsize=14,
                    fontweight="bold",
                    backgroundcolor="white",
                )
        if cbar:
            base_map.append_colorbar(ax=ax, label=unit)
        if isinstance(write, str):
            plt.savefig(write)
        elif write:
            plt.savefig(f"{self.experiment}_{self.varname.lower()}_{aggregation.lower()}_map.png")
        return base_map

    def mask_ocean(self, mask: xr.Dataset | None = None) -> xr.Dataset:
        """Mask ocean from the data."""
        if not mask:
            mask = xr.open_dataset(gar_path / "MM" / "MM_d02km_static_landmask.nc")
        return self.wrf_product.where(mask.landmask.squeeze() == 1)

    def select_altitude_range(self, min_altitude: float, max_altitude: float) -> xr.Dataset:
        """Select data from a certain altitude range."""
        return self.wrf_product.where(
            self.wrf_product.altitude >= min_altitude & self.wrf_product.altitude <= max_altitude
        )

    # NOTE: This method is not working properly.
    # def set_units(self) -> xr.DataArray:
    #     """Add units to xarray data."""
    #     ds = self.wrf_product
    #     var = self.varname
    #     ds[var] = ds[var] * units(ds[var].attrs["units"])
    #     return self.wrf_product

    def cat(self: Experiment, other: xr.Dataset) -> xr.Dataset:
        """Concatenate two experiments."""
        return xr.concat([self.wrf_product, other.wrf_product], dim="time")


def compute_rh(
    mixing_ratio: xr.Dataset, temperature: xr.Dataset, pressure: xr.Dataset
) -> xr.Dataset:
    """Compute relative humidity from mixing ratio."""
    psfc = pressure.psfc * units("mbar")
    temperature = temperature.t2 * units("K")
    mixing_ratio = mixing_ratio.q2 * units("kg/kg")
    rel_humidity: xr.Dataset = relative_humidity_from_mixing_ratio(
        pressure=psfc,
        temperature=temperature,
        mixing_ratio=mixing_ratio,
    )
    return xr.Dataset(
        rel_humidity,
        coords=temperature.coords,
        attrs=temperature.attrs,
    )


def plot_pcp(
    data: pd.Dataset,
    title: str | None = None,
    sample_rate: str = "D",
    **kwargs: dict[str, Any],
) -> None:
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
    path: Path | str = str(measured_path),
    ds_number: str = "??",
    type_of_data: str = "AWS-P",
) -> list[str]:
    """Glob for measurement data of the darwin measurement network."""
    if isinstance(path, str):
        path = Path(path)
    return list(path.glob(f"{ds_number}_{type_of_data}*.csv"))


def open_measurements(path: Path | str) -> MeasurementFrame:
    """Open a csv-file containing measurements from the darwin measurement
    network.

    Args:
        path: Path to a csv-file containing measurements

    Returns:
        The data in the specified csv-file
    """
    measurements = pd.read_csv(path)
    measurements.index = pd.DatetimeIndex(
        measurements.datetime, tz=timezone("Pacific/Galapagos")
    ).tz_convert(datetime.UTC)
    return MeasurementFrame(measurements)


class MeasurementFrame(pd.DataFrame):
    """A class to hold measurements from the darwin measurement network."""

    def __init__(self: MeasurementFrame, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        """Initiate an instance of class Measurement.

        *args: extra argument list **kwargs: extra keyword arguments
        """
        super().__init__(*args, **kwargs)

    def join_wrfdata(
        self: MeasurementFrame, wrfdata: pd.DataFrame, join: str = "inner"
    ) -> pd.DataFrame:
        """Join the data with WRF data.

        Args:
            self: The object
            wrfdata: A DataFrame containing WRF data
            join: Join type

        Returns:
            The combined data
        """
        return pd.concat([self.data, wrfdata], axis=1, join=join)


def load_measurements(
    path: Path, variable: list[str] | str | None = None
) -> pd.Series[float] | pd.DataFrame:
    """Load measured data from darwin measurement network.

    Parameters
    ----------
    path :
        Path to a csv-file containing measurements
    variable :
        name(s) of the variable to load

    Returns
    -------
    pandas.Series
        a pandas Series of the data in specified csv-file
    """
    measured = pd.read_csv(path, parse_dates=["datetime"], index_col=["datetime"])
    if not variable:
        return measured
    if variable == "prcp":
        filter_col = [
            col
            for col in measured
            if col in ["PCP_diff_radar", "PCP_acoustic", "PCP_tot_bucket", "PCP"]
        ]
    else:
        filter_col = variable
    measured.attrs["name"] = path.name.split("_")[-3]
    measured.attrs["coordinates"] = [
        value
        for key, value in coordinates.items()
        if key.rsplit("_", maxsplit=1)[-1] in measured.attrs["name"].lower()
    ]
    return measured[filter_col]


if __name__ == "__main__":
    ds = open_dataset_from_file(gar_path / "MM" / "MM_d02km_d_2d_t2.nc")
    ds.plot_map("prcp", write="test.png")
