"""Plot trends and mean values for MM data. May convert to core of the package.

CLI Usage:
    trends_plevels.py --var t2c --type mean --start 2001 --end 2010 --frequency y
    trends_plevels.py --var t2c --type trend --start 2001 --end 2010 --frequency y
    trends_plevels.py --var t2c --type timeseries --start 2001 --end 2010 --frequency y
    trends_plevels.py -v t2c -t timeseries -l 100 -s 2001 -e 2010 -f y
"""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from sys import exit
from warnings import warn

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import salem
import seaborn as sns
import statsmodels.api as sm
import xarray as xr

logging.getLogger("matplotlib.font_manager").disabled = True
xr.set_options(keep_attrs=True)

# load data
mm_path = Path.home() / "data" / "GAR" / "MM"

# constants
pvalue_threshold = 0.05
ndims = {"2d": 2, "3d_press": 3, "3d": 3}
nlevels = {"1": 1, "2": 2, "3": 3}
bar_vars = ["prcp", "et", "wateravailability"]

fixed_units = {
    "k": "K",
    "w m-2": "W/m**2",
    "pa": "Pa",
}
dry_wet_map = {
    1: (1, "Wet"),
    2: (1, "Wet"),
    3: (1, "Wet"),
    4: (1, "Wet"),
    5: (1, "Wet"),
    6: (2, "Dry"),
    7: (2, "Dry"),
    8: (2, "Dry"),
    9: (2, "Dry"),
    10: (2, "Dry"),
    11: (2, "Dry"),
    12: (1, "Wet"),
}
season_names = ["JFM", "AMJ", "JAS", "OND"]
season_map = {
    1: (1, season_names[0]),
    2: (1, season_names[0]),
    3: (1, season_names[0]),
    4: (2, season_names[1]),
    5: (2, season_names[1]),
    6: (2, season_names[1]),
    7: (3, season_names[2]),
    8: (3, season_names[2]),
    9: (3, season_names[2]),
    10: (4, season_names[3]),
    11: (4, season_names[3]),
    12: (4, season_names[3]),
}
dry_wet_names = ["Wet", "Dry"]
dry_wet_map = {
    1: (1, dry_wet_names[0]),
    2: (1, dry_wet_names[0]),
    3: (1, dry_wet_names[0]),
    4: (1, dry_wet_names[0]),
    5: (1, dry_wet_names[0]),
    6: (2, dry_wet_names[0]),
    7: (2, dry_wet_names[1]),
    8: (2, dry_wet_names[1]),
    9: (2, dry_wet_names[1]),
    10: (2, dry_wet_names[1]),
    11: (2, dry_wet_names[1]),
    12: (1, dry_wet_names[1]),
}

def parse_args() -> Namespace:
    """Parse command line arguments.

    Returns:
        Namespace: The parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--var", "-v", type=str, nargs="+")
    parser.add_argument("--level", "-l", type=int, nargs="+", default=None)
    parser.add_argument("--colorbarlabel", "-cl", type=str, default="")
    parser.add_argument("--type", "-t", type=str, default="mean")
    parser.add_argument("--start", "-s", type=int, default=1980)
    parser.add_argument("--end", "-e", type=int, default=2023)
    parser.add_argument("--frequency", "-f", type=str, choices=["m", "y"], default="y")
    parser.add_argument("--lowest_alt", "-la", type=float, default=0.1)
    parser.add_argument("--highest_alt", "-ha", type=float, default=2000.0)
    parser.add_argument("--no_mask", "-nm", action="store_false", dest="mask")
    parser.add_argument("--no_plot", "-np", action="store_false", dest="plot")
    parser.add_argument("--mask_sea", "-ms", action="store_true")
    return parser.parse_args()


def plot_regression_residuals(
    xdata: np.ndarray,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    title: str,
) -> None:
    _, ax_residuals = plt.subplots(1, 1)
    ax_residuals.scatter(xdata, model.resid, color="black")
    ax_residuals.grid(visible=True)
    ax_residuals.set_ylabel("residuals")
    ax_residuals.set_xlabel("year")
    plt.savefig(f"plots/residuals/{title}.png")


class _DarwinAccessorBase:
    def __init__(
        self,
        xarray_obj: xr.Dataset | xr.DataArray,
        topography: Path | None = None,
        landmask: Path | None = None,
    ) -> None:
        self._obj = xarray_obj
        self.attrs = xarray_obj.attrs
        self._seasons = season_names
        self._season_map = season_map
        self._dry_wet = dry_wet_names
        self._dry_wet_map = dry_wet_map
        self._time_varname = "time"
        if isinstance(xarray_obj, xr.DataArray):
            self.varname = str(xarray_obj.name)
        elif "VARNAME" in self.attrs:
            self.varname = self.attrs["VARNAME"]
        else:
            names = list(xarray_obj.keys())
            self.varname = str(names[-1])
        if self.varname in ["t2", "theta"] or self.varname == "geopotential":
            self.cmap = "RdBu_r"
        else:
            self.cmap = "Blues"

        self.varname: str = xarray_obj.VARNAME
        self.units: str = xarray_obj[xarray_obj.VARNAME].attrs["units"]
        if self.units in fixed_units:
            self.units = fixed_units[self.units]
        if "experiment" in xarray_obj.attrs:
            self.experiment = xarray_obj.attrs["experiment"]
        if "year" in xarray_obj.attrs:
            self.year = xarray_obj.attrs["year"]
        if topography:
            self.topography = salem.open_xr_dataset(topography)
        if landmask:
            self.landmask = xr.open_dataset(landmask)
        if isinstance(xarray_obj, xr.DataArray):
            xarray_obj = xarray_obj.to_dataset(name="var")
            if hasattr(xarray_obj.attrs, "pyproj_srs"):
                xarray_obj.attrs["pyproj_srs"] = xarray_obj["var"].pyproj_srs

    def _map_seasons(self, data: np.ndarray, *, dry_wet: bool = False) -> np.ndarray:
        if dry_wet:
            return np.array([self._dry_wet_map[i][0] for i in data])
        return np.array([self._season_map[i][0] for i in data])

    def convert_units(self) -> xr.Dataset | xr.DataArray:
        """Convert the units of the data.

        Args:
            units (str): The units to convert to.
        """
        if self.varname in ["t2", "theta"]:
            if isinstance(self._obj, xr.DataArray):
                self._obj -= 273.15
            else:
                self._obj[self.varname] -= 273.15
            self.units = r"°C"
        elif self.varname == "geopotential":
            if isinstance(self._obj, xr.DataArray):
                self._obj /= 9.80665
            else:
                self._obj[self.varname] /= 9.80665
            self.units = "m"
        elif self.varname in bar_vars:
            if isinstance(self._obj, xr.DataArray):
                self._obj *= 24
            else:
                self._obj[self.varname] *= 24
            self.units = "mm/d"
        return self._obj

    def mask_land(
        self,
        landmask: xr.Dataset,
        hgt: xr.Dataset,
        lowest_alt: float = 0.1,
        highest_alt: float = 2000.0,
    ) -> xr.Dataset:
        hgtmask_lower = np.where(hgt.hgt >= lowest_alt, 1, np.nan)
        hgtmask_upper = np.where(hgt.hgt <= highest_alt, 1, np.nan)
        hgtmask = np.where(hgtmask_lower * hgtmask_upper > 0, 1, np.nan)
        hgtmask = np.where(landmask.landmask * hgtmask > 0, 1, np.nan)
        self._obj[self.varname] = self._obj[self.varname] * hgtmask
        if isinstance(self._obj, xr.DataArray):
            self._obj = xr.Dataset({self.varname: self._obj})
        return self._obj

    def assign_seasons_as_coords(
        self, *, dry_wet: bool = False
    ) -> xr.DataArray | xr.Dataset:
        """Assign seasons as coordinates to the dataarray.

        Args:
            dry_wet (bool, optional): Whether to assign dry and wet season instead of
            seasons. Defaults to False.

        Returns:
            Dataset: The filtered dataset.
        """
        # if isinstance(self._obj, xr.Dataset):
        #     msg = "Assigning seasons as coordinates is not implemented for Datasets"
        #     raise NotImplementedError(msg)
        if "seasons" in self._obj.coords:
            return self._obj

        new_coords = self._map_seasons(
            self._obj.time.dt.month.to_numpy(), dry_wet=dry_wet
        )
        self._obj = self._obj.assign_coords(season=("time", new_coords)).swap_dims(
            {"time": "season"}
        )
        self._obj.attrs["time_varname"] = "season"
        return self._obj

    def plot_map(
        self,
        ax: plt.Axes | None = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """Plot the map.

        Args:
            ax (plt.Axes, optional): The axes to plot on. Defaults to None.
            cmap (str, optional): The colormap to use. Defaults to None.
            vmin (float, optional): The minimum value for the colormap. Defaults to
            None.
            vmax (float, optional): The maximum value for the colormap. Defaults to
            None.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)
        base_map = self._obj.salem.get_map()
        if isinstance(self._obj, xr.DataArray):
            base_map.set_data(self._obj)
        else:
            base_map.set_data(self._obj[self._obj.data_vars[0]])
        if cmap is not None:
            base_map.set_cmap(cmap)
        if vmin is not None:
            base_map.set_vmin(vmin)
        if vmax is not None:
            base_map.set_vmax(vmax)
        base_map.set_scale_bar()
        base_map.plot(ax=ax)
        base_map.append_colorbar(ax=ax, label=self._obj.attrs["units"])

    def regression(
        self,
        xdata: np.ndarray | None = None,
        ydata: np.ndarray | None = None,
        variable: str | None = None,
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        """Regression of the data.

        Returns:
            Regression: The regression object.
        """
        if variable is None and ydata is None:
            warn(
                "No variable or ydata provided. Using first data variable in dataset.",
                stacklevel=1,
            )
            variable = str(self._obj.data_vars[0].name)
        xdata = self._obj.time.dt.year.to_numpy() if xdata is None else xdata
        ydata = self._obj[variable].to_numpy() if ydata is None else ydata
        model = sm.OLS(ydata, sm.add_constant(xdata)).fit()
        return model.predict(sm.add_constant(xdata))

    @staticmethod
    def _add_axis_annotations_for_regression(
        ax: plt.Axes,
        title: str,
        labels: list[str],
        ylabel: str,
        xlabel: str,
    ) -> plt.Axes:
        """Add axis annotations for regression.

        Args:
            ax (plt.Axes): The axes to add annotations to.
            title (str): The title of the plot.
            labels (list[str]): The labels for the regression.
            ylabel (str): The label for the y-axis.
            xlabel (str): The label for the x-axis.
        """
        yposition = 0.98
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                xy=(0.98, yposition - i * 0.07),
                xycoords="axes fraction",
                fontsize=13,
                horizontalalignment="right",
                verticalalignment="top",
            )
        ax.set_title(title)
        ax.grid(visible=True)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        return ax

    def get_timeseries_axis_labels(self, i: int, *, is_2d: bool) -> tuple[str, str]:
        match self.varname:
            case "t2":
                ylabel = "temperature in °C"
            case "theta":
                ylabel = f"{self._obj.level} hPa potential temperature in °C"
            case "geopotential":
                if self._obj.level:
                    ylabel = f"{self._obj.level} hPa geopotential height in m"
                else:
                    ylabel = "geopotential height difference in m"
            case "prcp":
                ylabel = "mean precipitation in mm/d"
            case "et":
                ylabel = "mean actual evapotranspiration in mm/d"
            case "wateravailability":
                ylabel = "mean water availability in mm/d"
            case "q":
                ylabel = f"{self._obj.level} hPa mixing ratio in kg/kg"
            case "q2":
                ylabel = "mixing ratio in g/kg"
            case "w":
                if self._obj.level:
                    ylabel = f"{self._obj.level} hPa vertical velocity in m/s"
                else:
                    ylabel = "vertical velocity in m/s"
            case "qvapor":
                ylabel = f"{self._obj.level} hPa water vapor mixing ratio in kg/kg"
            case "qsolid":
                ylabel = f"{self._obj.level} hPa solid water mixing ratio in kg/kg"
            case "qliquid":
                ylabel = f"{self._obj.level} hPa liquid water mixing ratio in kg/kg"
            case _:
                ylabel = "unknown"
        xlabel = "year"
        if is_2d:
            xlabel = xlabel if i in [2, 3] else ""
            ylabel = ylabel if i in [0, 2] else ""
        else:
            xlabel = xlabel if i == 1 else ""
        return xlabel, ylabel

    def __plot_timeseries_ols(
        self,
        ax: plt.Axes,
        xdata: np.ndarray,
        ydata: np.ndarray,
    ) -> float:
        """Plot the regression for a time series.

        Args:
            ax (plt.Axes): The axes to plot on.
            xdata (np.ndarray): The x-data.
            ydata (np.ndarray): The y-data.
            regression (sm.regression.linear_model.RegressionResultsWrapper): The
            regression object.
            is_2d (bool): Whether the data is 2D.
            i (int): The index of the subplot.
        """
        results = sm.OLS(ydata, sm.add_constant(xdata)).fit()
        sns.regplot(
            x=xdata,
            y=ydata,
            ax=ax,
            x_ci="ci",
            ci=95,
            color="black",
            scatter=True,
            line_kws={"color": "red"},
            marker=".",
            # robust=True,
            truncate=False,
        )
        r = np.zeros_like(results.params)
        t_test = results.t_test(r)
        return t_test.pvalue

    @staticmethod
    def __get_gls_ar1(
        ols_model: sm.regression.linear_model.RegressionResultsWrapper,
        xdata: np.ndarray,
        ydata: np.ndarray,
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        ols_residuals = ols_model.resid
        rho_est = sm.tsa.acf(ols_residuals, fft=False, nlags=1)[1]

        n = len(ols_residuals)
        sigma = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sigma[i, j] = rho_est ** abs(i - j)

        sigma_inv = np.linalg.inv(sigma)
        return sm.GLS(ydata, sm.add_constant(xdata), sigma=sigma_inv).fit()

    @staticmethod
    def __get_gls_hs(
        ols_model: sm.regression.linear_model.RegressionResultsWrapper,
        xdata: np.ndarray,
        ydata: np.ndarray,
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        ols_residuals = ols_model.resid
        resid_squared = ols_residuals**2
        ols_variance_model = sm.OLS(
            resid_squared, sm.add_constant(xdata), data=data
        ).fit()
        predicted_variances = ols_variance_model.fittedvalues
        sigma = np.diag(predicted_variances)
        sigma_inv = np.linalg.inv(sigma)
        return sm.GLS(ydata, sm.add_constant(xdata), sigma=sigma_inv).fit()

    def plot_linear_regression(
        self,
        ax: plt.Axes,
        xdata: np.ndarray,
        ydata: np.ndarray,
        model_type: str = "ols",
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        models = {}
        models["ols"] = sm.OLS(ydata, sm.add_constant(xdata)).fit()
        if model_type == "gls-ar1":
            models["gls-ar1"] = self.__get_gls_ar1(models["ols"], xdata, ydata)
        elif model_type == "gls-hs":
            models["gls-hs"] = self.__get_gls_hs(models["ols"], xdata, ydata)
        elif model_type == "rls":
            models["rls"] = sm.RLM(ydata, sm.add_constant(xdata)).fit()

        pred_summary = (
            models[model_type]
            .get_prediction(sm.add_constant(xdata))
            .summary_frame(alpha=0.05)
        )
        if self.varname in bar_vars and xdata[0] == 1980:
            ax.bar(xdata, ydata, color="black")
        else:
            ax.scatter(xdata, ydata, color="black", marker=".")
        # plot trend line
        # ax.autoscale(False)
        ax.plot(
            np.linspace(xdata.min(), xdata.max(), 100),
            models[model_type].params[1] * np.linspace(xdata.min(), xdata.max(), 100)
            + models[model_type].params[0],
            color="red",
            label="regression",
        )
        # ax.plot(
        #     np.linspace(xdata.min(), xdata.max(), 100),
        #     np.linspace(xdata.min(), xdata.max(), 100),
        #     color="black",
        #     linestyle="--",
        #     label="1:1 line",
        # )
        ax.legend()
        ax.plot(xdata, pred_summary["mean"], color="red")
        # ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="black")
        ax.fill_between(
            xdata,
            pred_summary["mean_ci_lower"],
            pred_summary["mean_ci_upper"],
            color="red",
            alpha=0.2,
        )
        return models[model_type]

    def plot_facet_map(
        self,
        filestem: str,
        landmask: xr.Dataset,
        cmap: str,
        ax: matplotlib.axes.Axes | npt.NDArray | None = None,
    ) -> None:
        """Plot time series faceted by coordinates.

        This is useful for plotting seasonal data.

        Args:
            ax (plt.Axes, optional): The axes to plot on. Default: None.
            model_type (str, optional): The type of model to use. Default: "ols".
        """
        varname = self.varname
        if isinstance(self._obj, xr.Dataset):
            da = self._obj[varname].squeeze()
        else:
            da = self._obj.squeeze()
        if len(da.shape) != 3:
            msg = "Can only plot time series trends from coordinates for 3D data"
            raise ValueError(msg)
        if ax is None:
            _, axes = plt.subplots(2, 2)
        elif isinstance(ax, np.ndarray):
            axes = ax
        else:
            axes = np.array([ax])
        xdata = np.unique(self._obj.time.dt.year.to_numpy())
        for i, time_val in enumerate(
            np.unique(da.coords[self._time_varname].to_numpy())
        ):
            print(i, time_val)
            axi = axes[i // 2, i % 2] if axes.ndim > 1 else axes[i]
            data = da
            years = da.time.dt.year.to_numpy()
            ds = (
                data.assign_coords(year=(self._time_varname, years))
                .swap_dims({self._time_varname: "year"})
                .groupby("year")
                .mean("year")
            ).to_numpy()
            vals = ds
            vals2 = vals.reshape(vals.shape[0], -1)
            regressions = np.polyfit(np.unique(years), vals2, 1)
            if self.varname in bar_vars:
                cmap = "RdBu"
            data = regressions[0, :].reshape(vals.shape[1], vals.shape[2])
            vmean = 0
            print(f"Trends mean: {np.round(np.nanmean(data), 4)}")
            print(f"Trends sd: {np.round(np.nanstd(data), 5)}")
            # create xarray dataset
            vdist = max(np.abs(np.nanmin(data)), np.nanmax(data))
            base_map = landmask.salem.get_map()
            base_map.set_data(data)
            base_map.set_cmap(cmap)
            if args.type == "trend" and vdist:
                base_map.set_vmin(-vdist)
                base_map.set_vmax(vdist)
            base_map.set_scale_bar()
            base_map.plot(ax=ax)
            if i == 3:
                base_map.append_colorbar(ax=ax, label=args.colorbarlabel)
            plt.savefig(f"plots/{filestem}.png")
            print(f"Output: {output}")
        plt.tight_layout()

    def __group_by_coord(self, years: np.ndarray) -> xr.Dataset | xr.DataArray:
        self._obj = (
            self._obj.assign_coords(year=(self._time_varname, years))
            .swap_dims({self._time_varname: "year"})
            .groupby("year")
            .mean("year")
        ).squeeze()
        return self

    def plot_facet_from_coords(
        self,
        ax: matplotlib.axes.Axes | npt.NDArray | None = None,
        xvar: str | None = None,
        yvar: str | None = None,
        *,
        model_type: str = "ols",
    ) -> None:
        """Plot time series faceted by coordinates.

        This is useful for plotting seasonal data.

        Args:
            ax (plt.Axes, optional): The axes to plot on. Default: None.
            model_type (str, optional): The type of model to use. Default: "ols".
        """
        varname = self.varname
        if len(self._obj[varname].shape) != 1:
            msg = "Can only plot time series faceted by coordinates for 1D data"
            raise ValueError(msg)
        if ax is None:
            _, axes = plt.subplots(2, 2)
        elif isinstance(ax, np.ndarray):
            axes = ax
        else:
            axes = np.array([ax])
        ds = self._obj
        for i, time_val in enumerate(
            # np.unique(self._obj.coords[self._time_varname].to_numpy())
            np.arange(1, 5)
        ):
            axi = axes[i // 2, i % 2] if axes.ndim > 1 else axes[i]
            years = self._obj.loc[{"season": time_val}].time.dt.year.to_numpy()
            data = (
                self._obj.loc[{"season": time_val}]
                .assign_coords(year=("season", years))
                .swap_dims({"season": "year"})
            )
            xdata = (
                np.unique(years)
                if xvar is None
                else (data[xvar].groupby("year").mean("year")).to_numpy()
            )
            data = (
                data[self.varname].groupby("year").mean("year").to_numpy()
                if yvar is None
                else data[yvar].groupby("year").mean("year").to_numpy()
            )
            print(f"Data mean: {np.nanmean(data)}")
            reg_results = self.plot_linear_regression(axi, xdata, data, model_type)
            # plot_regression_residuals(
            #     xdata, reg_results, f"residuals_{self.varname}_{time_val}_{model_type}"
            # )
            # print(reg_results.summary())
            is_2d = axes.ndim == ndims["2d"]
            if xvar is None and yvar is None:
                xlabel, ylabel = self.get_timeseries_axis_labels(i, is_2d=is_2d)
            else:
                xlabel = str(xvar)
                ylabel = str(yvar)

            print(f"p-value: {reg_results.t_test('x1 = 0').pvalue}")
            pval_label = (
                "p < 0.05"
                if reg_results.t_test("x1 = 0").pvalue < pvalue_threshold
                else "p >= 0.05"
            )
            if self.units == "K":
                self.units = "°C"
            elif "h-1" in self.units:
                self.units = self.units.replace(" h-1", "/d")
            elif " kg-1" in self.units:
                self.units = self.units.replace(" kg-1", " /kg")
            labels = [
                f"$\\beta$ = {10 * reg_results.params[1]:.2f} {self.units}/dec",
                f"$r^2$ = {reg_results.rsquared:.2f}",
                pval_label,
            ]
            self._add_axis_annotations_for_regression(
                axi,
                labels=labels,
                title=self._dry_wet[i] if not is_2d else self._seasons[i],
                xlabel=xlabel,
                ylabel=ylabel,
            )
        plt.tight_layout()

    def crop_margins(self, *, mask: bool = True) -> xr.Dataset | xr.DataArray:
        crop_cells = slice(40, -40) if mask else slice(5, -5)
        return self._obj.isel(west_east=crop_cells, south_north=crop_cells)

    def mask_height_zone(
        self, landmask: xr.Dataset, hgt: xr.Dataset, lower: float, upper: float
    ) -> xr.Dataset | xr.DataArray:
        """Mask data based on height zone and landmask.

        Args:
            landmask (xr.Dataset): The landmask.
            hgt (xr.Dataset): The height.
            lower (float): The lower bound of the height zone.
            upper (float): The upper bound of the height zone.

        Returns:
            Dataset: The masked dataset.
        """
        hgtmask_lower = np.where(hgt.hgt >= lower, 1, np.nan)
        hgtmask_upper = np.where(hgt.hgt <= upper, 1, np.nan)
        hgtmask = np.where(hgtmask_lower * hgtmask_upper > 0, 1, np.nan)
        hgtmask = np.where(landmask.landmask * hgtmask > 0, 1, np.nan)
        return self._obj * hgtmask

    def mask_sea(self, landmask: xr.Dataset) -> xr.Dataset | xr.DataArray:
        """Mask data based on sea.

        Args:
            landmask (xr.Dataset): The landmask.

        Returns:
            Dataset: The masked dataset.
        """
        # reverse landmask
        seamask = np.where(landmask.landmask == 0, 1, np.nan)
        return self._obj * seamask


@xr.register_dataset_accessor("darwin")
class DatasetAccessor(_DarwinAccessorBase):
    """Accessor for Datasets."""

    def add_seasons(self) -> xr.Dataset:
        """Add seasons to the dataset.

        Returns:
            Dataset: The dataset with seasons.
        """
        if isinstance(self._obj, xr.DataArray):
            msg = "Can only add seasons to Datasets"
            raise TypeError(msg)
        if "seasons" in self._obj.data_vars:
            return self._obj

        new_data = self._map_seasons(self._obj.time.dt.month.to_numpy())
        self._obj["seasons"] = xr.DataArray(
            new_data, dims=("time"), coords={"time": self._obj.time}
        )
        return self._obj


@xr.register_dataarray_accessor("darwin")
class DataArrayAccessor(_DarwinAccessorBase):
    """Accessor for DataArrays."""


if __name__ == "__main__":
    # load data
    args = parse_args()
    dim = "2d" if not args.level else "3d_press"
    data = {}

    landmask = salem.open_xr_dataset(mm_path / "MM_d02km_static_landmask.nc")
    hgt = salem.open_xr_dataset(mm_path / "MM_d02km_static_hgt.nc")
    landmask = landmask.darwin.crop_margins(mask=args.mask).isel(time=0)
    hgt = hgt.darwin.crop_margins(mask=args.mask).isel(time=0)

    is_level_comparison = args.level is not None and len(args.level) == nlevels["2"]
    fig = plt.figure(
        figsize=(10, 4) if args.type == "timeseries" else (10, 8),
        frameon=False,
        layout="tight",
        facecolor="white",
        edgecolor="white",
    )
    ax = fig.add_subplot(111)
    filestem = args.type
    if args.level:
        filestem += f"_p-{'-'.join(str(lev) for lev in args.level)}"
    else:
        filestem += f"_hgt-{int(args.lowest_alt)}-{int(args.highest_alt)}"
    var_part = "-".join(var for var in args.var)
    filestem += f"_v-{var_part}_y-{args.start}-{args.end}"
    if not args.mask:
        filestem += "_nomask"
    if args.mask_sea:
        filestem += "_masksea"
    output = f"plots/{filestem}.png"

    # prepare data
    for var in args.var:
        ds = salem.open_xr_dataset(
            mm_path / f"MM_d02km_{args.frequency}_{dim}_{var}.nc"
        )
        ds.attrs["level"] = (
            "-".join(str(lev) for lev in args.level) if args.level else ""
        )
        if args.level and not is_level_comparison:
            ds = ds.sel(pressure=args.level[0])
        elif args.level and is_level_comparison:
            ds1 = ds.sel(pressure=args.level[0])
            ds2 = ds.sel(pressure=args.level[1])
            ds = ds2 - ds1

        ds = ds.sel(time=slice(f"{args.start}-01-01", f"{args.end}-12-31"))
        ds = ds.darwin.crop_margins(mask=args.mask)
        if args.mask:
            ds = ds.darwin.mask_height_zone(
                landmask, hgt, args.lowest_alt, args.highest_alt
            )
        elif args.mask_sea:
            ds = ds.darwin.mask_sea(landmask)

        # select and clean data
        ds.darwin.convert_units()
        units = "mm/d" if var in bar_vars else "C"
        ds.attrs["units"] = units
        data[var] = ds

    if len(args.var) == 1:
        ds = data[args.var[0]]
    else:
        ds = xr.merge([data[var] for var in args.var])
    if args.type == "full_year":
        ds = ds.mean(dim=("south_north", "west_east")).squeeze()
        fig, ax = plt.subplots(
            1, 1, figsize=(12, 5), sharex=True, sharey=True, layout="tight"
        )
        data = ds.groupby("time.year").mean("time")
        reg_results = data.darwin.plot_linear_regression(
            ax,
            xdata=data.year.to_numpy(),
            ydata=data[args.var[0]].to_numpy(),
        )
        print(reg_results.summary())

        print(reg_results.t_test("x1 = 0").pvalue)
        pval_label = (
            "p < 0.05"
            if reg_results.t_test("x1 = 0").pvalue < pvalue_threshold
            else "p >= 0.05"
        )
        if data.units == "K" or data.units == "C":
            data.units = "°C"
        elif "/d" in data.units:
            data.units = data.units.replace("/d", "")
            reg_results.params[1] = reg_results.params[1] * 365
        labels = [
            f"$\\beta$ = {10 * reg_results.params[1]:.2f} {data.units} / dec",
            f"$r^2$ = {reg_results.rsquared:.2f}",
            pval_label,
        ]
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
        if not args.plot:
            exit()
        fig.savefig(output)
        print(f"Output: {output}")
        exit()
    if args.type == "timeseries":
        ds = ds.mean(dim=("south_north", "west_east")).squeeze()
        ds.darwin.assign_seasons_as_coords()
        fig, ax = plt.subplots(
            2, 2, figsize=(12, 8), sharex=True, sharey=True, layout="tight"
        )
        ds.darwin.plot_facet_from_coords(ax=ax)
        if not args.plot:
            exit()
        fig.savefig(output)
        print(f"Output: {output}")
        exit()
    elif args.type == "regression":
        fig, ax = plt.subplots(
            2, 2, figsize=(12, 8), sharex=False, sharey=False, layout="tight"
        )
        ds = ds.mean(dim=("south_north", "west_east")).squeeze()
        ds.darwin.assign_seasons_as_coords()
        ds.darwin.plot_facet_from_coords(ax=ax, xvar=args.var[0], yvar=args.var[1])
        if not args.plot:
            exit()
        fig.savefig(output)
        print(f"Output: {output}")
        exit()
    elif args.type == "mean":
        timstd = np.mean(
            ds.mean(dim=("south_north", "west_east"))
            .std(dim="time")
            .squeeze()[args.var]
            .to_numpy()
        )
        ds = ds.mean(dim="time").squeeze()[args.var].to_numpy()
        mean = np.nanmean(ds)
        std = np.nanstd(ds)
        print(f"Mean: {np.round(mean, 2)}")
        print(f"Horizontal Std: {np.round(std, 3)}, {np.round(std/mean*100, 2)} %")
        print(f"Time Std: {np.round(timstd, 3)}, {np.round(timstd/mean*100, 2)} %")
        vdist = None
    elif args.type == "trend":
        ds.darwin.plot_facet_map(
            ax=ax,
            cmap="RdBu_r",
            filestem=filestem,
            landmask=landmask,
        )
    else:
        msg = f"Unknown type: {args.type}"
        raise ValueError(msg)
    ds = xr.Dataset(
        data_vars={
            args.var: (
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
    if not args.plot:
        exit()
    _, ax = plt.subplots(1, 1)
    base_map = landmask.salem.get_map()
    base_map.set_data(ds[args.var])
    base_map.set_cmap(ds.darwin.cmap)
    if args.type == "trend" and vdist:
        base_map.set_vmin(-vdist)
        base_map.set_vmax(vdist)
    base_map.set_scale_bar()
    base_map.plot(ax=ax)
    base_map.append_colorbar(ax=ax, label=args.colorbarlabel)
    plt.savefig(f"plots/{filestem}.png")
    print(f"Output: {output}")
