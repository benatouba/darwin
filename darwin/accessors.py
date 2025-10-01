import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import salem
import seaborn as sns
import statsmodels.api as sm
import xarray as xr
from metpy.units import units as mpunits
from pint import Unit

from darwin.defaults import (
    bar_vars,
    dry_wet_map,
    dry_wet_names,
    fixed_units,
    pvalue_threshold,
    season_map,
    season_names,
    start_year,
)
from darwin.plot import DarwinBaseMap
from darwin.utils import days_in_slot, nino_anomaly, time_slots


class _DarwinAccessorBase:
    def __init__(
        self,
        xarray_obj: xr.Dataset | xr.DataArray,
        topography: Path | None = None,
        landmask: Path | None = None,
    ) -> None:
        if "bnds" in xarray_obj.dims:
            xarray_obj = xarray_obj.drop_dims("bnds")
        self._obj: xr.Dataset | xr.DataArray = xarray_obj
        self.attrs: dict[str, str] = xarray_obj.attrs
        self._seasons = season_names
        self._season_map = season_map
        self._dry_wet = dry_wet_names
        self._dry_wet_map = dry_wet_map
        self._time_varname = "time"
        if "VARNAME" in self.attrs:
            self.varname: str = self.attrs["VARNAME"]
        else:
            names = list(xarray_obj.keys())
            self.varname = str(names[-1])
        if self.varname in ("t2", "theta", "t2max") or self.varname == "geopotential":
            self.cmap = "RdBu_r"
        else:
            self.cmap = "Blues"
        if topography:
            self.topography: xr.Dataset = salem.open_xr_dataset(topography)
        if landmask:
            self.landmask: xr.Dataset = xr.open_dataset(landmask)
        if isinstance(xarray_obj, xr.DataArray):
            xarray_obj = xarray_obj.to_dataset(name="var")
            if hasattr(xarray_obj.attrs, "pyproj_srs"):
                xarray_obj.attrs["pyproj_srs"] = xarray_obj["var"].pyproj_srs

    def _map_seasons(self, data: list[int], *, dry_wet: bool = False) -> list[int]:
        if dry_wet:
            return [self._dry_wet_map[i][0] for i in data]
        return [self._season_map[i][0] for i in data]

    def convert_units(self) -> xr.Dataset | xr.DataArray:
        """Convert the units of the data.

        Args:
            units (str): The units to convert to.
        """
        units_str = self._obj[self.varname].units
        if units_str in fixed_units:
            self._obj[self.varname].attrs["units"] = fixed_units[units_str]
        da: xr.DataArray = self._obj[self.varname].metpy.quantify()
        if da.metpy.units == mpunits.kelvin:
            da = da.metpy.convert_units("°C")
        elif da.metpy.units == Unit("mm/h"):
            da = da.metpy.convert_units("mm/d")
        elif da.metpy.units == Unit("kg/kg"):
            da = da.metpy.convert_units("g/kg")
        elif da.metpy.units == Unit("Pa"):
            da = da.metpy.convert_units("hPa")
        elif da.metpy.units == Unit("m**2/s**2"):
            da = da / mpunits.Quantity(9.80665, "m/s**2")
        if self.varname == "w":
            da = da.metpy.convert_units("cm/s")

        self._obj[self.varname] = da
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

    def assign_seasons_as_coords(self, *, dry_wet: bool = False) -> xr.DataArray | xr.Dataset:
        """Assign seasons as coordinates to the dataarray.

        Args:
            dry_wet (bool, optional): Whether to assign dry and wet season instead of
            seasons. Defaults to False.

        Returns:
            Dataset: The filtered dataset.
        """
        if "seasons" in self._obj.coords:
            return self._obj

        new_coords = np.array(
            self._map_seasons(self._obj.time.dt.month.to_numpy(), dry_wet=dry_wet)
        )
        self._obj = self._obj.assign_coords(season=("time", new_coords)).swap_dims(
            {"time": "season"}
        )
        self._obj.attrs["time_varname"] = "season"
        return self._obj

    def plot_map(
        self,
        ax: plt.Axes,
        **kwargs: dict[str, Any],
    ) -> None:
        """Plot the map.

        Args:
            ax: The axes to plot on. Defaults to None.
            kwargs: Additional keyword arguments to pass to the salem Map class.
        """
        base_map = DarwinBaseMap(self._obj.salem.get_map())
        if isinstance(self._obj, xr.DataArray):
            base_map.plot_map(ax, self._obj, **kwargs)
        else:
            base_map.plot_map(ax, self._obj[self._obj.data_vars[0]], **kwargs)

    @staticmethod
    def _add_axis_annotations_for_regression(
        ax: plt.Axes,
        title: str,
        labels: list[str],
        ylabel: str,
        xlabel: str,
        **kwargs: dict[str, Any],
    ) -> plt.Axes:
        """Add axis annotations for regression.

        Args:
            ax (plt.Axes): The axes to add annotations to.
            title (str): The title of the plot.
            labels (list[str]): The labels for the regression.
            ylabel (str): The label for the y-axis.
            xlabel (str): The label for the x-axis.
            kwargs: Additional keyword arguments to pass to `ax.annotate`.
        """
        yposition = 0.98
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                xy=(0.92, yposition - i * 0.07),
                xycoords="axes fraction",
                horizontalalignment="right",
                verticalalignment="top",
                **kwargs,
            )
        ax.set_title(title)
        ax.grid(zorder=0)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        return ax

    def get_timeseries_axis_labels(self, i: int, *, is_2d: bool) -> tuple[str, str]:
        ylabels = {
            "t2": "air temperature in °C",
            "t2max": "max. air temperature in °C",
            "theta": "potential air temperature in °C",
            "geopotential": "geopotential height in m",
            "prcp": "precipitation in mm",
            "et": "actual evapotranspiration in mm",
            "wateravailability": "net precipitation in mm",
            "net precipitation": "net precipitation in mm",
            "q2": "mixing ratio in g kg$^{-1}$",
            "q": "mixing ratio in g kg$^{-1}$",
            "w": "vertical velocity in m s$^{-1}$",
            "qvapor": "water vapour mixing ratio in g kg$^{-1}$",
            "qsolid": "solid water mixing ratio in g kg$^{-1}$",
            "qliquid": "liquid water mixing ratio in g kg$^{-1}$",
            "pblh": "planetary boundary layer height in m",
        }
        ylabel = ylabels.get(self.varname, self.varname)
        if self._obj.level:
            if self.varname == "geopotential":
                ylabel = "layer thickness in m"
            ylabel = f"{self._obj.level} hPa {ylabel}"
        xlabel = "year"
        if is_2d:
            xlabel = xlabel if i in (2, 3) else ""
            ylabel = ylabel if i in (0, 2) else ""
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
        xdata: npt.ArrayLike,
        ydata: npt.ArrayLike,
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
        xdata: npt.ArrayLike,
        ydata: npt.ArrayLike,
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        ols_residuals = ols_model.resid
        resid_squared = ols_residuals**2
        ols_variance_model = sm.OLS(resid_squared, sm.add_constant(xdata)).fit()
        predicted_variances = ols_variance_model.fittedvalues
        sigma = np.diag(predicted_variances)
        sigma_inv = np.linalg.inv(sigma)
        return sm.GLS(ydata, sm.add_constant(xdata), sigma=sigma_inv).fit()

    def plot_linear_regression(
        self,
        ax: plt.Axes,
        xdata: npt.ArrayLike,
        ydata: npt.ArrayLike,
        nino_class: npt.ArrayLike,
        model_type: str = "ols",
        *,
        fill_between: bool = True,
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
            models[model_type].get_prediction(sm.add_constant(xdata)).summary_frame(alpha=0.05)
        )
        color_map = {-1: "blue", 0: "black", 1: "red"}
        colors = [color_map[nino_cls] for nino_cls in nino_class]
        if self.varname in bar_vars and xdata[0] == start_year:
            ax.bar(xdata, ydata, color=colors)
        else:
            ax.scatter(xdata, ydata, color=colors)
        # plot trend line
        # ax.autoscale(False)
        ax.plot(
            np.linspace(xdata.min(), xdata.max(), 100),
            models[model_type].params[1] * np.linspace(xdata.min(), xdata.max(), 100)
            + models[model_type].params[0],
            color="grey",
            label="trend",
        )
        # ax.plot(
        #     np.linspace(xdata.min(), xdata.max(), 100),
        #     np.linspace(xdata.min(), xdata.max(), 100),
        #     color="black",
        #     linestyle="--",
        #     label="1:1 line",
        # )
        # ax.legend()
        # ax.plot(xdata, pred_summary["mean"], color="red")
        # ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color="black")
        if fill_between:
            ax.fill_between(
                xdata,
                pred_summary["mean_ci_lower"],
                pred_summary["mean_ci_upper"],
                color="grey",
                alpha=0.2,
            )
        if self.varname == "w":
            ax.axhline(0, color="grey", linewidth=1.5, linestyle="--")
        return models[model_type]

    def plot_facet_map(
        self,
        output: Path,
        varname: str | None = None,
        overlay_dim: str | None = None,
        cmap: str = "RdBu_r",
        colorbarlabel: str | None = None,
        ax: mpl.axes.Axes | npt.NDArray | None = None,
    ) -> None:
        """Plot time series faceted by coordinates.

        This is useful for plotting seasonal data.

        Args:
            output: The stem of the output file path.
            varname: The variable name to plot.
            cmap: The colormap to use.
            colorbarlabel: The label for the colorbar.
            ax: The axes to plot on.
        """
        if varname is None:
            varname = self.varname
        if isinstance(self._obj, xr.Dataset):
            da = self._obj[varname].squeeze()
        else:
            da = self._obj.squeeze()
        if len(da.shape) == 2:
            pass
        elif len(da.shape) != 3:
            msg = "Can only plot time series trends from coordinates for 3D data"
            raise ValueError(msg)
        if ax is None:
            _, axes = plt.subplots(2, 2)
        elif isinstance(ax, np.ndarray):
            axes = ax
        else:
            axes = np.array([ax])
        # for i, time_val in enumerate(
        #     np.unique(da.coords[self._time_varname].to_numpy())
        # ):
        #     axi = axes[i // 2, i % 2] if axes.ndim > 1 else axes[i]
        data = da
        # years = da.time.dt.year.to_numpy()
        # ds = (
        #     data.assign_coords(year=(self._time_varname, years))
        #     .swap_dims({self._time_varname: "year"})
        #     .groupby("year")
        #     .mean("year")
        # ).to_numpy()
        # vals = ds
        # vals2 = vals.reshape(vals.shape[0], -1)
        # regressions = np.polyfit(np.unique(years), vals2, 1)
        # if self.varname in bar_vars:
        #     cmap = "RdBu"
        # data = regressions[0, :].reshape(vals.shape[1], vals.shape[2])
        # vmean = 0
        # print(f"Trends mean: {np.round(np.nanmean(data), 4)}")
        # print(f"Trends sd: {np.round(np.nanstd(data), 5)}")
        # create xarray dataset
        vdist = max(np.abs(np.nanmin(data)), np.nanmax(data))
        if "time" in self._obj[varname].sizes:
            base_map = self._obj[varname][0].salem.get_map()
        else:
            base_map = self._obj[varname].salem.get_map()
        base_map.set_data(data)
        base_map.set_cmap(cmap)
        if varname == "trend" and vdist:
            base_map.set_vmin(-vdist)
            base_map.set_vmax(vdist)
        base_map.set_scale_bar()
        lons = self._obj["lon"].to_numpy()
        lats = self._obj["lat"].to_numpy()
        if overlay_dim:
            significants = np.where(self._obj[overlay_dim].to_numpy() < 0.05, np.True_, np.False_)
            for i in np.arange(da.shape[0]):
                for j in np.arange(da.shape[1]):
                    if significants[i, j]:
                        base_map.set_points(
                            lons[i, j],
                            lats[i, j],
                            marker=".",
                            color="k",
                            markersize=3,
                        )
        base_map.plot(ax=ax)
        # if i == 3:
        base_map.append_colorbar(ax=ax, label=colorbarlabel)
        # plt.tight_layout()
        plt.savefig(output)
        print(f"Output: {output}")
        sys.exit()

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
        ax: mpl.axes.Axes | npt.NDArray | None = None,
        xvar: str | None = None,
        yvar: str | None = None,
        model_type: str = "ols",
        *,
        xlog: bool = False,
        ylog: bool = False,
        fill_between: bool = True,
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
            _, axes = plt.subplots(2, 2, sharex=True, sharey=True)
        elif isinstance(ax, np.ndarray):
            axes = ax
        else:
            axes = np.array([ax])
        for i, time_val in enumerate(
            # np.unique(self._obj.coords[self._time_varname].to_numpy())
            np.arange(1, 5)
        ):
            axi = axes[i // 2, i % 2] if axes.ndim > 1 else axes[i]
            years = self._obj.loc[{"season": time_val}].time.dt.year.to_numpy()
            ndays_list = [days_in_slot(*time_slots[i], year) for year in np.unique(years)]
            data = (
                self._obj.loc[{"season": time_val}]
                .assign_coords(year=("season", years))
                .swap_dims({"season": "year"})
            )
            season_to_month_nums: dict[int, tuple[int, int, int]] = {
                1: (1, 2, 3),
                2: (4, 5, 6),
                3: (7, 8, 9),
                4: (10, 11, 12),
            }
            months = season_to_month_nums[int(time_val)]
            # select the months of nino_anomaly for this time val
            nino_select = nino_anomaly.loc[nino_anomaly.index.month.isin(months)]
            nino_select.loc[:, "year"] = nino_select.index.year
            nino_select_max = nino_select.groupby("year").max("year")
            nino_select_min = nino_select.groupby("year").min("year")
            nino_select = nino_select.groupby("year").mean("year")
            nino_select["class"] = 0
            nino_select.loc[nino_select_min["value"] <= -1.13, "class"] = -1
            nino_select.loc[nino_select_max["value"] >= 1.13, "class"] = 1
            # nino_select["no_nino"] = 1
            # nino_select.loc[nino_select_max["value"] >= 1, "no_nino"] = np.nan
            nino_select["nino"] = np.where(nino_select["class"] == 1, 1, np.nan)
            nino_select["nina"] = np.where(nino_select["class"] == -1, 1, np.nan)
            nino_select["no_nina"] = np.where(nino_select["class"] != -1, 1, np.nan)
            nino_select["no_nino"] = np.where(nino_select["class"] != 1, 1, np.nan)
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
            if self.varname in bar_vars:
                data *= mpunits.Quantity(ndays_list, "d")
                data = data.magnitude
            if xvar is not None and self.varname in bar_vars:
                xdata *= mpunits.Quantity(ndays_list, "d")
                xdata = xdata.magnitude
            print(f"Data mean: {np.nanmean(data)}")
            print(f"No Nino OLS model {self._seasons[i]}")
            no_nino_data = data * nino_select["no_nino"].to_numpy()
            nino_data = data * nino_select["nino"].to_numpy()
            no_nino_ols = sm.OLS(
                no_nino_data,
                sm.add_constant(xdata),
                missing="drop",
            ).fit()
            print(f"No Nino y mean: {np.nanmean(no_nino_data)}")
            print(f"No Nino y anomaly: {np.nanmean(no_nino_data) - np.nanmean(data)}")
            print(f"No Nino R^2: {no_nino_ols.rsquared:.2f}")
            print(f"No Nino p-value: {no_nino_ols.t_test('x1 = 0').pvalue:.2f}")
            print(f"No Nino beta: {no_nino_ols.params[1]:.3f}")
            nino_data = data * nino_select["nino"].to_numpy()
            print(f"Nino y mean: {np.nanmean(nino_data)}")
            print(f"Nino y anomaly: {np.nanmean(nino_data) - np.nanmean(data)}")
            nina_data = data * nino_select["nina"].to_numpy()
            print(f"Nina y mean: {np.nanmean(nina_data)}")
            print(f"Nina y anomaly: {np.nanmean(nina_data) - np.nanmean(data)}")
            reg_results = self.plot_linear_regression(
                axi,
                xdata,
                data,
                nino_select["class"].to_numpy(),
                model_type,
                fill_between=fill_between,
            )
            if xlog:
                axi.set_xscale("log")
            if ylog:
                axi.set_yscale("log")
            # plot_regression_residuals(
            #     xdata, reg_results, f"residuals_{self.varname}_{time_val}_{model_type}"
            # )
            # print(reg_results.summary())
            is_2d = axes.ndim == ndims["2d"]
            if xvar is None and yvar is None:
                xlabel, ylabel = self.get_timeseries_axis_labels(i, is_2d=is_2d)
            else:
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
                xlabel = labels[xvar] if i in (2, 3) else ""
                ylabel = labels[yvar] if i in (0, 2) else ""

            print(f"p-value: {reg_results.t_test('x1 = 0').pvalue}")
            pval_label = (
                p_less if reg_results.t_test("x1 = 0").pvalue < pvalue_threshold else p_greater
            )
            beta = mpunits.Quantity(reg_results.params[1], self._obj[self.varname].metpy.units)
            beta_units = f"{beta.units:~P}"
            if beta.units == mpunits.degree_Celsius:
                beta_units = "K"
            split_units = beta_units.split("/")
            if len(split_units) == 2:
                if split_units[1] in ("y", "a", "h", "d"):
                    beta_units = split_units[0]
                else:
                    beta_units = f"{split_units[0]}$\\,${split_units[1]}$^{{-1}}$"
            beta_units = beta_units + r"$\,$a$^{-1}$"
            mean = data.mean()
            if split_units[0] == "K":
                split_units[0] = "°C"
            labels = [
                f"$\\overline{{y}}$ = {mean:.1f} {split_units[0]}",
                f"$\\beta$ = {beta.magnitude:.3f} {beta_units}",
                f"$R^2$ = {reg_results.rsquared:.2f}",
                pval_label,
            ]
            labels = [" $|$  ".join(labels)]
            self._add_axis_annotations_for_regression(
                axi,
                labels=labels,
                title=self._dry_wet[i] if not is_2d else self._seasons[i],
                xlabel=xlabel,
                ylabel=ylabel,
                fontsize=10.7,
            )
        ylims = plt.gca().get_ylim()
        plt.gca().set_ylim(ylims[0], ylims[1] + (ylims[1] - ylims[0]) * 0.2)
        plt.tight_layout(pad=0.3, h_pad=0, w_pad=0)

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
