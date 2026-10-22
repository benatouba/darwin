import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.typing as npt
import salem
import statsmodels.api as sm
import xarray as xr

mpl.rcParams["axes.axisbelow"] = True


# NOTE: shouldn't be needed
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


class DarwinBaseMap(salem.Map):
    """BaseMap class for darwin."""

    def __init__(self, base_map: salem.Map) -> None:
        super().__init__()

    def plot_map(
        self,
        ax: plt.Axes,
        data: xr.DataArray,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        """Plot the map.

        Args:
            ax: The axes to plot on.
            data: The data to plot.
            cmap: The colormap to use. If None, the default (viridis) is used.
            vmin: The minimum value for the colormap. If None, the minimum value
            in the data is used.
            vmax: The maximum value for the colormap. If None, the maximum value
            in the data is used.
        """
        self.set_data(data)
        if cmap is not None:
            self.set_cmap(cmap)
        if vmin is not None:
            self.set_vmin(vmin)
        if vmax is not None:
            self.set_vmax(vmax)
        self.set_scale_bar()
        self.plot(ax=ax)
        self.append_colorbar(ax=ax, label=data.attrs["units"])
