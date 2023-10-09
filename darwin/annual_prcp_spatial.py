from matplotlib.colors import LinearSegmentedColormap
import salem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


class SalemExtend(salem.DatasetAccessor):
    def __init__(self):
        super().__init__()

    def plot_map(self, model, observation, lats, lons, cmap, scalebar=False):
        # berlin = "/home/wang/Data/CER/DEU_adm/DEU_adm1.shp"
        smap = self.salem.get_map(data=model, cmap=cmap, vmin=380, vmax=1020)
        x, y = smap.grid.transform(lons, lats)
        smap.set_shapefile(countries=False)
        # smap.set_levels(levels=levs)
        dl = salem.DataLevels(observation, cmap=cmap, extend="max", vmin=380, vmax=1020)
        ax.scatter(x, y, color=dl.to_rgb(), s=20, edgecolors="k", linewidths=1, zorder=10)
        # smap.set_shapefile(shape=berlin, linewidth=0.5)
        smap.set_lonlat_contours(xinterval=1)
        if scalebar:
            smap.set_scale_bar(location=(0.85, 0.08))
        smap.visualize(addcbar=False)
        return smap


def read_CER(file):
    ds = salem.open_wrf_dataset(file)
    c1 = ds.lat.values.shape[0]
    c2 = ds.lat.values.shape[1]
    ds = ds.isel(west_east=np.arange(5, c1 - 5), south_north=np.arange(5, c2 - 5))
    return ds


color_map = LinearSegmentedColormap.from_list(
    "mycmap",
    ["white", "steelblue", "c", "khaki", "orange", "orangered", "r", "darkred"],
)


class Station():
    def __init__(self, f):
        self.data = self.__read_csv__(f)
        return self.data

    def __str__(self):
        self.data = self.Stations_id.astype(str)
        print(self.data.Stations_id.values)

    def __read_csv__(self, f):
        return pd.read_csv(f)

    def __get_data__(self):
        ob = self.data[self.data.columns[:-1]]
        return ob.sum()


pattern = "/home/wang/Data/CER/CERv2/{0}/y/2d/CERv2_{1}_d02km_y_2d_prcp_2017.nc"
tests = ["test2", "test6", "test9", "test10", "test14", "test15", "test16"]
titles = [
    "Test2_REF",
    "Test6_CU1 (KF)",
    "Test9_COMB1 (KF+YSU)",
    "Test10_COMB2 (KF+UCM)",
    "Test14_COMB6 (KF+Milbrandt)",
    "Test15_COMB7 (KF+AThompson)",
    "Test16_CU3 (KFCuP)",
]
fig = plt.figure(figsize=(16, 12))
for i, test in enumerate(tests):
    f = pattern.format(test, test)
    ds_test = read_CER(f)
    ax = plt.subplot(2, 4, i + 1)
    smap = plot_map(
        ax,
        ds_test,
        ds_test.prcp * 24 * 365,
        ob_sum.values,
        df_info.geoBreite,
        df_info.geoLaenge,
        color_map,
    )
    ax.set_title(titles[i])

f = "/home/wang/Data/era5/Other/ERA5_daily_europe_2017.nc"
ds1 = salem.open_xr_dataset(f)
ds_era5 = ds_test.salem.transform(ds1)

ax = plt.subplot(248)
smap = plot_map(
    ax,
    ds_test,
    ds_era5.tp.sum(axis=0) * 1000,
    ob_sum.values,
    df_info.geoBreite,
    df_info.geoLaenge,
    color_map,
    scalebar=True,
)
ax.set_title("ERA5")

cax = plt.axes([0.3, 0.03, 0.4, 0.02])
smap.colorbarbase(cax, orientation="horizontal")
