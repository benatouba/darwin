"""Convert GAR V1 product netCDF files to geotiffs in EPSG:32616.

Usage:

    convert.py <input_file> <output_path>

Arguments:
    input_file: Input file
    output_path: Output file
"""

import os
from pathlib import Path

import numpy as np
import pyproj
import rioxarray
import typer
import xarray as xr

app = typer.Typer()

wrf_proj = pyproj.CRS.from_proj4(
    """
    +proj=merc +lat_0=-0.405 +lon_0=-90.31006622 +R=6370000 +k_0=1 +ellps=sphere"
    """
)
# wgs_proj = pyproj.Proj(proj="latlong", datum="WGS84")
tmerc_proj = pyproj.CRS.from_proj4(
    """
    +proj=merc +k_0=1.0 +units=m +lat_0=0 +lon_0=-90.31006622 +x_0=0.0 +y_0=0.0
    +ellps=WGS84 +datum=WGS84 +no_defs
    """
)


def nc_to_geotiff(da: xr.DataArray, output: Path) -> None:
    """Convert a xarray DataArray file to a geotiff.

    Args:
        da: xr.DataArray
        output: Output file path
    """
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da.rio.write_crs("EPSG:32616", inplace=True)
    output = output.with_suffix(".tif") if output.suffix != ".tif" else output
    da.rio.to_raster(output)
    print(f"Wrote to {output}")


def to_mercator(ds: xr.Dataset) -> xr.DataArray:
    """Convert a dataset to mercator projection EPSG:32616.

    Args:
        ds: The dataset to convert.

    Returns:
        The converted dataset.
    """
    # e, n = pyproj.transform(wgs_proj, wrf_proj, -90.31006622, -0.405)
    dx, dy = int(ds.attrs["DX"]), int(ds.attrs["DY"])
    _, nx, ny = ds.sizes.values()
    transformer = pyproj.Transformer.from_crs(tmerc_proj, wrf_proj)
    x0, y0 = transformer.transform(-112541, 185336.5 - 460000)
    xx, yy = np.meshgrid(np.arange(nx) * dx + x0, np.arange(ny) * dy + y0)
    our_es, our_ns = transformer.transform(xx, yy)
    var = ds.attrs["VARNAME"]
    return xr.DataArray(
        data=ds[var].values,
        dims=["time", "y", "x"],
        coords={"time": ds.time, "x": our_es[0, :], "y": our_ns[:, 0]},
        attrs=ds.attrs,
        name=var,
    )


@app.command()
def convert(
    input_file: Path = typer.Argument(..., help="Input file"),  # noqa: B008
    output_path: Path = typer.Argument(..., help="Output file"),  # noqa: B008
) -> None:
    """Convert a netCDF file to a geotiff.

    Args:
        input_file: Input file
        output_path: Output file
    """
    ds = xr.open_dataset(input_file)
    ds = to_mercator(ds)
    nc_to_geotiff(ds, output_path)


def main():
    folder = Path.cwd()
    files = list(folder.glob("static/*.nc"))
    for f in files:
        output = folder / "geotiff" / f.relative_to(Path.cwd()).with_suffix("")
        if output.exists():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Converting {f}")
        convert(f, output)


if __name__ == "__main__":
    folder = Path("/local_data/webdav/GAR/V1/d02km/")
    os.chdir(folder)
    main()
