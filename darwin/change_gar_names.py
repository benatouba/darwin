#!/usr/bin/env python
"""Set attributes of WRF products produced by WAVE."""
from __future__ import annotations

import datetime
from argparse import ArgumentParser, Namespace
from pathlib import Path, PosixPath
from pprint import PrettyPrinter
from typing import TYPE_CHECKING

from darwin.core import FilePath, open_dataset
from darwin.utils import glob_files, remove_nonalphanumerics

if TYPE_CHECKING:
    from xarray import Dataset


def parse_input(parser: ArgumentParser) -> Namespace:
    """Parse command line inputs.

        parser (): argparse.ArgumentParser object

    Returns:
        A list of parsed arguments.
    """
    parser.add_argument(
        "-f",
        "--folder",
        help="Topfolder to search for files to process",
        default="~/data/GAR/MM/",
    )
    parser.add_argument(
        "-g",
        "--glob",
        help="glob pattern to search for files",
        default="*_????.nc",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrite existing attributes",
        action="store_true",
    )
    return vars(parser.parse_args())


pp = PrettyPrinter(indent=2)


def change_all_projections(path: str | PosixPath, glob: str, *, overwrite: bool = False) -> None:
    """path: path from darwin's base folder or absolute path."""
    files = glob_files(path, glob)
    pp.pprint(f"Base folder: {path.as_posix()}")
    pp.pprint("Found the following files:")
    pp.pprint(files)
    files = [FilePath(f) for f in files]
    for f in files:
        pp.pprint("Working on:")
        pp.pprint(f.as_posix())
        with open_dataset(
            from_path=f.as_posix(),
            engine="xarray",
            decode_cf=False,
        ) as ds:
            ds_new = ds.copy(deep=True)
            if "months" in ds_new.time.attrs["units"] or "years" in ds_new.time.attrs["units"]:
                pp.pprint("correcting time")
                ds_new = correct_time(ds_new)
            elif "year" in ds.attrs:
                pp.pprint("correcting time")
                ds_new = correct_time(ds_new)
                # pp.pprint("year attribute already set, skipping set overwrite option to overwrite")
                # continue
            pp.pprint("setting global attributes")
            ds_new = assign_projection_info(ds_new)
            pp.pprint("setting time attributes")
            ds_new = set_calendar(ds_new)
            pp.pprint("setting projection")
            ds_new = set_projection(ds_new)
            pp.pprint("Saving dataset")
            extra_attrs = {
                "experiment": f.experiment,
                "frequency": f.frequency,
                "dimensionality": f.dimensionality,
            }
            if hasattr(f, "year"):
                extra_attrs["year"] = f.year
            if hasattr(f, "frequency"):
                extra_attrs["frequency"] = f.frequency

            ds_new = add_extra_attrs(ds_new, extra_attrs)
            temp_path = Path(f"{f}_temp")
            ds_new.to_netcdf(temp_path, mode="w")
            if overwrite:
                temp_path.rename(f)
            pp.pprint(f"Dataset {f.name} processed")


def set_calendar(ds: Dataset) -> Dataset:
    """Set calendar attribute to standard.

    Args:
        ds: xarray.Dataset

    Returns:
        xarray.Dataset with standard calendar attribute.
    """
    ds["time"].attrs["calendar"] = "standard"
    return ds
    # return ds.convert_calendar("standard")


def add_extra_attrs(ds: Dataset, attrs: dict) -> Dataset:
    """Add extra attributes to dataset.

    Args:
        ds: xarray.Dataset
        attrs: dictionary with attributes to add

    Returns:
        xarray.Dataset with extra attributes.
    """
    for key, value in attrs.items():
        ds.attrs[key] = str(value)
    return ds


def set_projection(ds: Dataset) -> Dataset:
    """Set projection attributes.

    Args:
        ds: xarray.Dataset

    Returns:
        xarray.Dataset with projection attributes.
    """
    var = ds.attrs["VARNAME"]
    ds[var].attrs["coordinates"] = "lon lat"
    return ds


def build_pyproj(projection: dict) -> str:
    """Build pyproj string from projection dictionary.

    Args:
        projection: Dictionary with projection attributes

    Returns:
        Projection string in pyproj format.
    """
    string = " ".join(f"+{key}={value!s}" for key, value in projection.items())
    return f"{string} +no_defs"


def split_attribute(attr: str) -> str:
    """Split attribute by comma or space.

    Args:
        attr: Attribute to split

    Returns:
        List with split attribute.
    """
    # Try to split by comma
    split = attr.split(",")
    if len(split) == 1:
        split = attr.split(" ")
    return split


# def check_proj_string(proj_string: str) -> str:
#     return not proj_string.startswith("{")


def assign_projection_info(ds: Dataset) -> Dataset:
    """Assign projection attributes to dataset.

    Args:
        ds: xarray.Dataset

    Returns:
        xarray.Dataset with projection attributes.
    """
    xx = ds.coords["west_east"].to_numpy()
    yy = ds.coords["south_north"].to_numpy()

    nx = xx.size
    ny = yy.size

    x00 = xx.min()
    y00 = yy.min()
    x01 = xx.min()
    y01 = yy.max()

    attributes = {
        # "TITLE": title,
        # "DATA_NOTES": data_notes,
        # "WRF_VERSION": wrf_version,
        # "CREATED_BY": created_by,
        # "INSTITUTION": institution,
        # "CREATION_DATE": creation_date,
        # "SOFTWARE_NOTES": software_notes,
        # "VARNAME": varname,
        # "DOMAIN": str(domain),
        # "NESTED": nested,
        "TIME_ZONE": "UTC",
        # "PRODUCT_LEVEL": str(product_level),
        # "LEVEL_INFO": f"{level_info} S: static",
        "PROJ_DATUM": "wgs-84",
        "Conventions": "CF-1.8",
        "GRID_INFO": (
            "Grid spacing: GRID_DX and GRID_DY (unit: m), Down left corner: GRID_X00 and "
            "GRID_Y00 (unit: m), Upper Left Corner: GRID_X01 and GRID_Y01 (unit: m)"
        ),
        "GRID_DX": get_grid_distance_attribute(ds.attrs, "x"),
        "GRID_DY": get_grid_distance_attribute(ds.attrs, "y"),
        "GRID_X00": x00,
        "GRID_Y00": y00,
        "GRID_X01": x01,
        "GRID_Y01": y01,
        "GRID_NX": nx,
        "GRID_NY": ny,
        "CEN_LON": 0.0,
    }

    proj_split = (
        split_attribute(ds.attrs["PROJ_ENVI_STRING"])
        if hasattr(ds.attrs, "PROJ_ENVI_STRING")
        else None
    )
    if proj_split:
        projection = {
            # "proj_id": remove_nonalphanumerics(proj_split[0]),
            # "a": float(proj_split[1]),
            # "b": float(proj_split[2]),
            "k_0": 1.0,
            "units": "m",
            "lat_0": float(proj_split[3]),
            "lon_0": float(proj_split[4]),
            "x_0": float(proj_split[5]),
            "y_0": float(proj_split[6]),
            "ellps": remove_nonalphanumerics(str(proj_split[-2])),
            "datum": remove_nonalphanumerics(str(proj_split[-2])),
            "name": remove_nonalphanumerics(str(proj_split[-1])),
        }
        projection["proj"] = "merc" if projection["name"].lower() == "wrfmercator" else "lcc"
        if projection["name"].lower() == "wrfmercator":
            proj_name = "WRF Mercator"
        elif projection["name"].lower() == "wrflambertconformal":
            projection["sp1"] = float(proj_split[7])
            projection["sp2"] = float(proj_split[8])
            proj_name = "Lambert Conformal Conic"
        pyproj_srs = build_pyproj(projection)
        attributes["PROJ_NAME"] = proj_name
        attributes["LON_0"] = projection["lon_0"]
        attributes["LAT_0"] = projection["lat_0"]
        attributes["TRUELAT1"] = projection["lat_0"]
        # attributes["PROJ_ENVI_STRING"] = pyproj_srs
        attributes["pyproj_srs"] = pyproj_srs
    else:
        attributes["pyproj_srs"] = (
            "+k_0=1.0 +units=m +lat_0=2.0 +lon_0=-90.31006622 "
            "+x_0=0.0 +y_0=0.0 +ellps=WGS84 +datum=WGS84 +name=WRFMercator +proj=merc +no_defs"
        )
    # lcc_attrs = {
    #     "PROJ_SEMIMAJOR_AXIS": projlis[1],
    #     "PROJ_SEMIMINOR_AXIS": projlis[2],
    #     "PROJ_CENTRAL_LAT": projlis[3],
    #     "PROJ_CENTRAL_LON": projlis[4],
    #     "PROJ_FALSE_EASTING": projlis[5],
    #     "PROJ_FALSE_NORTHING": projlis[6],
    #     "PROJ_STANDARD_PAR1": projlis[7],
    #     "PROJ_STANDARD_PAR2": projlis[8],
    # }
    for key, value in attributes.items():
        pp.pprint(f"add {value} to attribute {key}")
        ds.attrs[key] = str(value)
    return ds


def get_grid_distance_attribute(attrs: dict, dimension: str) -> str:
    """Return the grid distance attribute from the dataset attributes.

    Args:
        attrs: Dataset attributes.
        dimension: Dimension to get the grid distance attribute for.

    Returns:
        Grid distance attribute.
    """
    msg = "No grid distance attribute found."
    if dimension.lower() == "x":
        if "DX" in attrs:
            return attrs["DX"]
        if "GRID_DX" in attrs:
            return attrs["GRID_DX"]
        raise ValueError(msg)
    if dimension.lower() == "y":
        if "DY" in attrs:
            return attrs["DY"]
        if "GRID_DY" in attrs:
            return attrs["GRID_DY"]
        raise ValueError(msg)
    msg = f"Dimension {dimension} not supported."
    raise ValueError(msg)


# if projection == "lcc":
#     for key, value in lcc_attrs.items():
#         ds.attrs[key] = value
#
# pyproj_srs = (
#     f"+proj={projection['proj']} +lat_0={str(projection['lat_0'])} "
#     f"+lon_0={str(projection['lon_0'])} +k=1 "
#     f"+x_0={str(projection['x_0'])} +y_0={str(projection['y_0'])} +ellps={projection['ellps']} "
#     f"+datum={projection['ellps']} +units=m +no_defs"
# )
def get_first_day_of_month_doys(year: int) -> list:
    """Return the day of the year for the first day of each month.

    Args:
        year: Year to get the first day of each month for.

    Returns:
        List of day of the year for the first day of each month.
    """
    first_day_of_month_doys = []
    for month in range(1, 13):
        first_day = datetime.date(year, month, 1)
        day_of_year = first_day.timetuple().tm_yday
        first_day_of_month_doys.append(day_of_year - 1)
    return first_day_of_month_doys


def correct_time(ds: Dataset) -> Dataset:
    """Correct the time dimension values and units for a dataset.

    This function is used to change the time units to days if they are in months or years.
    The values are changed accordingly.

    Args:
        ds: Dataset to correct.

    Returns:
        Dataset with corrected time dimension.
    """
    pp.pprint(ds.time.attrs["units"])
    if "months" in ds.time.attrs["units"]:
        pp.pprint("months to days in time units")
        old_attrs = ds.time.attrs
        ds = ds.assign_coords(
            {"time": get_first_day_of_month_doys(int(ds.time.attrs["units"].split(" ")[2][:4]))},
        )
        ds.time.attrs = old_attrs
        ds.time.attrs["units"] = old_attrs["units"].replace("months", "days")
    elif "years" in ds.time.attrs["units"]:
        pp.pprint("years to days in time units")
        ds.time.attrs["units"] = ds.time.attrs["units"].replace("years", "days")
    pp.pprint(ds.time)
    return ds


if __name__ == "__main__":
    parser = ArgumentParser(description="Description of your program")
    args = parse_input(parser)
    filepath = FilePath(args["folder"])
    change_all_projections(filepath, args["glob"], overwrite=args["overwrite"])