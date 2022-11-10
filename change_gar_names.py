"""Set attributes of WRF products produced by WAVE."""
from argparse import ArgumentParser
from os import path
from pprint import PrettyPrinter

from xarray import Dataset

from darwin import FilePath, open_dataset
from utils import glob_files, remove_nonalphanumerics


def parse_input(parser):
    """Parse command line inputs.

        parser (): argparse.ArgumentParser object

    Returns:
        A list of parsed arguments.
    """
    parser.add_argument(
        "-f",
        "--folder",
        help="Topfolder to search for files to process",
        default=".",
    )
    parser.add_argument(
        "-g",
        "--glob",
        help="glob pattern to search for files",
        default="**/*.nc",
    )
    return vars(parser.parse_args())


pp = PrettyPrinter(indent=2)


def change_all_projections(path, *args, **kwargs):
    """path: path from darwin's base folder or absolute path."""
    files = glob_files(path, *args, **kwargs)
    pp.pprint(f"Base folder: {path.as_posix()}")
    pp.pprint("Found the following files:")
    pp.pprint(files)
    for f in files:
        f = FilePath(f)
        pp.pprint("Working on:")
        pp.pprint(f.as_posix())
        with open_dataset(
            from_path=f.as_posix(),
            engine="xarray",
            decode_cf=False,
        ) as ds:
            pp.pprint("setting global attributes")
            ds = assign_projection_info(ds)
            pp.pprint("setting time attributes")
            ds = set_calendar(ds)
            pp.pprint("setting projection")
            ds = set_projection(ds)
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

            ds = add_extra_attrs(ds, extra_attrs)
            temp_path = f.parent / "temp.nc"
            ds.to_netcdf(temp_path)
            temp_path.rename(f)
            pp.pprint("Dataset processed")


def set_calendar(ds):
    ds["time"].attrs["calendar"] = "standard"
    return ds


def add_extra_attrs(ds: Dataset, attrs: dict) -> Dataset:
    for key, value in attrs.items():
        ds.attrs[key] = str(value)
    return ds


def set_projection(ds):
    var = ds.attrs["VARNAME"]
    ds[var].attrs["coordinates"] = "lon lat"
    return ds


def build_pyproj(projection: dict) -> str:
    string = " ".join(f"+{key}={str(value)}" for key, value in projection.items())
    return f"{string} +no_defs"


def split_attribute(attr: str) -> str:
    # Try to split by comma
    split = attr.split(",")
    if len(split) == 1:
        split = attr.split(" ")
    return split


# def check_proj_string(proj_string: str) -> str:
#     return not proj_string.startswith("{")


def assign_projection_info(ds):
    xx = ds.coords["west_east"].values
    yy = ds.coords["south_north"].values

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
        "GRID_DX": ds.attrs["DX"],
        "GRID_DY": ds.attrs["DY"],
        "GRID_X00": x00,
        "GRID_Y00": y00,
        "GRID_X01": x01,
        "GRID_Y01": y01,
        "GRID_NX": nx,
        "GRID_NY": ny,
        "CEN_LON": 0.0,
    }

    proj_split = split_attribute(ds.attrs["PROJ_ENVI_STRING"])
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
    projection["proj"] = (
        "merc" if projection["name"].lower() == "wrfmercator" else "lcc"
    )
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


if __name__ == "__main__":
    parser = ArgumentParser(description="Description of your program")
    path = FilePath(args["folder"])
    change_all_projections(path, args["glob"])
