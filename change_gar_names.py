from argparse import ArgumentParser
from pprint import PrettyPrinter
from utils import remove_nonalphanumerics

import darwin

parser = ArgumentParser(description="Description of your program")
parser.add_argument(
    "-f",
    "--folder",
    help="Topfolder to search for files to process",
    default=darwin.FilePath("."),
)
args = vars(parser.parse_args())


pp = PrettyPrinter(indent=2)


def change_all_projections(path):
    """path: path from darwin's base folder or absolute path"""
    path = darwin.FilePath(path)
    if not path.is_absolute():
        path = darwin.base_folder / path
    files = list(path.glob("**/*.nc"))
    print(f"Base folder: {path.as_posix()}")
    print("Found the following files:")
    pp.pprint(files)
    for f in files:
        print("Working on:")
        print(f.as_posix())
        with darwin.open_dataset(
            from_path=f.as_posix(), engine="xarray", mode="a"
        ) as ds:
            print(ds.coords["west_east"].size)
            print("setting global attributes")
            ds = assign_projection_info(ds)
            print("setting time attributes")
            ds = set_calendar(ds)
            print("setting projection")
            ds = set_projection(ds)
            print("Saving dataset")
            f_new = f.parent / "new" / f.name
            ds.to_netcdf(f"{f_new}")
            print("Dataset processed")
        break


def set_calendar(ds):
    ds["time"].attrs["calendar"] = "standard"
    return ds


def set_projection(ds):
    var = ds.attrs["VARNAME"]
    ds[var].attrs["coordinates"] = "lon lat"
    return ds


def build_pyproj(projection: dict) -> str:
    return " ".join(f"+{key}={str(value)}" for key, value in projection.items())


def split_attribute(attr: str) -> str:
    # Try to split by comma
    split = attr.split(",")
    if len(split) == 1:
        split = attr.split(" ")
    return split


def check_proj_string(proj_string: str) -> str:
    return not proj_string.startswith("{")


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
        "CEN_LAT": ds.coords["south_north"].median().data,
        "CEN_LON": ds.coords["west_east"].median().data,
    }

    if not check_proj_string(ds.attrs["PROJ_ENVI_STRING"]):
        proj_split = split_attribute(ds.attrs["PROJ_ENVI_STRING"])
        projection = {
            "proj_id": remove_nonalphanumerics(proj_split[0]),
            "a": float(proj_split[1]),
            "b": float(proj_split[2]),
            "lat_0": float(proj_split[3]),
            "lon_0": float(proj_split[4]),
            "x_0": float(proj_split[5]),
            "y_0": float(proj_split[6]),
            "ellps": remove_nonalphanumerics(proj_split[-2]),
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
        attributes["PROJ_ENVI_STRING"] = pyproj_srs
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
        print(f"add {value} to attribute {key}")
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
    change_all_projections(args["folder"])
