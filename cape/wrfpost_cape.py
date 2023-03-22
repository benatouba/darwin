import argparse
import glob
import os.path as path
import subprocess
from calendar import monthrange
from datetime import datetime, timedelta
from platform import python_version
from shutil import copyfile

import numpy as np
import pandas as pd
import salem
import wrf
import xarray as xr
from netCDF4 import Dataset, date2num


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--in_dir",
        help="The directory where the pp files are stored without yearly subfolder",
    )
    parser.add_argument(
        "-w",
        "--working_dir",
        help="The working directory to store and unzip pp files, and to store the output",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        help="The working directory to store and unzip pp files, and to store the output",
    )
    parser.add_argument(
        "-y", "--year", type=int, help="The year you want to calculate CAPE"
    )
    parser.add_argument(
        "-m",
        "--month",
        nargs="+",
        type=int,
        help="The month(s) you want to calculated CAPE",
    )
    return parser.parse_args()


def run_cmd(in_str, dry_run=False):
    """
    Run a cmd in the shell.

    :param in_str: the command to run
    :param dry_run: if true the command is printed instead executed
    """
    if not dry_run:
        return_code = subprocess.run(in_str, shell=True, check=True)
        if return_code != 0:
            print("Returncode: ", return_code)
                    #  raise Exception("Cant execute: " + in_str)
    else:
        print("Call would be:" + in_str)


def cape_calculate(f):
    """Calculate 2d CAPE."""
    wrf_nc = Dataset(f)
    cape_2d = wrf.getvar(wrf_nc, "cape_2d", timeidx=wrf.ALL_TIMES).isel(
        mcape_mcin_lcl_lfc=0, Time=np.arange(0, 24)
    )
    wrf_nc.close()
    cape_2d = cape_2d.to_dataset()
    return cape_2d


def copy_one_day(fg_dir, working_dir, date):
    source_dir = path.join(
        fg_dir,
        datetime.strftime(date, "%Y"),
        datetime.strftime(date, "%Y-%m"),
        datetime.strftime(date, "%Y-%m-%d"),
    )
    print("from dir" + source_dir)
    source_wrfpost = glob.glob(path.join(source_dir, "wrfpost*"))[0]
    print(source_wrfpost)
    local_wrfpost = path.join(working_dir, path.split(source_wrfpost)[-1])
    print("copying ", path.join(source_dir, "wrfpost*"))
    copyfile(source_wrfpost, local_wrfpost)
    if "gz" in local_wrfpost:
        run_cmd("gunzip " + local_wrfpost)
        print("unziping ", local_wrfpost)
        local_wrfpost = local_wrfpost[: -len(".gz")]
    return local_wrfpost


def write_cape_to_nc(cape_data, sample_file, out_file):
    ds = salem.open_wrf_dataset(sample_file)
    dates = pd.to_datetime(cape_data.Time.values)
    dataset = Dataset(out_file, "w", format="NETCDF4_CLASSIC")

    # set global attributes
    dataset.TITLE = "HAR v2 d10km"
    dataset.DATA_NOTES = (
        "File generated with the output of successive model "
        "runs of 36H (first 12 hours discarded for spin-up)"
    )
    dataset.WRF_VERSION = (
        ds.attrs["TITLE"].split()[2] + " " + ds.attrs["TITLE"].split()[3]
    )
    dataset.CREATED_BY = "Xun Wang - xun.wang@tu-berlin.de"
    dataset.INSTITUTION = (
        "Technische Universitaet Berlin, Institute of Ecology, Chair of Climatology"
    )
    now = datetime.now()
    dataset.CREATION_DATE = now.strftime("%d-%m-%Y %H:%M:%S")
    dataset.SOFTWARE_NOTES = "Python " + python_version()
    dataset.VARNAME = "cape"
    domain_num = path.split(sample_file)[-1].split("_")[1]
    dataset.DOMAIN = domain_num.strip("d0")
    dataset.NESTED = "YES" if domain_num != "d01" else "NO"
    dataset.TIME_ZONE = "UTC"
    dataset.PRODUCT_LEVEL = "H"
    dataset.LEVEL_INFO = (
        "H: original simulation output; D: daily; M: monthly; Y: yearly; S: static"
    )
    dataset.PROJ_NAME = ds.attrs["MAP_PROJ_CHAR"]
    dataset.PROJ_CENTRAL_LON = ds.attrs["STAND_LON"]
    dataset.PROJ_CENTRAL_LAT = ds.attrs["MOAD_CEN_LAT"]
    dataset.PROJ_STANDARD_PAR1 = ds.attrs["TRUELAT1"]
    dataset.PROJ_STANDARD_PAR2 = ds.attrs["TRUELAT2"]
    dataset.PROJ_SEMIMAJOR_AXIS = "6370000.0"
    dataset.PROJ_SEMIMINOR_AXIS = "6370000.0"
    dataset.PROJ_FALSE_EASTING = "0.0"
    dataset.PROJ_FALSE_NORTHING = "0.0"
    dataset.PROJ_DATUM = "WGS-84"
    dataset.PYPROJ_SRS = ds.attrs["pyproj_srs"]
    dataset.GRID_INFO = (
        "Grid spacing: GRID_DX and GRID_DY (unit: m), "
        "Down left corner: GRID_X00 and GRID_Y00 (unit: m), \
                         Upper Left Corner: GRID_X01 and GRID_Y01 (unit: m)"
    )
    dataset.GRID_DX = ds.attrs["DX"]
    dataset.GRID_DY = ds.attrs["DY"]
    dataset.GRID_X00 = ds.west_east.values.min()
    dataset.GRID_Y00 = ds.south_north.values.min()
    dataset.GRID_X01 = ds.west_east.values.min()
    dataset.GRID_Y01 = ds.south_north.values.max()
    dataset.GRID_NX = len(ds.west_east)
    dataset.GRID_NY = len(ds.south_north)

    # create dimensions
    dataset.createDimension("time", None)
    dataset.createDimension("west_east", ds.west_east.size)
    dataset.createDimension("south_north", ds.south_north.size)

    # create coordinates
    times = dataset.createVariable("time", np.float64, ("time",))
    west_easts = dataset.createVariable("west_east", np.float32, ("west_east",))
    south_norths = dataset.createVariable("south_north", np.float32, ("south_north",))
    lon = dataset.createVariable("lon", np.float32, ("south_north", "west_east"))
    lat = dataset.createVariable("lat", np.float32, ("south_north", "west_east"))

    # create variables
    cape = dataset.createVariable(
        "cape", np.float32, ("time", "south_north", "west_east")
    )

    # set attributes
    # for coordinate
    times.long_name = "Time"
    times.units = f"hour since {str(dates[0])}"
    times.calendar = "standard"

    south_norths.long_name = "y-coordinate in Cartesian system"
    south_norths.units = "m"
    west_easts.long_name = "x-coordinate in Cartesian system"
    west_easts.units = "m"

    lat.long_name = "Latitude"
    lat.units = "degree_north"
    lon.long_name = "Longitude"
    lon.units = "degree_east"

    # for variables
    cape.long_name = "convective available potential energy"
    cape.units = "J kg-1"
    cape.agg_method = "mean"
    cape.coordinates = "lon lat"

    # set values
    cape[:] = cape_data.cape_2d.values

    dates_new = [
        datetime(dates.year[0], dates.month[0], dates.day[0])
        + n * timedelta(hours=1)
        for n in range(cape_data.cape_2d.shape[0])
    ]
    times[:] = date2num(dates_new, units=times.units, calendar=times.calendar)
    south_norths[:] = ds.south_north.values
    west_easts[:] = ds.west_east.values
    lat[:] = ds.lat.values
    lon[:] = ds.lon.values
    dataset.close()


def extract_cape_month(fg_dir, working_dir, out_dir, year, month):
    days = monthrange(year, month)[1]
    day = 2
    local_wrf = copy_one_day(fg_dir, working_dir, datetime(year, month, day))
    cape_all = cape_calculate(local_wrf)
    run_cmd("rm -f " + local_wrf)

    for day in range(3, days):
        local_wrf = copy_one_day(fg_dir, working_dir, datetime(year, month, day))
        cape_one_day = cape_calculate(local_wrf)
        cape_all = xr.combine_by_coords([cape_all, cape_one_day])
        if day != days:
            run_cmd("rm -f " + local_wrf)

    out_name_pattern = "HARv2_d10km_h_2d_cape_{0}.nc"
    out_name = out_name_pattern.format(datetime(year, month, 1).strftime("%Y-%m"))
    write_cape_to_nc(cape_all, local_wrf, path.join(out_dir, out_name))
    run_cmd("rm -f " + local_wrf)


def extract_cape(fg_dir, working_dir, out_dir, year, month):
    for mon in month:
        extract_cape_month(fg_dir, working_dir, out_dir, year, mon)


def main():
    args = get_args()
    extract_cape(args.in_dir, args.working_dir, args.out_dir, args.year, args.month)


if __name__ == "__main__":
    main()
