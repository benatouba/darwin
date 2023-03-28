"""Calculate and extract CAPE products from wrfpost files."""
import argparse
import os.path as path
import subprocess
from calendar import monthrange
from datetime import datetime, timedelta
from glob import glob
from shutil import copyfile

import numpy as np
import pandas as pd
import salem
import wrf
import xarray as xr
from netCDF4 import Dataset, date2num

from .metadata import get_attributes


def get_args():
    """Parse arguments."""
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
    parser.add_argument("-y", "--year", type=int, help="The year you want to calculate CAPE")
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
    """Calculate 2-dimensional CAPE.

    Args:
        f (string): Input file path.

    Returns:
        netcdf.Dataset
    """
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
    print("Source: " + source_dir)
    source_wrfpost = glob(path.join(source_dir, "wrfpost*"))[0]
    print(source_wrfpost)
    local_wrfpost = path.join(working_dir, path.split(source_wrfpost)[-1])
    print("copying ", path.join(source_dir, "wrfpost*"))
    copyfile(source_wrfpost, local_wrfpost)
    if "gz" in local_wrfpost:
        run_cmd("gunzip " + local_wrfpost)
        print("unziping ", local_wrfpost)
        local_wrfpost = local_wrfpost[: -len(".gz")]
    return local_wrfpost


def write_cape_to_nc(cape_data, sample, out_file):
    dates = pd.to_datetime(cape_data.Time.values)
    dataset = Dataset(out_file, "w", format="NETCDF4_CLASSIC")

    dataset["attrs"] = get_attributes(dataset)
    # create dimensions
    dataset.createDimension("time", None)
    dataset.createDimension("west_east", sample.west_east.size)
    dataset.createDimension("south_north", sample.south_north.size)

    # create coordinates
    times = dataset.createVariable("time", np.float64, ("time",))
    west_east = dataset.createVariable("west_east", np.float32, ("west_east",))
    south_north = dataset.createVariable("south_north", np.float32, ("south_north",))
    lon = dataset.createVariable("lon", np.float32, ("south_north", "west_east"))
    lat = dataset.createVariable("lat", np.float32, ("south_north", "west_east"))

    # create variables
    cape = dataset.createVariable("cape", np.float32, ("time", "south_north", "west_east"))

    # set attributes
    # for coordinate
    times.long_name = "Time"
    times.units = f"hour since {str(dates[0])}"
    times.calendar = "standard"

    south_north.long_name = "y-coordinate in Cartesian system"
    south_north.units = "m"
    west_east.long_name = "x-coordinate in Cartesian system"
    west_east.units = "m"

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
        datetime(dates.year[0], dates.month[0], dates.day[0]) + n * timedelta(hours=1)
        for n in range(cape_data.cape_2d.shape[0])
    ]
    times[:] = date2num(dates_new, units=times.units, calendar=times.calendar)
    south_north[:] = sample.south_north.values
    west_east[:] = sample.west_east.values
    lat[:] = sample.lat.values
    lon[:] = sample.lon.values
    dataset.close()


def extract_cape_month(fg_dir, working_dir, out_dir, year, month):
    length_of_month = monthrange(year, month)[1]
    cape_all = None
    for day in range(1, length_of_month + 1):
        local_wrf = copy_one_day(fg_dir, working_dir, datetime(year, month, day))
        cape_one_day = cape_calculate(local_wrf)
        if cape_all:
            cape_all = xr.combine_by_coords([cape_all, cape_one_day])
        else:
            cape_all = cape_one_day
        if day <= length_of_month:
            run_cmd("rm -f " + local_wrf)
            return

    out_name= f"GAR_d02km_h_2d_cape_{datetime(year, month, 1).strftime('%Y-%m')}.nc"
    write_cape_to_nc(cape_all, salem.open_wrf_dataset(local_wrf), path.join(out_dir, out_name))
    run_cmd("rm -f " + local_wrf)


if __name__ == "__main__":
    args = get_args()
    for mon in args.month:
        extract_cape_month(args.in_dir, args.working_dir, args.out_dir, args.year, args.month)
