"""A script to download CDS data from the CDS API."""

import argparse
import cdsapi

c = cdsapi.Client()

parser = argparse.ArgumentParser(description="Download CDS data.")
parser.add_argument(
    "--variable",
    "-v",
    type=str,
    help="The variable to download.",
    required=True,
)
parser.add_argument(
    "--start",
    "-s",
    type=str,
    help="The start date to download.",
    required=True,
)
parser.add_argument(
    "--end",
    "-e",
    type=str,
    help="The end date to download.",
    required=True,
)
parser.add_argument(
    "--dataset",
    "-ds",
    type=str,
    help="The level to download.",
    choices=["sfc", "pl"],
    required=True,
)
parser.add_argument(
    "--time",
    "-t",
    type=list,
    help="The time to download.",
    default=[
        "00:00",
        "01:00",
        "02:00",
        "03:00",
        "04:00",
        "05:00",
        "06:00",
        "07:00",
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00",
        "18:00",
        "19:00",
        "20:00",
        "21:00",
        "22:00",
        "23:00",
    ],
)
args = parser.parse_args()

c.retrieve(
    "reanalysis-era5-single-levels"
    if args.dataset == "sfc"
    else "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": args.variable,
        "date": f"{args.start}/{args.end}",
        "time": args.time,
        "format": "netcdf",
    },
    f"ERA5_{args.dataset}_{args.variable}_{args.start}-{args.end}.nc",
)