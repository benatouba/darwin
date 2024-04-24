"""Combine files to a certain or the measurement period."""
import argparse
from datetime import date
from pathlib import Path
from pprint import PrettyPrinter

from cdo import Cdo
from icecream import ic

from darwin.defaults import measured_vars
from darwin.utils import glob_files


def parse_input(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse command line inputs.

        parser (): argparse.ArgumentParser object

    Returns:
        A list of parsed arguments.
    """
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     help="Model to process",
    #     default="MM",
    # )
    parser.add_argument(
        "-f",
        "--folder",
        help="Topfolder to search for files to process",
        default="~/data/GAR/MM",
    )
    parser.add_argument(
        "-s",
        "--start",
        help="Start date of period to process",
        type=date.fromisoformat,
        default="1980-01-01",
    )
    parser.add_argument(
        "-e",
        "--end",
        help="End date of period to process",
        type=date.fromisoformat,
        default="2023-12-31",
    )
    parser.add_argument(
        "-g",
        "--glob",
        help="glob pattern to search for files",
        default="*.nc",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrite existing attributes",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--vars",
        help="Variables to process",
        nargs="+",
        default=list(measured_vars.keys()),
    )
    return vars(parser.parse_args())


cdo = Cdo(returnNoneOnError=True)
pp = PrettyPrinter(indent=2)


def get_files(
    folder: Path = "~/data/GAR/GAR",
    year: int = 2023,
    variables: tuple = ("prcp", "t2", "q2", "rh2"),
) -> dict:
    """Get files for a certain year and variables.

    Args:
        year: year to get files for
        variables: The variables to get files for
        folder: The folder to get files from

    Returns:
        A dictionary with the variables as keys and the files as values.
    """
    files = {}
    filelist = glob_files(folder, f"*_[dmy]_2d_*_{year}.nc*")
    ic(filelist)
    for f in filelist:
        for v in variables:
            if f"_{v}_" in f.as_posix():
                files[v] = f
    return files


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser()
    args = parse_input(parser)
    start_year = args["start"].year
    end_year = args["end"].year
    folder = Path(args["folder"]).expanduser()

    files = []
    [
        files.append(get_files(folder, y, list(measured_vars.keys())))
        for y in range(start_year, end_year + 1)
    ]
    ic(files)
    for v in args["vars"]:
        inputs = " ".join([str(f[str(v)]) for f in files])
        ofile = str(files[0][v])
        if start_year != end_year:
            inputs = f"-cat {inputs}"
            ofile = str(files[0][v]).replace(f"_{start_year}", "")
        for f, i in zip(files, range(len(files))):
            pp.pprint(f"{v} {start_year + i}: {f[v]}")
        pp.pprint(f"Processing {v} to {ofile}")
        cdo.seldate(
            f"{args['start']},{args['end']}",
            input=inputs,
            output=ofile,
            options="-L",
        )
        [f.pop(v) for f in files]
        del inputs
        del ofile


if __name__ == "__main__":
    main()