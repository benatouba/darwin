from argparse import ArgumentParser
from os import rename
from pprint import PrettyPrinter

from cdo import Cdo

from darwin import FilePath
from utils import glob_files

parser = ArgumentParser(description="Description of your program")
parser.add_argument(
    "-f",
    "--folder",
    help="Topfolder to search for files to process",
    default=".",
)
parser.add_argument(
    "-G",
    "--glob",
    help="glob pattern to search for files",
    default="**/*.nc",
)
parser.add_argument(
    "-g",
    "--grid",
    help="Path to a grid file as expected by CDO",
    default="./grid.txt",
)
parser.add_argument(
    "-w",
    "--weights",
    help="Path to a weights file as expected by CDO",
    default=None,
)
parser.add_argument(
    "-o",
    "--options",
    help="extra options for CDO in format like '-f nc' (default=None)",
    default=None,
)
parser.add_argument(
    "-O",
    "--overwrite",
    help="Overwrite files? (default=False)",
    default=False,
)
args = vars(parser.parse_args())

cdo = Cdo()
pp = PrettyPrinter()


def reproject_all(
    path,
    glob_pattern,
    gridFile,
    weightsFile=None,
    overwrite=args["overwrite"],
    *args,
    **kwargs,
):
    """path: path from darwin's base folder or absolute path."""
    files = glob_files(path, glob_pattern, **kwargs)
    print(f"Base folder: {path.as_posix()}")
    print("Found the following files:")
    pp.pprint(files)
    for f in files:
        print("Working on:")
        print(f.as_posix())
        input_ = f.as_posix()
        output = (f.parent / f"{f.stem}_remapped{f.suffix}").as_posix()
        if args:
            " ".join(str(arg) for arg in args)
        else:
            args = ""
        grid_weights_arg = str(gridFile)
        if weightsFile:
            grid_weights_arg = f"{gridFile},{weightsFile}"
        cdo.remapbil(grid_weights_arg, input=input_, output=output, options=args)
        if overwrite:
            rename(output, input_)


if __name__ == "__main__":
    path = FilePath(args["folder"])
    reproject_all(path, args["glob"], args["grid"], args["weights"], args["overwrite"])
