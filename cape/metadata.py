"""Hold metadata for products."""
from datetime import datetime
from os.path import path
from platform import python_version

from netCDF4 import Dataset

from .wrfpost_cape import sample_file

now = datetime.now()
domain_num = path.split(sample_file)[-1].split("_")[1]


def get_attributes(ds: Dataset):
    """Add attribute metadata to dataset.

    Args:
        ds (netcdf.Dataset): The dataset to add attributes to.

    Returns:
        netcdf.Dataset: The dataset with added attributes.
    """
    return {
        "TITLE": "GAR d02km",
        "DATA_NOTES": "File generated with the output of successive model "
        "runs of 36H (first 12 hours discarded for spin-up)",
        "WRF_VERSION": f"{ds.attrs['TITLE'].split()[2]} {ds.attrs['TITLE'].split()[3]}",
        "CREATED_BY": "Benjamin Schmidt - benjamin.schmidt@tu-berlin.de",
        "INSTITUTION": "Technische Universitaet Berlin, Institute of Ecology, Chair of Climatology",
        "CREATION_DATE": now.strftime("%d-%m-%Y %H:%M:%S"),
        "SOFTWARE_NOTES": f"Python {python_version()}",
        "VARNAME": "cape",
        "DOMAIN": domain_num.strip("d0"),
        "NESTED": "YES" if domain_num != "d01" else "NO",
        "TIME_ZONE": "UTC",
        "PRODUCT_LEVEL": "H",
        "LEVEL_INFO": "H: original simulation output; D: daily; M: monthly; Y: yearly; S: static",
        "PROJ_NAME": ds.attrs["MAP_PROJ_CHAR"],
        "PROJ_CENTRAL_LON": ds.attrs["STAND_LON"],
        "PROJ_CENTRAL_LAT": ds.attrs["MOAD_CEN_LAT"],
        "PROJ_STANDARD_PAR1": ds.attrs["TRUELAT1"],
        "PROJ_STANDARD_PAR2": ds.attrs["TRUELAT2"],
        "PROJ_SEMIMAJOR_AXIS": "6370000.0",
        "PROJ_SEMIMINOR_AXIS": "6370000.0",
        "PROJ_FALSE_EASTING": "0.0",
        "PROJ_FALSE_NORTHING": "0.0",
        "PROJ_DATUM": "WGS-84",
        "PYPROJ_SRS": ds.attrs["pyproj_srs"],
        "GRID_INFO": "Grid spacing: GRID_DX and GRID_DY (unit: m), "
        "Down left corner: GRID_X00 and GRID_Y00 (unit: m), "
        "Upper Left Corner: GRID_X01 and GRID_Y01 (unit: m)",
        "GRID_DX": ds.attrs["DX"],
        "GRID_DY": ds.attrs["DY"],
        "GRID_X00": ds.west_east.values.min(),
        "GRID_Y00": ds.south_north.values.min(),
        "GRID_X01": ds.west_east.values.min(),
        "GRID_Y01": ds.south_north.values.max(),
        "GRID_NX": len(ds.west_east),
        "GRID_NY": len(ds.south_north),
    }
