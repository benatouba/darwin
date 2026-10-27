"""Build a checksum-pinned January 2020 GAR 2D temperature seed store.

The NetCDF input is downloaded separately and can be deleted after the
extraction; only one monthly timestep is retained on disk.
"""

import hashlib
from pathlib import Path

import netCDF4 as nc
import numpy as np
import zarr

from darwin.publisher.multiyear_store import (
    TIME_UNITS,
    _normalize_time,
    assert_native_grid,
)

SOURCE_SHA256 = "392067fad109006f311cb1121f8574459c637bd690cfa4126016aaabaeda5056"
STORE_NAME = "GAR_d02km_m_2d_t2_2020-01.zarr"


def build_temperature_store(source: str | Path, out_dir: str | Path) -> Path:
    """Verify the downloaded GAR source then publish its first monthly field."""
    source = Path(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Temperature source checksum does not match reviewed manifest")
    with nc.Dataset(str(source)) as dataset:
        variable = dataset.variables["t2"]
        if str(variable.units).lower() != "k" or variable.shape != (12, 230, 245):
            raise ValueError("Unexpected temperature field or units")
        values = np.asarray(variable[0:1])
        xs = np.asarray(dataset.variables["west_east"][:])
        ys = np.asarray(dataset.variables["south_north"][:])
        time_var = dataset.variables["time"]
        hours = _normalize_time(str(time_var.units), np.asarray(time_var[:]), 2020)
        time = np.asarray(hours[0:1], dtype=np.int64)
        time_units = TIME_UNITS
        crs = str(dataset.pyproj_srs)
        assert_native_grid(xs, ys, crs, source.name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / STORE_NAME
    from darwin.publisher.immutable import staged_artifact

    with staged_artifact(path) as staged:
        group = zarr.open_group(str(staged), mode="w-")
        group.create_array("t2", data=values, chunks=values.shape)
        group.create_array("x", data=xs)
        group.create_array("y", data=ys)
        group.create_array("time", data=time)
        group.attrs["time_units"] = time_units
        group.attrs["crs"] = crs
        group.attrs["source_sha256"] = SOURCE_SHA256
        group["t2"].attrs["units"] = "K"
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    print(build_temperature_store(args.source, args.out_dir))
