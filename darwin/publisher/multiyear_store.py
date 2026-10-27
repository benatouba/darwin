"""Build one 12-month year-product analysis store from a checksummed GAR file.

Time normalization: both source conventions (``days since`` month starts
and ``months since`` 0..11) map onto the canonical axis — int64 hours
since 1970-01-01 00:00:00 UTC — so future hourly data fits without another
migration. Month starts stay exact integers.

Grid identity: every file must carry the single verified native grid
(x0=-244000, dx=2000, nx=245; y0=-274001.375, dy=2000, ny=230). Any
deviation fails loudly; the publisher never reprojects silently.
"""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import zarr

GRID_SPEC = {
    "x0": -244000.0,
    "dx": 2000.0,
    "nx": 245,
    "y0": -274001.375,
    "dy": 2000.0,
    "ny": 230,
}

# The single verified native GAR CRS, carried by 90 of the 92 staged files
# (the two 2025 files omit the global attribute). Any file declaring a
# different CRS fails the build instead of being reprojected silently.
NATIVE_CRS = (
    "+k_0=1.0 +units=m +lat_0=2.0 +lon_0=-90.31006622 +x_0=0.0 +y_0=0.0"
    " +ellps=WGS84 +datum=WGS84 +name=WRFMercator +proj=merc +no_defs"
)

SOURCE_UNITS = {"prcp": "mm h-1", "t2": "k", "u10": "m s-1", "v10": "m s-1", "ws10": "m s-1"}
STORE_UNITS = {"prcp": "mm h-1", "t2": "K", "u10": "m s-1", "v10": "m s-1", "ws10": "m s-1"}

# Canonical store time axis (S11 #31): int64 hours since the Unix epoch.
EPOCH = datetime(1970, 1, 1)
TIME_UNITS = "hours since 1970-01-01 00:00:00"


def _reference_axis(first: float, step: float, count: int) -> np.ndarray:
    """Build the exact native axis the source grid must equal."""
    return np.array([first + step * i for i in range(count)], dtype=np.float64)


def assert_native_grid(xs: np.ndarray, ys: np.ndarray, crs: str, name: str) -> None:
    """Refuse any source grid that is not the single verified native grid.

    S12 (#31): no per-year reprojection — a foreign grid or CRS fails the
    build instead of being reprojected silently.
    """
    if not np.array_equal(
        xs, _reference_axis(GRID_SPEC["x0"], GRID_SPEC["dx"], GRID_SPEC["nx"])
    ) or not np.array_equal(
        ys, _reference_axis(GRID_SPEC["y0"], GRID_SPEC["dy"], GRID_SPEC["ny"])
    ):
        raise ValueError(f"{name}: grid is not the verified native grid")
    if crs != NATIVE_CRS:
        raise ValueError(f"{name}: unexpected CRS {crs!r}")


def _month_start_epoch_hours(year: int) -> list[int]:
    """Epoch hours of the 12 month starts for a (leap-aware) year."""
    return [
        int((datetime(year, month, 1) - EPOCH).total_seconds() // 3600)
        for month in range(1, 13)
    ]


def _normalize_time(raw_units: str, raw_values: np.ndarray, year: int) -> list[int]:
    """Validate a source time axis, returning canonical epoch hours."""
    amount, _, base = raw_units.partition(" since ")
    origin = datetime.fromisoformat(base.strip())
    if len(raw_values) != 12:
        raise ValueError(f"expected 12 monthly steps, got {len(raw_values)}")
    if amount.strip() == "months":
        if list(int(v) for v in raw_values) != list(range(12)):
            raise ValueError(f"non-sequential monthly axis: {list(raw_values)}")
        if (origin.year, origin.month) != (year, 1):
            raise ValueError(f"monthly axis origin {origin} is not {year}-01")
    elif amount.strip() == "days":
        for index, offset in enumerate(raw_values):
            date = origin + timedelta(days=float(offset))
            if (date.year, date.month, date.day) != (year, index + 1, 1):
                raise ValueError(f"step {index} is {date}, not {year}-{index + 1:02d}-01")
    else:
        raise ValueError(f"unsupported source time units: {raw_units!r}")
    return _month_start_epoch_hours(year)


def build_year_store(
    source: str | Path,
    *,
    variable: str,
    year: int,
    expected_sha256: str,
    out_dir: str | Path,
) -> Path:
    """Verify, normalize and publish one year-file as a 12-step zarr store."""
    if variable not in SOURCE_UNITS:
        raise ValueError(f"multiyear publishing covers prcp/t2/u10/v10/ws10, not {variable!r}")
    source = Path(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"{source.name}: checksum does not match reviewed manifest")
    with nc.Dataset(str(source)) as dataset:
        if variable not in dataset.variables:
            raise ValueError(f"{source.name}: no {variable!r} field in source")
        field = dataset.variables[variable]
        if str(field.units) != SOURCE_UNITS[variable] or field.shape != (12, 230, 245):
            raise ValueError(f"{source.name}: unexpected field {field.shape} {field.units}")
        xs = np.asarray(dataset.variables["west_east"][:], dtype=np.float64)
        ys = np.asarray(dataset.variables["south_north"][:], dtype=np.float64)
        time_var = dataset.variables["time"]
        offsets = _normalize_time(str(time_var.units), np.asarray(time_var[:]), year)
        values = np.asarray(field[:])
        try:
            crs = str(dataset.pyproj_srs)
        except AttributeError:
            crs = NATIVE_CRS
        assert_native_grid(xs, ys, crs, source.name)
    product_id = f"GAR_d02km_m_2d_{variable}_{year}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{product_id}.zarr"
    from darwin.publisher.immutable import staged_artifact

    provenance = {
        "builder": "darwin.publisher.multiyear_store.build_year_store",
        "sources": [{"id": source.name, "sha256": expected_sha256}],
        "parameters": {"variable": variable, "year": year},
    }
    with staged_artifact(path, provenance=provenance) as staged:
        group = zarr.open_group(str(staged), mode="w-")
        group.create_array(variable, data=values, chunks=values.shape)
        group.create_array("x", data=xs)
        group.create_array("y", data=ys)
        group.create_array("time", data=np.asarray(offsets, dtype=np.int64))
        group.attrs["time_units"] = TIME_UNITS
        group.attrs["crs"] = crs
        group.attrs["source_sha256"] = expected_sha256
        group[variable].attrs["units"] = STORE_UNITS[variable]
    return path


def hash_store(path: str | Path) -> str:
    """Canonical sha256 over a published analysis store's data payload.

    Contract (mirrored by the dashboard seed verifier
    backend/app/catalogue.store_payload_hash; both docstrings carry it
    verbatim): sha256 over, for each array in sorted name order, the array
    name, dtype string, shape, and C-order raw bytes. Store attrs and
    consolidation metadata are excluded so future annotation never changes
    the hash. Drift between the two implementations fails loudly at dev
    seed (hash mismatch), never silently.
    """
    digest = hashlib.sha256()
    group = zarr.open_group(str(path), mode="r")
    for name in sorted(group.array_keys()):
        arr = group[name]
        assert isinstance(arr, zarr.Array)
        digest.update(name.encode())
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(np.ascontiguousarray(arr[...]).tobytes())
    return digest.hexdigest()
