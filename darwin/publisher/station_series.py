"""Per-station comparison series stores (spec #32).

Generalizes ``comparison_series.build_comparison_series`` (hardcoded to
CerroCrocker / January) to any station, month, and variable. The original
function is untouched; this module adds a parameterized builder plus a
temperature-means mode.

Two obs modes:

* ``"sum"`` (precipitation): the raw P-file bucket column is
  cumulative-within-hour, so per-reading tip increments are derived first
  as positive diffs with across-boundary carry (a drop means the bucket
  was emptied, not negative rain). Hourly tip sums then reuse the
  comparison-engine thresholds verbatim, matching the declared prcp
  ``sum`` pairing.
* ``"mean"`` (temperature): hourly ``T`` readings are converted to
  Kelvin at build time (``+ 273.15``) and stored as hourly/daily means,
  matching the t2 ``mean`` pairing the dashboard resolves for ``t2-T``.

S10/S11/S12 compliance: source sha256 is verified before reading; the
native grid is asserted (never reprojected); the sampled store carries
the canonical int64 epoch-hours axis. Obs stores carry no time axis
(year/month attrs only), like the existing AWS store.
"""

from __future__ import annotations

import csv
import hashlib
from calendar import monthrange
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

from darwin.publisher.comparison import (
    COARSER_COVERAGE_REQUIRED,
    HOURLY_READINGS_REQUIRED,
)
from darwin.publisher.comparison_series import MISSING_TOKENS
from darwin.publisher.multiyear_store import (
    NATIVE_CRS,
    STORE_UNITS,
    TIME_UNITS,
    assert_native_grid,
    _normalize_time,
)

#: Celsius-to-Kelvin offset applied to T readings at build time.
C_TO_K = 273.15


def _station_point(meta_path: Path) -> tuple[float, float]:
    """Station lon/lat from the degenerate meta.xml bounding box."""
    root = ElementTree.parse(str(meta_path)).getroot()
    box = root.find(".//boundingCoordinates")
    assert box is not None, f"no boundingCoordinates in {meta_path}"
    west = box.findtext("westBoundingCoordinate", "").rstrip("°")
    north = box.findtext("northBoundingCoordinate", "").rstrip("°")
    return float(west), float(north)


def _verify_source(path: Path, expected_sha256: str, label: str) -> None:
    """Fail fast when a source file is not the reviewed bytes."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"{label} sha256 mismatch: expected {expected_sha256[:16]}…, "
            f"got {digest[:16]}… (re-copy the reviewed source?)"
        )


def _read_column(
    csv_path: Path, column: str
) -> list[tuple[datetime, float | None]]:
    """Timestamped readings of one column; missing tokens become None."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = [name.strip() for name in rows[0]]
    lowered = [name.lower() for name in header]
    time_col = lowered.index("datetime")
    value_col = header.index(column)
    readings = []
    for row in rows[1:]:
        if len(row) <= max(time_col, value_col):
            continue
        stamp = datetime.fromisoformat(row[time_col].strip())
        raw = row[value_col].strip()
        readings.append((stamp, None if raw in MISSING_TOKENS else float(raw)))
    return readings


def _derive_tips(
    readings: list[tuple[datetime, float | None]],
) -> list[tuple[datetime, float]]:
    """Per-reading tip increments from cumulative bucket values.

    Positive diffs with across-boundary carry: the first valid reading
    (or the first after a gap) contributes 0.0, and a drop contributes
    0.0 because the bucket was emptied, not because it rained negatively.
    """
    tips: list[tuple[datetime, float]] = []
    previous: float | None = None
    for stamp, value in readings:
        if value is None:
            previous = None
            continue
        tips.append((stamp, max(0.0, value - previous) if previous is not None else 0.0))
        previous = value
    return tips


def build_station_series(
    repo_root: str | Path,
    out_dir: str | Path,
    *,
    gar_source: str,
    gar_variable: str,
    obs_csv: str,
    obs_meta: str,
    obs_year: int,
    obs_month: int,
    gar_store_name: str,
    obs_store_name: str,
    mode: str = "sum",
    obs_field: str = "PCP_tot_bucket",
    gar_sha256: str,
    obs_sha256: str,
) -> dict[str, Path]:
    """Build the GAR-sampled + obs-month Zarr stores, paths by key."""
    import netCDF4 as nc
    import numpy as np
    import zarr

    if mode not in ("sum", "mean"):
        raise ValueError(f"unknown mode {mode!r}: expected 'sum' or 'mean'")
    repo_root, out_dir = Path(repo_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gar_path = repo_root / gar_source
    obs_path = repo_root / obs_csv
    _verify_source(gar_path, gar_sha256, gar_source)
    _verify_source(obs_path, obs_sha256, obs_csv)

    station_lon, station_lat = _station_point(repo_root / obs_meta)
    source = nc.Dataset(str(gar_path))
    try:
        lon = np.asarray(source.variables["lon"][:])
        lat = np.asarray(source.variables["lat"][:])
        xs = np.asarray(source.variables["west_east"][:])
        ys = np.asarray(source.variables["south_north"][:])
        field = np.asarray(source.variables[gar_variable][:])
        time_units = str(source.variables["time"].units)
        time_values = np.asarray(source.variables["time"][:])
        crs = str(getattr(source, "pyproj_srs", None) or NATIVE_CRS)
    finally:
        source.close()
    assert_native_grid(xs, ys, crs, gar_source)
    gar_year = datetime.fromisoformat(time_units.partition(" since ")[2].strip()).year
    epoch_hours = _normalize_time(time_units, time_values, gar_year)
    nearest_flat_index = int(np.argmin((lon - station_lon) ** 2 + (lat - station_lat) ** 2))
    grid_y, grid_x = divmod(nearest_flat_index, lon.shape[1])
    monthly = np.asarray(field[:, grid_y, grid_x], dtype=np.float64)

    days = monthrange(obs_year, obs_month)[1]
    hours = days * 24
    month_value = float(monthly[obs_month - 1])
    store_units = STORE_UNITS[gar_variable]
    obs_units = "mm" if mode == "sum" else "K"

    gar_store_path = out_dir / gar_store_name
    gar_group = zarr.open_group(str(gar_store_path), mode="w")
    gar_group.create_array("monthly", data=monthly, attributes={"units": store_units})
    gar_group.create_array(
        "hourly", data=np.full(hours, month_value), attributes={"units": obs_units}
    )
    gar_group.create_array(
        "daily",
        data=np.full(days, month_value * (24.0 if mode == "sum" else 1.0)),
        attributes={"units": obs_units},
    )
    gar_group.create_array(
        "time",
        data=np.asarray(epoch_hours, dtype=np.int64),
        attributes={"units": TIME_UNITS},
    )
    gar_group.attrs["constant_series"] = True
    gar_group.attrs["method"] = "nearest"
    gar_group.attrs["variable"] = gar_variable
    gar_group.attrs["crs"] = crs
    gar_group.attrs["grid_y"] = grid_y
    gar_group.attrs["grid_x"] = grid_x
    gar_group.attrs["cell_lon"] = float(lon[grid_y, grid_x])
    gar_group.attrs["cell_lat"] = float(lat[grid_y, grid_x])
    gar_group.attrs["station_lon"] = station_lon
    gar_group.attrs["station_lat"] = station_lat
    gar_group.attrs["source_gar"] = gar_source
    gar_group.attrs["source_gar_sha256"] = gar_sha256
    gar_group.attrs["source_csv"] = obs_csv
    gar_group.attrs["source_csv_sha256"] = obs_sha256

    readings = _read_column(obs_path, obs_field)
    hourly = np.full(hours, np.nan)
    hourly_valid = np.zeros(hours, dtype=bool)
    if mode == "sum":
        buckets: list[list[float]] = [[] for _ in range(hours)]
        for stamp, value in _derive_tips(readings):
            if stamp.year != obs_year or stamp.month != obs_month:
                continue
            buckets[(stamp.day - 1) * 24 + stamp.hour].append(value)
        for index, tips in enumerate(buckets):
            if len(tips) >= HOURLY_READINGS_REQUIRED:
                hourly[index] = sum(tips)
                hourly_valid[index] = True
        daily = np.full(days, np.nan)
        daily_valid = np.zeros(days, dtype=bool)
        for day in range(days):
            window = [
                hourly[day * 24 + hour]
                for hour in range(24)
                if hourly_valid[day * 24 + hour]
            ]
            if len(window) / 24 >= COARSER_COVERAGE_REQUIRED:
                daily[day] = sum(window)
                daily_valid[day] = True
    else:
        buckets: list[list[float]] = [[] for _ in range(hours)]
        for stamp, value in readings:
            if stamp.year != obs_year or stamp.month != obs_month or value is None:
                continue
            buckets[(stamp.day - 1) * 24 + stamp.hour].append(value + C_TO_K)
        for index, samples in enumerate(buckets):
            if samples:
                hourly[index] = sum(samples) / len(samples)
                hourly_valid[index] = True
        daily = np.full(days, np.nan)
        daily_valid = np.zeros(days, dtype=bool)
        for day in range(days):
            window = [
                hourly[day * 24 + hour]
                for hour in range(24)
                if hourly_valid[day * 24 + hour]
            ]
            if len(window) / 24 >= COARSER_COVERAGE_REQUIRED:
                daily[day] = sum(window) / len(window)
                daily_valid[day] = True

    obs_store_path = out_dir / obs_store_name
    obs_group = zarr.open_group(str(obs_store_path), mode="w")
    obs_group.create_array("hourly", data=hourly, attributes={"units": obs_units})
    obs_group.create_array("hourly_valid", data=np.asarray(hourly_valid))
    obs_group.create_array("daily", data=daily, attributes={"units": obs_units})
    obs_group.create_array("daily_valid", data=np.asarray(daily_valid))
    obs_group.attrs["units"] = obs_units
    obs_group.attrs["year"] = obs_year
    obs_group.attrs["month"] = obs_month
    obs_group.attrs["source_csv"] = obs_csv
    obs_group.attrs["source_sha256"] = obs_sha256

    return {"gar_sampled": gar_store_path, "obs": obs_store_path}
