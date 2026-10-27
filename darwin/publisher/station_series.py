"""Version-2 station stores preserve native cadence and observation time provenance.

Monthly GAR values remain monthly: they cannot establish native hourly skill.
Observation aggregation preserves the existing sum/mean completeness policies;
timezone declarations, coordinates and interval bounds are now mandatory evidence.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
from calendar import monthrange
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree

from darwin.publisher.immutable import identity_path, staged_artifact

from darwin.publisher.comparison import (
    COARSER_COVERAGE_REQUIRED,
    HOURLY_READINGS_REQUIRED,
)
from darwin.publisher.multiyear_store import (
    NATIVE_CRS,
    STORE_UNITS,
    TIME_UNITS,
    assert_native_grid,
)
from darwin.publisher.series import (
    CALENDARS,
    EPOCH,
    SCHEMA_VERSION,
    TimeValidationError,
    interval_end,
    time_zone,
    timestamp,
)
from darwin.publisher.series import (
    TIME_UNITS as SECOND_UNITS,
)

MISSING_TOKENS = frozenset({"", "NA", "NaN", "NULL"})

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


def _read_column(csv_path: Path, column: str) -> list[tuple[datetime, float | None]]:
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
        value = None if raw in MISSING_TOKENS else float(raw)
        readings.append((stamp, value if value is not None and math.isfinite(value) else None))
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
    obs_timezone: str | None = None,
    precipitation_kind: str = "cumulative",
    station_id: str | None = None,
) -> dict[str, Path]:
    """Prepare monthly model + timestamped observation stores, never auto-publish.

    Naive CSV dates require ``obs_timezone`` from reviewed source metadata, not a
    guessed location. Offset-aware CSVs aggregate in UTC unless explicitly declared
    otherwise. ``precipitation_kind`` preserves the existing distinct cumulative
    versus increment policies; neither is newly scientifically approved here.
    Existing destinations are refused; historical stores are not upgraded in place.
    ``station_id`` is the reviewed station identity; it is recorded as absent
    (``None``) rather than inferred from file names when not supplied.
    """
    import netCDF4 as nc
    import numpy as np
    import zarr

    if mode not in ("sum", "mean"):
        raise ValueError(f"unknown mode {mode!r}: expected 'sum' or 'mean'")
    if precipitation_kind not in ("cumulative", "increments"):
        raise ValueError("precipitation_kind must be 'cumulative' or 'increments'")
    repo_root, out_dir = Path(repo_root), Path(out_dir)
    gar_store_path, obs_store_path = out_dir / gar_store_name, out_dir / obs_store_name
    stores = (gar_store_path, obs_store_path)
    occupied = (*stores, *map(identity_path, stores))
    if gar_store_path == obs_store_path or any(os.path.lexists(path) for path in occupied):
        raise FileExistsError(
            "use distinct new versioned destinations; existing stores are immutable"
        )

    gar_path = repo_root / gar_source
    obs_path = repo_root / obs_csv
    _verify_source(gar_path, gar_sha256, gar_source)
    _verify_source(obs_path, obs_sha256, obs_csv)

    meta_sha256 = hashlib.sha256((repo_root / obs_meta).read_bytes()).hexdigest()
    station_lon, station_lat = _station_point(repo_root / obs_meta)
    source = nc.Dataset(str(gar_path))
    try:
        lon = np.asarray(source.variables["lon"][:])
        lat = np.asarray(source.variables["lat"][:])
        xs = np.asarray(source.variables["west_east"][:])
        ys = np.asarray(source.variables["south_north"][:])
        field = np.ma.filled(source.variables[gar_variable][:], np.nan)
        source_units = str(source.variables[gar_variable].units)
        if "time" not in source.variables:
            raise TimeValidationError("coordinates", "missing model time coordinates")
        axis = source.variables["time"]
        calendar = str(getattr(axis, "calendar", "standard"))
        if calendar not in CALENDARS:
            raise TimeValidationError("calendar", f"unsupported source calendar: {calendar!r}")
        time_units = str(axis.units)
        time_values = np.asarray(axis[:])
        if time_units.startswith("months since "):
            origin = timestamp(time_units.partition(" since ")[2].strip(), "UTC")
            if origin != datetime(origin.year, 1, 1, tzinfo=UTC) or not np.array_equal(
                time_values, np.arange(12)
            ):
                raise TimeValidationError(
                    "cadence", "expected exact January-origin 0..11 monthly coordinates"
                )
            model_times = [datetime(origin.year, month, 1, tzinfo=UTC) for month in range(1, 13)]
        else:
            decoded = nc.num2date(time_values, time_units, calendar=calendar)
            model_times = [timestamp(value.isoformat(), "UTC") for value in decoded]
        if len(model_times) != 12 or any(
            stamp != datetime(model_times[0].year, index + 1, 1, tzinfo=UTC)
            for index, stamp in enumerate(model_times)
        ):
            raise TimeValidationError(
                "cadence", "expected twelve authoritative calendar-month starts"
            )
        model_bounds = [(stamp, interval_end(stamp, "monthly")) for stamp in model_times]
        source_bounds = getattr(axis, "bounds", None)
        if source_bounds:
            if source_bounds not in source.variables:
                raise TimeValidationError("bounds", "declared source time bounds are missing")
            decoded_bounds = nc.num2date(
                source.variables[source_bounds][:], time_units, calendar=calendar
            )
            actual_bounds = [
                tuple(timestamp(v.isoformat(), "UTC") for v in row) for row in decoded_bounds
            ]
            if actual_bounds != model_bounds:
                raise TimeValidationError(
                    "bounds", "source bounds differ from declared monthly intervals"
                )
        crs = str(getattr(source, "pyproj_srs", None) or NATIVE_CRS)
    finally:
        source.close()
    assert_native_grid(xs, ys, crs, gar_source)
    epoch_hours = [int((stamp - EPOCH).total_seconds() // 3600) for stamp in model_times]
    nearest_flat_index = int(np.argmin((lon - station_lon) ** 2 + (lat - station_lat) ** 2))
    grid_y, grid_x = divmod(nearest_flat_index, lon.shape[1])
    monthly = np.asarray(field[:, grid_y, grid_x], dtype=np.float64)
    if (
        field.shape != (12, len(ys), len(xs))
        or source_units.lower() != STORE_UNITS[gar_variable].lower()
    ):
        raise ValueError("unexpected monthly model shape or units")

    days = monthrange(obs_year, obs_month)[1]
    hours = days * 24
    store_units = STORE_UNITS[gar_variable]
    obs_units = "mm" if mode == "sum" else "K"
    aggregation_timezone = obs_timezone or "UTC"
    zone = time_zone(aggregation_timezone)
    raw_readings = _read_column(obs_path, obs_field)
    readings = [
        (timestamp(stamp, obs_timezone).astimezone(zone), value) for stamp, value in raw_readings
    ]
    if any(right[0] <= left[0] for left, right in pairwise(readings)):
        raise TimeValidationError(
            "order", "observation timestamps must be strictly increasing and unique"
        )
    selected = [
        (stamp, value)
        for stamp, value in readings
        if (stamp.year, stamp.month) == (obs_year, obs_month)
    ]
    if not selected:
        raise TimeValidationError("no_overlap", "no observations in the requested calendar month")
    start = datetime(obs_year, obs_month, 1, tzinfo=zone)
    end = interval_end(start, "monthly")
    if start.utcoffset() != end.utcoffset():
        raise TimeValidationError(
            "timezone",
            "DST-changing aggregation windows are unsupported; explicitly normalize first",
        )
    hourly_times = [start + timedelta(hours=i) for i in range(hours)]
    daily_times = [start + timedelta(days=i) for i in range(days)]
    hourly = np.full(hours, np.nan)
    hourly_valid = np.zeros(hours, dtype=bool)
    if mode == "sum":
        buckets: list[list[float]] = [[] for _ in range(hours)]
        increments = _derive_tips(readings) if precipitation_kind == "cumulative" else readings
        for stamp, value in increments:
            if stamp.year != obs_year or stamp.month != obs_month:
                continue
            if value is None:
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
                hourly[day * 24 + hour] for hour in range(24) if hourly_valid[day * 24 + hour]
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
                hourly[day * 24 + hour] for hour in range(24) if hourly_valid[day * 24 + hour]
            ]
            if len(window) / 24 >= COARSER_COVERAGE_REQUIRED:
                daily[day] = sum(window) / len(window)
                daily_valid[day] = True

    station = {"id": station_id, "lon": station_lon, "lat": station_lat}
    builder = "darwin.publisher.station_series.build_station_series"
    model_provenance = {
        "builder": builder,
        "sources": [{"id": gar_source, "sha256": gar_sha256}],
        "parameters": {
            "variable": gar_variable,
            "method": "nearest",
            "cadence": "monthly",
            "station_id": station_id,
        },
    }
    obs_provenance = {
        "builder": builder,
        "sources": [
            {"id": obs_csv, "sha256": obs_sha256},
            {"id": obs_meta, "sha256": meta_sha256},
        ],
        "parameters": {
            "field": obs_field,
            "mode": mode,
            "precipitation_kind": precipitation_kind if mode == "sum" else None,
            "year": obs_year,
            "month": obs_month,
            "obs_timezone": obs_timezone,
            "station_id": station_id,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with staged_artifact(gar_store_path, provenance=model_provenance) as staged:
        _write_model_store(
            staged, monthly, store_units, epoch_hours, model_bounds,
            {
                "series_schema_version": SCHEMA_VERSION,
                "cadences": ["monthly"],
                "cadence": "monthly",
                "calendar": calendar,
                "timezone": "UTC",
                "aggregation_timezone": "UTC",
                "source_coverage": {
                    "start": model_times[0].isoformat(),
                    "end": model_bounds[-1][1].isoformat(),
                },
                "bounds_origin": "source" if source_bounds else "validated-calendar-month-cadence",
                "synthetic": False,
                "scientific_status": "unassessed",
                "method": "nearest",
                "variable": gar_variable,
                "crs": crs,
                "grid_y": grid_y,
                "grid_x": grid_x,
                "cell_lon": float(lon[grid_y, grid_x]),
                "cell_lat": float(lat[grid_y, grid_x]),
                "station_lon": station_lon,
                "station_lat": station_lat,
                "station": station,
                "sampling": {
                    "method": "nearest",
                    "crs": crs,
                    "grid_y": grid_y,
                    "grid_x": grid_x,
                    "cell_lon": float(lon[grid_y, grid_x]),
                    "cell_lat": float(lat[grid_y, grid_x]),
                },
                "source_gar": gar_source,
                "source_gar_sha256": gar_sha256,
            },
        )
    with staged_artifact(obs_store_path, provenance=obs_provenance) as staged:
        _write_observation_store(
            staged, hourly, hourly_valid, daily, daily_valid, obs_units,
            hourly_times, daily_times,
            {
                "units": obs_units,
                "year": obs_year,
                "month": obs_month,
                "source_csv": obs_csv,
                "source_sha256": obs_sha256,
                "series_schema_version": SCHEMA_VERSION,
                "cadences": ["hourly", "daily"],
                "calendar": "proleptic_gregorian",
                "timezone": "UTC",
                "aggregation_timezone": aggregation_timezone,
                "source_timezone": obs_timezone or "explicit-CSV-offsets",
                "source_coverage": {
                    "first_timestamp": readings[0][0].astimezone(UTC).isoformat(),
                    "last_timestamp": readings[-1][0].astimezone(UTC).isoformat(),
                    "reading_count": len(readings),
                    "selected_reading_count": len(selected),
                    "window": {
                        "start": start.astimezone(UTC).isoformat(),
                        "end": end.astimezone(UTC).isoformat(),
                    },
                },
                "station": station,
                "synthetic": False,
                "scientific_status": "unassessed",
                "aggregation": "positive-difference-tip-sum"
                if mode == "sum" and precipitation_kind == "cumulative"
                else "increment-sum" if mode == "sum" else "C-to-K-mean",
                "hourly_readings_required": HOURLY_READINGS_REQUIRED if mode == "sum" else 1,
                "daily_coverage_required": COARSER_COVERAGE_REQUIRED,
            },
        )
    return {"gar_sampled": gar_store_path, "obs": obs_store_path}


def _write_model_store(path, monthly, store_units, epoch_hours, model_bounds, metadata):
    import numpy as np
    import zarr

    gar_group = zarr.open_group(str(path), mode="w-")
    gar_group.create_array("monthly", data=monthly, attributes={"units": store_units})
    gar_group.create_array(
        "time", data=np.asarray(epoch_hours, dtype=np.int64), attributes={"units": TIME_UNITS}
    )
    gar_group.create_array(
        "time_bounds",
        data=np.asarray(
            [
                [int((bound - EPOCH).total_seconds() // 3600) for bound in pair]
                for pair in model_bounds
            ],
            dtype=np.int64,
        ),
        attributes={"units": TIME_UNITS},
    )
    gar_group.attrs.update(metadata)


def _write_observation_store(
    path, hourly, hourly_valid, daily, daily_valid, obs_units, hourly_times, daily_times, metadata,
):
    import numpy as np
    import zarr

    obs_group = zarr.open_group(str(path), mode="w-")
    obs_group.create_array("hourly", data=hourly, attributes={"units": obs_units})
    obs_group.create_array("hourly_valid", data=np.asarray(hourly_valid))
    obs_group.create_array("daily", data=daily, attributes={"units": obs_units})
    obs_group.create_array("daily_valid", data=np.asarray(daily_valid))
    obs_group.attrs.update(metadata)
    for cadence, stamps in (("hourly", hourly_times), ("daily", daily_times)):
        name = f"{cadence}_time"
        obs_group.create_array(
            name,
            data=np.asarray(
                [int((stamp.astimezone(UTC) - EPOCH).total_seconds()) for stamp in stamps],
                dtype=np.int64,
            ),
            attributes={"units": SECOND_UNITS},
        )
        obs_group.create_array(
            f"{name}_bounds",
            data=np.asarray(
                [
                    [
                        int((bound.astimezone(UTC) - EPOCH).total_seconds())
                        for bound in (stamp, interval_end(stamp, cadence))
                    ]
                    for stamp in stamps
                ],
                dtype=np.int64,
            ),
            attributes={"units": SECOND_UNITS},
        )
