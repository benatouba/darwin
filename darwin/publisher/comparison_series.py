"""Comparison series stores (T6): published GAR/AWS January series.

Reads the pinned GAR monthly prcp field and the CerroCrocker AWS
tip-bucket series, and writes two seed Zarr stores with verbatim values:
the GAR January-2020 monthly rate sampled nearest the Station location
record
uniformly disaggregated to constant hourly/daily series, and the AWS
January hourly/daily tip-sum depths with completeness masks. The
hourly/daily completeness rules reuse the comparison-engine
thresholds (``HOURLY_READINGS_REQUIRED``,
``COARSER_COVERAGE_REQUIRED``) with sum aggregation, matching the
declared prcp ``sum`` pairing. The dashboard bundles these stores as
seed data; the published ``s3://`` uris stay the long-term locations.
"""

from __future__ import annotations

import csv
from calendar import monthrange
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

from darwin.publisher.comparison import (
    COARSER_COVERAGE_REQUIRED,
    HOURLY_READINGS_REQUIRED,
)

GAR_SOURCE = "GAR_d02km_m_2d_prcp_2020.nc"
AWS_SOURCE = "data/11_AWS-PCerroCrocker_data.csv"
AWS_META = "data/11_AWS-PCerroCrocker_meta.xml"
GAR_STORE = "GAR-CerroCrocker-sampled_2020.zarr"
AWS_STORE = "AWS-CerroCrocker_2025-01.zarr"

#: January paired against the GAR January-2020 monthly rate.
AWS_YEAR = 2025
AWS_MONTH = 1

#: Readings that carry no tip information.
MISSING_TOKENS = frozenset({"", "NA", "NaN", "NULL"})


def _station_point(meta_path: Path) -> tuple[float, float]:
    """Station-location-record lon/lat from the degenerate meta.xml bounding box."""
    root = ElementTree.parse(str(meta_path)).getroot()
    box = root.find(".//boundingCoordinates")
    assert box is not None, f"no boundingCoordinates in {meta_path}"
    west = box.findtext("westBoundingCoordinate", "").rstrip("°")
    north = box.findtext("northBoundingCoordinate", "").rstrip("°")
    return float(west), float(north)


def _read_tip_readings(csv_path: Path) -> list[tuple[datetime, float | None]]:
    """Timestamped PCP_tot_bucket tip readings; missing tokens become None."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = [name.strip() for name in rows[0]]
    lowered = [name.lower() for name in header]
    time_col = lowered.index("datetime")
    value_col = header.index("PCP_tot_bucket")
    readings = []
    for row in rows[1:]:
        if len(row) <= max(time_col, value_col):
            continue
        stamp = datetime.fromisoformat(row[time_col].strip())
        raw = row[value_col].strip()
        readings.append((stamp, None if raw in MISSING_TOKENS else float(raw)))
    return readings


def build_comparison_series(
    repo_root: str | Path, out_dir: str | Path
) -> dict[str, Path]:
    """Build the GAR-sampled + AWS-January Zarr stores, paths by key."""
    import netCDF4 as nc
    import numpy as np
    import zarr

    repo_root, out_dir = Path(repo_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    station_lon, station_lat = _station_point(repo_root / AWS_META)
    source = nc.Dataset(str(repo_root / GAR_SOURCE))
    try:
        lon = np.asarray(source.variables["lon"][:])
        lat = np.asarray(source.variables["lat"][:])
        prcp = np.asarray(source.variables["prcp"][:])
        crs = str(source.pyproj_srs)
    finally:
        source.close()
    nearest_flat_index = int(np.argmin((lon - station_lon) ** 2 + (lat - station_lat) ** 2))
    grid_y, grid_x = divmod(nearest_flat_index, lon.shape[1])
    monthly = np.asarray(prcp[:, grid_y, grid_x], dtype=np.float64)

    days = monthrange(AWS_YEAR, AWS_MONTH)[1]
    hours = days * 24
    gar_path = out_dir / GAR_STORE
    gar_group = zarr.open_group(str(gar_path), mode="w")
    gar_group.create_array("monthly", data=monthly, attributes={"units": "mm h-1"})
    # Both disaggregated series are depths (hourly depth = rate x 1h) so
    # they pair unit-clean with the AWS tip-sum depths, both in mm.
    gar_group.create_array(
        "hourly", data=np.full(hours, monthly[0]), attributes={"units": "mm"}
    )
    gar_group.create_array(
        "daily", data=np.full(days, monthly[0] * 24.0), attributes={"units": "mm"}
    )
    gar_group.attrs["constant_series"] = True
    gar_group.attrs["method"] = "nearest"
    gar_group.attrs["crs"] = crs
    gar_group.attrs["grid_y"] = grid_y
    gar_group.attrs["grid_x"] = grid_x
    gar_group.attrs["cell_lon"] = float(lon[grid_y, grid_x])
    gar_group.attrs["cell_lat"] = float(lat[grid_y, grid_x])
    gar_group.attrs["station_lon"] = station_lon
    gar_group.attrs["station_lat"] = station_lat

    buckets: list[list[float]] = [[] for _ in range(hours)]
    for stamp, value in _read_tip_readings(repo_root / AWS_SOURCE):
        if stamp.year != AWS_YEAR or stamp.month != AWS_MONTH or value is None:
            continue
        buckets[(stamp.day - 1) * 24 + stamp.hour].append(value)
    hourly = np.full(hours, np.nan)
    hourly_valid = np.zeros(hours, dtype=bool)
    for index, readings in enumerate(buckets):
        if len(readings) >= HOURLY_READINGS_REQUIRED:
            hourly[index] = sum(readings)
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

    aws_path = out_dir / AWS_STORE
    aws_group = zarr.open_group(str(aws_path), mode="w")
    aws_group.create_array("hourly", data=hourly, attributes={"units": "mm"})
    aws_group.create_array("hourly_valid", data=np.asarray(hourly_valid))
    aws_group.create_array("daily", data=daily, attributes={"units": "mm"})
    aws_group.create_array("daily_valid", data=np.asarray(daily_valid))
    aws_group.attrs["units"] = "mm"
    aws_group.attrs["year"] = AWS_YEAR
    aws_group.attrs["month"] = AWS_MONTH

    return {"gar_sampled": gar_path, "aws_january": aws_path}
