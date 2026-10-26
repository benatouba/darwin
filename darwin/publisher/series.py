"""Validated interval-start series and timestamp joins, never positional pairing.

Version 2 station stores retain native cadence and explicit UTC coordinates/bounds.
Legacy stores cannot be upgraded by inferring dates from their length or name.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA_VERSION = 2
TIME_UNITS = "seconds since 1970-01-01 00:00:00"
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}


class TimeValidationError(ValueError):
    """An actionable time-contract failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def time_zone(name: str) -> ZoneInfo:
    """Validate an explicitly declared IANA timezone."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as error:
        raise TimeValidationError("timezone", f"unsupported timezone: {name!r}") from error


def timestamp(value: datetime | str, assumed_timezone: str | None = None) -> datetime:
    """Normalize to UTC; naive source dates need an explicit timezone declaration."""
    try:
        stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(stamp, datetime):
            raise ValueError("expected a datetime or ISO timestamp")
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            if assumed_timezone is None:
                raise TimeValidationError(
                    "timezone", "time coordinates require an explicit timezone"
                )
            zone = time_zone(assumed_timezone)
            aware = stamp.replace(tzinfo=zone)
            if aware.utcoffset() != aware.replace(fold=1).utcoffset():
                raise TimeValidationError(
                    "timezone", "ambiguous/nonexistent local time; supply an offset"
                )
            stamp = aware
        return stamp.astimezone(UTC)
    except (ValueError, OverflowError) as error:
        if isinstance(error, TimeValidationError):
            raise
        raise TimeValidationError("timestamp", f"invalid time coordinate: {value!r}") from error


def interval_end(start: datetime, cadence: str) -> datetime:
    """Advance a declared calendar interval; no finer-cadence reconstruction."""
    if cadence == "hourly":
        return start + timedelta(hours=1)
    if cadence == "daily":
        return start + timedelta(days=1)
    if cadence == "monthly":
        return start.replace(year=start.year + (start.month == 12), month=start.month % 12 + 1)
    if cadence == "annual":
        return start.replace(year=start.year + 1)
    raise TimeValidationError("cadence", f"unsupported cadence: {cadence!r}")


@dataclass(frozen=True)
class TimeSeries:
    """Values on declared interval starts; missing intervals remain missing.

    Optional bounds must equal the declared calendar intervals. ``valid`` is an
    already evaluated completeness/QC mask, not permission to fabricate values.
    Naive datetimes are never accepted here. ``timezone`` defines aggregation
    boundaries, while timestamps can carry any explicit equivalent UTC offset.
    """

    values: Sequence[float]
    times: Sequence[datetime | str]
    cadence: str
    bounds: Sequence[tuple[datetime | str, datetime | str]] | None = None
    valid: Sequence[bool] | None = None
    calendar: str = "proleptic_gregorian"
    timezone: str = "UTC"
    synthetic: bool = False

    def records(self) -> dict[datetime, tuple[float, bool, datetime]]:
        """Validate the entire series before any pairing or statistics."""
        if self.synthetic:
            raise TimeValidationError(
                "synthetic", "synthetic series are not scientific comparison evidence"
            )
        if self.calendar not in CALENDARS:
            raise TimeValidationError("calendar", f"unsupported calendar: {self.calendar!r}")
        zone = time_zone(self.timezone)
        if len(self.values) != len(self.times):
            raise TimeValidationError("length", "value and time coordinate length mismatch")
        for metadata in (self.bounds, self.valid):
            if metadata is not None and len(metadata) != len(self.times):
                raise TimeValidationError("length", "time metadata length mismatch")
        result = {}
        previous = None
        for index, (raw, value) in enumerate(zip(self.times, self.values, strict=True)):
            start = timestamp(raw)
            if self.calendar != "proleptic_gregorian" and start.year < 1583:
                raise TimeValidationError(
                    "calendar", "mixed Gregorian calendars before 1583 are unsupported"
                )
            local = start.astimezone(zone)
            aligned = local.minute == local.second == local.microsecond == 0
            if self.cadence in ("daily", "monthly", "annual"):
                aligned = aligned and local.hour == 0
            if self.cadence in ("monthly", "annual"):
                aligned = aligned and local.day == 1
            if self.cadence == "annual":
                aligned = aligned and local.month == 1
            end = interval_end(local, self.cadence).astimezone(UTC)
            if not aligned:
                raise TimeValidationError(
                    "cadence", "time coordinates do not match declared cadence boundaries"
                )
            if previous is not None and start <= previous:
                raise TimeValidationError(
                    "order", "time coordinates must be strictly increasing and unique"
                )
            if self.bounds is not None:
                bounds = tuple(timestamp(v) for v in self.bounds[index])
                if bounds != (start, end):
                    raise TimeValidationError(
                        "bounds", "interval bounds do not match declared cadence"
                    )
            eligible = (self.valid is None or bool(self.valid[index])) and math.isfinite(value)
            result[start] = (float(value), eligible, end)
            previous = start
        return result


def align_series(
    model: TimeSeries, observations: TimeSeries
) -> tuple[list[float], list[float], int]:
    """Join exact UTC intervals, reporting unmatched/invalid union records.

    No aggregation or interpolation is implicit. Exclusions count represented
    intervals, not an invented timeline between disjoint source records.
    """
    if not isinstance(model, TimeSeries) or not isinstance(observations, TimeSeries):
        raise TimeValidationError(
            "coordinates", "comparisons require TimeSeries with time coordinates"
        )
    if model.cadence != observations.cadence:
        raise TimeValidationError(
            "cadence", "incompatible native cadence; explicitly aggregate first"
        )
    left, right = model.records(), observations.records()
    common = sorted(left.keys() & right.keys())
    if not common:
        raise TimeValidationError(
            "no_overlap", "no overlapping time intervals in the source series"
        )
    pairs = []
    for start in common:
        g, g_valid, g_end = left[start]
        o, o_valid, o_end = right[start]
        if g_end != o_end:
            raise TimeValidationError(
                "bounds", "incompatible interval bounds; explicit reconciliation required"
            )
        if g_valid and o_valid:
            pairs.append((g, o))
    return (
        [g for g, _ in pairs],
        [o for _, o in pairs],
        len(left.keys() | right.keys()) - len(pairs),
    )


def load_series(path: str | Path, cadence: str) -> TimeSeries:
    """Read a version-2 station-store series, rejecting unsafe legacy metadata."""
    import zarr

    group = zarr.open_group(str(path), mode="r")
    if group.attrs.get("series_schema_version") != SCHEMA_VERSION:
        raise TimeValidationError(
            "schema", "unsupported series schema; rebuild from reviewed timestamped sources"
        )
    if cadence not in group.attrs.get("cadences", []):
        raise TimeValidationError(
            "cadence", f"native {cadence} series is unavailable; explicitly aggregate first"
        )
    time_name = "time" if cadence == "monthly" else f"{cadence}_time"
    required = [cadence, time_name, f"{time_name}_bounds"]
    if cadence in ("hourly", "daily"):
        required.append(f"{cadence}_valid")
    if any(name not in group for name in required) or any(
        name not in group.attrs
        for name in ("calendar", "timezone", "aggregation_timezone", "source_coverage", "synthetic")
    ):
        raise TimeValidationError(
            "coordinates", "missing time coordinates, bounds or source provenance"
        )
    axis = group[time_name]
    bounds_axis = group[f"{time_name}_bounds"]
    units = axis.attrs.get("units")
    factors = {TIME_UNITS: 1, "hours since 1970-01-01 00:00:00": 3600}
    if units not in factors or group.attrs["timezone"] != "UTC":
        raise TimeValidationError("timezone", "unsupported coordinate units or timezone")
    if axis.ndim != 1 or group[cadence].ndim != 1:
        raise TimeValidationError(
            "coordinates", "expected one-dimensional station series and time coordinates"
        )
    if bounds_axis.attrs.get("units") != units or bounds_axis.shape != (axis.shape[0], 2):
        raise TimeValidationError(
            "bounds", "time bounds shape or units do not match the coordinate axis"
        )
    factor = factors[units]
    result = TimeSeries(
        values=group[cadence][:].tolist(),
        times=[EPOCH + timedelta(seconds=float(v) * factor) for v in axis[:]],
        cadence=cadence,
        bounds=[
            tuple(EPOCH + timedelta(seconds=float(v) * factor) for v in row)
            for row in group[f"{time_name}_bounds"][:]
        ],
        valid=group[f"{cadence}_valid"][:].tolist() if f"{cadence}_valid" in group else None,
        calendar=group.attrs["calendar"],
        timezone=group.attrs["aggregation_timezone"],
        synthetic=group.attrs["synthetic"] or group.attrs.get("constant_series", False),
    )
    result.records()
    return result
