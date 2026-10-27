"""Darwin's own reader for native WRF history files and their calendars.

It decodes the WRF ``Times`` character array (``YYYY-MM-DD_hh:mm:ss`` per record),
checks ``XTIME`` offsets against it under a declared calendar, and reads the
grid/projection identity and per-variable metadata. Nothing is inferred from file
names, task directories or scheduler state. ``cftime`` is not required: only the
``standard`` (proleptic Gregorian) and ``noleap`` calendars are supported.

The grid identity digest uses the same canonical JSON as the CJC
``cjc.simulation-output/2`` contract, re-implemented here, not imported.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

CALENDARS = ("standard", "noleap")
PROJECTIONS = {1: "lambert_conformal", 2: "polar_stereographic", 3: "mercator", 6: "lat_lon"}
_CALENDAR_NAMES = {
    "standard": "standard",
    "gregorian": "standard",
    "proleptic_gregorian": "standard",
    "noleap": "noleap",
    "365_day": "noleap",
}
_WRF_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2}:\d{2}:\d{2})$")
_XTIME_UNITS = re.compile(r"^minutes since (\d{4}-\d{2}-\d{2})[ _T](\d{2}:\d{2}:\d{2})$")
_INT_ATTRIBUTES = {
    "map_proj": "MAP_PROJ",
    "grid_id": "GRID_ID",
    "parent_id": "PARENT_ID",
    "parent_grid_ratio": "PARENT_GRID_RATIO",
    "i_parent_start": "I_PARENT_START",
    "j_parent_start": "J_PARENT_START",
}
_FLOAT_ATTRIBUTES = {
    "dx": "DX",
    "dy": "DY",
    "truelat1": "TRUELAT1",
    "truelat2": "TRUELAT2",
    "stand_lon": "STAND_LON",
    "cen_lat": "CEN_LAT",
    "cen_lon": "CEN_LON",
    "moad_cen_lat": "MOAD_CEN_LAT",
    "pole_lat": "POLE_LAT",
    "pole_lon": "POLE_LON",
}
_NOLEAP_CUMULATIVE = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


class NativeHistoryError(ValueError):
    """A native file that cannot be read or does not look like WRF history."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def calendar_date(text: str, calendar: str) -> date:
    """Parse ``YYYY-MM-DD`` and reject dates that do not exist in the calendar."""
    if calendar not in CALENDARS:
        raise ValueError(f"unsupported calendar {calendar!r}")
    day = date.fromisoformat(text)
    if calendar == "noleap" and (day.month, day.day) == (2, 29):
        raise ValueError(f"{text} does not exist in the noleap calendar")
    return day


def parse_time(text: str) -> datetime:
    """Parse a contract timestamp (``YYYY-MM-DDThh:mm:ssZ``) as naive calendar fields."""
    return datetime.fromisoformat(text.removesuffix("Z"))


def format_time(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds") + "Z"


def calendar_seconds(text: str, calendar: str) -> int:
    """Seconds since 0001-01-01T00:00 in the calendar for a contract timestamp."""
    moment = parse_time(text)
    calendar_date(moment.date().isoformat(), calendar)
    if calendar == "noleap":
        days = (moment.year - 1) * 365 + _NOLEAP_CUMULATIVE[moment.month - 1] + moment.day - 1
    else:
        days = moment.toordinal() - 1
    return days * 86400 + moment.hour * 3600 + moment.minute * 60 + moment.second


def day_axis(start: str, history_seconds: int, calendar: str) -> tuple[str, list[str]]:
    """Right-closed ``(start, end]`` axis of one calendar day and its end timestamp."""
    moment = parse_time(start)
    if moment.time().isoformat() != "00:00:00":
        raise ValueError("a day must start at midnight")
    following = calendar_date(moment.date().isoformat(), calendar) + timedelta(days=1)
    if calendar == "noleap" and (following.month, following.day) == (2, 29):
        following += timedelta(days=1)
    end = f"{following.isoformat()}T00:00:00Z"
    times = [
        format_time(moment + timedelta(seconds=s))
        for s in range(history_seconds, 86400, history_seconds)
    ]
    return end, [*times, end]


def grid_digest(fields: dict) -> str:
    """Digest of the canonical grid/projection declaration (excluding id/sha256)."""
    canonical = {k: v for k, v in fields.items() if k not in {"id", "sha256"}}
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def read_wrf_history(path: str | Path) -> dict:
    """Return ``times``, ``grid``, ``calendar_attribute``, ``xtime`` and ``variables``.

    ``times`` are decoded from the native ``Times`` character array in file order.
    ``grid`` holds the WRF projection attributes and mass dimensions plus the
    projection name; it is the field set whose canonical digest identifies a grid.
    """
    import netCDF4
    import numpy as np

    try:
        with netCDF4.Dataset(path) as dataset:
            dataset.set_auto_mask(False)
            attrs = {name: dataset.getncattr(name) for name in dataset.ncattrs()}
            if "Times" not in dataset.variables:
                raise NativeHistoryError("native_structure", "history has no WRF Times variable")
            variable = dataset["Times"]
            variable.set_auto_chartostring(False)
            times = []
            for row in np.atleast_2d(np.asarray(variable[:])):
                text = b"".join(row.tolist()).decode("ascii", "replace").rstrip("\x00 ")
                match = _WRF_TIME.match(text)
                if match is None:
                    raise NativeHistoryError("native_structure", f"malformed WRF time {text!r}")
                times.append(f"{match[1]}T{match[2]}Z")
            calendar_attribute = None
            xtime = None
            if "XTIME" in dataset.variables:
                variable = dataset["XTIME"]
                calendar_attribute = getattr(variable, "calendar", None)
                match = _XTIME_UNITS.match(_text(getattr(variable, "units", "")).strip())
                if match is None:
                    raise NativeHistoryError(
                        "native_structure", "XTIME units are not minutes since"
                    )
                base = f"{match[1]}T{match[2]}Z"
                xtime = [(base, float(v)) for v in np.asarray(variable[:], dtype="f8")]
            variables = {
                name: {
                    "name": name,
                    "units": _text(getattr(item, "units", "")).strip(),
                    "dimensions": list(item.dimensions),
                    "stagger": _text(getattr(item, "stagger", "")).strip(),
                }
                for name, item in dataset.variables.items()
            }
            try:
                grid = {k: int(attrs[v]) for k, v in _INT_ATTRIBUTES.items()}
                grid |= {k: float(attrs[v]) for k, v in _FLOAT_ATTRIBUTES.items()}
                grid |= {
                    name: dataset.dimensions[name].size
                    for name in ("west_east", "south_north", "bottom_top")
                }
            except KeyError as error:
                raise NativeHistoryError(
                    "native_structure", f"missing WRF grid metadata {error}"
                ) from error
    except OSError as error:
        raise NativeHistoryError(
            "native_unreadable", f"cannot read native history {path}"
        ) from error
    if grid["map_proj"] not in PROJECTIONS:
        raise NativeHistoryError("native_structure", f"unsupported MAP_PROJ {grid['map_proj']}")
    grid["projection"] = PROJECTIONS[grid["map_proj"]]
    return {
        "times": times,
        "grid": grid,
        "calendar_attribute": calendar_attribute,
        "xtime": xtime,
        "variables": variables,
    }


def check_calendar(history: dict, calendar: str) -> None:
    """Native times must exist in ``calendar`` and agree with ``XTIME`` offsets under it."""
    attribute = history["calendar_attribute"]
    if attribute is not None:
        if _CALENDAR_NAMES.get(_text(attribute).strip().lower()) != calendar:
            raise NativeHistoryError("calendar_mismatch", "XTIME calendar differs from declaration")
    try:
        seconds = [calendar_seconds(time, calendar) for time in history["times"]]
        if history["xtime"] is not None:
            if len(history["xtime"]) != len(seconds):
                raise ValueError("XTIME and Times have different record counts")
            for moment, (base, minutes) in zip(seconds, history["xtime"], strict=True):
                if abs(moment - calendar_seconds(base, calendar) - minutes * 60) > 30:
                    raise ValueError("WRF Times and XTIME disagree under the calendar")
    except ValueError as error:
        raise NativeHistoryError("calendar_mismatch", str(error)) from error
