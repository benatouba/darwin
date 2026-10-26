"""Supported provisional AWS screening API and matching ``python -m`` CLI.

No raw-data edits, gap filling, publication, model comparison or acceptance.
Policy and station metadata are explicit, hashed JSON inputs. Reports contain
counts and sample-level reasons, not redistributable observation values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPORT_SCHEMA = "darwin.aws-screening-report/1"
UNITS = {
    "T": "degC",
    "RH": "%",
    "WS": "m s-1",
    "WD": "degree",
    "SLR": "W m-2",
    "Pabs": "hPa",
    "PCP_tot_bucket": "mm per 10 min",
    "PCP_diff_radar": "mm per 10 min",
    "PCP_acoustic": "mm per 10 min",
}
RAIN = {"PCP_tot_bucket", "PCP_diff_radar", "PCP_acoustic"}


class ScreeningError(ValueError):
    """Actionable input failure, with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ScreeningError(code, message)


def _keys(value: dict, names: str) -> None:
    _require(
        isinstance(value, dict) and set(value) == set(names.split()),
        "schema",
        f"expected fields: {names}",
    )


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _read_json(path: Path) -> tuple[dict, dict]:
    data = path.read_bytes()
    try:
        result = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ScreeningError("json", f"invalid JSON: {path.name}") from exc
    _require(isinstance(result, dict), "schema", "JSON root must be an object")
    return result, _identity(data)


def _identity(data: bytes) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _validate_policy(p: dict) -> None:
    _keys(
        p,
        "schema id status cadence_seconds utc_offset_minutes timestamp_label "
        "time_basis_status ranges rh_cap_percent step_limits persistence_seconds "
        "saturated_rh_persistence_seconds saturated_rh_percent pressure_reference "
        "night gauge minimum_interval_fraction hour_mean_min_records day_mean_min_hours",
    )
    _require(
        p["schema"] == "darwin.aws-screening-policy/1", "version", "unsupported policy version"
    )
    _require(isinstance(p["id"], str) and bool(p["id"]), "policy", "policy id is required")
    _require(p["status"] == "provisional_screening_only", "policy", "screening is not acceptance")
    _require(
        type(p["cadence_seconds"]) is int and p["cadence_seconds"] == 600,
        "cadence",
        "only ten-minute source intervals are supported",
    )
    _require(
        type(p["utc_offset_minutes"]) is int and -720 <= p["utc_offset_minutes"] <= 840,
        "time_basis",
        "declare a fixed source UTC offset in minutes",
    )
    _require(
        p["timestamp_label"] == "interval_end"
        and p["time_basis_status"] == "inferred_not_operator_confirmed",
        "time_basis",
        "this version supports the explicitly inferred end-label convention",
    )
    _keys(p["ranges"], " ".join(sorted(set(UNITS) - {"Pabs"})))
    for bounds in p["ranges"].values():
        _require(
            isinstance(bounds, list)
            and len(bounds) == 2
            and _number(bounds[0])
            and (bounds[1] is None or (_number(bounds[1]) and bounds[1] >= bounds[0])),
            "policy",
            "invalid range bounds",
        )
    _keys(p["step_limits"], "T RH")
    _keys(
        p["pressure_reference"],
        "method sea_level_hpa sea_level_kelvin lapse_kelvin_per_m "
        "gravity_m_s2 dry_air_gas_constant tolerance_hpa",
    )
    _keys(p["night"], "method elevation_below_degrees shortwave_above_w_m2")
    _keys(p["gauge"], "ratio_above larger_total_above_mm")
    _require(
        p["pressure_reference"]["method"] == "isa_troposphere"
        and p["night"]["method"] == "noaa_julian_century_geometric_v1",
        "method",
        "unsupported pressure or solar method",
    )
    for group in (p["step_limits"], p["pressure_reference"], p["gauge"]):
        for key, value in group.items():
            if key != "method":
                _require(
                    _number(value) and value > 0, "policy", f"invalid positive parameter {key}"
                )
    for key in (
        "rh_cap_percent",
        "persistence_seconds",
        "saturated_rh_persistence_seconds",
        "saturated_rh_percent",
        "minimum_interval_fraction",
    ):
        _require(_number(p[key]) and p[key] > 0, "policy", f"invalid positive parameter {key}")
    _require(
        p["minimum_interval_fraction"] <= 1
        and p["saturated_rh_persistence_seconds"] >= p["persistence_seconds"]
        and _number(p["ranges"]["RH"][1])
        and p["ranges"]["RH"][0]
        <= p["saturated_rh_percent"]
        <= p["rh_cap_percent"]
        <= p["ranges"]["RH"][1],
        "policy",
        "inconsistent completeness or humidity parameters",
    )
    for key, upper in (("hour_mean_min_records", 6), ("day_mean_min_hours", 24)):
        _require(type(p[key]) is int and 1 <= p[key] <= upper, "policy", f"invalid {key}")
    night = p["night"]
    _require(
        _number(night["elevation_below_degrees"])
        and -90 < night["elevation_below_degrees"] < 90
        and _number(night["shortwave_above_w_m2"])
        and night["shortwave_above_w_m2"] >= 0,
        "policy",
        "invalid night-radiation thresholds",
    )


def _validate_station(m: dict) -> None:
    _keys(m, "schema id latitude longitude elevation_m units source_sha256 metadata_source_sha256")
    _require(
        m["schema"] == "darwin.aws-screening-station/1", "version", "unsupported station version"
    )
    _require(isinstance(m["id"], str) and bool(m["id"]), "station", "station id is required")
    for key, lo, hi in (
        ("latitude", -90, 90),
        ("longitude", -180, 180),
        ("elevation_m", -500, 11000),
    ):
        _require(_number(m[key]) and lo <= m[key] <= hi, "station", f"invalid {key}")
    _require(
        isinstance(m["units"], dict)
        and bool(m["units"])
        and all(key in UNITS and unit == UNITS[key] for key, unit in m["units"].items()),
        "units",
        "declare supported source units; conversion is not implicit",
    )
    for key in ("source_sha256", "metadata_source_sha256"):
        value = m[key]
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            "identity",
            f"invalid {key}",
        )


def solar_elevation(time: datetime, latitude: float, longitude: float) -> float:
    """Meeus/NOAA geometric solar elevation; east-positive longitude, no refraction.

    Julian-century equations as documented by NOAA's Solar Calculation Details.
    Used only for the conservative night screen, not observed sunshine or cloud.
    """
    _require(
        time.tzinfo is not None and 1900 <= time.year <= 2100,
        "solar_time",
        "solar method requires an aware time in 1900–2100",
    )
    _require(
        _number(latitude)
        and -90 <= latitude <= 90
        and _number(longitude)
        and -180 <= longitude <= 180,
        "solar_location",
        "invalid solar coordinates",
    )
    time = time.astimezone(UTC)
    century = (time.timestamp() / 86400 + 2440587.5 - 2451545) / 36525
    mean_long = (280.46646 + century * (36000.76983 + century * 0.0003032)) % 360
    mean_anomaly = math.radians(357.52911 + century * (35999.05029 - 0.0001537 * century))
    eccentricity = 0.016708634 - century * (0.000042037 + 0.0000001267 * century)
    center = (
        math.sin(mean_anomaly) * (1.914602 - century * (0.004817 + 0.000014 * century))
        + math.sin(2 * mean_anomaly) * (0.019993 - 0.000101 * century)
        + math.sin(3 * mean_anomaly) * 0.000289
    )
    omega = math.radians(125.04 - 1934.136 * century)
    apparent_long = math.radians(mean_long + center - 0.00569 - 0.00478 * math.sin(omega))
    obliquity = math.radians(
        23
        + (26 + (21.448 - century * (46.815 + century * (0.00059 - century * 0.001813))) / 60) / 60
        + 0.00256 * math.cos(omega)
    )
    declination = math.asin(math.sin(obliquity) * math.sin(apparent_long))
    y = math.tan(obliquity / 2) ** 2
    long_rad = math.radians(mean_long)
    equation_minutes = 4 * math.degrees(
        y * math.sin(2 * long_rad)
        - 2 * eccentricity * math.sin(mean_anomaly)
        + 4 * eccentricity * y * math.sin(mean_anomaly) * math.cos(2 * long_rad)
        - 0.5 * y * y * math.sin(4 * long_rad)
        - 1.25 * eccentricity * eccentricity * math.sin(2 * mean_anomaly)
    )
    minutes = time.hour * 60 + time.minute + time.second / 60
    hour_angle = math.radians(((minutes + equation_minutes + 4 * longitude) % 1440) / 4 - 180)
    lat = math.radians(latitude)
    sine = math.sin(lat) * math.sin(declination) + math.cos(lat) * math.cos(declination) * math.cos(
        hour_angle
    )
    return math.degrees(math.asin(max(-1, min(1, sine))))


def _utc_midnight(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ScreeningError("window", "use UTC midnight ISO timestamps") from exc
    _require(
        value.endswith("Z")
        and result.utcoffset() == timedelta(0)
        and result.time().isoformat() == "00:00:00",
        "window",
        "window boundaries must be UTC midnight",
    )
    return result


def _stamp(time: datetime) -> str:
    return time.isoformat().replace("+00:00", "Z")


def _records(
    data: bytes, fields: list[str], start: datetime, end: datetime, p: dict
) -> tuple[list, dict]:
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig")))
    header = next(reader, [])
    required = {"datetime", *fields}
    _require(
        all(header.count(field) == 1 for field in required),
        "columns",
        "missing or duplicate selected CSV columns",
    )
    positions = {field: header.index(field) for field in required}
    ignored = sorted(set(header) - required)
    column_info = {
        "selected": sorted(required),
        "ignored": ignored,
        "duplicate_ignored": [field for field in ignored if header.count(field) > 1],
    }
    halo = timedelta(seconds=p["saturated_rh_persistence_seconds"] + p["cadence_seconds"])
    records, previous = [], None
    for row_number, row in enumerate(reader, 2):
        _require(len(row) == len(header), "columns", f"malformed CSV row {row_number}")
        try:
            time = datetime.fromisoformat(row[positions["datetime"]])
        except (ValueError, TypeError) as exc:
            raise ScreeningError(
                "timestamp", f"invalid source timestamp at row {row_number}"
            ) from exc
        _require(
            time.tzinfo is None, "time_basis", "source timestamps must be naive local end labels"
        )
        time = time.replace(tzinfo=UTC) - timedelta(minutes=p["utc_offset_minutes"])
        if not start - halo <= time <= end + halo:
            continue
        _require(
            previous is None or time > previous,
            "timestamp_order",
            f"duplicate or nonmonotonic timestamp at row {row_number}",
        )
        _require(
            time.second == 0 and time.microsecond == 0 and time.minute % 10 == 0,
            "off_grid",
            f"off-grid timestamp at row {row_number}",
        )
        previous = time
        values, flags = {}, {field: set() for field in fields}
        for field in fields:
            raw = row[positions[field]].strip()
            if raw.lower() in ("", "null", "none", "nan", "na"):
                values[field] = None
                flags[field].add("missing")
                continue
            try:
                value = float(raw)
            except ValueError as exc:
                raise ScreeningError(
                    "numeric", f"invalid {field} number at row {row_number}"
                ) from exc
            _require(math.isfinite(value), "numeric", f"nonfinite {field} at row {row_number}")
            values[field] = value
        records.append({"time": time, "values": values, "flags": flags})
    return records, column_info


def _flag_samples(records: list, p: dict, m: dict) -> None:
    pressure = p["pressure_reference"]
    base = 1 - pressure["lapse_kelvin_per_m"] * m["elevation_m"] / pressure["sea_level_kelvin"]
    _require(
        base > 0, "pressure_reference", "elevation outside supported pressure-reference atmosphere"
    )
    reference = pressure["sea_level_hpa"] * base ** (
        pressure["gravity_m_s2"]
        / (pressure["dry_air_gas_constant"] * pressure["lapse_kelvin_per_m"])
    )
    ranges = {
        **p["ranges"],
        "Pabs": [reference - pressure["tolerance_hpa"], reference + pressure["tolerance_hpa"]],
    }
    for index, row in enumerate(records):
        for field, value in row["values"].items():
            if value is None:
                continue
            lo, hi = ranges[field]
            if value < lo or (hi is not None and value > hi):
                row["flags"][field].add("range")
            if field in p["step_limits"] and index:
                previous = records[index - 1]
                old = previous["values"][field]
                if (
                    old is not None
                    and (row["time"] - previous["time"]).total_seconds() == p["cadence_seconds"]
                ):
                    if abs(value - old) > p["step_limits"][field]:
                        row["flags"][field].add("step")
            if field == "SLR" and value > p["night"]["shortwave_above_w_m2"]:
                middle = row["time"] - timedelta(seconds=p["cadence_seconds"] / 2)
                if (
                    solar_elevation(middle, m["latitude"], m["longitude"])
                    < p["night"]["elevation_below_degrees"]
                ):
                    row["flags"][field].add("night_radiation")
    for field in set(m["units"]) & {"T", "RH", "WS"}:
        run = []
        for index in range(len(records) + 1):
            row = records[index] if index < len(records) else None
            if run and (
                row is None
                or row["values"][field] is None
                or row["values"][field] != run[-1]["values"][field]
                or (row["time"] - run[-1]["time"]).total_seconds() != p["cadence_seconds"]
            ):
                duration = (run[-1]["time"] - run[0]["time"]).total_seconds()
                limit = (
                    p["saturated_rh_persistence_seconds"]
                    if field == "RH" and run[0]["values"][field] >= p["saturated_rh_percent"]
                    else p["persistence_seconds"]
                )
                if duration > limit:
                    for item in run:
                        item["flags"][field].add("persistence")
                run = []
            if row is not None and row["values"][field] is not None:
                run.append(row)


def screen_csv(
    source: str | Path, *, station_metadata: str | Path, policy: str | Path, start: str, end: str
) -> dict:
    """Screen ``(start, end]`` UTC days without modifying any input.

    Metadata declares exact units and source digest. Only selected-window and
    adjacent persistence-context values are inspected; full source bytes are
    hashed. Counts are eligibility evidence, not aggregated observations or scores.
    """
    p, policy_identity = _read_json(Path(policy))
    m, metadata_identity = _read_json(Path(station_metadata))
    _validate_policy(p)
    _validate_station(m)
    first, last = _utc_midnight(start), _utc_midnight(end)
    _require(first < last, "window", "start must precede end")
    _require(1900 <= first.year <= last.year <= 2100, "window", "unsupported screening years")
    data = Path(source).read_bytes()
    source_identity = _identity(data)
    _require(
        source_identity["sha256"] == m["source_sha256"],
        "source_digest",
        "source differs from station metadata",
    )
    fields = sorted(m["units"])
    records, column_info = _records(data, fields, first, last, p)
    _flag_samples(records, p, m)
    days, gauge_checks = {}, []
    for offset in range((last - first).days):
        left = first + timedelta(days=offset)
        right = left + timedelta(days=1)
        rows = [row for row in records if left < row["time"] <= right]
        gauges = sorted(set(fields) & RAIN)
        for a, b in itertools.combinations(gauges, 2):
            complete = len(rows) == 144 and all(
                row["flags"][f] <= {"gauge_disagreement"} for row in rows for f in (a, b)
            )
            state = "unevaluated"
            if complete:
                totals = [math.fsum(row["values"][f] for row in rows) for f in (a, b)]
                lo, hi = min(totals), max(totals)
                disagreement = (
                    hi > p["gauge"]["larger_total_above_mm"] and hi > p["gauge"]["ratio_above"] * lo
                )
                state = "flagged" if disagreement else "passed"
                if disagreement:
                    for row in rows:
                        for field in (a, b):
                            row["flags"][field].add("gauge_disagreement")
            gauge_checks.append({"day": left.date().isoformat(), "fields": [a, b], "status": state})
        daily = {}
        for field in fields:
            available = sum(row["values"][field] is not None for row in rows)
            eligible = [row for row in rows if not row["flags"][field]]
            hours = Counter((row["time"] - timedelta(microseconds=1)).hour for row in eligible)
            eligible_hours = sum(
                count >= (6 if field in RAIN else p["hour_mean_min_records"])
                for count in hours.values()
            )
            flag_counts = Counter(flag for row in rows for flag in row["flags"][field])
            daily[field] = {
                "expected": 144,
                "available": available,
                "eligible": len(eligible),
                "flags": dict(sorted(flag_counts.items())),
                "eligible_hours": eligible_hours,
                "daily_aggregation_eligible": eligible_hours
                >= (24 if field in RAIN else p["day_mean_min_hours"]),
            }
        days[left.date().isoformat()] = daily
    issues, corrections = [], []
    for row in records:
        if not first < row["time"] <= last:
            continue
        for field in fields:
            flags = row["flags"][field]
            if flags:
                issues.append(
                    {"time": _stamp(row["time"]), "field": field, "reasons": sorted(flags)}
                )
            value = row["values"][field]
            if (
                field == "RH"
                and value is not None
                and p["rh_cap_percent"] < value <= p["ranges"]["RH"][1]
            ):
                corrections.append(
                    {
                        "time": _stamp(row["time"]),
                        "field": "RH",
                        "operation": "cap_percent",
                        "parameter": p["rh_cap_percent"],
                        "eligible": not flags,
                    }
                )
    present_times = {row["time"] for row in records if first < row["time"] <= last}
    missing_times = [
        _stamp(first + timedelta(seconds=index * 600))
        for index in range(1, (last - first).days * 144 + 1)
        if first + timedelta(seconds=index * 600) not in present_times
    ]
    totals = {}
    for field in fields:
        counts = [day[field] for day in days.values()]
        eligible = sum(day["eligible"] for day in counts)
        expected = len(days) * 144
        totals[field] = {
            "available": sum(day["available"] for day in counts),
            "eligible": eligible,
            "expected": expected,
            "interval_score_eligible": eligible / expected >= p["minimum_interval_fraction"],
        }
    try:
        installed = version("darwin")
    except PackageNotFoundError:
        installed = None
    return {
        "schema": REPORT_SCHEMA,
        "scientific_status": "not_assessed",
        "purpose": "provisional_screening_only",
        "station": m,
        "source": source_identity,
        "policy": {"id": p["id"], **policy_identity},
        "station_metadata": metadata_identity,
        "software": {
            "darwin": installed,
            "python": sys.version.split()[0],
            "implementation_sha256": _identity(Path(__file__).read_bytes())["sha256"],
        },
        "recipe": {
            "start": start,
            "end": end,
            "ownership": "(start,end]",
            "time_basis_status": p["time_basis_status"],
            "timestamp_validation_scope": "selection_plus_context",
            "context_seconds": p["saturated_rh_persistence_seconds"] + p["cadence_seconds"],
        },
        "source_columns": column_info,
        "daily": days,
        "totals": totals,
        "issues": issues,
        "corrections": corrections,
        "missing_times": missing_times,
        "gauge_checks": gauge_checks,
        "limitations": [
            "Provider visual QC is inherited, not independently certified.",
            "No daily-climate-record QC, fog conversion, cloud-event selection or scientific acceptance.",
            "Solar geometry excludes refraction and terrain; pressure uses a standard-atmosphere approximation.",
            "Missing timestamps are counted as absent, never imputed; no numerical aggregates are produced.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the same public operation; emit JSON report or a structured error."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "station-metadata", "policy", "start", "end"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        report = screen_csv(
            args.source,
            station_metadata=args.station_metadata,
            policy=args.policy,
            start=args.start,
            end=args.end,
        )
    except (ScreeningError, OSError, UnicodeError, csv.Error) as exc:
        print(
            json.dumps({"error": {"code": getattr(exc, "code", "input"), "message": str(exc)}}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
