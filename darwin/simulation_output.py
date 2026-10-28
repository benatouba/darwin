"""Consume CJC simulation-output views without CJC, setups or Slurm installed.

``cjc.simulation-output/1`` is inspected as a transport declaration with optional
byte checks. ``cjc.simulation-output/2`` is additionally verified against the
native WRF history records (``Times``, ``XTIME``, grid/projection, variables) and
can be assembled, strictly, through the ``assemble_simulation_output`` operation of
``darwin.operations``. Nothing here grants scientific acceptance or publication.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timedelta
from importlib.resources import files
from itertools import pairwise
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError, validators

from darwin.wrf_history import (
    PROJECTIONS,
    NativeHistoryError,
    calendar_seconds,
    check_calendar,
    day_axis,
    grid_digest,
    read_wrf_history,
)

CONTRACT = "cjc.simulation-output/1"
CONTRACT_V2 = "cjc.simulation-output/2"
SUPPORTED_CONTRACTS = (CONTRACT, CONTRACT_V2)
_SCHEMAS = {CONTRACT: "cjc-output-v1.schema.json", CONTRACT_V2: "cjc-output-v2.schema.json"}

__all__ = [
    "CONTRACT",
    "CONTRACT_V2",
    "SUPPORTED_CONTRACTS",
    "SimulationOutputError",
    "artifact_ids",
    "assemble_operation",
    "assemble_views",
    "check_views",
    "contract_schema",
    "inspect_manifest",
    "load_manifest",
    "read_wrf_history",
    "validate_manifest",
    "verify_completion_references",
    "verify_native",
]


class SimulationOutputError(ValueError):
    """A stable contract, storage-adapter or assembly failure."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def contract_schema(contract: str = CONTRACT) -> dict:
    """Return the bundled producer-owned structural schema for compatibility tests."""
    if contract not in _SCHEMAS:
        raise SimulationOutputError("unsupported_contract", f"no schema for {contract!r}")
    return json.loads(files("darwin").joinpath(_SCHEMAS[contract]).read_text())


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SimulationOutputError("invalid_manifest", f"duplicate JSON member: {key}")
        result[key] = value
    return result


def load_manifest(source: str | Path) -> dict:
    """Read JSON without silently accepting duplicate object members."""
    try:
        document = json.loads(Path(source).read_text(), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SimulationOutputError("invalid_input", "cannot read a JSON manifest") from exc
    return validate_manifest(document)


def _strict_validate(document: dict) -> None:
    strict = validators.extend(
        Draft202012Validator,
        type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
            "integer", lambda checker, value: type(value) is int
        ),
    )
    strict(contract_schema(document["contract"])).validate(document)


def validate_manifest(document: dict) -> dict:
    """Return an independent checked copy; verify semantic rules beyond JSON Schema."""
    if not isinstance(document, dict):
        raise SimulationOutputError("invalid_manifest", "manifest must be a JSON object")
    if document.get("contract") not in SUPPORTED_CONTRACTS:
        raise SimulationOutputError(
            "unsupported_contract", "unsupported simulation-output contract"
        )
    try:
        _strict_validate(document)
        if document["contract"] == CONTRACT:
            _check_v1(document)
        else:
            _check_v2(document)
    except (ValidationError, ValueError, OverflowError) as exc:
        raise SimulationOutputError("invalid_manifest", str(exc)) from exc
    return copy.deepcopy(document)


def _check_v1(document: dict) -> None:
    start, end = (
        datetime.fromisoformat(document["start"]),
        datetime.fromisoformat(document["end"]),
    )
    if start.time().isoformat() != "00:00:00" or end - start != timedelta(days=1):
        raise ValueError("expected a midnight-to-midnight UTC day")
    cadence = document["history_seconds"]
    if 86400 % cadence:
        raise ValueError("cadence must divide a day")
    expected = [
        (start + timedelta(seconds=i)).isoformat(timespec="seconds").replace("+00:00", "Z")
        for i in range(cadence, 86401, cadence)
    ]
    selections = document["selections"]
    times = [selection["time"] for selection in selections]
    if times != sorted(set(times)) or not set(times) <= set(expected):
        raise ValueError("timestamps must be unique, ordered and on the right-closed day axis")
    missing = sorted(set(expected) - set(times))
    if missing != document["missing_times"]:
        raise ValueError("explicit gaps disagree with selections")
    if document["completeness"] == "complete":
        if missing or document["partial_reason"] is not None:
            raise ValueError("complete views cannot have gaps or partial reasons")
    elif not missing or document["partial_reason"] is None:
        raise ValueError("partial views require gaps and a reason")
    for collection, reference in [("artifacts", "artifact_id"), ("segments", "segment_id")]:
        ids = [item["id"] for item in document[collection]]
        if len(ids) != len(set(ids)) or set(ids) != {s[reference] for s in selections}:
            raise ValueError("duplicate, missing or unused descriptor")
    names = [variable["name"] for variable in document["variables"]]
    records = [(s["artifact_id"], s["record_index"]) for s in selections]
    if len(names) != len(set(names)) or len(records) != len(set(records)):
        raise ValueError("duplicate variable or reused native record")


def _check_v2(document: dict) -> None:
    """Mirror the producer's documented v2 rules from Darwin's side of the contract."""
    if 86400 % document["history_seconds"]:
        raise ValueError("invalid_cadence: cadence must divide a day")
    end, expected = day_axis(document["start"], document["history_seconds"], document["calendar"])
    if document["end"] != end:
        raise ValueError("invalid_day: end must be the next calendar midnight")
    for collection, key in [
        ("artifacts", "id"),
        ("segments", "id"),
        ("run_roots", "id"),
        ("variables", "name"),
    ]:
        keys = [item[key] for item in document[collection]]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate {collection} identity")
    for variable in document["variables"]:
        if variable["dimensions"][0] != "Time":
            raise ValueError(f"variable_dimensions: {variable['name']} is not per record")
    _check_v2_grid(document)
    times = [s["time"] for s in document["selections"]]
    if times != sorted(set(times)):
        raise ValueError("duplicate_or_unordered_time: exactly one owner per timestamp")
    if not set(times) <= set(expected):
        raise ValueError("outside_axis: timestamp is not on the right-closed day axis")
    if document["missing_times"] != sorted(set(expected) - set(times)):
        raise ValueError("incorrect_gaps: missing timestamps disagree with selections")
    _check_v2_status(document)
    _check_v2_references(document)


def _check_v2_grid(document: dict) -> None:
    grid = document["grid"]
    if grid is None:
        return
    if grid["projection"] != PROJECTIONS.get(grid["map_proj"]):
        raise ValueError("projection_mismatch: MAP_PROJ and projection disagree")
    if grid_digest(grid) != grid["sha256"]:
        raise ValueError("grid_digest: grid identity digest does not match its fields")
    if grid["grid_id"] != int(document["domain"][1:]):
        raise ValueError("domain_mismatch: GRID_ID differs from the view domain")


def _check_v2_status(document: dict) -> None:
    status, reason, segments = document["status"], document["reason"], document["segments"]
    has_data = bool(document["selections"])
    missing = document["missing_times"]
    if status == "complete" and (missing or reason is not None):
        raise ValueError("false_complete: complete views cannot contain gaps or reasons")
    if status != "complete" and reason is None:
        raise ValueError("unexplained_status: non-complete views need a reason")
    if status == "partial" and not (has_data and missing):
        raise ValueError("invalid_partial: partial views need selections and gaps")
    if status in {"failed", "missing"} and (
        has_data or document["artifacts"] or document["grid"] is not None or document["variables"]
    ):
        raise ValueError(f"{status}_with_data: no artifacts, grid or variables allowed")
    if has_data and (document["grid"] is None or not document["variables"]):
        raise ValueError("undescribed_data: data views need native grid and variables")
    if status == "failed" and not any(s["outcome"] == "failed" for s in segments):
        raise ValueError("unevidenced_failure: failed views need a failed segment")
    if status == "missing" and (segments or document["run_roots"]):
        raise ValueError("missing_with_execution: use failed when a segment ran")
    if status != "missing" and document["execution_evidence"] is None:
        raise ValueError("missing_execution_evidence: executed views need evidence")
    for segment in segments:
        if segment["outcome"] == "failed" and segment["completion"] is not None:
            raise ValueError("failed_completion: failed segments have no completion record")


def _check_v2_references(document: dict) -> None:
    roots = {r["id"]: r for r in document["run_roots"]}
    segments = {s["id"]: s for s in document["segments"]}
    artifacts = {a["id"]: a for a in document["artifacts"]}
    for root in roots.values():
        if (root["role"] == "recovery") != (root["recovers"] is not None):
            raise ValueError("recovery_reference: only recovery roots name a recovered root")
        if root["recovers"] is not None and root["recovers"] not in roots.keys() - {root["id"]}:
            raise ValueError("recovery_reference: recovered root must be declared")
    referenced = {a["run_root_id"] for a in artifacts.values()}
    referenced |= {s["run_root_id"] for s in segments.values()}
    if not referenced <= roots.keys():
        raise ValueError("run_root_reference: undeclared run root")
    used = {s["run_root_id"] for s in segments.values()}
    used |= {r["recovers"] for r in roots.values() if r["recovers"]}
    if used != roots.keys():
        raise ValueError("run_root_reference: unused run root descriptor")
    if {s["artifact_id"] for s in document["selections"]} != artifacts.keys():
        raise ValueError("artifact_reference: missing or unused artifact descriptor")
    records = set()
    for selection in document["selections"]:
        segment = segments.get(selection["segment_id"])
        if segment is None:
            raise ValueError("segment_reference: undeclared segment")
        if segment["outcome"] != "completed":
            raise ValueError("failed_segment_data: records of failed segments are not data")
        artifact = artifacts[selection["artifact_id"]]
        if artifact["run_root_id"] != segment["run_root_id"]:
            raise ValueError("run_root_reference: artifact and segment roots differ")
        index = selection["record_index"]
        if index >= len(artifact["records"]) or artifact["records"][index] != selection["time"]:
            raise ValueError("record_time: selection disagrees with the native record time")
        key = (selection["artifact_id"], index)
        if key in records:
            raise ValueError("reused_record: one native record cannot represent two times")
        records.add(key)


def _check_locations(locations) -> None:
    if not isinstance(locations, dict) or not all(
        isinstance(k, str) and isinstance(v, (str, Path)) for k, v in locations.items()
    ):
        raise SimulationOutputError("invalid_locations", "locations must map IDs to paths")


def _verify_bytes(view: dict, locations: dict) -> None:
    _check_locations(locations)
    for artifact in view["artifacts"]:
        if artifact["id"] not in locations:
            raise SimulationOutputError("missing_artifact", f"no location for {artifact['id']}")
        digest = hashlib.sha256()
        size = 0
        try:
            with Path(locations[artifact["id"]]).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
        except OSError as exc:
            raise SimulationOutputError(
                "missing_artifact", f"cannot read {artifact['id']}"
            ) from exc
        if size != artifact["bytes"] or digest.hexdigest() != artifact["sha256"]:
            raise SimulationOutputError(
                "corrupt_artifact", f"byte identity mismatch for {artifact['id']}"
            )


def _require_v2(document: dict, purpose: str) -> dict:
    view = validate_manifest(document)
    if view["contract"] != CONTRACT_V2:
        raise SimulationOutputError(
            "unsupported_contract",
            f"{purpose} requires {CONTRACT_V2}; {CONTRACT} declares no native metadata",
        )
    return view


def verify_native(document: dict, locations: dict[str, str | Path]) -> dict[str, dict]:
    """Check bytes, then that native files still say what a v2 view declares.

    Returns the native history of each artifact, keyed by artifact ID. Errors:
    ``native_records`` (record times), ``native_grid`` (grid/projection),
    ``native_variable`` (units/dimensions/stagger), ``calendar_mismatch`` and
    ``native_structure``/``native_unreadable`` for files that are not WRF history.
    """
    view = _require_v2(document, "native verification")
    _verify_bytes(view, locations)
    histories = {}
    names = [v["name"] for v in view["variables"]]
    declared = {k: v for k, v in (view["grid"] or {}).items() if k not in {"id", "sha256"}}
    for artifact in view["artifacts"]:
        try:
            history = read_wrf_history(locations[artifact["id"]])
            if history["times"] != artifact["records"]:
                raise SimulationOutputError(
                    "native_records", f"{artifact['id']} record times differ from the view"
                )
            if history["grid"] != declared:
                raise SimulationOutputError(
                    "native_grid", f"{artifact['id']} grid/projection differs from the view"
                )
            check_calendar(history, view["calendar"])
            if [history["variables"].get(name) for name in names] != view["variables"]:
                raise SimulationOutputError(
                    "native_variable", f"{artifact['id']} variable metadata differs from the view"
                )
        except NativeHistoryError as error:
            raise SimulationOutputError(error.code, str(error)) from error
        histories[artifact["id"]] = history
    return histories


def verify_completion_references(
    document: dict, locations: dict[str, str | Path]
) -> dict[str, str]:
    """Read-only check of each completed segment's record in its own run root.

    ``locations`` maps segment IDs to completion records. Bytes must match the
    declared digest and the record's ``fingerprint`` must equal its run root's
    ``bundle_fingerprint``; recovery roots are never merged or rewritten. Returns
    ``{segment_id: run_root_id}`` for every checked record.
    """
    view = _require_v2(document, "completion verification")
    _check_locations(locations)
    fingerprints = {root["id"]: root["bundle_fingerprint"] for root in view["run_roots"]}
    checked = {}
    for segment in view["segments"]:
        if segment["completion"] is None:
            continue
        if segment["id"] not in locations:
            raise SimulationOutputError("missing_completion", f"no location for {segment['id']}")
        try:
            payload = Path(locations[segment["id"]]).read_bytes()
        except OSError as exc:
            raise SimulationOutputError(
                "missing_completion", f"cannot read {segment['id']}"
            ) from exc
        if hashlib.sha256(payload).hexdigest() != segment["completion"]["sha256"]:
            raise SimulationOutputError(
                "changed_completion", f"{segment['id']} completion record changed"
            )
        try:
            record = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise SimulationOutputError(
                "changed_completion", "completion record is not JSON"
            ) from exc
        if not isinstance(record, dict) or (
            record.get("fingerprint") != fingerprints[segment["run_root_id"]]
        ):
            raise SimulationOutputError(
                "fingerprint_mismatch", f"{segment['id']} completion belongs to another bundle"
            )
        checked[segment["id"]] = segment["run_root_id"]
    return checked


def inspect_manifest(
    document: dict,
    *,
    locations: dict[str, str | Path] | None = None,
    completions: dict[str, str | Path] | None = None,
) -> dict:
    """Report declared coverage and optionally verify resolved artifacts.

    Locations are a deployment-adapter input, not manifest-controlled paths. Files
    must be sealed externally: a checksum check cannot prevent a later mutation.
    For v2, resolved artifacts are also re-read natively; completion locations
    (segment ID to record path) additionally check completion record bytes.
    """
    view = validate_manifest(document)
    if view["contract"] == CONTRACT:
        if completions is not None:
            raise SimulationOutputError(
                "invalid_locations", f"{CONTRACT} declares no completion records"
            )
        if locations is not None:
            _verify_bytes(view, locations)
        return {
            "schema": "darwin.simulation-output-inspection/1",
            "contract": CONTRACT,
            "view_id": view["id"],
            "simulation_id": view["simulation"]["id"],
            "domain": view["domain"],
            "completeness": view["completeness"],
            "selected_times": [s["time"] for s in view["selections"]],
            "missing_times": view["missing_times"],
            "payload_verification": "byte_identity_checked"
            if locations is not None
            else "not_checked",
            "native_record_metadata": "not_checked",
            "referenced_receipts": "not_authenticated",
            "scientific_status": "not_assessed",
        }
    if locations is not None:
        verify_native(view, locations)
    if completions is not None:
        verify_completion_references(view, completions)
    return {
        "schema": "darwin.simulation-output-inspection/2",
        "contract": CONTRACT_V2,
        "view_id": view["id"],
        "simulation_id": view["simulation"]["id"],
        "domain": view["domain"],
        "calendar": view["calendar"],
        "completeness": view["status"],
        "reason": view["reason"],
        "selected_times": [s["time"] for s in view["selections"]],
        "missing_times": view["missing_times"],
        "run_roots": [
            {"id": r["id"], "role": r["role"], "recovers": r["recovers"]} for r in view["run_roots"]
        ],
        "payload_verification": "byte_identity_checked" if locations is not None else "not_checked",
        "native_record_metadata": "checked" if locations is not None else "not_checked",
        "referenced_receipts": "completion_bytes_checked"
        if completions is not None
        else "not_authenticated",
        "scientific_status": "not_assessed",
    }


OPERATION = "assemble_simulation_output"


def assemble_operation(views: list[dict], variables: list[str]) -> dict:
    """The explicit ``darwin.operations`` step that reads selected native records.

    Full view manifests are embedded so a recipe records exactly which native
    records, segments, run roots and declared coverage it was built from.
    """
    return {
        "operation": OPERATION,
        "parameters": {"views": copy.deepcopy(views), "variables": list(variables)},
    }


def artifact_ids(views: list[dict]) -> list[str]:
    """Native artifact IDs referenced by the views, in first-use order."""
    ordered = []
    for document in views:
        for artifact in document.get("artifacts", []):
            if artifact["id"] not in ordered:
                ordered.append(artifact["id"])
    return ordered


def view_digest(document: dict) -> str:
    """SHA-256 of a view's canonical JSON (sorted keys, compact separators)."""
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _bind(seen: dict, key: tuple, value, code: str) -> None:
    if seen.setdefault(key, value) != value:
        raise SimulationOutputError(code, f"{key[0]} {key[1]!r} differs between views")


def _check_identities(checked: list[dict]) -> None:
    """One simulation/domain/calendar/cadence; shared IDs never change meaning."""
    first = checked[0]
    ids = [view["id"] for view in checked]
    if len(ids) != len(set(ids)):
        raise SimulationOutputError("view_conflict", "a view may appear only once")
    seen: dict = {}
    for view in checked:
        for field, code in [
            ("simulation", "incompatible_assembly"),
            ("domain", "incompatible_assembly"),
            ("calendar", "incompatible_calendar"),
            ("history_seconds", "incompatible_cadence"),
        ]:
            if view[field] != first[field]:
                raise SimulationOutputError(code, f"views declare different {field}")
        for collection, label in [
            ("run_roots", "run_root"),
            ("segments", "segment"),
            ("artifacts", "artifact"),
        ]:
            for item in view[collection]:
                _bind(seen, (label, item["id"]), item, f"{label}_conflict")


def _check_definitions(data: list[dict], variables: list[str]) -> None:
    """Identical grid/projection and variable definitions; units are never converted."""
    if not isinstance(variables, list) or not variables or len(set(variables)) != len(variables):
        raise SimulationOutputError("invalid_recipe", "name each variable to read exactly once")
    reference = {v["name"]: v for v in data[0]["variables"]}
    for view in data:
        if view["grid"] != data[0]["grid"]:
            raise SimulationOutputError(
                "incompatible_grid", f"{view['id']} has a different grid/projection"
            )
        declared = {v["name"]: v for v in view["variables"]}
        for name in variables:
            if name not in declared or name not in reference:
                raise SimulationOutputError("variable", f"{name} is not declared by {view['id']}")
            if declared[name]["units"] != reference[name]["units"]:
                raise SimulationOutputError(
                    "incompatible_units", f"{name} units differ; conversion must be explicit"
                )
            if declared[name] != reference[name]:
                raise SimulationOutputError(
                    "incompatible_variable", f"{name} dimensions or staggering differ"
                )


def _check_timeline(checked: list[dict]) -> list[str]:
    """Chronological, adjacent days; covered records contiguous. Returns covered times."""
    calendar = checked[0]["calendar"]
    for previous, current in pairwise(checked):
        if current["start"] == previous["start"]:
            raise SimulationOutputError(
                "time_overlap", f"{previous['id']} and {current['id']} cover the same day"
            )
        if calendar_seconds(current["start"], calendar) < calendar_seconds(
            previous["start"], calendar
        ):
            raise SimulationOutputError("time_order", "views must be in chronological order")
        if current["start"] != previous["end"]:
            raise SimulationOutputError(
                "time_gap", f"no view covers {previous['end']} to {current['start']}"
            )
    axis = [
        time
        for view in checked
        for time in day_axis(view["start"], view["history_seconds"], calendar)[1]
    ]
    covered = [s["time"] for view in checked for s in view["selections"]]
    positions = [axis.index(time) for time in covered]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise SimulationOutputError(
            "time_gap", "missing records inside the assembled range; gaps are never filled"
        )
    return covered


def check_views(views: list[dict], variables: list[str]) -> tuple[list[dict], dict]:
    """Validate an ordered view set for assembly without opening native files.

    Returns the checked views and their ``source_coverage``. Every inconsistency
    fails; nothing is converted, reconciled, interpolated or filled. Leading or
    trailing missing records are reported in the coverage; interior gaps fail.
    """
    if not isinstance(views, list) or not views:
        raise SimulationOutputError("invalid_recipe", "assembly needs at least one view")
    checked = [_require_v2(document, "assembly") for document in views]
    _check_identities(checked)
    data = [view for view in checked if view["selections"]]
    if not data:
        raise SimulationOutputError("empty_selection", "the views contain no native records")
    _check_definitions(data, variables)
    covered = _check_timeline(checked)
    coverage = {
        "status": "complete"
        if all(view["status"] == "complete" for view in checked)
        else "partial",
        "covered_times": covered,
        "missing_times": [time for view in checked for time in view["missing_times"]],
        "views": [
            {
                "id": view["id"],
                "sha256": view_digest(view),
                "status": view["status"],
                "reason": view["reason"],
            }
            for view in checked
        ],
    }
    return checked, coverage


def _plain(value):
    return value.tolist() if hasattr(value, "tolist") else value


def assemble_views(views: list[dict], variables: list[str], locations: dict) -> tuple:
    """Read exactly the selected native records into one dataset; no repair.

    Returns ``(dataset, coverage)``. Bytes, record times, grid, calendar and
    variable metadata are verified against every view first. Values keep their
    native dtype and attributes; only declared, contiguous records are included.
    """
    import netCDF4  # noqa: PLC0415 - data readers load only when records are read
    import numpy as np  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    checked, coverage = check_views(views, variables)
    for view in checked:
        verify_native(view, locations)
    data = [view for view in checked if view["selections"]]
    declared = {v["name"]: v for v in data[0]["variables"]}
    roots = {a["id"]: a["run_root_id"] for view in data for a in view["artifacts"]}
    selections = [s for view in data for s in view["selections"]]
    records = {name: [] for name in variables}
    attributes: dict = {}
    try:
        with ExitStack() as stack:
            handles = {}
            for artifact_id in artifact_ids(data):
                handle = stack.enter_context(netCDF4.Dataset(locations[artifact_id]))
                handle.set_auto_maskandscale(False)
                handles[artifact_id] = handle
                for name in variables:
                    native = handle[name]
                    attrs = {k: _plain(native.getncattr(k)) for k in native.ncattrs()}
                    if attributes.setdefault(name, attrs) != attrs:
                        raise SimulationOutputError(
                            "incompatible_variable", f"{name} attributes differ in {artifact_id}"
                        )
            for selection in selections:
                native = handles[selection["artifact_id"]]
                for name in variables:
                    records[name].append(np.array(native[name][selection["record_index"]]))
    except OSError as error:
        raise SimulationOutputError("native_unreadable", str(error)) from error
    data_vars = {}
    for name in variables:
        attrs = dict(attributes[name])
        encoding = {k: attrs.pop(k) for k in ("_FillValue", "missing_value") if k in attrs}
        encoding.setdefault("_FillValue", None)
        variable = xr.Variable(declared[name]["dimensions"], np.stack(records[name]), attrs)
        variable.encoding = encoding
        data_vars[name] = variable
    times = [s["time"] for s in selections]
    calendar, origin = checked[0]["calendar"], checked[0]["start"]
    units = f"seconds since {origin.removesuffix('Z').replace('T', ' ')}"
    if calendar == "standard":
        naive = [t.removesuffix("Z") for t in times]
        time = xr.Variable(("Time",), np.array(naive, "datetime64[ns]"))
        time.encoding = {"units": units, "calendar": "proleptic_gregorian", "dtype": "int64"}
    else:
        # Kept encoded: noleap times are not representable as datetime64 offsets.
        offset = calendar_seconds(origin, calendar)
        seconds = [calendar_seconds(t, calendar) - offset for t in times]
        time = xr.Variable(("Time",), np.array(seconds, "int64"))
        time.attrs = {"units": units, "calendar": calendar}
    dataset = xr.Dataset(
        data_vars,
        coords={
            "Time": time,
            "Times": ("Time", [t.removesuffix("Z").replace("T", "_") for t in times]),
            "source_artifact": ("Time", [s["artifact_id"] for s in selections]),
            "source_record_index": (
                "Time",
                np.array([s["record_index"] for s in selections], "int64"),
            ),
            "source_segment": ("Time", [s["segment_id"] for s in selections]),
            "source_run_root": ("Time", [roots[s["artifact_id"]] for s in selections]),
        },
        attrs={
            "source_contract": CONTRACT_V2,
            "simulation_id": checked[0]["simulation"]["id"],
            "domain": checked[0]["domain"],
            "calendar": calendar,
            "history_seconds": checked[0]["history_seconds"],
            "grid": json.dumps(data[0]["grid"], sort_keys=True),
            "source_views": json.dumps(coverage["views"], sort_keys=True),
            "source_coverage_status": coverage["status"],
            "source_missing_times": json.dumps(coverage["missing_times"]),
            "scientific_status": "unassessed",
        },
    )
    return dataset, coverage


def _read_mapping(path: str | None):
    if path is None:
        return None
    mapping = json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)
    if not isinstance(mapping, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
    ):
        raise SimulationOutputError("invalid_locations", "locations must map IDs to path strings")
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?")
    parser.add_argument(
        "--schema", action="store_true", help="export the supported structural schema"
    )
    parser.add_argument(
        "--contract", choices=SUPPORTED_CONTRACTS, default=CONTRACT, help="schema to export"
    )
    parser.add_argument(
        "--locations", help="JSON object mapping opaque artifact IDs to local paths"
    )
    parser.add_argument(
        "--completions", help="JSON object mapping v2 segment IDs to completion record paths"
    )
    args = parser.parse_args(argv)
    try:
        if args.schema:
            print(json.dumps(contract_schema(args.contract), sort_keys=True))
            return 0
        if args.manifest is None:
            parser.error("a manifest path is required")
        report = inspect_manifest(
            load_manifest(args.manifest),
            locations=_read_mapping(args.locations),
            completions=_read_mapping(args.completions),
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    except (SimulationOutputError, OSError, ValueError) as exc:
        print(
            json.dumps({"code": getattr(exc, "code", "invalid_input"), "message": str(exc)}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
