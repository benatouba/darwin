"""Inspect CJC output declarations and byte identities without CJC or Slurm installed.

This is a transport-contract adapter, not yet a native NetCDF record reader.
No operation here grants scientific acceptance or publication permission.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timedelta
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError, validators

CONTRACT = "cjc.simulation-output/1"


class SimulationOutputError(ValueError):
    """A stable contract or storage-adapter failure."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def contract_schema() -> dict:
    """Return the bundled producer-owned structural schema for compatibility tests."""
    return json.loads(files("darwin").joinpath("cjc-output-v1.schema.json").read_text())


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


def validate_manifest(document: dict) -> dict:
    """Return an independent checked copy; verify semantic rules beyond JSON Schema."""
    if not isinstance(document, dict):
        raise SimulationOutputError("invalid_manifest", "manifest must be a JSON object")
    if document.get("contract") != CONTRACT:
        raise SimulationOutputError(
            "unsupported_contract", "unsupported simulation-output contract"
        )
    schema = contract_schema()
    strict = validators.extend(
        Draft202012Validator,
        type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
            "integer", lambda checker, value: type(value) is int
        ),
    )
    try:
        strict(schema).validate(document)
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
    except (ValidationError, ValueError, OverflowError) as exc:
        raise SimulationOutputError("invalid_manifest", str(exc)) from exc
    return copy.deepcopy(document)


def inspect_manifest(document: dict, *, locations: dict[str, str | Path] | None = None) -> dict:
    """Report declared coverage and optionally verify resolved artifact bytes.

    Locations are a deployment-adapter input, not manifest-controlled paths. Files
    must be sealed externally: a checksum check cannot prevent a later mutation.
    """
    view = validate_manifest(document)
    if locations is not None:
        if not isinstance(locations, dict) or not all(
            isinstance(k, str) and isinstance(v, (str, Path)) for k, v in locations.items()
        ):
            raise SimulationOutputError("invalid_locations", "locations must map IDs to paths")
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
    return {
        "schema": "darwin.simulation-output-inspection/1",
        "contract": CONTRACT,
        "view_id": view["id"],
        "simulation_id": view["simulation"]["id"],
        "domain": view["domain"],
        "completeness": view["completeness"],
        "selected_times": [s["time"] for s in view["selections"]],
        "missing_times": view["missing_times"],
        "payload_verification": "byte_identity_checked" if locations is not None else "not_checked",
        "native_record_metadata": "not_checked",
        "referenced_receipts": "not_authenticated",
        "scientific_status": "not_assessed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?")
    parser.add_argument(
        "--schema", action="store_true", help="export the supported structural schema"
    )
    parser.add_argument(
        "--locations", help="JSON object mapping opaque artifact IDs to local paths"
    )
    args = parser.parse_args(argv)
    try:
        if args.schema:
            print(json.dumps(contract_schema(), sort_keys=True))
            return 0
        if args.manifest is None:
            parser.error("a manifest path is required")
        document = load_manifest(args.manifest)
        locations = (
            json.loads(Path(args.locations).read_text(), object_pairs_hook=_unique_object)
            if args.locations
            else None
        )
        if locations is not None and (
            not isinstance(locations, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in locations.items())
        ):
            raise SimulationOutputError(
                "invalid_locations", "locations must map IDs to path strings"
            )
        print(json.dumps(inspect_manifest(document, locations=locations), sort_keys=True))
        return 0
    except (SimulationOutputError, OSError, ValueError) as exc:
        print(
            json.dumps({"code": getattr(exc, "code", "invalid_input"), "message": str(exc)}),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
