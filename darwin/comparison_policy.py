"""Policy-bound station comparisons: supported Python interface and matching CLI.

A comparison takes exact model/observation store identities, a station location and
sampling identity, a named comparison policy revision, a cadence and a half-open
UTC timeframe. The policy (``darwin.comparison-policy/1``) is accepted only when a
setups ``study-requirements/1`` document pins its exact bytes as the
``operation_recipe``; draft requirements run only when explicitly allowed and stay
labelled draft. Results report scientific status from validated setups
``acceptance-assessment/1`` records, never from the comparison itself: running a
comparison, a draft policy or a UI selection grants no acceptance.

The version-1 ``darwin.comparison`` interface (``native-intervals-v1``) is a
separate policy and is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from bisect import bisect_right
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from functools import cache
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import jsonschema

from darwin import study_contracts
from darwin.publisher import series
from darwin.publisher.series import (
    TimeSeries,
    TimeValidationError,
    interval_end,
    load_series,
    timestamp,
)
from darwin.study_contracts import StudyContractError, sha256

POLICY_CONTRACT = "darwin.comparison-policy/1"
RECIPE_CONTRACT = "darwin.comparison-recipe/2"
OPERATION = "compare-policy-intervals"

#: Pairings this implementation can compute: (model, observation, units, kind).
IMPLEMENTED_PAIRINGS = {
    ("t2", "T", "K", "scalar"),
    ("prcp", "PCP_tot_bucket", "mm", "precipitation"),
}
#: Cadences with a station-store series contract. Annual is rejected explicitly:
#: no store carries annual series and no reviewed annual completeness rule exists.
SUPPORTED_CADENCES = ("hourly", "daily", "monthly")
METRICS = frozenset(
    {
        "paired_count",
        "means",
        "sample_variance",
        "bias",
        "mae",
        "rmse",
        "correlation",
        "histogram",
        "ecdf",
        "precipitation",
    }
)
#: Declared by some study specifications but not implemented for station stores.
UNSUPPORTED_METRICS = {
    "circular_direction": "no station-store contract carries wind direction/speed pairs",
}
DYNAMIC_VERSIONS = {"latest", "current", "HEAD"}


class ComparisonError(TimeValidationError):
    """Structured refusal: stable ``code``, message and machine-readable ``details``."""

    def __init__(self, code: str, message: str, details: Mapping | None = None) -> None:
        super().__init__(code, message)
        self.details = dict(details or {})

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "details": self.details}


def store_identity(path: str | Path) -> str:
    """Content digest of a local store: every file's relative path and bytes."""
    root = Path(path)
    if not root.is_dir():
        raise ComparisonError("store", "comparison requires a readable local station store")
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ComparisonError("store", "station stores must not contain symlinks")
        if item.is_file():
            name = item.relative_to(root).as_posix().encode()
            data = item.read_bytes()
            digest.update(len(name).to_bytes(8, "big") + name)
            digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def _software() -> dict:
    digest = hashlib.sha256()
    for module in (sys.modules[__name__], study_contracts, series):
        digest.update(Path(module.__file__).read_bytes())
    digest.update(files("darwin").joinpath("comparison-policy-v1.schema.json").read_bytes())
    return {"darwin": version("darwin"), "implementation_sha256": digest.hexdigest()}


@cache
def _policy_schema() -> dict:
    return json.loads(files("darwin").joinpath("comparison-policy-v1.schema.json").read_text())


_REQUEST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "model",
        "observations",
        "station",
        "sampling",
        "model_variable",
        "observation_field",
        "cadence",
        "start",
        "end",
    ],
    "properties": {
        "model": {"$ref": "#/$defs/identity"},
        "observations": {"$ref": "#/$defs/identity"},
        "station": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "lon", "lat"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "lon": {"type": "number"},
                "lat": {"type": "number"},
            },
        },
        "sampling": {"type": "object", "required": ["method"]},
        "model_variable": {"type": "string", "minLength": 1},
        "observation_field": {"type": "string", "minLength": 1},
        "cadence": {"type": "string", "minLength": 1},
        "start": {"type": "string", "minLength": 1},
        "end": {"type": "string", "minLength": 1},
    },
    "$defs": {
        "identity": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "version", "sha256"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "version": {"type": "string", "minLength": 1},
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        }
    },
}


def _validate(document: object, schema: dict, code: str) -> None:
    try:
        json.dumps(document, allow_nan=False)
        jsonschema.Draft202012Validator(schema).validate(document)
    except (ValueError, TypeError, jsonschema.ValidationError) as error:
        raise ComparisonError(code, getattr(error, "message", str(error))) from error


def _parse_policy(data: bytes, allow_synthetic: bool) -> dict:
    try:
        policy = json.loads(data)
    except (ValueError, TypeError) as error:
        raise ComparisonError("invalid_policy", str(error)) from error
    if not isinstance(policy, dict) or policy.get("contract") != POLICY_CONTRACT:
        raise ComparisonError("unsupported_contract", f"expected {POLICY_CONTRACT}")
    _validate(policy, _policy_schema(), "invalid_policy")
    if not allow_synthetic and policy["id"].startswith(study_contracts.SYNTHETIC_PREFIX):
        raise ComparisonError("synthetic_not_trusted", "synthetic policies are test-only")
    for cadence in policy["cadences"]:
        _supported_cadence(cadence)
    for metric in policy["metrics"]:
        if metric in UNSUPPORTED_METRICS:
            raise ComparisonError(
                "unsupported_metric", f"{metric}: {UNSUPPORTED_METRICS[metric]}", {"metric": metric}
            )
        if metric not in METRICS:
            raise ComparisonError(
                "unsupported_metric", f"unknown metric {metric!r}", {"metric": metric}
            )
    if "histogram" in policy["metrics"] and "histogram_bins" not in policy:
        raise ComparisonError("invalid_policy", "histogram metrics need declared histogram_bins")
    for pairing in policy["pairings"]:
        key = (
            pairing["model_variable"],
            pairing["observation_field"],
            pairing["units"],
            pairing["kind"],
        )
        if key not in IMPLEMENTED_PAIRINGS:
            raise ComparisonError(
                "unsupported_pairing",
                "pairing/units/kind are not implemented",
                {"pairing": pairing},
            )
        if (pairing["kind"] == "precipitation") != ("wet_threshold_mm" in pairing):
            raise ComparisonError(
                "invalid_policy", "precipitation pairings (only) must declare wet_threshold_mm"
            )
    for exclusion in policy["exclusions"]:
        if _utc(exclusion["start"], "invalid_policy") >= _utc(exclusion["end"], "invalid_policy"):
            raise ComparisonError("invalid_policy", "exclusion start must precede end")
    return policy


def _supported_cadence(cadence: str) -> None:
    if cadence == "annual":
        raise ComparisonError(
            "unsupported_cadence",
            "annual comparisons are not supported: no annual station series or reviewed "
            "annual completeness rule exists",
            {"cadence": cadence},
        )
    if cadence not in SUPPORTED_CADENCES:
        raise ComparisonError(
            "unsupported_cadence", f"unknown cadence {cadence!r}", {"cadence": cadence}
        )


def _utc(value: str, code: str) -> datetime:
    try:
        return timestamp(value)
    except TimeValidationError as error:
        raise ComparisonError(code, f"{value!r}: {error}") from error


def _study(data: bytes, allow_synthetic: bool) -> tuple[dict, dict]:
    try:
        document = study_contracts.parse_requirements(data, allow_synthetic=allow_synthetic)
        report = study_contracts.inspect_requirements(data, allow_synthetic=allow_synthetic)
    except StudyContractError as error:
        raise ComparisonError(error.code, str(error)) from error
    return document, report


def _expected_starts(start: datetime, end: datetime, cadence: str, zone: str) -> list[datetime]:
    try:
        TimeSeries(values=[0.0], times=[start], cadence=cadence, timezone=zone).records()
    except TimeValidationError as error:
        raise ComparisonError(
            "timeframe", f"start is not a {cadence} interval boundary in {zone}"
        ) from error
    starts, cursor = [], start
    local = series.time_zone(zone)
    while cursor < end:
        starts.append(cursor)
        cursor = timestamp(interval_end(cursor.astimezone(local), cadence))
    if cursor != end:
        raise ComparisonError("timeframe", f"end is not a {cadence} interval boundary")
    return starts


def _metric(value: float | None, reason: str | None = None) -> dict:
    return {
        "value": value,
        "defined": value is not None,
        "reason": None if value is not None else reason,
    }


def _scalar_metrics(model: list[float], obs: list[float], policy: dict) -> dict:
    declared, count = set(policy["metrics"]), len(model)
    errors = [m - o for m, o in zip(model, obs, strict=True)]
    none = "no_pairs"
    result: dict = {}
    if "paired_count" in declared:
        result["paired_count"] = count
    if "means" in declared:
        result["means"] = {
            "model": _metric(statistics.fmean(model) if count else None, none),
            "observations": _metric(statistics.fmean(obs) if count else None, none),
        }
    if "sample_variance" in declared:
        result["sample_variance"] = {
            name: _metric(
                statistics.variance(values) if count >= 2 else None, "fewer_than_two_pairs"
            )
            for name, values in (("model", model), ("observations", obs))
        }
    if "bias" in declared:
        result["bias"] = _metric(statistics.fmean(errors) if count else None, none)
    if "mae" in declared:
        result["mae"] = _metric(statistics.fmean(abs(e) for e in errors) if count else None, none)
    if "rmse" in declared:
        result["rmse"] = _metric(
            math.sqrt(statistics.fmean(e * e for e in errors)) if count else None, none
        )
    if "correlation" in declared:
        if count < 2:
            result["correlation"] = _metric(None, "fewer_than_two_pairs")
        elif len(set(model)) == 1 or len(set(obs)) == 1:
            result["correlation"] = _metric(None, "zero_variance")
        else:
            result["correlation"] = _metric(statistics.correlation(model, obs))
    if "histogram" in declared:
        result["histogram"] = _histogram(model, obs, policy["histogram_bins"])
    if "ecdf" in declared:
        result["ecdf"] = {"model": _ecdf(model), "observations": _ecdf(obs)}
    return result


def _histogram(model: list[float], obs: list[float], bins: int) -> dict:
    """Shared equal-width edges over both series; the last bin is closed."""
    values = model + obs
    if not values or min(values) == max(values):
        reason = "no_pairs" if not values else "zero_range"
        return {
            "defined": False,
            "reason": reason,
            "edges": None,
            "model_counts": None,
            "observation_counts": None,
        }
    lo, hi = min(values), max(values)
    edges = [lo + (hi - lo) * i / bins for i in range(bins)] + [hi]

    def counts(sample: list[float]) -> list[int]:
        found = [0] * bins
        for value in sample:
            found[min(bisect_right(edges, value) - 1, bins - 1)] += 1
        return found

    return {
        "defined": True,
        "reason": None,
        "edges": edges,
        "model_counts": counts(model),
        "observation_counts": counts(obs),
    }


def _ecdf(values: list[float]) -> list[list[float]]:
    """Right-continuous empirical CDF at each distinct value."""
    total, running, points = len(values), 0, []
    for value, count in sorted(Counter(values).items()):
        running += count
        points.append([value, running / total])
    return points


def _precipitation(model: list[float], obs: list[float], threshold: float) -> dict:
    """Totals, wet frequency (> threshold) and contingency with observations as truth."""
    count = len(model)
    model_wet = [value > threshold for value in model]
    obs_wet = [value > threshold for value in obs]
    pairs = list(zip(model_wet, obs_wet, strict=True))
    return {
        "wet_threshold_mm": threshold,
        "total_model": sum(model),
        "total_observations": sum(obs),
        "wet_frequency_model": sum(model_wet) / count if count else None,
        "wet_frequency_observations": sum(obs_wet) / count if count else None,
        "hits": pairs.count((True, True)),
        "misses": pairs.count((False, True)),
        "false_alarms": pairs.count((True, False)),
        "correct_negatives": pairs.count((False, False)),
    }


def _check_store(name: str, request: dict, path: Path, group, study: dict) -> dict:
    attrs = dict(group.attrs)
    if attrs.get("station") is None or attrs["station"].get("id") is None:
        raise ComparisonError(
            "station_identity", f"{name} store does not record a station identity"
        )
    if attrs["station"] != request["station"]:
        raise ComparisonError(
            "station_mismatch",
            f"{name} store station differs from the requested station",
            {"store": attrs["station"], "request": request["station"]},
        )
    if name == "model" and attrs.get("sampling") != request["sampling"]:
        raise ComparisonError(
            "sampling_mismatch",
            "model store sampling differs from the requested sampling identity",
            {"store": attrs.get("sampling"), "request": request["sampling"]},
        )
    upstream = attrs.get("source_gar_sha256" if name == "model" else "source_sha256")
    pinned = {source["sha256"] for source in study["sources"]}
    if upstream not in pinned:
        raise ComparisonError(
            "source_not_in_requirements",
            f"{name} store upstream source is not pinned by the requirements",
            {"upstream_sha256": upstream},
        )
    return {"upstream_sha256": upstream, "store_declared_status": attrs.get("scientific_status")}


def compare_with_policy(
    request: Mapping,
    *,
    model_path: str | Path,
    observations_path: str | Path,
    policy: bytes,
    requirements: bytes,
    assessments: Mapping[str, Mapping] | None = None,
    allow_draft: bool = False,
    allow_synthetic: bool = False,
) -> dict:
    """Compare two exact station stores under one requirements-pinned policy revision.

    ``policy`` and ``requirements`` are exact bytes. ``assessments`` optionally maps
    ``"model"``/``"observations"`` to ``{"assessment": bytes, "evidence": {id: bytes},
    "prior": bytes | None}`` records validated against the same requirements and the
    requested dataset identity. Raises ``ComparisonError`` with a stable code.
    """
    import zarr

    request = json.loads(json.dumps(request))
    _validate(request, _REQUEST_SCHEMA, "invalid_request")
    parsed_policy = _parse_policy(policy, allow_synthetic)
    study, report = _study(requirements, allow_synthetic)
    policy_identity = {
        "id": parsed_policy["id"],
        "version": parsed_policy["revision"],
        "sha256": sha256(policy),
    }
    if study["operation_recipe"] != policy_identity:
        raise ComparisonError(
            "policy_not_bound",
            "requirements do not pin this exact comparison policy revision",
            {"requirements": study["operation_recipe"], "policy": policy_identity},
        )
    if report["review_status"] != "reviewed" and not allow_draft:
        raise ComparisonError(
            "requirements_not_reviewed", "draft requirements run only with explicit allow_draft"
        )
    for name in ("model", "observations"):
        if request[name]["version"] in DYNAMIC_VERSIONS:
            raise ComparisonError("dynamic_version", f"{name} needs an exact version")
    if request["station"]["id"] not in study["stations"]:
        raise ComparisonError("station_not_in_requirements", "station is not in the requirements")
    matches = [
        pairing
        for pairing in parsed_policy["pairings"]
        if (pairing["model_variable"], pairing["observation_field"])
        == (request["model_variable"], request["observation_field"])
    ]
    if not matches:
        raise ComparisonError(
            "pairing_not_in_policy", "variable pairing is not declared by the policy"
        )
    pairing = matches[0]
    if pairing["requirement_variable"] not in study["variables"]:
        raise ComparisonError("variable_not_in_requirements", "policy variable is not required")
    cadence = request["cadence"]
    _supported_cadence(cadence)
    if cadence not in parsed_policy["cadences"]:
        raise ComparisonError("cadence_not_in_policy", f"{cadence} is not declared by the policy")
    # A precipitation metric declared for the policy applies to precipitation pairings.
    declared_precip = (
        "precipitation" in parsed_policy["metrics"] and pairing["kind"] == "precipitation"
    )
    start, end = _utc(request["start"], "timeframe"), _utc(request["end"], "timeframe")
    if start >= end:
        raise ComparisonError("timeframe", "start must precede end")

    paths = {"model": Path(model_path), "observations": Path(observations_path)}
    sources, records = {}, {}
    for name, path in paths.items():
        digest = store_identity(path)
        if digest != request[name]["sha256"]:
            raise ComparisonError(
                "source_mismatch",
                f"{name} store bytes differ from the requested exact version",
                {"expected": request[name]["sha256"], "actual": digest},
            )
        group = zarr.open_group(str(path), mode="r")
        sources[name] = _check_store(name, request, path, group, study)
        try:
            native = load_series(path, cadence)
        except TimeValidationError as error:
            raise ComparisonError(error.code, f"{name} store: {error}") from error
        if group[cadence].attrs.get("units") != pairing["units"]:
            raise ComparisonError(
                "units", f"expected native units {pairing['units']}; convert explicitly first"
            )
        records[name] = (native, native.records())
    zone = records["observations"][0].timezone
    expected = _expected_starts(start, end, cadence, zone)

    exclusions = [
        (_utc(e["start"], "invalid_policy"), _utc(e["end"], "invalid_policy"), e["reason"])
        for e in parsed_policy["exclusions"]
    ]
    excluded: Counter = Counter()
    model_values, obs_values = [], []
    represented = {name: 0 for name in records}
    for stamp in expected:
        stop = timestamp(interval_end(stamp.astimezone(series.time_zone(zone)), cadence))
        entries = {}
        for name, (_, table) in records.items():
            entry = table.get(stamp)
            if entry is not None:
                represented[name] += 1
                if entry[2] != stop:
                    raise ComparisonError("bounds", f"{name} interval bounds differ from {cadence}")
            entries[name] = entry
        reason = next((r for s, e, r in exclusions if s < stop and stamp < e), None)
        if reason is not None:
            excluded[f"policy:{reason}"] += 1
            continue
        missing = next(
            (
                f"{name}_{'absent' if entries[name] is None else 'invalid'}"
                for name in ("model", "observations")
                if entries[name] is None or not entries[name][1]
            ),
            None,
        )
        if missing:
            excluded[missing] += 1
            continue
        model_values.append(entries["model"][0])
        obs_values.append(entries["observations"][0])
    if not all(represented.values()):
        raise ComparisonError(
            "no_overlap",
            "a store has no intervals in the requested timeframe",
            {"represented_intervals": represented},
        )
    policy_excluded = sum(v for k, v in excluded.items() if k.startswith("policy:"))
    eligible = len(expected) - policy_excluded
    paired = len(model_values)
    completeness = {
        "requested_intervals": len(expected),
        "eligible_intervals": eligible,
        "paired_intervals": paired,
        "paired_fraction": paired / eligible if eligible else None,
        "excluded": dict(excluded),
    }
    rule = parsed_policy["completeness"]
    if (
        paired < rule["minimum_paired_count"]
        or not eligible
        or paired / eligible < rule["minimum_paired_fraction"]
    ):
        raise ComparisonError(
            "insufficient_completeness",
            "paired intervals do not meet the policy completeness rule",
            {"completeness": completeness, "rule": rule},
        )

    metrics = _scalar_metrics(model_values, obs_values, parsed_policy)
    if declared_precip:
        metrics["precipitation"] = _precipitation(
            model_values, obs_values, pairing["wet_threshold_mm"]
        )

    validated = {}
    for name in ("model", "observations"):
        record = (assessments or {}).get(name)
        if record is None:
            validated[name] = None
            continue
        try:
            summary = study_contracts.validate_assessment(
                record["assessment"],
                requirements=requirements,
                dataset=request[name],
                evidence=record.get("evidence", {}),
                prior=record.get("prior"),
                allow_synthetic=allow_synthetic,
            )
        except StudyContractError as error:
            raise ComparisonError(error.code, f"{name} assessment: {error}") from error
        validated[name] = {
            "assessment": summary["assessment"],
            "status": summary["status"],
            "authority": summary["authority"],
        }
    for name, path in paths.items():
        sources[name]["identity"] = request[name]
        sources[name]["assessment"] = validated[name]
        sources[name]["scientific_status"] = (
            validated[name]["status"] if validated[name] else "unassessed"
        )
        if store_identity(path) != request[name]["sha256"]:
            raise ComparisonError("identity", "store changed during comparison")

    return {
        "metrics": metrics,
        "completeness": completeness,
        "review_status": report["review_status"],
        "scientific_status": "unassessed",
        "sources": sources,
        "recipe": {
            "contract": RECIPE_CONTRACT,
            "operation": OPERATION,
            "request": request,
            "policy": policy_identity,
            "requirements": report["requirements"] | {"review_status": report["review_status"]},
            "assessments": {
                name: summary["assessment"] if summary else None
                for name, summary in validated.items()
            },
            "flags": {"allow_draft": allow_draft, "allow_synthetic": allow_synthetic},
            "software": _software(),
        },
    }


def replay_with_policy(
    recipe: Mapping,
    *,
    model_path: str | Path,
    observations_path: str | Path,
    policy: bytes,
    requirements: bytes,
    assessments: Mapping[str, Mapping] | None = None,
) -> dict:
    """Reproduce a recipe exactly; any changed input, policy or software is refused."""
    if recipe.get("contract") != RECIPE_CONTRACT or recipe.get("operation") != OPERATION:
        raise ComparisonError("recipe", "unsupported comparison recipe")
    if recipe.get("software") != _software():
        raise ComparisonError("software", "recipe requires the exact Darwin implementation")
    result = compare_with_policy(
        recipe["request"],
        model_path=model_path,
        observations_path=observations_path,
        policy=policy,
        requirements=requirements,
        assessments=assessments,
        **recipe["flags"],
    )
    if result["recipe"] != recipe:
        raise ComparisonError("identity", "recipe identity mismatch")
    return result


def _assessment_bundle(path: Path) -> dict:
    """Read ``{"assessment", "evidence": {id: file}, "prior"}`` paths relative to the bundle."""
    bundle = json.loads(path.read_text())
    root = path.parent
    return {
        "assessment": (root / bundle["assessment"]).read_bytes(),
        "evidence": {
            key: (root / value).read_bytes() for key, value in bundle.get("evidence", {}).items()
        },
        "prior": (root / bundle["prior"]).read_bytes() if bundle.get("prior") else None,
    }


def main(argv: list[str] | None = None) -> int:
    """Run or replay through the same interface; JSON result or structured error."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--requirements", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--request", type=Path, help="comparison request JSON")
    action.add_argument("--recipe", type=Path, help="recipe JSON to reproduce exactly")
    parser.add_argument("--assessment", action="append", default=[], metavar="NAME=BUNDLE")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--test-mode", action="store_true", help="accept synthetic: fixtures")
    args = parser.parse_args(argv)
    try:
        assessments = {}
        for item in args.assessment:
            name, _, bundle = item.partition("=")
            assessments[name] = _assessment_bundle(Path(bundle))
        common = {
            "model_path": args.model,
            "observations_path": args.observations,
            "policy": args.policy.read_bytes(),
            "requirements": args.requirements.read_bytes(),
            "assessments": assessments or None,
        }
        if args.recipe:
            result = replay_with_policy(json.loads(args.recipe.read_text()), **common)
        else:
            result = compare_with_policy(
                json.loads(args.request.read_text()),
                allow_draft=args.allow_draft,
                allow_synthetic=args.test_mode,
                **common,
            )
    except ComparisonError as error:
        print(json.dumps(error.as_dict()), file=sys.stderr)
        return 2
    except TimeValidationError as error:
        print(
            json.dumps({"code": error.code, "message": str(error), "details": {}}), file=sys.stderr
        )
        return 2
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(
            json.dumps({"code": "invalid_input", "message": str(error), "details": {}}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
