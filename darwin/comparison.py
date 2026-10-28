"""Public version-1 native-interval comparison interface and reproducible CLI.

No aggregation, conversion or disaggregation is implicit. Inputs are immutable
local version-2 station stores; deployment adapters resolve/transfer them first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from darwin.publisher import comparison as metrics
from darwin.publisher import series
from darwin.publisher.series import TimeValidationError, align_series, load_series, timestamp

CONTRACT_VERSION = 1
POLICY = "native-intervals-v1"


def _software() -> dict:
    digest = hashlib.sha256()
    for module in (sys.modules[__name__], metrics, series):
        digest.update(Path(module.__file__).read_bytes())
    return {"darwin": version("darwin"), "implementation_sha256": digest.hexdigest()}


def _identity(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise TimeValidationError("store", "comparison requires a readable local station store")
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise TimeValidationError("store", "station stores must not contain mutable symlinks")
        if item.is_file():
            name = item.relative_to(root).as_posix().encode()
            data = item.read_bytes()
            digest.update(len(name).to_bytes(8, "big") + name)
            digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def histogram_edges(values: list[float], bins: int) -> list[float]:
    """Return the existing equal-width presentation bins."""
    lo, hi = min(values), max(values)
    width = (hi - lo) / bins if hi > lo else 1.0
    return [lo + i * width for i in range(bins + 1)]


def _histogram(values: list[float], bins: int = 5) -> list[float]:
    edges = histogram_edges(values, bins)
    counts = [0.0] * bins
    for value in values:
        index = min(int((value - edges[0]) / (edges[1] - edges[0])), bins - 1)
        counts[index] += 1
    return counts


def compare_stores(
    model: str | Path,
    observations: str | Path,
    *,
    model_variable: str,
    observation_field: str,
    cadence: str,
    policy: str,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Validate immutable stores, select intervals and compare using one implementation.

    ``start`` is inclusive, ``end`` exclusive; selected intervals must fit entirely
    inside that window. Recipe identities include metadata, masks and coordinates.
    ``native-intervals-v1`` consumes the stores' explicit completeness masks; it
    neither changes their aggregation policy nor grants scientific acceptance.
    """
    import zarr

    if policy != POLICY:
        raise TimeValidationError("policy", f"unsupported comparison policy: {policy!r}")
    supported = {("t2", "T"): "K", ("prcp", "PCP_tot_bucket"): "mm"}
    units = supported.get((model_variable, observation_field))
    if units is None:
        raise TimeValidationError("variable", "unsupported model/observation variable pairing")
    left, right = timestamp(start) if start else None, timestamp(end) if end else None
    if left is not None and right is not None and left >= right:
        raise TimeValidationError("timeframe", "start must precede end")
    sources, selected = {}, []
    for name, path in (("model", model), ("observations", observations)):
        identity = _identity(path)
        native = load_series(path, cadence)
        group = zarr.open_group(str(path), mode="r")
        if group[cadence].attrs.get("units") != units:
            raise TimeValidationError("units", f"expected native units {units}; convert explicitly")
        records = native.records()
        kept = [
            (stamp, value, valid, bound)
            for stamp, (value, valid, bound) in records.items()
            if (left is None or stamp >= left) and (right is None or bound <= right)
        ]
        selected.append(
            replace(
                native,
                values=[v for _, v, _, _ in kept],
                times=[t for t, _, _, _ in kept],
                valid=[v for _, _, v, _ in kept],
                bounds=[(t, b) for t, _, _, b in kept],
            )
        )
        sources[name] = {
            "sha256": identity,
            # This legacy interface does not validate signed dataset assessments.
            # Store metadata is a claim, never authority to accept a dataset.
            "scientific_status": "unassessed",
            "store_declared_status": group.attrs.get("scientific_status", "unassessed"),
        }
    result = asdict(metrics.compare(model_variable, observation_field, *selected))
    gar, obs, _ = align_series(*selected)
    count = result["paired_count"]
    result["completeness_ratio"] = count / (count + result["excluded_count"]) if count else 0.0
    result["correlation_defined"] = result["correlation"] is not None
    for name, path in (("model", model), ("observations", observations)):
        if _identity(path) != sources[name]["sha256"]:
            raise TimeValidationError("identity", "store changed during comparison")
    return {
        "statistics": result,
        "histogram": {"gar_bins": _histogram(gar), "obs_bins": _histogram(obs)} if gar else None,
        "ecdf": {
            "gar": [(i + 1) / count for i in range(count)],
            "obs": [(i + 1) / count for i in range(count)],
        }
        if gar
        else None,
        "precip_extras": asdict(metrics.precip_extras(gar, obs))
        if gar and model_variable == "prcp"
        else None,
        "recipe": {
            "schema_version": CONTRACT_VERSION,
            "operation": "compare-native-intervals",
            "parameters": {
                "model_variable": model_variable,
                "observation_field": observation_field,
                "cadence": cadence,
                "policy": policy,
                "start": start,
                "end": end,
            },
            "sources": sources,
            "software": _software(),
        },
    }


def replay(recipe: dict, model: str | Path, observations: str | Path) -> dict:
    """Reproduce a recipe using caller-resolved paths, with no path/version fallback."""
    if (
        recipe.get("schema_version") != CONTRACT_VERSION
        or recipe.get("operation") != "compare-native-intervals"
    ):
        raise TimeValidationError("recipe", "unsupported comparison recipe")
    if recipe.get("software") != _software():
        raise TimeValidationError("software", "recipe requires the exact Darwin implementation")
    result = compare_stores(model, observations, **recipe["parameters"])
    if result["recipe"] != recipe:
        raise TimeValidationError("identity", "recipe source identity mismatch")
    return result


def main() -> int:
    """Run the same public interface, emitting JSON results or structured errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--recipe", required=True, help="Darwin recipe JSON to reproduce")
    args = parser.parse_args()
    try:
        result = replay(json.loads(Path(args.recipe).read_text()), args.model, args.observations)
    except (TimeValidationError, OSError, KeyError, TypeError, ValueError) as error:
        print(
            json.dumps({"code": getattr(error, "code", "recipe"), "message": str(error)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
