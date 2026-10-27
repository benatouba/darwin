"""Supported, transport-independent dataset operations and reproducible CLI.

Only explicit selections, coordinate crops, sums and strict concatenation are
supported. No resampling, unit conversion, gap filling or acceptance is implicit.
Local paths are supplied separately from scientific recipes by deployment adapters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

from darwin.publisher.immutable import staged_artifact

CONTRACT = "darwin.operations/1"
OPERATIONS = ("select", "cut", "crop", "integrate", "assemble")
INTEGRATION_METHODS = ("sum-discrete-amounts",)

# Requested transformations that version 1 refuses instead of approximating.
UNSUPPORTED = {
    "resample": "temporal resampling is not implemented; no implicit aggregation",
    "interpolate": "interpolation is not implemented; no nearest/linear fallback",
    "regrid": "regridding/reprojection is not implemented",
    "convert_units": "unit conversion is not implemented; units are preserved exactly",
    "fill_gaps": "gap filling is not implemented; gaps fail assembly",
    "average": "averaging is not implemented; only declared discrete sums",
    "reconcile": "metadata reconciliation is not implemented; differences fail assembly",
}
UNSUPPORTED_INTEGRATION = {
    "rate-integral": "time/bounds-aware rate integration is not implemented",
    "trapezoid": "numerical rate integration is not implemented",
}


class OperationError(ValueError):
    """Actionable stable error code, independent of Python/CLI invocation."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def software() -> dict:
    """Exact algorithm and dependency identities needed for recipe replay."""
    return {
        "darwin": version("darwin"),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "xarray": version("xarray"),
        "numpy": version("numpy"),
        "netCDF4": version("netCDF4"),
    }


def source_identity(path: str | Path) -> dict:
    """Hash a local native file without putting its location in a recipe."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def make_recipe(sources: dict, operations: list[dict]) -> dict:
    """Bind explicit operations to caller-named source byte identities.

    Source mapping order defines assembly order and is recorded as a list. No
    hostnames, storage locations, credentials or scientific approval are inferred.
    """
    recipe = {
        "contract": CONTRACT,
        "sources": [{"id": key, **source_identity(path)} for key, path in sources.items()],
        "operations": operations,
        "software": software(),
    }
    validate_recipe(recipe)
    return recipe


def validate_recipe(recipe: dict) -> None:
    """Reject unknown contracts and ambiguous parameters before opening inputs."""
    if not isinstance(recipe, dict) or recipe.get("contract") != CONTRACT:
        raise OperationError("unsupported_contract", f"expected {CONTRACT}")
    try:
        json.dumps(recipe, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise OperationError("invalid_recipe", "recipe must contain finite JSON values") from error
    _refuse_unsupported(recipe.get("operations"))
    from importlib.resources import files

    import jsonschema

    schema = json.loads(files("darwin").joinpath("operations-v1.schema.json").read_text())
    try:
        jsonschema.Draft202012Validator(schema).validate(recipe)
    except jsonschema.ValidationError as error:
        raise OperationError("invalid_recipe", error.message) from error
    ids = [source["id"] for source in recipe["sources"]]
    if len(ids) != len(set(ids)):
        raise OperationError("invalid_recipe", "duplicate source identities")


def _refuse_unsupported(operations) -> None:
    """Name unsupported operations/methods instead of a generic schema failure."""
    if not isinstance(operations, list):
        return
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        name = operation.get("operation")
        if isinstance(name, str) and name not in OPERATIONS:
            reason = UNSUPPORTED.get(name, "not in the supported operation inventory")
            raise OperationError("unsupported_operation", f"{name}: {reason}")
        parameters = operation.get("parameters")
        method = parameters.get("method") if isinstance(parameters, dict) else None
        if name == "integrate" and isinstance(method, str) and method not in INTEGRATION_METHODS:
            reason = UNSUPPORTED_INTEGRATION.get(method, "not a supported integration method")
            raise OperationError("unsupported_method", f"integrate {method}: {reason}")


def _axis(dataset, name):
    if name not in dataset.coords or dataset[name].dims != (name,):
        raise OperationError("coordinate", f"{name!r} must be an explicit 1D dimension coordinate")
    index = dataset.indexes[name]
    if not index.is_unique or not index.is_monotonic_increasing:
        raise OperationError("coordinate", f"{name!r} must be unique and increasing")
    return index


def _assemble(datasets, dimension, cadence_seconds):
    import numpy as np
    import xarray as xr

    first = datasets[0]
    for dataset in datasets:
        _axis(dataset, dimension)
        if set(dataset.data_vars) != set(first.data_vars) or set(dataset.coords) != set(
            first.coords
        ):
            raise OperationError("incompatible_assembly", "variable/coordinate inventories differ")
        if dataset.attrs != first.attrs:
            raise OperationError(
                "incompatible_assembly", "global metadata differ; reconcile explicitly"
            )
        for name in first.variables:
            left, right = first[name], dataset[name]
            if left.dims != right.dims or left.attrs != right.attrs or left.dtype != right.dtype:
                raise OperationError("incompatible_assembly", f"variable metadata differ: {name}")
            for field in ("calendar", "units"):
                if dimension in left.dims and left.encoding.get(field) != right.encoding.get(field):
                    raise OperationError("incompatible_assembly", f"encoded {field} differ: {name}")
            if dimension not in left.dims and not left.identical(right):
                raise OperationError(
                    "incompatible_assembly", f"static coordinate/grid/field differ: {name}"
                )
    try:
        result = xr.concat(
            datasets,
            dim=dimension,
            data_vars="minimal",
            coords="minimal",
            compat="identical",
            join="exact",
            combine_attrs="identical",
        )
    except (ValueError, TypeError) as error:
        raise OperationError("incompatible_assembly", str(error)) from error
    index = _axis(result, dimension)
    if len(index) > 1:
        try:
            differences = [float((b - a).total_seconds()) for a, b in zip(index[:-1], index[1:])]
        except AttributeError as error:
            raise OperationError(
                "coordinate", "assembly requires a decoded datetime axis"
            ) from error
        if not np.allclose(differences, cadence_seconds, rtol=0, atol=0):
            raise OperationError("time_gap", "time axis differs from explicitly declared cadence")
    return result


def _apply(datasets, operation):
    name, parameters = operation["operation"], operation["parameters"]
    if name == "assemble":
        return [_assemble(datasets, parameters["dimension"], parameters["cadence_seconds"])]
    if len(datasets) != 1:
        raise OperationError(
            "assembly_required", "assemble multiple inputs before other operations"
        )
    dataset = datasets[0]
    if name == "select":
        missing = set(parameters["variables"]) - set(dataset.data_vars)
        if missing:
            raise OperationError("variable", f"unknown variables: {sorted(missing)}")
        return [dataset[parameters["variables"]]]
    if name in {"cut", "crop"}:
        import numpy as np

        dimension = parameters["dimension"]
        _axis(dataset, dimension)
        lower, upper = parameters["start"], parameters["end"]
        if name == "cut":
            if dataset[dimension].dtype.kind != "M":
                raise OperationError(
                    "calendar", "cut currently requires a standard datetime64 axis"
                )
            lower, upper = np.datetime64(lower), np.datetime64(upper)
        elif not all(isinstance(value, (int, float)) for value in (lower, upper)):
            raise OperationError("selection", "crop requires explicit numeric coordinate bounds")
        if lower >= upper:
            raise OperationError("selection", "start must precede end")
        # Half-open selections, including numeric crops; never nearest-neighbour fallback.
        result = dataset.sel(
            {dimension: (dataset[dimension] >= lower) & (dataset[dimension] < upper)}
        )
        if result.sizes[dimension] == 0:
            raise OperationError(
                "empty_selection", f"no coordinates in requested {dimension} interval"
            )
        return [result]
    if name == "integrate":
        # This is a declared sum of discrete amounts, NOT a rate/time integral.
        dimension = parameters["dimension"]
        _axis(dataset, dimension)
        variables = parameters["variables"]
        if any(v not in dataset or dimension not in dataset[v].dims for v in variables):
            raise OperationError(
                "variable", "all sum variables must contain the selected dimension"
            )
        result = dataset[variables].sum(dim=dimension, skipna=False, keep_attrs=True)
        for variable in variables:
            previous = result[variable].attrs.get("cell_methods", "")
            result[variable].attrs["cell_methods"] = f"{previous} {dimension}: sum".strip()
        return [result]
    raise OperationError("unsupported_operation", name)


def execute(recipe: dict, locations: dict, destination: str | Path) -> dict:
    """Replay a recipe into a new atomic bundle (data.nc + recipe + result).

    Completion describes execution only. This API cannot publish, grant access or
    accept science; the returned scientific status is always unassessed.
    """
    import xarray as xr

    validate_recipe(recipe)
    if recipe["software"] != software():
        raise OperationError(
            "software_mismatch", "recipe requires the exact implementation and dependencies"
        )
    sources = recipe["sources"]
    if set(locations) != {item["id"] for item in sources}:
        raise OperationError("missing_source", "provide exactly the recipe's source ID mappings")
    try:
        with ExitStack() as stack:
            datasets = []
            for source in sources:
                path = locations[source["id"]]
                if source_identity(path) != {k: source[k] for k in ("sha256", "size_bytes")}:
                    raise OperationError("source_mismatch", f"source bytes changed: {source['id']}")
                datasets.append(stack.enter_context(xr.open_dataset(path, engine="netcdf4")))
            for operation in recipe["operations"]:
                datasets = _apply(datasets, operation)
            if len(datasets) != 1:
                raise OperationError(
                    "assembly_required", "a product requires one assembled dataset"
                )
            with staged_artifact(destination) as staged:
                staged.mkdir()
                datasets[0].to_netcdf(staged / "data.nc", engine="netcdf4")
                for source in sources:
                    if source_identity(locations[source["id"]]) != {
                        k: source[k] for k in ("sha256", "size_bytes")
                    }:
                        raise OperationError(
                            "source_mismatch", f"source changed during execution: {source['id']}"
                        )
                (staged / "recipe.json").write_text(
                    json.dumps(recipe, sort_keys=True, indent=2) + "\n"
                )
                report = {
                    "contract": "darwin.operation-result/1",
                    "recipe_sha256": source_identity(staged / "recipe.json")["sha256"],
                    "artifact": {"id": "data.nc", **source_identity(staged / "data.nc")},
                    "execution_completeness": "complete",
                    "scientific_status": "unassessed",
                }
                (staged / "result.json").write_text(
                    json.dumps(report, sort_keys=True, indent=2) + "\n"
                )
        return report
    except OperationError:
        raise
    except FileExistsError as error:
        raise OperationError("destination_exists", str(error)) from error
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise OperationError("invalid_input", str(error)) from error


def inventory() -> dict:
    """Machine-readable version-1 operation inventory, shared by Python and CLI.

    Anything absent from ``supported`` is refused with a structured error; the
    ``unsupported`` reasons name commonly requested transformations explicitly.
    """
    return {
        "contract": CONTRACT,
        "input_format": "CF-NetCDF via xarray/netCDF4",
        "supported": [
            {"operation": "select", "parameters": ["variables"]},
            {
                "operation": "cut",
                "parameters": ["dimension", "start", "end"],
                "scope": "half-open interval on a standard datetime64 dimension coordinate",
            },
            {
                "operation": "crop",
                "parameters": ["dimension", "start", "end"],
                "scope": "half-open numeric interval on a 1-D increasing dimension coordinate",
            },
            {
                "operation": "integrate",
                "parameters": ["dimension", "variables", "method"],
                "methods": list(INTEGRATION_METHODS),
                "scope": "sum of discrete amounts; missing values propagate",
            },
            {
                "operation": "assemble",
                "parameters": ["dimension", "cadence_seconds"],
                "scope": "ordered concatenation with identical inventories and strict cadence",
            },
        ],
        "unsupported": dict(UNSUPPORTED),
        "unsupported_integration_methods": dict(UNSUPPORTED_INTEGRATION),
    }


def main() -> None:
    """Run an exact recipe with deployment-local source locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path, nargs="?")
    parser.add_argument("--locations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--inventory", action="store_true", help="print the supported operation inventory"
    )
    args = parser.parse_args()
    if args.inventory:
        if args.recipe or args.locations or args.output:
            parser.error("--inventory takes no recipe, --locations or --output")
        print(json.dumps(inventory(), sort_keys=True))
        return
    if not (args.recipe and args.locations and args.output):
        parser.error("recipe, --locations and --output are required")
    try:
        report = execute(
            json.loads(args.recipe.read_text()), json.loads(args.locations.read_text()), args.output
        )
    except (OperationError, OSError, ValueError) as error:
        print(
            json.dumps({"code": getattr(error, "code", "invalid_input"), "message": str(error)}),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
