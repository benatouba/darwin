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
from typing import NoReturn

from darwin.publisher.immutable import staged_artifact

CONTRACT = "darwin.operations/1"
OPERATIONS = ("select", "cut", "crop", "integrate", "assemble", "assemble_simulation_output")
# Modules whose bytes define the replayed algorithms (see ``software``).
IMPLEMENTATION = ("operations.py", "simulation_output.py", "wrf_history.py")
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


def implementation_digest(files: dict[str, bytes]) -> str:
    """Unambiguous sha256 of named files: each framed by its name and byte length."""
    digest = hashlib.sha256()
    for name, data in files.items():
        encoded = name.encode()
        digest.update(f"{len(encoded)}:".encode() + encoded + f"{len(data)}:".encode())
        digest.update(data)
    return digest.hexdigest()


def software() -> dict:
    """Exact algorithm and dependency identities needed for recipe replay."""
    return {
        "darwin": version("darwin"),
        "implementation_sha256": implementation_digest(
            {name: (Path(__file__).parent / name).read_bytes() for name in IMPLEMENTATION}
        ),
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


def _recipe_bytes(recipe: dict) -> bytes:
    return (json.dumps(recipe, sort_keys=True, indent=2) + "\n").encode()


def _recipe_sha256(recipe: dict) -> str:
    """sha256 of the exact ``recipe.json`` bytes an operation bundle stores."""
    return hashlib.sha256(_recipe_bytes(recipe)).hexdigest()


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
    _check_simulation_output(recipe)


def _check_simulation_output(recipe: dict) -> dict | None:
    """Bind an ``assemble_simulation_output`` step to exactly its views' native bytes.

    It must be the first operation, and the recipe sources must be exactly the
    artifacts the embedded views declare, with the declared sizes and digests.
    Returns the source coverage, or None for recipes without the step.
    """
    from darwin.simulation_output import OPERATION, SimulationOutputError, check_views

    steps = [i for i, op in enumerate(recipe["operations"]) if op["operation"] == OPERATION]
    if not steps:
        return None
    if steps != [0]:
        raise OperationError("invalid_recipe", f"{OPERATION} must be the only first operation")
    parameters = recipe["operations"][0]["parameters"]
    try:
        views, coverage = check_views(parameters["views"], parameters["variables"])
    except SimulationOutputError as error:
        raise OperationError(error.code, str(error)) from error
    declared = {
        artifact["id"]: {"sha256": artifact["sha256"], "size_bytes": artifact["bytes"]}
        for view in views
        for artifact in view["artifacts"]
    }
    bound = {
        s["id"]: {"sha256": s["sha256"], "size_bytes": s["size_bytes"]} for s in recipe["sources"]
    }
    if bound != declared:
        raise OperationError(
            "invalid_recipe", "sources must be exactly the views' artifacts and byte identities"
        )
    return coverage


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


def _read_simulation_output(recipe: dict, locations: dict):
    from darwin.simulation_output import SimulationOutputError, assemble_views

    parameters = recipe["operations"][0]["parameters"]
    try:
        return assemble_views(parameters["views"], parameters["variables"], locations)[0]
    except SimulationOutputError as error:
        raise OperationError(error.code, str(error)) from error


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
    coverage = _check_simulation_output(recipe)
    operations = recipe["operations"][1:] if coverage is not None else recipe["operations"]
    try:
        with ExitStack() as stack:
            datasets = []
            for source in sources:
                path = locations[source["id"]]
                if source_identity(path) != {k: source[k] for k in ("sha256", "size_bytes")}:
                    raise OperationError("source_mismatch", f"source bytes changed: {source['id']}")
                if coverage is None:
                    datasets.append(stack.enter_context(xr.open_dataset(path, engine="netcdf4")))
            if coverage is not None:
                datasets = [_read_simulation_output(recipe, locations)]
            for operation in operations:
                datasets = _apply(datasets, operation)
            if len(datasets) != 1:
                raise OperationError(
                    "assembly_required", "a product requires one assembled dataset"
                )
            provenance = {
                "builder": "darwin.operations.execute",
                "recipe": {"sha256": _recipe_sha256(recipe)},
                "sources": [{"id": item["id"], "sha256": item["sha256"]} for item in sources],
                "software": recipe["software"],
            }
            with staged_artifact(destination, provenance=provenance) as staged:
                staged.mkdir()
                datasets[0].to_netcdf(staged / "data.nc", engine="netcdf4")
                for source in sources:
                    if source_identity(locations[source["id"]]) != {
                        k: source[k] for k in ("sha256", "size_bytes")
                    }:
                        raise OperationError(
                            "source_mismatch", f"source changed during execution: {source['id']}"
                        )
                (staged / "recipe.json").write_bytes(_recipe_bytes(recipe))
                report = {
                    "contract": "darwin.operation-result/1",
                    "recipe_sha256": source_identity(staged / "recipe.json")["sha256"],
                    "artifact": {"id": "data.nc", **source_identity(staged / "data.nc")},
                    # The operation ran over every given source; the coverage of
                    # those sources (a partial simulation output) bounds it.
                    "execution_completeness": "complete"
                    if coverage is None
                    else coverage["status"],
                    "scientific_status": "unassessed",
                }
                if coverage is not None:
                    report["source_coverage"] = coverage
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
            {
                "operation": "assemble_simulation_output",
                "parameters": ["views", "variables"],
                "scope": (
                    "first step only: exact native WRF records selected by chronological, "
                    "adjacent cjc.simulation-output/2 views with identical simulation, "
                    "calendar, cadence, grid and variable definitions; partial coverage "
                    "is reported, never filled"
                ),
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

    def usage(message: str) -> NoReturn:
        print(json.dumps({"code": "invalid_arguments", "message": message}), file=sys.stderr)
        raise SystemExit(2)

    if args.inventory:
        if args.recipe or args.locations or args.output:
            usage("--inventory takes no recipe, --locations or --output")
        print(json.dumps(inventory(), sort_keys=True))
        return
    if not (args.recipe and args.locations and args.output):
        usage("recipe, --locations and --output are required")
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
