"""Versioned native published-grid operations, independent of HTTP and storage adapters.

This contract preserves the dashboard's legacy grid definitions explicitly. It is
not a new study policy or scientific acceptance. Locations are supplied separately
from recipes, and every replay verifies exact source bytes and software.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
import zarr

from darwin.errors import CodedError, emit_error

CONTRACT = "darwin.grid-operations/1"
DEFINITION = "dashboard-native-grid/1"


class GridError(CodedError):
    """Structured scientific-operation refusal, shared by Python and CLI."""


def software() -> dict:
    """Pin the implementation, interpreter and numerical/storage dependencies."""
    import platform

    return {
        "darwin": version("darwin"),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "zarr": version("zarr"),
        "numcodecs": version("numcodecs"),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def source_identity(path: str | Path) -> dict:
    """Hash every relative file name and byte, including metadata (not payload only)."""
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise GridError("source_unavailable", "A local immutable Zarr directory is required")
    digest = hashlib.sha256()
    size = 0
    for file in sorted(root.rglob("*")):
        if file.is_symlink():
            raise GridError("source_type", "Symlinks are not immutable source files")
        if not file.is_file():
            continue
        name = file.relative_to(root).as_posix().encode()
        data = file.read_bytes()
        digest.update(f"{len(name)}:".encode() + name + f"{len(data)}:".encode() + data)
        size += len(data)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def make_recipe(sources: dict, operation: dict, *, scientific_status: dict) -> dict:
    """Bind explicit definitions to exact sources and caller-reported source labels.

    Each source supplies path, variable, product_id, version and published status.
    The derived product is always unassessed; source labels never grant acceptance.
    """
    recipe = {
        "contract": CONTRACT,
        "definition": DEFINITION,
        "sources": [
            {
                **{k: v for k, v in source.items() if k != "path"},
                "id": key,
                **source_identity(source["path"]),
            }
            for key, source in sources.items()
        ],
        "operation": operation,
        "software": software(),
        "scientific_status": {"derived_acceptance": "unassessed", "sources": scientific_status},
    }
    validate_recipe(recipe)
    return recipe


def validate_recipe(recipe: dict) -> None:
    """Reject unversioned, implicit or unsupported transformations."""
    try:
        _validate_recipe(recipe)
    except GridError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise GridError("recipe", f"Malformed grid recipe: {exc}") from exc


def _validate_recipe(recipe: dict) -> None:
    if set(recipe) != {
        "contract",
        "definition",
        "sources",
        "operation",
        "software",
        "scientific_status",
    }:
        raise GridError("recipe", "Use only the declared grid recipe fields")
    if recipe.get("contract") != CONTRACT or recipe.get("definition") != DEFINITION:
        raise GridError("contract", "Use darwin.grid-operations/1 and dashboard-native-grid/1")
    sources = recipe.get("sources")
    if not isinstance(sources, list) or not sources:
        raise GridError("recipe", "At least one exact source is required")
    ids = [s.get("id") for s in sources]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise GridError("recipe", "Source ids must be unique nonempty strings")
    for source in sources:
        if (
            not isinstance(source.get("sha256"), str)
            or len(source["sha256"]) != 64
            or not isinstance(source.get("size_bytes"), int)
            or not isinstance(source.get("variable"), str)
            or not source.get("product_id")
            or source.get("version") is None
        ):
            raise GridError(
                "recipe", "Source requires exact bytes, variable, product_id and version"
            )
    status = recipe.get("scientific_status", {})
    if status.get("derived_acceptance") != "unassessed" or set(status.get("sources", {})) != set(
        ids
    ):
        raise GridError(
            "scientific_status", "Preserve each source label; derived results are unassessed"
        )
    op = recipe.get("operation", {})
    name = op.get("name")
    allowed = {"subset", "layer", "aggregate", "regional_summary"}
    if name not in allowed:
        raise GridError("unsupported_operation", f"Supported operations: {sorted(allowed)}")
    common = {"name", "source", "bbox", "polygon", "level"}
    extra = {
        "subset": set(),
        "layer": {"timestep"},
        "aggregate": {"method", "start", "end", "missing_data", "partner"},
        "regional_summary": {"mode", "mask"},
    }[name]
    if set(op) - common - extra:
        raise GridError("unsupported_parameter", "Operation contains undeclared parameters")
    if op.get("source") not in ids:
        raise GridError("recipe", "Operation source must name a pinned source")
    bbox = op.get("bbox")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox)
        or bbox[0] >= bbox[2]
        or bbox[1] >= bbox[3]
    ):
        raise GridError("geometry", "bbox requires finite xmin < xmax and ymin < ymax")
    polygon = op.get("polygon")
    if polygon is not None:
        if (
            len(polygon) != 1
            or len(polygon[0]) < 4
            or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in polygon[0])
        ):
            raise GridError(
                "geometry", "Only one finite polygon exterior without holes is supported"
            )
    if name == "aggregate":
        if op.get("method") not in ("mean", "depth_sum", "component_mean"):
            raise GridError("unsupported_method", "Choose mean, depth_sum or component_mean")
        if op.get("missing_data") not in ("require-complete", "propagate"):
            raise GridError("missing_data", "Declare require-complete or propagate explicitly")
        if op.get("method") == "component_mean" and op.get("partner") not in ids:
            raise GridError("recipe", "component_mean requires a pinned partner")
    if name == "regional_summary":
        if op.get("mode") not in ("land-only", "whole-domain"):
            raise GridError("mode", "Declare land-only or whole-domain explicitly")
        if op["mode"] == "land-only" and op.get("mask") not in ids:
            raise GridError("recipe", "land-only requires an exact mask source")


def polygon_rect_fraction(ring, cell) -> float:
    """Legacy Sutherland-Hodgman clipping and shoelace footprint fraction."""
    poly = ring
    x0, y0, x1, y1 = cell
    for axis, bound, keep_le in ((0, x1, True), (0, x0, False), (1, y1, True), (1, y0, False)):
        out = []
        for k in range(len(poly)):
            p, q = poly[k], poly[(k + 1) % len(poly)]
            pin = p[axis] <= bound if keep_le else p[axis] >= bound
            qin = q[axis] <= bound if keep_le else q[axis] >= bound
            if pin:
                out.append(p)
            if pin != qin:
                t = (bound - p[axis]) / (q[axis] - p[axis])
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            if qin and not pin:
                out.append(q)
        poly = out
        if not poly:
            return 0.0
    area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))) / 2.0
    return area / ((x1 - x0) * (y1 - y0))


def inside_mask(region, xs, ys, dx, dy, i0, j0, nj, ni):
    """Boolean legacy download mask for a previously selected native cell window.

    ``region`` is a structural object with ``polygon`` (rings or None). New callers
    should normally use a recipe; this kernel supports compatibility tests.
    """
    mask = np.ones((nj, ni), dtype=bool)
    if region.polygon is not None:
        for ii in range(ni):
            for jj in range(nj):
                cell = (xs[i0 + ii], ys[j0 + jj], xs[i0 + ii] + dx, ys[j0 + jj] + dy)
                mask[jj, ii] = polygon_rect_fraction(region.polygon[0], cell) > 0.0
    return mask


def count_missing_inside(window, mask) -> int:
    """Count NaN native timesteps only at selected download cells."""
    return int(np.isnan(window[:, mask]).sum())


def selected_native_level(group, array, level: int | None) -> dict | None:
    """Select one native index, retaining coordinate and bounds without interpreting them."""
    if array.ndim != 4:
        if level is not None:
            raise GridError("level", "Correction: product has no native level axis")
        return None
    if (
        "x" not in group
        or "y" not in group
        or array.shape[-2:] != (group["y"].shape[0], group["x"].shape[0])
    ):
        raise GridError("level", "Correction: native level axes do not match the published grid")
    if level is None or not 0 <= level < array.shape[1]:
        raise GridError("level", "Correction: select exactly one valid native vertical level")
    metadata = {"index": level}
    if "level" in group:
        coordinate = group["level"]
        if coordinate.shape != (array.shape[1],):
            raise GridError(
                "level", "Correction: native level coordinate does not match the level axis"
            )
        metadata.update(value=coordinate[level].item(), attributes=dict(coordinate.attrs))
        bounds_name = coordinate.attrs.get("bounds")
        if isinstance(bounds_name, str) and bounds_name in group:
            bounds = group[bounds_name]
            if bounds.ndim != 2 or bounds.shape[0] != array.shape[1]:
                raise GridError(
                    "level", "Correction: native level bounds do not match the level axis"
                )
            metadata.update(bounds=bounds[level].tolist(), bounds_attributes=dict(bounds.attrs))
    return metadata


def _axes(group, array):
    xs, ys = [np.asarray(group[name][:], dtype="f8") for name in ("x", "y")]
    for axis in (xs, ys):
        if (
            axis.ndim != 1
            or len(axis) < 2
            or not np.isfinite(axis).all()
            or axis[1] <= axis[0]
            or not np.allclose(np.diff(axis), axis[1] - axis[0], rtol=1e-12, atol=1e-12)
        ):
            raise GridError("grid", "This definition requires a regular increasing native x/y grid")
    if array.shape[-2:] != (len(ys), len(xs)):
        raise GridError("grid", "Variable does not match native coordinates")
    return xs.tolist(), ys.tolist(), xs[1] - xs[0], ys[1] - ys[0]


def _window(xs, ys, dx, dy, bbox, *, footprints=False):
    x0, y0, x1, y1 = bbox
    if footprints:
        return (
            max(0, math.floor((x0 - xs[0]) / dx) - 1),
            min(len(xs), math.ceil((x1 - xs[0]) / dx) + 1),
            max(0, math.floor((y0 - ys[0]) / dy) - 1),
            min(len(ys), math.ceil((y1 - ys[0]) / dy) + 1),
        )
    i0, i1 = max(0, math.ceil((x0 - xs[0]) / dx)), min(len(xs), math.floor((x1 - xs[0]) / dx) + 1)
    j0, j1 = max(0, math.ceil((y0 - ys[0]) / dy)), min(len(ys), math.floor((y1 - ys[0]) / dy) + 1)
    if i0 >= i1 or j0 >= j1:
        raise GridError("selection", "Correction: region outside domain")
    return i0, i1, j0, j1


def _aligned(group, array, other, other_array, *, time):
    for axis in ("x", "y", "time") if time else ("x", "y"):
        if axis not in other or not np.array_equal(group[axis][:], other[axis][:]):
            raise GridError("source_alignment", f"Sources have different {axis} coordinates")
    if array.shape[-2:] != other_array.shape[-2:]:
        raise GridError("source_alignment", "Sources have different grid shapes")
    if time and group.attrs.get("time_units", "") != other.attrs.get("time_units", ""):
        raise GridError("source_alignment", "Sources have different time units")


def _summary(values, weights):
    cells = []
    for ii in range(weights.shape[1]):
        for jj in range(weights.shape[0]):
            vals = values[:, jj, ii].tolist()
            if weights[jj, ii] > 0 and all(math.isfinite(v) for v in vals):
                cells.append((float(weights[jj, ii]), vals))
    if not cells:
        raise GridError(
            "selection", "Correction: region selects no grid cells; move it onto the product domain"
        )
    total = sum(w for w, _ in cells)
    series = [sum(w * vals[m] for w, vals in cells) / total for m in range(len(cells[0][1]))]
    mean = math.fsum(series) / len(series)
    var = math.fsum((v - mean) ** 2 for v in series) / len(series)
    lo, hi = min(series), max(series)
    width = (hi - lo) / 10 if hi > lo else 1.0
    edges = [lo + i * width for i in range(11)]
    counts = [0.0] * 10
    for value in series:
        counts[min(int((value - edges[0]) / (edges[1] - edges[0])), 9)] += 1.0
    return {
        "valid_cells": len(cells),
        "time_series": series,
        "period": {"mean": mean, "min": lo, "max": hi, "std": math.sqrt(var)},
        "distribution": {"edges": edges, "counts": counts},
    }


def replay(recipe: dict, locations: dict) -> dict:
    """Execute a pinned recipe; arrays stay NumPy until a delivery adapter encodes them."""
    try:
        result = _replay(recipe, locations)
        # Refuse concurrent mutation too; locations must stay immutable during reads.
        for source in recipe["sources"]:
            if source_identity(locations[source["id"]]) != {
                k: source[k] for k in ("sha256", "size_bytes")
            }:
                raise GridError(
                    "source_mismatch", f"Source changed during execution: {source['id']}"
                )
        completeness = [
            s.get("execution_completeness") for s in recipe["scientific_status"]["sources"].values()
        ]
        result["execution_completeness"] = (
            "complete" if all(v == "complete" for v in completeness) else "partial"
        )
        return result
    except GridError:
        raise
    except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise GridError("invalid_input", str(exc)) from exc


def _replay(recipe: dict, locations: dict) -> dict:
    validate_recipe(recipe)
    if recipe.get("software") != software():
        raise GridError(
            "software_mismatch", "Replay requires the exact recorded software and implementation"
        )
    if set(locations) != {s["id"] for s in recipe["sources"]}:
        raise GridError("source_locations", "Supply exactly the recipe's source locations")
    groups, arrays = {}, {}
    for source in recipe["sources"]:
        path = locations[source["id"]]
        if source_identity(path) != {k: source[k] for k in ("sha256", "size_bytes")}:
            raise GridError("source_mismatch", f"Exact source bytes changed: {source['id']}")
        group = zarr.open_group(str(path), mode="r")
        if source["variable"] not in group:
            raise GridError("variable", f"Source lacks {source['variable']}")
        groups[source["id"]], arrays[source["id"]] = group, group[source["variable"]]
    op = recipe["operation"]
    group, array = groups[op["source"]], arrays[op["source"]]
    if array.ndim not in (2, 3, 4):
        raise GridError("grid", "Expected (y,x), (time,y,x) or (time,level,y,x)")
    selected = selected_native_level(group, array, op.get("level"))
    xs, ys, dx, dy = _axes(group, array)
    name, polygon = op["name"], op.get("polygon")
    i0, i1, j0, j1 = _window(xs, ys, dx, dy, op["bbox"], footprints=name == "regional_summary")
    values = np.asarray(
        array[:, op["level"], j0:j1, i0:i1] if selected is not None else array[..., j0:j1, i0:i1],
        dtype="f8",
    )
    weights = np.ones((max(0, j1 - j0), max(0, i1 - i0)))
    for ii in range(weights.shape[1]):
        for jj in range(weights.shape[0]):
            x, y = xs[i0 + ii], ys[j0 + jj]
            if polygon is not None:
                weights[jj, ii] = polygon_rect_fraction(polygon[0], (x, y, x + dx, y + dy))
            elif name == "regional_summary":
                ax0, ay0, ax1, ay1 = op["bbox"]
                weights[jj, ii] = (
                    max(0.0, min(ax1, x + dx) - max(ax0, x))
                    * max(0.0, min(ay1, y + dy) - max(ay0, y))
                    / (dx * dy)
                )
    result = {
        "x": xs[i0:i1],
        "y": ys[j0:j1],
        "dx": float(dx),
        "dy": float(dy),
        "level": selected,
        "attributes": dict(array.attrs),
        "time_units": str(group.attrs.get("time_units", "")),
        "scientific_status": recipe["scientific_status"],
        "recipe": recipe,
    }
    if name == "regional_summary":
        if values.ndim != 3:
            raise GridError("grid", "Regional summaries require a native time series")
        if op["mode"] == "land-only":
            mask_id = op["mask"]
            _aligned(group, array, groups[mask_id], arrays[mask_id], time=False)
            mask = np.asarray(arrays[mask_id][:])
            if mask.ndim != 2:
                raise GridError("mask", "Land mask must be a static (y,x) layer")
            weights *= mask[j0:j1, i0:i1] > 0.5
        result.update(_summary(values, weights))
        return result
    if name == "layer":
        timestep = op.get("timestep")
        if values.ndim == 2:
            if timestep is not None:
                raise GridError("timestep", "Correction: timestep out of range")
        else:
            t = 0 if timestep is None and selected is None else timestep
            if not isinstance(t, int) or not 0 <= t < values.shape[0]:
                raise GridError("timestep", "Correction: timestep out of range")
            values = values[t]
        result["values"] = np.where(weights > 0, values, np.nan)
        return result
    if values.ndim != 3 or "time" not in group or group["time"].shape != (len(values),):
        raise GridError("time", "Operation requires a matching native time axis")
    times = np.asarray(group["time"][:], dtype="f8")
    result["time"] = times.tolist()
    if name == "subset":
        result["values"] = np.where(weights > 0, values, np.nan)
        return result
    start, end = op.get("start"), op.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end < len(times):
        raise GridError("time_selection", "Choose inclusive native start/end indices")
    window = values[start : end + 1]
    missing = int(np.isnan(window[:, weights > 0]).sum())
    partner = None
    if op["method"] == "component_mean":
        pgroup, parray = groups[op["partner"]], arrays[op["partner"]]
        _aligned(group, array, pgroup, parray, time=True)
        variables = {s["id"]: s["variable"] for s in recipe["sources"]}
        if {variables[op["source"]], variables[op["partner"]]} != {
            "u10",
            "v10",
        } or parray.ndim != 3:
            raise GridError("wind_components", "Use published earth-relative u10/v10 components")
        partner = np.asarray(parray[start : end + 1, j0:j1, i0:i1], dtype="f8")
        missing += int(np.isnan(partner[:, weights > 0]).sum())
    if op["missing_data"] == "require-complete" and missing:
        raise GridError(
            "missing_data",
            f"Correction: require-complete refuses this aggregation: "
            f"{missing} missing values inside the download area",
        )
    if partner is not None:
        means = {
            variables[op["source"]]: window.mean(axis=0),
            variables[op["partner"]]: partner.mean(axis=0),
        }
        u, v = means["u10"], means["v10"]
        layers = {
            "u_mean": u,
            "v_mean": v,
            "wspd": np.hypot(u, v),
            "wdir": (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0,
        }
        result["layers"] = {k: np.where(weights > 0, val, np.nan) for k, val in layers.items()}
    else:
        if op["method"] == "mean":
            reduced = window.mean(axis=0)
        else:
            if not result["time_units"].startswith("hours since "):
                raise GridError(
                    "time_units", "depth_sum requires the published epoch-hours time axis"
                )
            durations = [times[i + 1] - times[i] for i in range(start, end)]
            if end + 1 < len(times):
                durations.append(times[end + 1] - times[end])
            elif durations:
                durations.append(durations[-1])
            else:
                raise GridError(
                    "step_duration",
                    "depth_sum needs step durations: single-step window at store end",
                )
            if any(not math.isfinite(v) or v <= 0 for v in durations):
                raise GridError(
                    "step_duration", "depth_sum requires positive finite step durations"
                )
            reduced = (window * np.asarray(durations)[:, None, None]).sum(axis=0)
        result["values"] = np.where(weights > 0, reduced, np.nan)
    result["time"] = [float(times[start])]
    return result


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    """Replay: python -m darwin.grid_operations recipe.json --locations paths.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--locations", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = replay(json.loads(args.recipe.read_text()), json.loads(args.locations.read_text()))
        print(json.dumps(_jsonable(result), sort_keys=True, allow_nan=False))
    except GridError as exc:
        emit_error(exc.code, str(exc))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        emit_error("invalid_input", str(exc))


if __name__ == "__main__":
    main()
