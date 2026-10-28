"""Native-grid golden cases, source/software refusals and public CLI equivalence."""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest
import zarr

from darwin import grid_operations as grid


@pytest.fixture
def stores(tmp_path):
    locations = {}
    for key, variable, values in (
        ("rate", "prcp", np.arange(24, dtype="f8").reshape(3, 2, 4)),
        ("u", "u10", np.full((3, 2, 4), 3.0)),
        ("v", "v10", np.full((3, 2, 4), 4.0)),
        ("mask", "land", np.array([[1, 0, 1, 1], [1, 1, 0, 1]], dtype="f8")),
        ("vertical", "temp", np.arange(48, dtype="f8").reshape(3, 2, 2, 4)),
    ):
        path = tmp_path / f"{key}.zarr"
        group = zarr.open_group(path, mode="w")
        group.create_array("x", data=np.array([0.0, 2.0, 4.0, 6.0]))
        group.create_array("y", data=np.array([0.0, 2.0]))
        group.create_array("time", data=np.array([0.0, 24.0, 72.0]))
        group.create_array(variable, data=values)
        group[variable].attrs["units"] = "mm h-1" if key == "rate" else "m s-1"
        group.attrs["time_units"] = "hours since 2020-01-01 00:00:00"
        if key == "vertical":
            group.create_array("level", data=np.array([10.0, 20.0]))
            group["level"].attrs.update(units="m", bounds="level_bnds")
            group.create_array("level_bnds", data=np.array([[5.0, 15.0], [15.0, 25.0]]))
        locations[key] = {
            "path": str(path),
            "variable": variable,
            "product_id": key,
            "version": "v1",
        }
    return locations


def recipe(stores, op, keys=("rate",)):
    sources = {key: stores[key] for key in keys}
    return grid.make_recipe(
        sources,
        {"source": keys[0], "bbox": [0.0, 0.0, 6.0, 2.0], **op},
        scientific_status={
            key: {"acceptance": "unassessed", "execution_completeness": "partial"} for key in keys
        },
    )


def run(stores, spec):
    return grid.replay(spec, {s["id"]: stores[s["id"]]["path"] for s in spec["sources"]})


def test_closed_window_and_native_level(stores):
    result = run(stores, recipe(stores, {"name": "subset", "bbox": [2.0, 0.0, 4.0, 2.0]}))
    np.testing.assert_array_equal(result["values"], np.arange(24).reshape(3, 2, 4)[:, :, 1:3])
    assert result["x"] == [2.0, 4.0]
    vertical = run(stores, recipe(stores, {"name": "subset", "level": 1}, ("vertical",)))
    np.testing.assert_array_equal(vertical["values"], np.arange(48).reshape(3, 2, 2, 4)[:, 1])
    assert vertical["level"] == {
        "index": 1,
        "value": 20.0,
        "attributes": {"units": "m", "bounds": "level_bnds"},
        "bounds": [15.0, 25.0],
        "bounds_attributes": {},
    }


def test_polygon_partial_cells_and_native_layer(stores):
    polygon = [[[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0], [1.0, 0.0]]]
    assert grid.polygon_rect_fraction(polygon[0], (0.0, 0.0, 2.0, 2.0)) == 0.5
    result = run(stores, recipe(stores, {"name": "layer", "timestep": 1, "polygon": polygon}))
    assert result["values"][0, :2].tolist() == [8.0, 9.0]
    assert np.isnan(result["values"][0, 2:]).all()
    assert np.isnan(result["values"][1]).all()


@pytest.mark.parametrize(
    "method,expected",
    [
        ("mean", [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]),
        ("depth_sum", [[1152.0, 1272.0, 1392.0, 1512.0], [1632.0, 1752.0, 1872.0, 1992.0]]),
    ],
)
def test_temporal_golden(stores, method, expected):
    result = run(
        stores,
        recipe(
            stores,
            {
                "name": "aggregate",
                "method": method,
                "start": 0,
                "end": 2,
                "missing_data": "propagate",
            },
        ),
    )
    np.testing.assert_array_equal(result["values"], expected)
    assert result["time"] == [0.0]


def test_wind_golden_and_no_positional_pairing(stores):
    spec = recipe(
        stores,
        {
            "name": "aggregate",
            "method": "component_mean",
            "start": 0,
            "end": 2,
            "missing_data": "require-complete",
            "partner": "v",
        },
        ("u", "v"),
    )
    result = run(stores, spec)
    np.testing.assert_array_equal(result["layers"]["wspd"], 5.0)
    np.testing.assert_allclose(result["layers"]["wdir"], 216.86989764584402)
    zarr.open_group(stores["v"]["path"], mode="a")["x"][:] = [1.0, 3.0, 5.0, 7.0]
    spec["sources"][1].update(grid.source_identity(stores["v"]["path"]))
    with pytest.raises(grid.GridError, match="different x") as exc:
        run(stores, spec)
    assert exc.value.code == "source_alignment"


def test_fractional_summary_complete_case_and_distribution(stores):
    group = zarr.open_group(stores["rate"]["path"], mode="a")
    group["prcp"][1, 0, 1] = np.nan
    spec = recipe(
        stores, {"name": "regional_summary", "bbox": [1.0, 0.0, 5.0, 2.0], "mode": "whole-domain"}
    )
    result = run(stores, spec)
    # Fractions .5,1,.5; center cell has one missing step and is excluded entirely.
    assert result["valid_cells"] == 2
    assert result["time_series"] == [1.0, 9.0, 17.0]
    assert result["period"] == {"mean": 9.0, "min": 1.0, "max": 17.0, "std": math_sqrt(128 / 3)}
    assert sum(result["distribution"]["counts"]) == 3
    assert result["distribution"]["counts"] == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert result["scientific_status"]["derived_acceptance"] == "unassessed"


def math_sqrt(value):
    return value**0.5


def test_land_mask_multisource(stores):
    spec = recipe(
        stores, {"name": "regional_summary", "mode": "land-only", "mask": "mask"}, ("rate", "mask")
    )
    result = run(stores, spec)
    assert result["valid_cells"] == 2  # bbox height covers first footprint row only
    assert result["time_series"] == [1.0, 9.0, 17.0]


def test_require_complete_ignores_polygon_exterior(stores):
    group = zarr.open_group(stores["rate"]["path"], mode="a")
    group["prcp"][0, 1, 3] = np.nan
    op = {
        "name": "aggregate",
        "method": "mean",
        "start": 0,
        "end": 2,
        "missing_data": "require-complete",
        "polygon": [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]],
    }
    run(stores, recipe(stores, op))
    group["prcp"][0, 0, 0] = np.nan
    with pytest.raises(grid.GridError, match="1 missing values"):
        run(stores, recipe(stores, op))


@pytest.mark.parametrize(
    "change,code",
    [
        ("bytes", "source_mismatch"),
        ("metadata", "source_mismatch"),
        ("software", "software_mismatch"),
    ],
)
def test_exact_replay_refusals(stores, change, code):
    spec = recipe(stores, {"name": "subset"})
    if change == "bytes":
        zarr.open_group(stores["rate"]["path"], mode="a")["prcp"][0, 0, 0] = 999.0
    elif change == "metadata":
        zarr.open_group(stores["rate"]["path"], mode="a").attrs["note"] = "changed"
    else:
        spec["software"]["darwin"] = "another-version"
    with pytest.raises(grid.GridError) as exc:
        run(stores, spec)
    assert exc.value.code == code


@pytest.mark.parametrize(
    "op",
    [
        {"name": "interpolate"},
        {"name": "subset", "regrid": True},
        {"name": "aggregate", "method": "mean", "start": 0, "end": 2},
        {"name": "regional_summary", "mode": "land-only"},
        {"name": "subset", "polygon": [[], []]},
    ],
)
def test_unsupported_explicitly_refused(stores, op):
    with pytest.raises(grid.GridError):
        recipe(stores, op)


def test_cli_matches_public_python_and_structured_error(stores, tmp_path):
    spec = recipe(stores, {"name": "regional_summary", "mode": "whole-domain"})
    path, locations = tmp_path / "recipe.json", tmp_path / "locations.json"
    path.write_text(json.dumps(spec))
    locations.write_text(json.dumps({"rate": stores["rate"]["path"]}))
    command = [
        sys.executable,
        "-m",
        "darwin.grid_operations",
        str(path),
        "--locations",
        str(locations),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == run(stores, spec)
    changed = copy.deepcopy(spec)
    changed["software"]["numpy"] = "different"
    path.write_text(json.dumps(changed))
    failed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert failed.returncode == 2
    assert json.loads(failed.stderr)["code"] == "software_mismatch"
