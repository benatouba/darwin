"""Public operation contracts reject implicit repair and preserve exact inputs."""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from darwin.operations import OperationError, execute, make_recipe, source_identity


def write_source(path, start=0, *, units="mm", x=(1.0, 2.0), offset=0):
    dataset = xr.Dataset(
        {"prcp": (("time", "x"), np.array([[1.0, 2.0], [3.0, 4.0]]) + offset)},
        coords={
            "time": np.array([start, start + 1], dtype="timedelta64[h]")
            + np.datetime64("2020-01-01T00:00:00"),
            "x": list(x),
        },
        attrs={"projection": "test-grid"},
    )
    dataset.prcp.attrs["units"] = units
    dataset.time.encoding.update(units="hours since 2020-01-01", calendar="proleptic_gregorian")
    dataset.to_netcdf(path)
    return path


ASSEMBLE = {"operation": "assemble", "parameters": {"dimension": "time", "cadence_seconds": 3600}}
SELECT = {"operation": "select", "parameters": {"variables": ["prcp"]}}


def test_python_cli_equivalence_and_location_independence(tmp_path):
    source = write_source(tmp_path / "native.nc")
    recipe = make_recipe(
        {"native-version-1": source},
        [
            SELECT,
            {
                "operation": "cut",
                "parameters": {
                    "dimension": "time",
                    "start": "2020-01-01T00:00",
                    "end": "2020-01-01T01:00",
                },
            },
            {"operation": "crop", "parameters": {"dimension": "x", "start": 1, "end": 2}},
            {
                "operation": "integrate",
                "parameters": {
                    "dimension": "time",
                    "variables": ["prcp"],
                    "method": "sum-discrete-amounts",
                },
            },
        ],
    )
    before = source_identity(source)
    locations = {"native-version-1": str(source)}
    report = execute(recipe, locations, tmp_path / "python")
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    (tmp_path / "locations.json").write_text(json.dumps(locations))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "darwin.operations",
            str(tmp_path / "recipe.json"),
            "--locations",
            str(tmp_path / "locations.json"),
            "--output",
            str(tmp_path / "cli"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["scientific_status"] == "unassessed"
    assert report["execution_completeness"] == "complete"
    assert source_identity(source) == before
    assert str(tmp_path) not in json.dumps(recipe)
    with (
        xr.open_dataset(tmp_path / "python/data.nc") as left,
        xr.open_dataset(tmp_path / "cli/data.nc") as right,
    ):
        xr.testing.assert_identical(left, right)
        assert left.prcp.values.tolist() == [1.0]
        assert left.prcp.attrs["cell_methods"] == "time: sum"
    relocated = tmp_path / "relocated.nc"
    relocated.write_bytes(source.read_bytes())
    execute(recipe, {"native-version-1": relocated}, tmp_path / "relocated-product")


def test_strict_assembly(tmp_path):
    a = write_source(tmp_path / "a.nc")
    b = write_source(tmp_path / "b.nc", 2, offset=10)
    recipe = make_recipe({"a": a, "b": b}, [ASSEMBLE])
    execute(recipe, {"a": a, "b": b}, tmp_path / "out")
    with xr.open_dataset(tmp_path / "out/data.nc") as result:
        assert result.sizes["time"] == 4
        assert result.prcp.values[:, 0].tolist() == [1, 3, 11, 13]


@pytest.mark.parametrize(
    "change,code",
    [
        ({"start": 3}, "time_gap"),
        ({"start": 1}, "coordinate"),
        ({"start": 2, "units": "K"}, "incompatible_assembly"),
        ({"start": 2, "x": (1.0, 3.0)}, "incompatible_assembly"),
    ],
)
def test_assembly_refuses_gaps_overlaps_units_and_grids(tmp_path, change, code):
    a = write_source(tmp_path / "a.nc")
    b = write_source(tmp_path / "b.nc", **change)
    recipe = make_recipe({"a": a, "b": b}, [ASSEMBLE])
    with pytest.raises(OperationError) as caught:
        execute(recipe, {"a": a, "b": b}, tmp_path / "out")
    assert caught.value.code == code
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda r: r.update(contract="darwin.operations/999"), "unsupported_contract"),
        (lambda r: r["operations"][0]["parameters"].update(silent_repair=True), "invalid_recipe"),
        (lambda r: r["sources"][0].update(sha256="0" * 64), "source_mismatch"),
        (lambda r: r["software"].update(darwin="unavailable"), "software_mismatch"),
    ],
)
def test_recipe_refusals(tmp_path, mutation, code):
    source = write_source(tmp_path / "source.nc")
    recipe = make_recipe({"native": source}, [copy.deepcopy(SELECT)])
    mutation(recipe)
    with pytest.raises(OperationError) as caught:
        execute(recipe, {"native": source}, tmp_path / "out")
    assert caught.value.code == code


def test_no_overwrite_and_missing_mapping(tmp_path):
    source = write_source(tmp_path / "native.nc")
    recipe = make_recipe({"native": source}, [SELECT])
    destination = tmp_path / "out"
    execute(recipe, {"native": source}, destination)
    before = (destination / "data.nc").read_bytes()
    with pytest.raises(OperationError, match="exists"):
        execute(recipe, {"native": source}, destination)
    assert (destination / "data.nc").read_bytes() == before
    with pytest.raises(OperationError) as caught:
        execute(recipe, {}, tmp_path / "missing")
    assert caught.value.code == "missing_source"


def test_nan_amounts_are_not_silently_dropped(tmp_path):
    path = tmp_path / "nan.nc"
    xr.Dataset({"prcp": ("time", [1.0, float("nan")])}, coords={"time": [0, 1]}).to_netcdf(path)
    recipe = make_recipe(
        {"native": path},
        [
            {
                "operation": "integrate",
                "parameters": {
                    "dimension": "time",
                    "variables": ["prcp"],
                    "method": "sum-discrete-amounts",
                },
            }
        ],
    )
    execute(recipe, {"native": path}, tmp_path / "out")
    with xr.open_dataset(tmp_path / "out/data.nc") as result:
        assert np.isnan(result.prcp.item())
