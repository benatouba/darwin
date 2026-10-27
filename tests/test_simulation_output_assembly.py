"""Strict assembly of CJC v2 views through the supported ``darwin.operations`` interface.

Inputs are the byte-for-byte copied producer fixtures; expected values are read
independently from the native files with netCDF4, never from Darwin's reader.
"""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr

from darwin.operations import OperationError, execute, inventory, make_recipe
from darwin.simulation_output import (
    SimulationOutputError,
    artifact_ids,
    assemble_operation,
    check_views,
)
from darwin.wrf_history import grid_digest

FIXTURES = Path(__file__).parent / "data" / "cjc-simulation-output-v2"
LOCATIONS = {
    key: str(FIXTURES / value)
    for key, value in json.loads((FIXTURES / "locations.json").read_text())["artifacts"].items()
}


def view(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def recipe_for(views, variables=("T2", "U"), extra=(), locations=LOCATIONS):
    sources = {key: locations[key] for key in artifact_ids(views)}
    return make_recipe(sources, [assemble_operation(views, list(variables)), *extra]), sources


def native_record(artifact_id, variable, index):
    with netCDF4.Dataset(LOCATIONS[artifact_id]) as dataset:
        dataset.set_auto_maskandscale(False)
        return np.asarray(dataset[variable][index])


def digests():
    return {key: hashlib.sha256(Path(p).read_bytes()).hexdigest() for key, p in LOCATIONS.items()}


def test_split_recovery_days_assemble_from_exact_native_records(tmp_path):
    before = digests()
    views = [view("complete"), view("recovery-split")]
    recipe, sources = recipe_for(views)
    report = execute(recipe, sources, tmp_path / "product")
    assert digests() == before
    assert report["scientific_status"] == "unassessed"
    assert report["source_coverage"]["status"] == "complete"
    assert report["source_coverage"]["missing_times"] == []
    with xr.open_dataset(tmp_path / "product" / "data.nc") as product:
        expected_times = [
            "2001-01-01T06",
            "2001-01-01T12",
            "2001-01-01T18",
            "2001-01-02T00",
            "2001-01-02T06",
            "2001-01-02T12",
            "2001-01-02T18",
            "2001-01-03T00",
        ]
        assert list(product["Time"].values) == [np.datetime64(t, "ns") for t in expected_times]
        assert list(product["Times"].values) == [
            t.replace("T", "_") + ":00:00" for t in expected_times
        ]
        # Recovery records come from run-b; nothing was copied into run-a.
        assert list(product["source_run_root"].values) == ["run-a"] * 6 + ["run-b"] * 2
        assert list(product["source_artifact"].values[-2:]) == ["run-b-d02-0102"] * 2
        assert list(product["source_record_index"].values[-3:]) == [2, 0, 1]
        assert product["U"].dims == ("Time", "bottom_top", "south_north", "west_east_stag")
        assert product["U"].attrs["units"] == "m s-1"
        assert product["U"].attrs["stagger"] == "X"
        assert product["T2"].dtype == np.float32
        np.testing.assert_array_equal(
            product["T2"].values[0], native_record("run-a-d02-0101", "T2", 1)
        )
        np.testing.assert_array_equal(
            product["U"].values[6], native_record("run-b-d02-0102", "U", 0)
        )
        assert product.attrs["calendar"] == "standard"
        assert json.loads(product.attrs["grid"])["projection"] == "mercator"
        assert product.attrs["scientific_status"] == "unassessed"
    stored = json.loads((tmp_path / "product" / "recipe.json").read_text())
    assert stored["operations"][0]["parameters"]["views"] == views


def test_partial_status_and_gaps_are_preserved_not_filled(tmp_path):
    views = [view("partial"), view("failed"), view("missing")]
    recipe, sources = recipe_for(views, ["T2"])
    report = execute(recipe, sources, tmp_path / "product")
    coverage = report["source_coverage"]
    assert coverage["status"] == "partial"
    assert coverage["covered_times"] == ["2001-01-01T06:00:00Z", "2001-01-01T12:00:00Z"]
    assert coverage["missing_times"][:2] == ["2001-01-01T18:00:00Z", "2001-01-02T00:00:00Z"]
    assert len(coverage["missing_times"]) == 10
    assert [v["status"] for v in coverage["views"]] == ["partial", "failed", "missing"]
    assert coverage["views"][0]["reason"].startswith("segment p2 failed")
    with xr.open_dataset(tmp_path / "product" / "data.nc") as product:
        assert product.sizes["Time"] == 2
        assert product.attrs["source_coverage_status"] == "partial"
        assert json.loads(product.attrs["source_missing_times"]) == coverage["missing_times"]


def test_cli_and_python_assembly_are_equivalent_and_composable(tmp_path):
    cut = {
        "operation": "cut",
        "parameters": {"dimension": "Time", "start": "2001-01-01T12:00", "end": "2001-01-02T00:00"},
    }
    recipe, sources = recipe_for([view("complete")], ["T2"], [cut])
    expected = execute(recipe, sources, tmp_path / "python")
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    (tmp_path / "locations.json").write_text(json.dumps(sources))
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
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == expected
    assert (tmp_path / "python/data.nc").read_bytes() == (tmp_path / "cli/data.nc").read_bytes()
    with xr.open_dataset(tmp_path / "python/data.nc") as product:
        assert product.sizes["Time"] == 2


def test_inventory_names_the_simulation_output_operation():
    names = [entry["operation"] for entry in inventory()["supported"]]
    assert "assemble_simulation_output" in names


def regrid(document, **fields):
    grid = {**document["grid"], **fields}
    grid["sha256"] = grid_digest(grid)
    document["grid"] = grid
    return document


def gap_in(document):
    removed = document["selections"].pop(1)
    document.update(status="partial", reason="12:00 record lost", missing_times=[removed["time"]])
    return document


def overlap(document):
    document["id"] = "another-view-of-the-same-day"
    return document


@pytest.mark.parametrize(
    ("second", "code"),
    [
        (lambda d: d.update(calendar="noleap") or d, "incompatible_calendar"),
        (lambda d: regrid(d, dx=1000.0, dy=1000.0), "incompatible_grid"),
        (lambda d: d["variables"][2].update(units="km h-1") or d, "incompatible_units"),
        (lambda d: d["variables"][2].update(stagger="") or d, "incompatible_variable"),
        (lambda d: d["simulation"].update(id="another-simulation") or d, "incompatible_assembly"),
        (gap_in, "time_gap"),
    ],
)
def test_incompatible_inputs_are_refused_before_assembly(tmp_path, second, code):
    views = [view("complete"), second(view("recovery-split"))]
    with pytest.raises(OperationError) as error:
        recipe_for(views)
    assert error.value.code == code
    # The same refusal guards replay of a recipe bound before any check.
    recipe, sources = recipe_for([view("complete"), view("recovery-split")])
    recipe["operations"][0]["parameters"]["views"] = views
    with pytest.raises(OperationError) as error:
        execute(recipe, sources, tmp_path / "product")
    assert error.value.code == code
    assert not (tmp_path / "product").exists()


def test_overlapping_and_unordered_days_need_explicit_resolution(tmp_path):
    for views, code in [
        ([view("complete"), overlap(view("complete"))], "time_overlap"),
        ([view("recovery-split"), view("complete")], "time_order"),
        ([view("partial"), view("missing")], "time_gap"),
    ]:
        with pytest.raises(OperationError) as error:
            recipe_for(views)
        assert error.value.code == code


def test_no_implicit_conversion_interpolation_or_gap_filling():
    for name in ("convert_units", "interpolate", "fill_gaps", "average", "reconcile"):
        with pytest.raises(OperationError) as error:
            recipe_for([view("complete")], extra=[{"operation": name, "parameters": {}}])
        assert error.value.code == "unsupported_operation"


def test_unsupported_versions_and_unknown_variables_are_structured_errors(tmp_path):
    legacy = json.loads((Path(__file__).parent / "data/cjc-output-v1.fixture.json").read_text())
    native = tmp_path / "native"
    native.write_bytes(b"x")
    with pytest.raises(OperationError) as error:
        make_recipe({"synthetic-native": native}, [assemble_operation([legacy], ["T2"])])
    assert error.value.code == "unsupported_contract"
    with pytest.raises(OperationError) as error:
        recipe_for([view("complete")], ["QVAPOR"])
    assert error.value.code == "variable"
    with pytest.raises(SimulationOutputError) as error:
        check_views([view("failed")], ["T2"])
    assert error.value.code == "empty_selection"


def test_recipe_sources_must_be_exactly_the_view_artifacts(tmp_path):
    operation = assemble_operation([view("complete")], ["T2"])
    with pytest.raises(OperationError) as error:
        make_recipe({"run-a-d02-0101": LOCATIONS["run-a-d02-0101"]}, [operation])
    assert error.value.code == "invalid_recipe"
    swapped = {
        "run-a-d02-0101": LOCATIONS["run-a-d02-0102"],
        "run-a-d02-0102": LOCATIONS["run-a-d02-0101"],
    }
    with pytest.raises(OperationError) as error:
        make_recipe(swapped, [operation])
    assert error.value.code == "invalid_recipe"
    select = {"operation": "select", "parameters": {"variables": ["T2"]}}
    with pytest.raises(OperationError) as error:
        make_recipe(
            {k: LOCATIONS[k] for k in artifact_ids([view("complete")])}, [select, operation]
        )
    assert error.value.code == "invalid_recipe"


def test_changed_or_missing_native_bytes_are_refused(tmp_path):
    recipe, sources = recipe_for([view("complete")])
    copied = {}
    for key, path in sources.items():
        copied[key] = str(tmp_path / Path(path).name)
        shutil.copyfile(path, copied[key])
    Path(copied["run-a-d02-0101"]).write_bytes(b"replaced")
    with pytest.raises(OperationError) as error:
        execute(recipe, copied, tmp_path / "product")
    assert error.value.code == "source_mismatch"
    with pytest.raises(OperationError) as error:
        execute(recipe, {"run-a-d02-0101": copied["run-a-d02-0101"]}, tmp_path / "product")
    assert error.value.code == "missing_source"
    assert not (tmp_path / "product").exists()


def test_recipe_view_cannot_be_edited_after_binding(tmp_path):
    recipe, sources = recipe_for([view("complete")])
    edited = copy.deepcopy(recipe)
    edited["operations"][0]["parameters"]["views"][0]["selections"].pop()
    with pytest.raises(OperationError) as error:
        execute(edited, sources, tmp_path / "product")
    assert error.value.code == "invalid_manifest"
