"""Darwin consumes CJC-owned ``cjc.simulation-output/2`` fixtures without importing CJC.

The fixtures under ``tests/data/cjc-simulation-output-v2`` are byte-for-byte copies
of the producer's ``cjc/workflow/fixtures/simulation-output-v2`` directory.
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

from darwin.simulation_output import (
    SimulationOutputError,
    inspect_manifest,
    load_manifest,
    read_wrf_history,
    validate_manifest,
    verify_completion_references,
)

FIXTURES = Path(__file__).parent / "data" / "cjc-simulation-output-v2"
# Independent source of truth: the digests published in the producer's fixture set.
PRODUCER_SHA256 = {
    "run-a/wrfout_d02_2001-01-01_000000.nc": (
        "7884d204d8e8aa5ec69c5c3ac0c3b58369ebbdedfdcf10dc4fc5dd458307d3df"
    ),
    "run-a/wrfout_d02_2001-01-02_000000.nc": (
        "889aeca537c9df0a8649a441c68b415fb1523afce3873ae5bffe2949b98fd847"
    ),
    "run-b/wrfout_d02_2001-01-02_180000.nc": (
        "d102bbc401f3cb0ca0157d9a055cd36117d03798841a8ab378ec8f1566a1a9c2"
    ),
    "partial/wrfout_d02_2001-01-01_000000.nc": (
        "14a66498bba7c59fa68f699f8f249182b19a299ebb7627fcba2ebf63a169b220"
    ),
}


def view(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def fixture_locations(kind="artifacts", root=FIXTURES):
    mapping = json.loads((FIXTURES / "locations.json").read_text())[kind]
    return {key: str(root / value) for key, value in mapping.items()}


def test_copied_producer_payloads_are_unchanged():
    for relative, digest in PRODUCER_SHA256.items():
        assert hashlib.sha256((FIXTURES / relative).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    ("name", "status", "covered"),
    [
        ("complete", "complete", 4),
        ("recovery-split", "complete", 4),
        ("partial", "partial", 2),
        ("failed", "failed", 0),
        ("missing", "missing", 0),
    ],
)
def test_v2_views_report_declared_status_and_actual_coverage(name, status, covered):
    document = load_manifest(FIXTURES / f"{name}.json")
    report = inspect_manifest(document)
    assert report["contract"] == "cjc.simulation-output/2"
    assert report["completeness"] == status
    assert len(report["selected_times"]) == covered
    assert report["missing_times"] == document["missing_times"]
    assert report["payload_verification"] == "not_checked"
    assert report["native_record_metadata"] == "not_checked"
    assert report["scientific_status"] == "not_assessed"


def test_partial_view_keeps_its_gaps_and_reason():
    report = inspect_manifest(view("partial"))
    assert report["selected_times"] == ["2001-01-01T06:00:00Z", "2001-01-01T12:00:00Z"]
    assert report["missing_times"] == ["2001-01-01T18:00:00Z", "2001-01-02T00:00:00Z"]
    assert report["reason"] == "segment p2 failed after restarting from the 12:00 checkpoint"


@pytest.mark.parametrize("name", ["complete", "recovery-split", "partial"])
def test_native_records_grid_calendar_and_variables_are_verified(name):
    report = inspect_manifest(view(name), locations=fixture_locations())
    assert report["payload_verification"] == "byte_identity_checked"
    assert report["native_record_metadata"] == "checked"
    assert report["scientific_status"] == "not_assessed"


def test_native_wrf_times_are_decoded_from_the_char_array():
    history = read_wrf_history(FIXTURES / "run-b/wrfout_d02_2001-01-02_180000.nc")
    assert history["times"] == ["2001-01-02T18:00:00Z", "2001-01-03T00:00:00Z"]
    assert history["grid"]["projection"] == "mercator"
    assert history["grid"]["grid_id"] == 2
    assert history["variables"]["U"]["stagger"] == "X"


def mutate(document, change):
    document = copy.deepcopy(document)
    change(document)
    return document


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(contract="cjc.simulation-output/3"),
        lambda d: d.update(status="partial"),
        lambda d: d.update(reason="spurious"),
        lambda d: d.update(calendar="julian"),
        lambda d: d.update(end="2001-01-04T00:00:00Z"),
        lambda d: d["missing_times"].append("2001-01-01T18:00:00Z"),
        lambda d: d["grid"].update(dx=1000.0),
        lambda d: d["grid"].update(grid_id=1),
        lambda d: d["grid"].update(projection="lambert_conformal"),
        lambda d: d["variables"][0].update(dimensions=["south_north", "west_east"]),
        lambda d: d["selections"][0].update(record_index=0),
        lambda d: d["selections"][0].update(segment_id="s2"),
        lambda d: d["selections"].reverse(),
        lambda d: d["artifacts"][0].update(run_root_id="run-b"),
        lambda d: d["run_roots"][1].update(recovers=None),
        lambda d: d["run_roots"][1].update(recovers="run-z"),
        lambda d: d["segments"][1].update(completion=d["segments"][0]["completion"]),
        lambda d: d.update(scientific_acceptance="accepted"),
    ],
)
def test_invalid_v2_declarations_are_refused(change):
    with pytest.raises(SimulationOutputError) as error:
        validate_manifest(mutate(view("recovery-split"), change))
    assert error.value.code in {"invalid_manifest", "unsupported_contract"}


def test_unknown_contract_is_unsupported():
    with pytest.raises(SimulationOutputError) as error:
        validate_manifest(mutate(view("complete"), lambda d: d.update(contract="x/9")))
    assert error.value.code == "unsupported_contract"


def test_failed_and_missing_views_cannot_carry_data():
    data = view("complete")
    with pytest.raises(SimulationOutputError):
        validate_manifest(mutate(view("failed"), lambda d: d.update(grid=data["grid"])))
    with pytest.raises(SimulationOutputError):
        validate_manifest(mutate(view("missing"), lambda d: d.update(segments=data["segments"])))


def copy_payloads(tmp_path):
    root = tmp_path / "fixtures"
    shutil.copytree(FIXTURES, root)
    return root


def test_changed_and_missing_artifacts_are_structured_errors(tmp_path):
    root = copy_payloads(tmp_path)
    locations = fixture_locations(root=root)
    target = Path(locations["run-a-d02-0101"])
    target.write_bytes(target.read_bytes() + b"\0")
    with pytest.raises(SimulationOutputError) as error:
        inspect_manifest(view("complete"), locations=locations)
    assert error.value.code == "corrupt_artifact"
    target.unlink()
    with pytest.raises(SimulationOutputError) as error:
        inspect_manifest(view("complete"), locations=locations)
    assert error.value.code == "missing_artifact"
    with pytest.raises(SimulationOutputError) as error:
        inspect_manifest(view("complete"), locations={})
    assert error.value.code == "missing_artifact"


def rewrite_native(path, change):
    with netCDF4.Dataset(path, "a") as dataset:
        change(dataset)


def redeclare(document, artifact_id, path):
    payload = Path(path).read_bytes()
    for artifact in document["artifacts"]:
        if artifact["id"] == artifact_id:
            artifact.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
    return document


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (
            lambda ds: ds["Times"].__setitem__((1, slice(11, 13)), np.array(list("07"), "S1")),
            "native_records",
        ),
        (
            lambda ds: ds["Times"].__setitem__((1, slice(0, 4)), np.array(list("20x1"), "S1")),
            "native_structure",
        ),
        (lambda ds: ds.setncattr("DX", np.float32(1000.0)), "native_grid"),
        (lambda ds: ds.setncattr("MAP_PROJ", np.int32(1)), "native_grid"),
        (lambda ds: ds["T2"].setncattr("units", "degC"), "native_variable"),
        (lambda ds: ds["XTIME"].setncattr("calendar", "noleap"), "calendar_mismatch"),
        (lambda ds: ds["XTIME"].__setitem__(1, 300.0), "calendar_mismatch"),
    ],
)
def test_native_metadata_disagreement_is_refused(tmp_path, change, code):
    root = copy_payloads(tmp_path)
    locations = fixture_locations(root=root)
    rewrite_native(locations["run-a-d02-0101"], change)
    document = redeclare(view("complete"), "run-a-d02-0101", locations["run-a-d02-0101"])
    with pytest.raises(SimulationOutputError) as error:
        inspect_manifest(document, locations=locations)
    assert error.value.code == code


def test_split_recovery_roots_keep_their_own_completion_records(tmp_path):
    document = view("recovery-split")
    completions = fixture_locations("completions")
    assert verify_completion_references(document, completions) == {
        "s1": "run-a",
        "s3": "run-b",
    }
    swapped = {"s1": completions["s3"], "s3": completions["s1"]}
    with pytest.raises(SimulationOutputError) as error:
        verify_completion_references(document, swapped)
    assert error.value.code == "changed_completion"
    relabelled = mutate(document, lambda d: d["run_roots"][1].update(bundle_fingerprint="other"))
    with pytest.raises(SimulationOutputError) as error:
        verify_completion_references(relabelled, completions)
    assert error.value.code == "fingerprint_mismatch"
    with pytest.raises(SimulationOutputError) as error:
        verify_completion_references(document, {"s1": completions["s1"]})
    assert error.value.code == "missing_completion"


def test_cli_verifies_native_records_and_completions(tmp_path):
    (tmp_path / "artifacts.json").write_text(json.dumps(fixture_locations()))
    (tmp_path / "completions.json").write_text(json.dumps(fixture_locations("completions")))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "darwin.simulation_output",
            str(FIXTURES / "recovery-split.json"),
            "--locations",
            str(tmp_path / "artifacts.json"),
            "--completions",
            str(tmp_path / "completions.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report == inspect_manifest(
        view("recovery-split"),
        locations=fixture_locations(),
        completions=fixture_locations("completions"),
    )
    assert report["referenced_receipts"] == "completion_bytes_checked"


def test_cli_exports_both_bundled_schemas():
    def schema(*extra):
        command = [sys.executable, "-m", "darwin.simulation_output", "--schema", *extra]
        return json.loads(subprocess.run(command, capture_output=True, check=True).stdout)

    assert schema()["$id"] == "cjc.simulation-output/1"
    assert schema("--contract", "cjc.simulation-output/2")["$id"] == "cjc.simulation-output/2"
