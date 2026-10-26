"""Darwin consumes the CJC-owned fixture without importing CJC."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from darwin.simulation_output import (
    SimulationOutputError,
    inspect_manifest,
    load_manifest,
    validate_manifest,
)


@pytest.fixture
def document():
    view = json.loads((Path(__file__).parent / "data/cjc-output-v1.fixture.json").read_text())
    assert view["artifacts"][0]["sha256"] == hashlib.sha256(b"x").hexdigest()
    return view


@pytest.mark.parametrize("coverage", ["complete", "partial", "absent"])
def test_coverage(document, coverage):
    if coverage != "complete":
        document.update(completeness="partial", partial_reason="incomplete attempt")
        document["missing_times"] = [document["selections"].pop()["time"]]
        document["segments"].pop()
        if coverage == "absent":
            document["missing_times"].insert(0, document["selections"].pop()["time"])
            document["artifacts"] = []
            document["segments"] = []
    report = inspect_manifest(document)
    assert report["scientific_status"] == "not_assessed"
    assert report["payload_verification"] == "not_checked"
    assert report["native_record_metadata"] == "not_checked"
    assert report["missing_times"] == document["missing_times"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("contract", "cjc.simulation-output/2"),
        ("history_seconds", True),
        ("history_seconds", 43200.0),
        ("history_seconds", 7),
        ("history_seconds", 0),
        ("start", "2001-02-30T00:00:00Z"),
        ("start", "2001-01-01T00:00:00+00:00"),
        ("end", "2001-01-03T00:00:00Z"),
        ("missing_times", ["2001-01-02T00:00:00Z"]),
        ("partial_reason", "not complete"),
        ("completeness", "partial"),
        ("artifacts", []),
        ("segments", []),
        ("variables", []),
        ("scientific_acceptance", "accepted"),
    ],
)
def test_invalid_declarations(document, key, value):
    document[key] = value
    with pytest.raises(SimulationOutputError):
        validate_manifest(document)


@pytest.mark.parametrize("failure", ["time", "record", "descriptor", "variable", "digest"])
def test_semantics(document, failure):
    if failure == "time":
        document["selections"].reverse()
    elif failure == "record":
        document["selections"][1]["record_index"] = 0
    elif failure == "descriptor":
        document["segments"] *= 2
    elif failure == "variable":
        document["variables"] *= 2
    else:
        document["grid"]["sha256"] += "\n"
    with pytest.raises(SimulationOutputError):
        validate_manifest(document)


def test_byte_checks_and_cli_equivalence(document, tmp_path):
    native = tmp_path / "native"
    native.write_bytes(b"x")
    locations = {"synthetic-native": str(native)}
    expected = inspect_manifest(document, locations=locations)
    assert expected["payload_verification"] == "byte_identity_checked"
    source = tmp_path / "manifest.json"
    mapping = tmp_path / "locations.json"
    source.write_text(json.dumps(document))
    mapping.write_text(json.dumps(locations))
    command = [
        sys.executable,
        "-m",
        "darwin.simulation_output",
        str(source),
        "--locations",
        str(mapping),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected
    native.write_bytes(b"y")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stderr)["code"] == "corrupt_artifact"
    native.unlink()
    with pytest.raises(SimulationOutputError) as error:
        inspect_manifest(document, locations=locations)
    assert error.value.code == "missing_artifact"


def test_validation_copy_and_duplicate_json(document, tmp_path):
    checked = validate_manifest(document)
    checked["simulation"]["mode"] = "daily_reinit"
    assert document["simulation"]["mode"] == "continuous"
    source = tmp_path / "manifest.json"
    source.write_text('{"contract":"ignored","contract":"cjc.simulation-output/1"}')
    result = subprocess.run(
        [sys.executable, "-m", "darwin.simulation_output", str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["code"] == "invalid_manifest"


def test_invalid_input_and_locations(document, tmp_path):
    with pytest.raises(SimulationOutputError) as error:
        load_manifest(tmp_path / "absent.json")
    assert error.value.code == "invalid_input"
    with pytest.raises(SimulationOutputError) as error:
        inspect_manifest(document, locations={"synthetic-native": 5})
    assert error.value.code == "invalid_locations"
