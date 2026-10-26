"""Public store comparison and CLI use the same validated implementation."""

import json
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np
import pytest
import zarr

from darwin.comparison import compare_stores, replay
from darwin.publisher.series import TimeValidationError


@pytest.fixture
def stores(tmp_path):
    paths = [tmp_path / "model.zarr", tmp_path / "obs.zarr"]
    axis = np.asarray(
        [datetime(2025, 1, 1, tzinfo=UTC).timestamp() + i * 3600 for i in range(3)], dtype="int64"
    )
    for path, values in zip(paths, ([10.0, 20.0, 30.0], [11.0, 20.0, 29.0]), strict=True):
        group = zarr.open_group(str(path), mode="w")
        group.create_array("hourly", data=np.asarray(values), attributes={"units": "K"})
        group.create_array("hourly_valid", data=np.asarray([True, False, True]))
        for name, data in (
            ("hourly_time", axis),
            ("hourly_time_bounds", np.stack([axis, axis + 3600], axis=1)),
        ):
            group.create_array(
                name, data=data, attributes={"units": "seconds since 1970-01-01 00:00:00"}
            )
        group.attrs.update(
            {
                "series_schema_version": 2,
                "cadences": ["hourly"],
                "calendar": "proleptic_gregorian",
                "timezone": "UTC",
                "aggregation_timezone": "UTC",
                "source_coverage": {"start": "2025-01-01T00:00:00Z"},
                "synthetic": False,
                "scientific_status": "unassessed",
            }
        )
    return paths


def run(paths, **kwargs):
    return compare_stores(
        *paths,
        model_variable="t2",
        observation_field="T",
        cadence="hourly",
        policy="native-intervals-v1",
        **kwargs,
    )


def test_public_comparison_replays_and_cli_matches(stores, tmp_path):
    result = run(stores)
    assert result["statistics"]["paired_count"] == 2
    assert result["statistics"]["excluded_count"] == 1
    assert result["statistics"]["bias"] == 0
    assert result["recipe"]["sources"]["model"]["scientific_status"] == "unassessed"
    assert replay(result["recipe"], *stores) == result
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(result["recipe"]))
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "darwin.comparison",
            "--model",
            str(stores[0]),
            "--observations",
            str(stores[1]),
            "--recipe",
            str(recipe),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(cli.stdout) == result


def test_replay_rejects_changed_metadata(stores):
    recipe = run(stores)["recipe"]
    zarr.open_group(str(stores[0]), mode="a").attrs["scientific_status"] = "accepted"
    with pytest.raises(TimeValidationError, match="identity"):
        replay(recipe, *stores)


def test_unsupported_policy_and_unit_conversion_fail(stores):
    with pytest.raises(TimeValidationError, match="policy"):
        compare_stores(
            *stores, model_variable="t2", observation_field="T", cadence="hourly", policy="guess"
        )
    zarr.open_group(str(stores[0]), mode="a")["hourly"].attrs["units"] = "degC"
    with pytest.raises(TimeValidationError, match="units"):
        run(stores)


def test_explicit_timeframe_is_half_open(stores):
    result = run(stores, start="2025-01-01T01:00:00Z", end="2025-01-01T03:00:00Z")
    assert result["statistics"]["paired_count"] == 1
    assert result["statistics"]["gar_mean"] == 30
