"""Policy-bound station comparisons through the supported Python/CLI interface (#16).

Every metric expectation below is a hand-worked example, written down as literals
or as the arithmetic done on paper, not by calling Darwin's kernels.
"""

import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import zarr

from darwin.comparison_policy import (
    ComparisonError,
    compare_with_policy,
    replay_with_policy,
    store_identity,
)

SETUPS = Path(__file__).parent / "fixtures" / "setups-contracts"
HOUR = 3600
START = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())
STATION = {"id": "synthetic:station", "lon": -90.3, "lat": -0.7}
SAMPLING = {
    "method": "nearest",
    "crs": "synthetic-crs",
    "grid_y": 3,
    "grid_x": 4,
    "cell_lon": -90.31,
    "cell_lat": -0.69,
}
MODEL_SOURCE = "a" * 64
OBS_SOURCE = "b" * 64


def _store(path, values, valid, *, units, upstream, sampling=True, station=STATION):
    group = zarr.open_group(str(path), mode="w")
    axis = np.asarray([START + i * HOUR for i in range(len(values))], dtype="int64")
    group.create_array(
        "hourly", data=np.asarray(values, dtype="float64"), attributes={"units": units}
    )
    group.create_array("hourly_valid", data=np.asarray(valid))
    for name, data in (
        ("hourly_time", axis),
        ("hourly_time_bounds", np.stack([axis, axis + HOUR], 1)),
    ):
        group.create_array(
            name, data=data, attributes={"units": "seconds since 1970-01-01 00:00:00"}
        )
    attrs = {
        "series_schema_version": 2,
        "cadences": ["hourly"],
        "calendar": "proleptic_gregorian",
        "timezone": "UTC",
        "aggregation_timezone": "UTC",
        "source_coverage": {"start": "2025-01-01T00:00:00+00:00"},
        "synthetic": False,
        "scientific_status": "unassessed",
        "station": station,
    }
    if sampling:
        attrs |= {"sampling": SAMPLING, "source_gar_sha256": upstream}
    else:
        attrs |= {"source_sha256": upstream}
    group.attrs.update(attrs)
    return path


POLICY = {
    "contract": "darwin.comparison-policy/1",
    "id": "synthetic:comparison-policy",
    "revision": "1",
    "pairings": [
        {
            "requirement_variable": "synthetic:air-temperature",
            "model_variable": "t2",
            "observation_field": "T",
            "units": "K",
            "kind": "scalar",
        },
        {
            "requirement_variable": "synthetic:precipitation",
            "model_variable": "prcp",
            "observation_field": "PCP_tot_bucket",
            "units": "mm",
            "kind": "precipitation",
            "wet_threshold_mm": 0.2,
        },
    ],
    "cadences": ["hourly", "daily"],
    "completeness": {"minimum_paired_count": 2, "minimum_paired_fraction": 0.5},
    "exclusions": [
        {
            "start": "2025-01-01T05:00:00Z",
            "end": "2025-01-01T06:00:00Z",
            "reason": "synthetic:maintenance",
        }
    ],
    "metrics": [
        "paired_count",
        "means",
        "sample_variance",
        "bias",
        "mae",
        "rmse",
        "correlation",
        "histogram",
        "ecdf",
        "precipitation",
    ],
    "histogram_bins": 2,
}


def _encode(document):
    return json.dumps(document, indent=2, sort_keys=True).encode()


def _requirements(policy_bytes, *, review_status="reviewed", **overrides):
    authorities = [
        {
            "kind": "reviewer",
            "id": "synthetic:reviewer",
            "authorization": {
                "id": "synthetic:reviewer-authorization",
                "version": "1",
                "sha256": "c" * 64,
            },
        }
    ]
    document = {
        "contract": "darwin-wrf-setups.study-requirements/1",
        "id": "synthetic:station-comparison-requirements",
        "revision": "3",
        "review_status": review_status,
        "sources": [
            {"id": "synthetic:gar", "version": "1", "sha256": MODEL_SOURCE},
            {"id": "synthetic:aws", "version": "1", "sha256": OBS_SOURCE},
        ],
        "stations": ["synthetic:station"],
        "variables": ["synthetic:air-temperature", "synthetic:precipitation"],
        "operation_recipe": {
            "id": "synthetic:comparison-policy",
            "version": "1",
            "sha256": hashlib.sha256(policy_bytes).hexdigest(),
        },
        "acceptance": {
            "criteria": [
                {
                    "id": "synthetic:criterion",
                    "description": "Test only; not an approved scientific check",
                    "check": "synthetic.test-only",
                    "parameters": {},
                    "evidence_type": "synthetic:comparison-recipe",
                }
            ],
            "authorized_authorities": authorities if review_status == "reviewed" else [],
        },
    }
    document.update(overrides)
    return _encode(document)


@pytest.fixture
def inputs(tmp_path):
    # Seven requested hours. Model hour 6 is absent; observation hour 4 is invalid;
    # hour 5 is excluded by the policy; hours 0-3 pair.
    model = _store(
        tmp_path / "model.zarr",
        [10.0, 12.0, 14.0, 9.0, 13.0, 50.0],
        [True] * 6,
        units="K",
        upstream=MODEL_SOURCE,
    )
    obs = _store(
        tmp_path / "obs.zarr",
        [11.0, 11.0, 15.0, 7.0, 99.0, 40.0, 12.0],
        [True, True, True, True, False, True, True],
        units="K",
        upstream=OBS_SOURCE,
        sampling=False,
    )
    policy = _encode(POLICY)
    request = {
        "model": {"id": "synthetic:model-store", "version": "1", "sha256": store_identity(model)},
        "observations": {
            "id": "synthetic:obs-store",
            "version": "1",
            "sha256": store_identity(obs),
        },
        "station": STATION,
        "sampling": SAMPLING,
        "model_variable": "t2",
        "observation_field": "T",
        "cadence": "hourly",
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-01-01T07:00:00Z",
    }
    return {
        "request": request,
        "model_path": model,
        "observations_path": obs,
        "policy": policy,
        "requirements": _requirements(policy),
        "allow_synthetic": True,
    }


def run(inputs, **changes):
    arguments = {**inputs, **changes}
    request = arguments.pop("request")
    return compare_with_policy(request, **arguments)


def error(inputs, **changes):
    with pytest.raises(ComparisonError) as caught:
        run(inputs, **changes)
    return caught.value


def test_hand_computed_scalar_metrics_and_exclusion_accounting(inputs):
    result = run(inputs)
    metrics = result["metrics"]
    # Pairs (model, obs): (10, 11) (12, 11) (14, 15) (9, 7); errors -1, 1, -1, 2.
    assert metrics["paired_count"] == 4
    assert metrics["means"]["model"] == {"value": 11.25, "defined": True, "reason": None}
    assert metrics["means"]["observations"]["value"] == 11.0
    assert metrics["bias"]["value"] == pytest.approx(0.25)
    assert metrics["mae"]["value"] == pytest.approx(1.25)
    assert metrics["rmse"]["value"] == pytest.approx(math.sqrt(7 / 4))
    # Squared deviations: model 1.5625+0.5625+7.5625+5.0625=14.75; obs 0+0+16+16=32.
    assert metrics["sample_variance"]["model"]["value"] == pytest.approx(14.75 / 3)
    assert metrics["sample_variance"]["observations"]["value"] == pytest.approx(32 / 3)
    # Cross products: 2.75*4 + (-2.25)*(-4) = 20.
    assert metrics["correlation"]["value"] == pytest.approx(20 / math.sqrt(14.75 * 32))
    assert metrics["histogram"] == {
        "defined": True,
        "reason": None,
        "edges": [7.0, 11.0, 15.0],
        "model_counts": [2, 2],
        "observation_counts": [1, 3],
    }
    assert metrics["ecdf"]["model"] == [[9.0, 0.25], [10.0, 0.5], [12.0, 0.75], [14.0, 1.0]]
    assert metrics["ecdf"]["observations"] == [[7.0, 0.25], [11.0, 0.75], [15.0, 1.0]]
    assert "precipitation" not in metrics
    assert result["completeness"] == {
        "requested_intervals": 7,
        "eligible_intervals": 6,
        "paired_intervals": 4,
        "paired_fraction": pytest.approx(4 / 6),
        "excluded": {
            "policy:synthetic:maintenance": 1,
            "model_absent": 1,
            "observations_invalid": 1,
        },
    }
    assert result["scientific_status"] == "unassessed"
    assert result["review_status"] == "reviewed"


def test_hand_computed_precipitation_contingency(tmp_path, inputs):
    model = _store(
        tmp_path / "pm.zarr", [0.0, 2.0, 0.0, 1.5], [True] * 4, units="mm", upstream=MODEL_SOURCE
    )
    obs = _store(
        tmp_path / "po.zarr",
        [0.0, 1.0, 0.5, 0.0],
        [True] * 4,
        units="mm",
        upstream=OBS_SOURCE,
        sampling=False,
    )
    request = inputs["request"] | {
        "model": {"id": "synthetic:pm", "version": "1", "sha256": store_identity(model)},
        "observations": {"id": "synthetic:po", "version": "1", "sha256": store_identity(obs)},
        "model_variable": "prcp",
        "observation_field": "PCP_tot_bucket",
        "end": "2025-01-01T04:00:00Z",
    }
    result = run(inputs, request=request, model_path=model, observations_path=obs)
    # Wet means > 0.2 mm: model F T F T, observations F T T F.
    assert result["metrics"]["precipitation"] == {
        "wet_threshold_mm": 0.2,
        "total_model": 3.5,
        "total_observations": 1.5,
        "wet_frequency_model": 0.5,
        "wet_frequency_observations": 0.5,
        "hits": 1,
        "misses": 1,
        "false_alarms": 1,
        "correct_negatives": 1,
    }


def test_undefined_metric_semantics_are_explicit(tmp_path, inputs):
    model = _store(tmp_path / "um.zarr", [3.0, 5.0], [True, True], units="K", upstream=MODEL_SOURCE)
    obs = _store(
        tmp_path / "uo.zarr",
        [4.0, 4.0],
        [True, True],
        units="K",
        upstream=OBS_SOURCE,
        sampling=False,
    )
    request = inputs["request"] | {
        "model": {"id": "synthetic:um", "version": "1", "sha256": store_identity(model)},
        "observations": {"id": "synthetic:uo", "version": "1", "sha256": store_identity(obs)},
        "end": "2025-01-01T02:00:00Z",
    }
    metrics = run(inputs, request=request, model_path=model, observations_path=obs)["metrics"]
    # Constant observations: correlation has no defined value, variance is zero.
    assert metrics["correlation"] == {"value": None, "defined": False, "reason": "zero_variance"}
    assert metrics["sample_variance"]["observations"]["value"] == 0.0
    assert metrics["sample_variance"]["model"]["value"] == 2.0
    assert metrics["bias"]["value"] == 0.0

    single = json.loads(inputs["policy"]) | {
        "completeness": {"minimum_paired_count": 1, "minimum_paired_fraction": 0}
    }
    policy = _encode(single)
    one = request | {"end": "2025-01-01T01:00:00Z"}
    metrics = run(
        inputs,
        request=one,
        model_path=model,
        observations_path=obs,
        policy=policy,
        requirements=_requirements(policy),
    )["metrics"]
    assert metrics["sample_variance"]["model"] == {
        "value": None,
        "defined": False,
        "reason": "fewer_than_two_pairs",
    }
    assert metrics["correlation"]["reason"] == "fewer_than_two_pairs"
    assert metrics["rmse"]["value"] == 1.0
    assert metrics["bias"]["value"] == -1.0


def test_completeness_rule_comes_only_from_the_policy(inputs):
    strict = json.loads(inputs["policy"]) | {
        "completeness": {"minimum_paired_count": 2, "minimum_paired_fraction": 0.7}
    }
    policy = _encode(strict)
    failure = error(inputs, policy=policy, requirements=_requirements(policy))
    assert failure.code == "insufficient_completeness"
    assert failure.details["completeness"]["paired_intervals"] == 4
    assert failure.details["completeness"]["eligible_intervals"] == 6


def test_annual_is_rejected_explicitly(inputs):
    failure = error(inputs, request=inputs["request"] | {"cadence": "annual"})
    assert failure.code == "unsupported_cadence"
    assert "annual" in str(failure)
    annual = json.loads(inputs["policy"]) | {"cadences": ["hourly", "annual"]}
    assert error(inputs, policy=_encode(annual)).code == "unsupported_cadence"


def test_cadence_must_be_declared_by_policy(inputs):
    hourly_only = _encode(json.loads(inputs["policy"]) | {"cadences": ["daily"]})
    assert (
        error(inputs, policy=hourly_only, requirements=_requirements(hourly_only)).code
        == "cadence_not_in_policy"
    )


def test_circular_diagnostics_are_rejected_not_approximated(inputs):
    circular = json.loads(inputs["policy"])
    circular["metrics"] = [*circular["metrics"], "circular_direction"]
    assert error(inputs, policy=_encode(circular)).code == "unsupported_metric"


def test_a_different_policy_revision_is_not_accepted_under_these_requirements(inputs):
    revised = _encode(json.loads(inputs["policy"]) | {"revision": "2"})
    assert error(inputs, policy=revised).code == "policy_not_bound"
    edited = inputs["policy"] + b"\n"
    assert error(inputs, policy=edited).code == "policy_not_bound"


def test_copied_setups_requirements_do_not_authorize_an_unpinned_policy(inputs):
    setups = (SETUPS / "fixtures" / "acceptance-assessment-v1" / "requirements.json").read_bytes()
    assert error(inputs, requirements=setups).code == "policy_not_bound"
    draft = (SETUPS / "study-requirements-v1.fixture.json").read_bytes()
    assert error(inputs, requirements=draft).code == "policy_not_bound"


def test_draft_requirements_need_explicit_opt_in_and_stay_labelled(inputs):
    draft = _requirements(inputs["policy"], review_status="draft")
    assert error(inputs, requirements=draft).code == "requirements_not_reviewed"
    result = run(inputs, requirements=draft, allow_draft=True)
    assert result["review_status"] == "draft"
    assert result["scientific_status"] == "unassessed"
    assert result["recipe"]["requirements"]["review_status"] == "draft"


def test_synthetic_fixtures_are_refused_outside_test_mode(inputs):
    assert error(inputs, allow_synthetic=False).code == "synthetic_not_trusted"


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"station": STATION | {"lon": -90.0}}, "station_mismatch"),
        ({"station": STATION | {"id": "synthetic:other"}}, "station_not_in_requirements"),
        ({"sampling": SAMPLING | {"grid_x": 5}}, "sampling_mismatch"),
        (
            {"model": {"id": "synthetic:model-store", "version": "1", "sha256": "0" * 64}},
            "source_mismatch",
        ),
        (
            {"model": {"id": "synthetic:model-store", "version": "latest", "sha256": "0" * 64}},
            "dynamic_version",
        ),
        ({"observation_field": "RH"}, "pairing_not_in_policy"),
        ({"start": "2025-01-01T00:30:00Z"}, "timeframe"),
        ({"start": "2025-01-01T00:00:00"}, "timeframe"),
        ({"start": "2025-01-02T00:00:00Z", "end": "2025-01-02T03:00:00Z"}, "no_overlap"),
        ({"cadence": "daily", "end": "2025-01-02T00:00:00Z"}, "cadence"),
        ({"unexpected": True}, "invalid_request"),
    ],
)
def test_structured_refusals(inputs, change, code):
    assert error(inputs, request=inputs["request"] | change).code == code


def test_upstream_sources_must_be_pinned_by_requirements(inputs):
    other = _requirements(
        inputs["policy"], sources=[{"id": "synthetic:gar", "version": "1", "sha256": MODEL_SOURCE}]
    )
    assert error(inputs, requirements=other).code == "source_not_in_requirements"


def test_store_units_are_never_converted(tmp_path, inputs):
    model = _store(tmp_path / "c.zarr", [10.0] * 7, [True] * 7, units="degC", upstream=MODEL_SOURCE)
    request = inputs["request"] | {
        "model": {"id": "synthetic:c", "version": "1", "sha256": store_identity(model)}
    }
    assert error(inputs, request=request, model_path=model).code == "units"


def test_replay_is_exact_and_refuses_changed_inputs(inputs):
    result = run(inputs)
    replay_args = {
        k: inputs[k] for k in ("model_path", "observations_path", "policy", "requirements")
    }
    assert replay_with_policy(result["recipe"], **replay_args) == result
    zarr.open_group(str(inputs["observations_path"]), mode="a").attrs["scientific_status"] = (
        "accepted"
    )
    with pytest.raises(ComparisonError) as caught:
        replay_with_policy(result["recipe"], **replay_args)
    assert caught.value.code == "source_mismatch"


def test_cli_matches_python_and_emits_structured_errors(inputs, tmp_path):
    result = run(inputs)
    files = {}
    for name in ("policy", "requirements"):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_bytes(inputs[name])
    files["request"] = tmp_path / "request.json"
    files["request"].write_text(json.dumps(inputs["request"]))
    files["recipe"] = tmp_path / "recipe.json"
    files["recipe"].write_text(json.dumps(result["recipe"]))
    base = [
        sys.executable,
        "-m",
        "darwin.comparison_policy",
        "--model",
        str(inputs["model_path"]),
        "--observations",
        str(inputs["observations_path"]),
        "--policy",
        str(files["policy"]),
        "--requirements",
        str(files["requirements"]),
    ]
    ran = subprocess.run(
        [*base, "--request", str(files["request"]), "--test-mode"], capture_output=True, text=True
    )
    assert ran.returncode == 0, ran.stderr
    assert json.loads(ran.stdout) == result
    replayed = subprocess.run(
        [*base, "--recipe", str(files["recipe"])], capture_output=True, text=True
    )
    assert json.loads(replayed.stdout) == result
    refused = subprocess.run(
        [*base, "--request", str(files["request"])], capture_output=True, text=True
    )
    assert refused.returncode == 2
    assert json.loads(refused.stderr)["code"] == "synthetic_not_trusted"


def _assessment(requirements, dataset, status="unassessed"):
    record = {
        "contract": "darwin-wrf-setups.acceptance-assessment/1",
        "id": "synthetic:model-assessment",
        "synthetic": True,
        "dataset": dataset,
        "requirements": {
            "id": "synthetic:station-comparison-requirements",
            "revision": "3",
            "sha256": hashlib.sha256(requirements).hexdigest(),
        },
        "status": status,
        "evidence": [],
        "authority": None,
        "supersedes": None,
    }
    return _encode(record)


def test_assessment_status_is_reported_only_from_bound_records(inputs):
    record = _assessment(inputs["requirements"], inputs["request"]["model"])
    result = run(inputs, assessments={"model": {"assessment": record, "evidence": {}}})
    assert result["sources"]["model"]["scientific_status"] == "unassessed"
    assert (
        result["sources"]["model"]["assessment"]["assessment"]["sha256"]
        == hashlib.sha256(record).hexdigest()
    )
    assert result["sources"]["observations"]["scientific_status"] == "unassessed"
    # A store's self-declared status is reported as such, never promoted.
    assert result["sources"]["model"]["store_declared_status"] == "unassessed"

    wrong = _assessment(inputs["requirements"], inputs["request"]["observations"])
    assert (
        error(inputs, assessments={"model": {"assessment": wrong, "evidence": {}}}).code
        == "dataset_mismatch"
    )
    other = _assessment(b"other requirements", inputs["request"]["model"])
    assert (
        error(inputs, assessments={"model": {"assessment": other, "evidence": {}}}).code
        == "requirements_mismatch"
    )
    # An acceptance claim citing no evidence is a malformed decision.
    claimed = json.loads(record) | {
        "status": "accepted",
        "authority": json.loads(inputs["requirements"])["acceptance"]["authorized_authorities"][0],
    }
    failure = error(inputs, assessments={"model": {"assessment": _encode(claimed), "evidence": {}}})
    assert failure.code == "invalid_assessment"


def test_version_one_interface_is_unchanged():
    import darwin.comparison as v1

    assert v1.CONTRACT_VERSION == 1
    assert v1.POLICY == "native-intervals-v1"
