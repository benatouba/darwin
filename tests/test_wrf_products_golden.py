"""Darwin reproduces CJC-produced reference outputs (copied ``cjc.product-golden/1``).

Expected values are the producer's files, never recomputed with Darwin code. This
is software parity on synthetic inputs, not real-data validation or acceptance.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from darwin.wrf_products import aggregate, derive, normalize_wrf

GOLDEN = Path(__file__).parent / "fixtures" / "cjc-product-golden-v1"
INDEX = json.loads((GOLDEN / "index.json").read_text())


def golden(name: str) -> xr.Dataset:
    with xr.open_dataset(GOLDEN / name) as dataset:
        return dataset.load()


def roundtrip(dataset: xr.Dataset, tmp_path: Path) -> xr.Dataset:
    """Compare after the same NetCDF serialization the reference went through."""
    path = tmp_path / "result.nc"
    dataset.to_netcdf(path, engine="netcdf4")
    with xr.open_dataset(path) as reopened:
        return reopened.load()


def assert_matches(actual: xr.Dataset, expected: xr.Dataset) -> None:
    assert sorted(actual.data_vars) == sorted(expected.data_vars)
    assert actual.attrs == expected.attrs
    for name, value in expected.variables.items():
        result = actual[name]
        assert result.dims == value.dims, name
        assert result.attrs == value.attrs, name
        if value.dtype.kind in "fc":
            np.testing.assert_allclose(result, value, rtol=1e-12, atol=0, equal_nan=True)
        else:
            np.testing.assert_array_equal(result, value)


def test_copied_fixtures_are_the_producer_bytes():
    assert INDEX["contract"] == "cjc.product-golden/1"
    for record in INDEX["files"]:
        digest = hashlib.sha256((GOLDEN / record["file"]).read_bytes()).hexdigest()
        assert digest == record["sha256"], record["file"]


def test_derived_products_match_cjc_reference(tmp_path):
    case = INDEX["cases"]["derived"]
    result = derive(
        normalize_wrf(golden(case["input"])),
        case["variables"],
        pressure_levels=case["pressure_levels"],
    )
    assert_matches(roundtrip(result, tmp_path), golden(case["output"]))


@pytest.mark.parametrize("name", ["daily", "daily-gap"])
def test_strict_aggregates_match_cjc_reference(tmp_path, name):
    derived = golden("derived.nc")
    if name == "daily-gap":
        derived = derived.drop_isel(time=[10])
    result = aggregate(derived, "2024-01-01", "2024-01-02", "D", 60)
    assert_matches(roundtrip(result, tmp_path), golden(INDEX["cases"][name]["output"]))


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    from darwin.native_diagnostics import build_native

    if shutil.which("gfortran") is None:
        pytest.fail("gfortran is required for native golden tests (devenv provides it)")
    return build_native(tmp_path_factory.mktemp("native"))


def test_native_diagnostics_match_cjc_reference_and_record_provenance(tmp_path, native_library):
    case = INDEX["cases"]["native"]
    result = derive(normalize_wrf(golden(case["input"])), case["variables"], native_library)
    library_sha256 = hashlib.sha256(native_library.read_bytes()).hexdigest()
    for name in ("mcape", "mcin", "lcl", "lfc"):
        assert result[name].attrs.pop("native_library_sha256") == library_sha256
        assert result[name].attrs["lookup_sha256"] == INDEX["native"]["lookup_sha256"]
    assert_matches(roundtrip(result, tmp_path), golden(case["output"]))
