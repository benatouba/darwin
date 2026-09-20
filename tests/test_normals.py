"""Migrated normal-calculation tests: verified gar-for-rob formulas, publisher provenance."""

import json

import numpy as np
import pytest
import xarray as xr

from darwin.publisher.catalogue import Catalogue
from darwin.publisher.candidate import run_manifest
from darwin.publisher.manifest import Manifest, ProductSpec, SourceEntry
from darwin.publisher.normals import (
    NormalAccumulator,
    NormalPeriod,
    YearInputs,
    calculate_year,
    derive_products,
)

PERIOD = NormalPeriod(1991, 2020)


def _tile(values, ntime):
    flat = np.asarray(values, dtype=float)
    repeats = (ntime + flat.size - 1) // flat.size
    return np.tile(flat, repeats)[:ntime]


def _year_inputs(year, temperature_values, precipitation_monthly_rates, ncell=2):
    time = xr.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="1h", use_cftime=False)
    ntime = time.size
    # Short daily temperature patterns tile across the year.
    temperature_flat = _tile(temperature_values, ntime)
    # Precipitation follows gar-for-rob: one monthly rate per calendar month.
    month_time = xr.date_range(f"{year}-01-01", periods=12, freq="MS", use_cftime=False)
    monthly_rates = _tile(precipitation_monthly_rates, 12)
    coords = {
        "cell": ("cell", np.arange(ncell)),
        "lon": ("cell", np.linspace(-90.0, -89.0, ncell)),
    }
    temperature = xr.DataArray(
        np.broadcast_to(temperature_flat.reshape(ntime, 1), (ntime, ncell)),
        dims=("time", "cell"),
        coords={"time": time, **coords},
        attrs={"units": "K"},
    )
    precipitation = xr.DataArray(
        np.broadcast_to(monthly_rates.reshape(12, 1), (12, ncell)),
        dims=("time", "cell"),
        coords={"time": month_time, **coords},
        attrs={"units": "mm h-1"},
    )
    return YearInputs(
        temperature=temperature,
        precipitation=precipitation,
        source_attributes={"TITLE": "GAR test source"},
        temperature_attributes={"units": "K"},
        precipitation_attributes={"units": "mm h-1"},
        coordinates=coords,
    )


def test_temperature_is_mean_of_daily_extrema_in_celsius():
    hours = np.arange(24 * 3)
    # Cold mornings (280 K), warm afternoons (290 K): daily max 290, min 280.
    values = np.where(hours % 24 < 12, 280.0, 290.0)
    inputs = _year_inputs(2021, values, np.zeros(12))
    normals = NormalAccumulator()
    normals.add(calculate_year(inputs))
    result = normals.normals()
    np.testing.assert_allclose(result.temperature_maximum.sel(month=1).values, 16.85)
    np.testing.assert_allclose(result.temperature_minimum.sel(month=1).values, 6.85)


def test_precipitation_uses_days_in_month_with_leap_february():
    leap = _year_inputs(2020, np.full(24 * 7, 280.0), np.ones(12))
    plain = _year_inputs(2021, np.full(24 * 7, 280.0), np.ones(12))
    normals = NormalAccumulator()
    normals.add(calculate_year(leap))
    normals.add(calculate_year(plain))
    result = normals.normals()
    february = result.precipitation_total.sel(month=2).values
    # (29*24 + 28*24) / 2 valid years at 1 mm/h.
    np.testing.assert_allclose(february, 684.0)


def test_layers_cover_month_1_to_12_with_units_and_semantics():
    inputs = _year_inputs(2021, np.full(24 * 7, 285.0), np.full(12, 0.5))
    normals = NormalAccumulator()
    normals.add(calculate_year(inputs))
    layers, _ = derive_products(normals.normals(), PERIOD)
    assert set(layers) == {"t2_max", "t2_min", "prcp_sum"}
    for variable, values in layers.items():
        assert values.coords["month"].values.tolist() == list(range(1, 13))
    assert layers["t2_max"].attrs["units"] == "C"
    assert layers["t2_min"].attrs["units"] == "C"
    assert layers["prcp_sum"].attrs["units"] == "mm"
    assert layers["t2_max"].attrs["long_name"] == "Mean daily maximum 2m Temperature"
    assert layers["t2_min"].attrs["long_name"] == "Mean daily minimum 2m Temperature"
    assert layers["prcp_sum"].attrs["long_name"] == "Mean total precipitation"
    assert all(layer.attrs["agg_method"] == "MEAN" for layer in layers.values())


def test_missing_values_excluded_per_grid_cell():
    values = np.where(np.arange(24 * 3) % 24 < 12, 280.0, 290.0)
    values[:24] = np.nan  # First day missing: excluded from the mean.
    inputs = _year_inputs(2021, values, np.zeros(12))
    normals = NormalAccumulator()
    normals.add(calculate_year(inputs))
    result = normals.normals()
    np.testing.assert_allclose(result.temperature_maximum.sel(month=1).values, 16.85)


def test_provenance_records_period_method_history_and_versions():
    inputs = _year_inputs(2021, np.full(24 * 7, 285.0), np.full(12, 0.5))
    normals = NormalAccumulator()
    normals.add(calculate_year(inputs))
    _, publication = derive_products(normals.normals(), NormalPeriod(1991, 2020))
    assert publication["climate_normal_period"] == "1991-01-01 through 2020-12-31"
    assert "daily maximum and minimum" in str(publication["processing_method"])
    assert publication["creator"] == "darwin.publisher"
    assert "darwin.publisher created monthly GAR climate-normal products" in str(
        publication["history"]
    )
    assert publication["TITLE"] == "GAR test source"
    versions = json.loads(str(publication["software_versions"]))
    assert set(versions) == {"darwin", "numpy", "python", "xarray"}


def test_normal_run_registers_three_candidates_bound_to_manifest_revision():
    manifest = Manifest(
        name="normals-1991-2020",
        release="v1",
        sources=(SourceEntry(path="GAR_d02km_h_2d_t2_1991.nc", sha256="abc"),),
        products=tuple(
            ProductSpec(product_id=f"GAR_d02km_m_2d_{variable}_1991-2020", kind="normal")
            for variable in ("t2_max", "t2_min", "prcp_sum")
        ),
        revision="rev-normal",
    )
    catalogue = Catalogue()
    run_manifest(manifest, catalogue)
    for variable in ("t2_max", "t2_min", "prcp_sum"):
        candidate = catalogue.get(f"GAR_d02km_m_2d_{variable}_1991-2020", "v1.1")
        assert candidate.status == "candidate"
        assert candidate.manifest_revision == "rev-normal"


def test_period_rejects_start_after_end():
    with pytest.raises(ValueError, match="starts after it ends"):
        NormalPeriod(2020, 1991)
