"""CJC's independent reference tests, copied with their literal expected values.

Sources (cjc ``integration/project-integration`` bcbb8e4): tests/workflow/
test_diagnostics.py, test_products.py, test_review_regressions.py, test_writer.py and
test_cape.py. Only imports and the native build entry point were adapted; the
expected values are the CJC authors' independent worked examples.
"""

import shutil
from pathlib import Path

import numpy as np
import pyproj
import pytest
import xarray as xr

from darwin.native_diagnostics import build_native, mcape_mcin
from darwin.wrf_diagnostics import accumulated_amount, inversion_layers, water_path
from darwin.wrf_products import aggregate, derive, normalize_wrf, write_products


# --- from tests/workflow/test_diagnostics.py ---


def test_inversion_is_adjacent_actual_temperature_increase() -> None:
    height = xr.DataArray([100, 500, 800, 1000, 1500, 2000, 3500], dims="bottom_top")
    temperature = xr.DataArray([299, 296, 294, 296, 297, 291, 280], dims="bottom_top")
    result = inversion_layers(temperature, height)
    np.testing.assert_allclose(
        result.inversion_delta_t, [np.nan, np.nan, 2, 1, np.nan, np.nan], equal_nan=True
    )
    assert result.inversion_base.isel(inversion_layer=2) == 800
    assert result.inversion_top.isel(inversion_layer=2) == 1000
    assert result.inversion_gradient.isel(inversion_layer=2) == 0.01
    assert result.inversion_count == 2


def test_inversion_rejects_non_monotonic_height() -> None:
    with pytest.raises(ValueError, match="increasing"):
        inversion_layers(
            xr.DataArray([290, 291], dims="bottom_top"), xr.DataArray([500, 400], dims="bottom_top")
        )


def test_amount_uses_preceding_boundary_and_detects_reset() -> None:
    time = np.array(["2024-01-01T00", "2024-01-01T01", "2024-01-01T02"], dtype="datetime64[h]")
    rain = xr.DataArray([20, 23, 25], dims="time", coords={"time": time})
    amount = accumulated_amount(rain)
    np.testing.assert_allclose(amount, [3, 2])
    np.testing.assert_array_equal(amount.time, time[1:])
    with pytest.raises(ValueError, match="reset"):
        accumulated_amount(xr.DataArray([20, 1, 2], dims="time", coords={"time": time}))


# --- from tests/workflow/test_products.py ---


def test_derived_precipitation_amount_has_interval_bounds() -> None:
    times = np.array(["2024-01-01T00", "2024-01-01T03", "2024-01-01T06"], dtype="datetime64[h]")
    data = xr.Dataset(
        {
            "RAINC": ("Time", [3, 4, 4]),
            "RAINNC": ("Time", [10, 13, 15]),
            "Times": ("Time", times.astype("datetime64[s]").astype(str)),
        }
    )
    result = derive(normalize_wrf(data), ["prcp", "prcp_rate"])
    np.testing.assert_allclose(result.prcp, [4, 2])
    np.testing.assert_allclose(result.prcp_rate, [4 / 3, 2 / 3])
    np.testing.assert_array_equal(result.time_bounds[:, 0], times[:-1])
    assert result.prcp.attrs["units"] == "mm"
    assert result.prcp.attrs["aggregation"] == "sum"


def test_daily_aggregate_masks_incomplete_cells_and_missing_timestamps() -> None:
    times = np.arange(
        np.datetime64("2024-01-01T01"), np.datetime64("2024-01-02T01"), np.timedelta64(1, "h")
    )
    values = np.ones((24, 2))
    values[5, 1] = np.nan
    source = xr.Dataset(
        {"prcp": (("time", "x"), values, {"aggregation": "sum", "units": "mm"})},
        coords={"time": times},
    )
    result = aggregate(source, "2024-01-01", "2024-01-02", "D", 60)
    assert result.prcp.values[0, 0] == 24
    assert np.isnan(result.prcp.values[0, 1])
    assert result.prcp_valid_fraction.values[0, 1] == pytest.approx(23 / 24)
    missing = aggregate(source.isel(time=slice(1, None)), "2024-01-01", "2024-01-02", "D", 60)
    assert np.isnan(missing.prcp).all()


# --- from tests/workflow/test_review_regressions.py ---


def test_hybrid_layer_masses_not_sigma_only() -> None:
    # Elevated column: hybrid eta pressure derivatives are 60000 and 80000 Pa,
    # each covering deta=.5; water masses are .01*30000/g + .02*40000/g.
    fields = xr.Dataset(
        {
            "DNW": ("bottom_top", [-0.5, -0.5]),
            "C1H": ("bottom_top", [1.0, 0.0]),
            "C2H": ("bottom_top", [0.0, 80000.0]),
            "MU": 0.0,
            "MUB": 60000.0,
        }
    )
    q = xr.DataArray([0.01, 0.02], dims="bottom_top")
    assert water_path(q, fields).item() == 1100 / 9.81


def test_net_radiation_includes_longwave_reflection_and_signed_condensation() -> None:
    times = np.array(["2024-01-01T00", "2024-01-01T01"], dtype="datetime64[h]")
    data = xr.Dataset(
        {
            "SWDOWN": ("time", [100, 100]),
            "ALBEDO": ("time", [0.2, 0.2]),
            "GLW": ("time", [300, 300]),
            "EMISS": ("time", [0.8, 0.8]),
            "TSK": ("time", [300, 300]),
        },
        coords={"time": times},
    )
    result = derive(data, ["netrad"])
    expected = 80 + 0.8 * (300 - 5.670374419e-8 * 300**4)
    np.testing.assert_allclose(result.netrad, expected)
    signed = accumulated_amount(
        xr.DataArray([10, 8], dims="time", coords={"time": times}), monotonic=False
    )
    assert signed.item() == -2


def test_inversion_absence_is_not_missing_profile() -> None:
    times = np.arange(
        np.datetime64("2024-01-01T01"), np.datetime64("2024-01-02T01"), np.timedelta64(1, "h")
    )
    source = xr.Dataset(
        {
            "inversion_base": (("time", "inversion_layer"), [[1000.0]] * 12 + [[np.nan]] * 12),
            "inversion_profile_valid": ("time", np.ones(24)),
            "inversion_interval_present": (
                ("time", "inversion_layer"),
                [[1.0]] * 12 + [[0.0]] * 12,
            ),
        },
        coords={"time": times},
    )
    result = aggregate(source, "2024-01-01", "2024-01-02", "D", 60)
    assert result.inversion_base.item() == 1000
    assert result.inversion_base_valid_fraction.item() == 1
    assert result.inversion_interval_present.item() == 0.5


def test_inversion_descriptor_roundtrip_retains_aggregation_companions(tmp_path: Path) -> None:
    times = np.arange(
        np.datetime64("2024-01-01T01"), np.datetime64("2024-01-02T01"), np.timedelta64(1, "h")
    )
    source = xr.Dataset(
        {
            "inversion_base": (("time", "inversion_layer"), [[1000.0]] * 12 + [[np.nan]] * 12),
            "inversion_profile_valid": ("time", np.ones(24)),
            "inversion_interval_present": (
                ("time", "inversion_layer"),
                [[1.0]] * 12 + [[0.0]] * 12,
            ),
        },
        coords={"time": times},
    )
    paths = write_products(source, tmp_path / "native", 2, "2024-01-01")
    descriptor = next(path for path in paths if "_inversion_base_" in path.name)
    with xr.open_dataset(descriptor) as ds:
        reduced = aggregate(ds, "2024-01-01", "2024-01-02", "D", 60)
        assert reduced.inversion_base.item() == 1000
        assert reduced.inversion_interval_present.item() == 0.5
        daily = write_products(reduced, tmp_path / "daily", 2, "2024-01-01", "d")
        assert len(daily) == 1


# --- from tests/workflow/test_writer.py ---


def test_native_mercator_writer_is_readable_and_no_clobber(tmp_path: Path) -> None:
    time = np.array(["2024-01-01T00", "2024-01-01T01", "2024-01-01T02"], dtype="datetime64[h]")
    projection = pyproj.CRS.from_proj4("+proj=merc +lat_ts=2 +lon_0=-87 +a=6370000 +b=6370000")
    transform = pyproj.Transformer.from_crs(projection, projection.geodetic_crs, always_xy=True)
    lon, lat = transform.transform(*np.meshgrid([0, 2000], [0, 2000]))
    source = xr.Dataset(
        {
            "RAINC": (("time", "south_north", "west_east"), np.zeros((3, 2, 2))),
            "RAINNC": (
                ("time", "south_north", "west_east"),
                np.broadcast_to(np.array([1, 4, 6])[:, None, None], (3, 2, 2)),
            ),
            "XLONG": (("time", "south_north", "west_east"), np.broadcast_to(lon, (3, 2, 2))),
            "XLAT": (("time", "south_north", "west_east"), np.broadcast_to(lat, (3, 2, 2))),
        },
        coords={"time": time},
        attrs={"MAP_PROJ": 3, "TRUELAT1": 2, "STAND_LON": -87},
    )
    products = derive(source, ["prcp"])
    paths = write_products(products, tmp_path, 2, "2024-01-01", source=source)
    with xr.open_dataset(paths[0]) as data:
        assert data.prcp.dims == ("time", "south_north", "west_east")
        assert data.attrs["DOMAIN"] == "d02km"
        assert data.attrs["product_status"] == "candidate"
        np.testing.assert_allclose(data.x, [0, 2000], atol=1e-6)
        np.testing.assert_allclose(data.y, [0, 2000], atol=1e-6)
        np.testing.assert_array_equal(data.time_bounds[:, 0], time[:-1])
    original = paths[0].read_bytes()
    with pytest.raises(FileExistsError):
        write_products(products, tmp_path, 2, "2024-01-01", source=source)
    assert paths[0].read_bytes() == original


# --- from tests/workflow/test_cape.py ---


def sounding(temperature: list[float]) -> xr.Dataset:
    n = len(temperature)
    pressure = np.geomspace(99000, 5000, n)
    heights = np.linspace(30, 20500, n)
    interfaces = np.linspace(0, 21000, n + 1)
    potential = np.array(temperature) * (100000 / pressure) ** (287 / 1004.5)
    shape = (1, n, 1, 1)
    dims = ("time", "bottom_top", "south_north", "west_east")
    surface = ("time", "south_north", "west_east")
    ds = xr.Dataset(
        {
            "P": (dims, np.zeros(shape)),
            "PB": (dims, pressure.reshape(shape)),
            "T": (dims, (potential - 300).reshape(shape)),
            "QVAPOR": (dims, np.geomspace(0.016, 1e-6, n).reshape(shape)),
            "PH": (
                ("time", "bottom_top_stag", "south_north", "west_east"),
                (interfaces * 9.81).reshape(1, n + 1, 1, 1),
            ),
            "PHB": (
                ("time", "bottom_top_stag", "south_north", "west_east"),
                np.zeros((1, n + 1, 1, 1)),
            ),
            "HGT": (surface, [[[0.0]]]),
            "PSFC": (surface, [[[101000.0]]]),
        },
        coords={"time": [np.datetime64("2024-01-01")]},
    )
    # Mass heights come from PH, not the nominal array used to construct the sounding.
    assert len(heights) == n
    return ds


def test_native_rip_2d_conventions_and_invalid_profile(tmp_path: Path) -> None:
    if shutil.which("gfortran") is None:
        pytest.fail("gfortran is required for native reference tests (devenv provides it)")
    library = build_native(tmp_path / "native")
    # Strongly stable: warm air aloft prevents a qualifying EL, NCAR returns missing.
    stable = mcape_mcin(sounding([300, *([330] * 40)]), library)
    assert np.isnan(stable.mcape).all()
    assert np.isnan(stable.mcin).all()
    unstable_profile = np.maximum(180, 300 - np.arange(41) * 5).tolist()
    unstable = mcape_mcin(sounding(unstable_profile), library)
    assert unstable.mcape.dims == ("time", "south_north", "west_east")
    assert unstable.mcape.values.item() > 100
    assert unstable.mcin.values.item() >= 0.1
    invalid = sounding(unstable_profile)
    invalid.QVAPOR.values[0, 5, 0, 0] = np.nan
    assert np.isnan(mcape_mcin(invalid, library).mcape).all()
