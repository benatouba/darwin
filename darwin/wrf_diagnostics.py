"""Native-grid WRF diagnostic definitions; no input/output side effects.

Migrated unchanged from CJC ``cjc.workflow.diagnostics`` (definition version
``gar-v2-python-1``); see docs/cjc-science-inventory.md. Do not change these
definitions without a new definition version and independent reference tests.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

G = 9.81
RD = 287.0
CP = 1004.5


def destagger(array: xr.DataArray, dim: str) -> xr.DataArray:
    """Average adjacent staggered cells positionally, avoiding xarray alignment loss."""
    axis = array.get_axis_num(dim)
    first = [slice(None)] * array.ndim
    second = first.copy()
    first[axis], second[axis] = slice(None, -1), slice(1, None)
    dims = [str(name).removesuffix("_stag") if name == dim else str(name) for name in array.dims]
    coords = {k: v for k, v in array.coords.items() if dim not in v.dims}
    return xr.DataArray(
        (array.values[tuple(first)] + array.values[tuple(second)]) / 2,
        dims=dims,
        coords=coords,
        attrs=array.attrs,
    )


def state(dataset: xr.Dataset) -> xr.Dataset:
    pressure = dataset.P + dataset.PB
    theta = dataset.T + 300
    temperature = theta * (pressure / 100000) ** (RD / CP)
    interface = (dataset.PH + dataset.PHB) / G
    height = destagger(interface, "bottom_top_stag")
    return xr.Dataset(
        {
            "pressure_pa": pressure,
            "theta": theta,
            "temperature": temperature,
            "height_msl": height,
            "height_agl": height - dataset.HGT,
            "interface_height": interface,
        }
    )


def inversion_layers(
    temperature: xr.DataArray, height_agl: xr.DataArray, search: tuple[float, float] = (200, 3000)
) -> xr.Dataset:
    """Detect each adjacent mass-level interval with dT/dz > 0, in AGL metres.

    Absence produces count 0. A missing profile produces missing count. No
    temperature/thickness threshold or smoothing is applied. Interval identity
    is retained; neighboring intervals are not merged into one inversion.
    """
    temperature, height_agl = xr.broadcast(temperature, height_agl)
    dims = ["inversion_layer" if d == "bottom_top" else d for d in temperature.dims]
    axis = temperature.get_axis_num("bottom_top")
    z, t = height_agl.values, temperature.values
    dz, dt = np.diff(z, axis=axis), np.diff(t, axis=axis)
    if np.any(np.isfinite(dz) & (dz <= 0)):
        raise ValueError("Profile heights must be strictly increasing")
    low = np.take(z, range(z.shape[axis] - 1), axis=axis)
    high = np.take(z, range(1, z.shape[axis]), axis=axis)
    valid = np.isfinite(dt) & np.isfinite(dz)
    present = valid & (dt > 0) & (low >= search[0]) & (high <= search[1])
    coords = {k: v for k, v in temperature.coords.items() if "bottom_top" not in v.dims}
    results = {}
    for name, value, units in [
        ("base", low, "m"),
        ("top", high, "m"),
        ("delta_t", dt, "K"),
        ("gradient", dt / dz, "K m-1"),
    ]:
        results[f"inversion_{name}"] = xr.DataArray(
            np.where(present, value, np.nan),
            dims=dims,
            coords=coords,
            attrs={"units": units, "definition": "adjacent-positive-dT-dz-v1"},
        )
    count_dims = [d for d in dims if d != "inversion_layer"]
    profile_valid = np.all(np.isfinite(t) & np.isfinite(z), axis=axis)
    results["inversion_count"] = xr.DataArray(
        np.where(profile_valid, np.sum(present, axis=axis), np.nan),
        dims=count_dims,
        coords=coords,
        attrs={"units": "1"},
    )
    results["inversion_profile_valid"] = xr.DataArray(
        profile_valid.astype("i1"), dims=count_dims, coords=coords, attrs={"units": "1"}
    )
    results["inversion_interval_present"] = xr.DataArray(
        np.where(valid, present.astype(float), np.nan),
        dims=dims,
        coords=coords,
        attrs={
            "units": "1",
            "aggregation": "mean",
            "definition": "positive-temperature-gradient-occurrence",
        },
    )
    return xr.Dataset(results)


def accumulated_amount(array: xr.DataArray, *, monotonic: bool = True) -> xr.DataArray:
    """Deaccumulate within one run; a reset is an error, never unexplained rainfall."""
    amount = array.diff("time", label="upper")
    if monotonic and bool((amount < -1e-6).any()):
        raise ValueError("Accumulation reset: split sources by independent model run")
    return amount.clip(min=0) if monotonic else amount


def earth_wind(
    u: xr.DataArray, v: xr.DataArray, cos: xr.DataArray, sin: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    return u * cos - v * sin, v * cos + u * sin


def water_path(mixing_ratio: xr.DataArray, dataset: xr.Dataset) -> xr.DataArray:
    """Total-air-mass approximation q*dp/g, documented; never an unweighted q sum."""
    # For WRF's hybrid coordinate, C1H = dB/deta and C2H = (1-C1H)*(p0-ptop).
    # dp_dry/deta = C1H * mu_dry + C2H. Sigma-only mu*dnw is not valid here.
    if not all(name in dataset for name in ["DNW", "MU", "MUB", "C1H", "C2H"]):
        raise ValueError("Hybrid water path requires DNW, MU, MUB, C1H and C2H")
    weight = (-dataset.DNW) * (dataset.C1H * (dataset.MU + dataset.MUB) + dataset.C2H)
    return (mixing_ratio * weight / G).sum("bottom_top", skipna=False)


def interpolate_pressure(
    value: xr.DataArray, pressure_pa: xr.DataArray, levels_hpa: list[float]
) -> xr.DataArray:
    """Linear-pressure legacy interpolation without below-ground/top extrapolation."""

    def column(v: np.ndarray, p: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(p)) or np.any(np.diff(p) >= 0):
            return np.full(len(levels_hpa), np.nan)
        return np.interp(np.array(levels_hpa) * 100, p[::-1], v[::-1], left=np.nan, right=np.nan)

    result = xr.apply_ufunc(
        column,
        value,
        pressure_pa,
        input_core_dims=[["bottom_top"]] * 2,
        output_core_dims=[["pressure"]],
        vectorize=True,
        output_dtypes=[float],
    )
    result = result.assign_coords(pressure=levels_hpa)
    result.pressure.attrs.update(units="hPa", positive="down", axis="Z")
    return result
