"""WRF-to-GAR product candidates: native diagnostics, pressure levels and strict aggregates.

Definitions deliberately distinguish rates, interval amounts, Earth-relative wind,
scheme PBLH and temperature inversions. Publication approval is outside this module.

Migrated from CJC ``cjc.workflow.products`` (definition version ``gar-v2-python-1``)
without changing numerical conventions; see docs/cjc-science-inventory.md and
docs/wrf-products.md. Native MCAPE/MCIN/SLP need the ``native`` extra and an
explicitly built kernel (``darwin.native_diagnostics``); the writer needs pyproj.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from darwin.publisher.immutable import file_sha256, identity_path, staged_artifact
from darwin.wrf_diagnostics import (
    accumulated_amount,
    destagger,
    earth_wind,
    interpolate_pressure,
    inversion_layers,
    state,
    water_path,
)

CONTRACT = "darwin.wrf-products/1"
DEFINITION_VERSION = "gar-v2-python-1"

DIRECT: dict[str, str] = {
    name.lower(): name
    for name in [
        "T2",
        "Q2",
        "PSFC",
        "TSK",
        "SST",
        "PBLH",
        "UST",
        "SNOWH",
        "SNOW",
        "CANWAT",
        "SWDOWN",
        "ALBEDO",
        "EMISS",
        "GRDFLX",
        "HFX",
        "LH",
        "SMOIS",
        "SH2O",
        "SMCREL",
        "TSLB",
        "QVAPOR",
        "CLDFRA",
        "SR",
        "VEGFRA",
        "LAI",
        "LU_INDEX",
        "LANDMASK",
        "HGT",
        "COSALPHA",
        "SINALPHA",
        "F",
        "MAPFAC_M",
        "ISLTYP",
        "IVGTYP",
    ]
}
INVERSION_DESCRIPTORS = (
    "inversion_base",
    "inversion_top",
    "inversion_delta_t",
    "inversion_gradient",
)
"""Inversion variables averaged only where an inversion is present."""
INVERSION_COMPANIONS = ("inversion_profile_valid", "inversion_interval_present")
PRESSURE_INTERPOLATED = (
    "theta",
    "temperature",
    "qvapor",
    "qliquid",
    "qsolid",
    "u",
    "v",
    "w",
    "ws",
    "geopotential",
)
DOMAIN_NAMES = {1: "d10km", 2: "d02km"}
PRODUCT_LEVELS = {"D": "d", "MS": "m", "YS": "y"}
"""Aggregation frequency to product level (native records are ``h``)."""
FRAGMENT_IDENTITY_ATTRS = ("source_sha256", "source_path", "sources", "recipe")
"""Per-input identity attributes an aggregate must never inherit from one fragment."""
DEFAULT_VARIABLES = [
    "t2",
    "q2",
    "psfc",
    "pblh",
    "u10",
    "v10",
    "ws10",
    "prcp",
    "prcp_rate",
    "temperature",
    "theta",
    "qvapor",
    "pressure",
    "height_agl",
    "inversion",
    "mcape",
    "mcin",
]


def wrf_times(dataset: xr.Dataset) -> np.ndarray:
    """Authoritative WRF record times (``time`` coordinate or ``Times`` strings)."""
    if "time" in dataset.coords:
        return dataset.time.values.astype("datetime64[s]")
    if "Times" not in dataset:
        raise ValueError("Missing authoritative WRF timestamps")
    values = dataset.Times.values
    result = []
    for row in values:
        value = (
            b"".join(row).decode()
            if getattr(row, "ndim", 0)
            else (row.decode() if isinstance(row, bytes) else str(row))
        )
        result.append(np.datetime64(value.replace("_", "T"), "s"))
    return np.asarray(result)


def normalize_wrf(dataset: xr.Dataset) -> xr.Dataset:
    times = wrf_times(dataset)
    if len(times) < 2 or np.any(np.diff(times) <= np.timedelta64(0, "s")):
        raise ValueError("WRF timestamps must be unique and increasing, including initial boundary")
    if "Time" in dataset.dims:
        dataset = dataset.rename({"Time": "time"})
    return dataset.assign_coords(time=times)


def derive(
    dataset: xr.Dataset,
    variables: list[str],
    rip_library: Path | None = None,
    search: tuple[float, float] = (200, 3000),
    pressure_levels: list[float] | None = None,
) -> xr.Dataset:
    """Derive explicitly selected diagnostics; unknown/missing inputs fail visibly."""
    result = xr.Dataset(
        attrs={
            **dataset.attrs,
            "diagnostic_version": DEFINITION_VERSION,
            "product_status": "candidate",
        }
    )
    cache: dict[str, xr.DataArray] = {}
    atmosphere: xr.Dataset | None = None

    def get_state() -> xr.Dataset:
        nonlocal atmosphere
        if atmosphere is None:
            atmosphere = state(dataset)
        return atmosphere

    def variable(name: str) -> xr.DataArray:
        if name in cache:
            return cache[name]
        units = None
        aggregation = "mean"
        if name in DIRECT:
            value = dataset[DIRECT[name]]
        elif name == "slp":
            if rip_library is None:
                raise ValueError("SLP requires the explicitly built NCAR library")
            from darwin.native_diagnostics import sea_level_pressure

            value = sea_level_pressure(dataset, rip_library)
            units = "hPa"
        elif name in ["u10", "v10", "ws10"]:
            u, v = earth_wind(dataset.U10, dataset.V10, dataset.COSALPHA, dataset.SINALPHA)
            value = {"u10": u, "v10": v, "ws10": np.hypot(u, v)}[name]
            units = "m s-1"
        elif name in ["u", "v", "ws"]:
            u, v = earth_wind(
                destagger(dataset.U, "west_east_stag"),
                destagger(dataset.V, "south_north_stag"),
                dataset.COSALPHA,
                dataset.SINALPHA,
            )
            value = {"u": u, "v": v, "ws": np.hypot(u, v)}[name]
            units = "m s-1"
        elif name == "w":
            value, units = destagger(dataset.W, "bottom_top_stag"), "m s-1"
        elif name in ["theta", "temperature", "pressure", "geopotential", "height_agl"]:
            key = {"pressure": "pressure_pa", "geopotential": "height_msl"}.get(name, name)
            value = get_state()[key]
            if name == "pressure":
                value = value / 100
            elif name == "geopotential":
                value = value * 9.81
            units = {
                "theta": "K",
                "temperature": "K",
                "pressure": "hPa",
                "geopotential": "m2 s-2",
                "height_agl": "m",
            }[name]
        elif name in ["qliquid", "qsolid"]:
            constituents = ["QCLOUD", "QRAIN"] if name == "qliquid" else ["QICE", "QSNOW", "QGRAUP"]
            available = [dataset[key] for key in constituents if key in dataset]
            if not available:
                raise ValueError(f"No constituent fields for {name}")
            value = available[0]
            for component in available[1:]:
                value = value + component
            units = "kg kg-1"
            value.attrs["constituents"] = ",".join(key for key in constituents if key in dataset)
        elif name.startswith("col_"):
            value, units = water_path(variable(name[4:]), dataset), "kg m-2"
        elif match := re.fullmatch(
            r"flux_(int_)?(qvapor|qliquid|qsolid)_(u|v|w|hor)_(2d|3d)", name
        ):
            integrated, constituent, component, dimension = match.groups()
            if bool(integrated) != (dimension == "2d") or (component == "hor" and not integrated):
                raise ValueError("Invalid flux diagnostic dimensionality")
            if component == "hor":
                first = variable(f"flux_int_{constituent}_u_2d")
                second = variable(f"flux_int_{constituent}_v_2d")
                value = (first**2 + second**2) ** 0.5
            elif integrated:
                value = water_path(variable(constituent) * variable(component), dataset)
            else:
                dry_density = get_state().pressure_pa / (
                    287 * get_state().temperature * (1 + dataset.QVAPOR / 0.622)
                )
                value = dry_density * variable(constituent) * variable(component)
            units = "kg m-1 s-1" if integrated else "kg m-2 s-1"
            value.attrs["definition"] = "dry-air-mass-basis; Earth-relative horizontal wind"
        elif name in [
            "prcp",
            "prcp_c",
            "prcp_nc",
            "prcp_rate",
            "prcp_fr",
            "snowfall",
            "graupel",
            "hail",
            "potevap",
            "et",
            "sfroff",
            "udroff",
        ]:
            if name in ["prcp", "prcp_rate"]:
                accumulated = dataset.RAINC + dataset.RAINNC
            elif name == "prcp_fr":
                # SNOWNC is total resolved frozen precipitation; graupel/hail
                # diagnostics can be subsets and must not be added again.
                accumulated = dataset.SNOWNC
            elif name == "et":
                accumulated = dataset.ACLHF / 2.5e6
            else:
                key = {
                    "prcp_c": "RAINC",
                    "prcp_nc": "RAINNC",
                    "snowfall": "SNOWNC",
                    "graupel": "GRAUPELNC",
                    "hail": "HAILNC",
                    "potevap": "POTEVP",
                    "sfroff": "SFROFF",
                    "udroff": "UDROFF",
                }[name]
                accumulated = dataset[key]
                if name == "potevap":
                    accumulated = accumulated * 1000  # WRF POTEVP in m, legacy conversion.
            value = accumulated_amount(accumulated, monotonic=name not in ["et", "potevap"])
            if name == "prcp_fr":
                value.attrs["definition"] = (
                    "SNOWNC resolved frozen amount; excludes parameterized convective frozen precipitation"
                )
            units, aggregation = "mm", "sum"
            if name == "prcp_rate":
                hours = dataset.time.diff("time") / np.timedelta64(1, "h")
                value = value / hours
                units, aggregation = "mm h-1", "mean"
        elif name == "qfx":
            value = dataset.QFX.isel(time=slice(1, None)) * (
                dataset.time.diff("time") / np.timedelta64(1, "s")
            )
            units, aggregation = "kg m-2", "sum"
            value.attrs["integration"] = "right-endpoint rectangle; flux may be signed"
        elif name in ["swup", "lwup", "lwdown", "netrad"]:
            if name == "swup":
                value = dataset.SWDOWN * dataset.ALBEDO
            elif name == "lwup":
                value = (
                    5.670374419e-8 * dataset.EMISS * dataset.TSK**4
                    + (1 - dataset.EMISS) * dataset.GLW
                )
            elif name == "lwdown":
                value = dataset.GLW
            else:
                value = dataset.SWDOWN + dataset.GLW - variable("swup") - variable("lwup")
            units = "W m-2"
        elif name == "low_cloud_fraction":
            # Outside-band missing cells must not erase a valid whole profile.
            value = dataset.CLDFRA.where(get_state().height_agl < 3000, 0).max(
                "bottom_top", skipna=False
            )
            units = "1"
            value.attrs["definition"] = "maximum-layer-cloud-fraction-below-3km; maximum-overlap"
        elif name in ["t2eta", "t2pbl"]:

            def extrapolate(t: np.ndarray, z: np.ndarray, pbl: float) -> float:
                if not np.all(np.isfinite(t)) or not np.all(np.isfinite(z)):
                    return float("nan")
                selected = np.flatnonzero(z < pbl) if name == "t2pbl" else np.arange(2)
                if len(selected) < 2:
                    selected = np.arange(2)
                slope, intercept = np.polyfit(z[selected], t[selected], 1)
                return float(intercept + 2 * slope)

            value = xr.apply_ufunc(
                extrapolate,
                get_state().temperature,
                get_state().height_agl,
                dataset.PBLH,
                input_core_dims=[["bottom_top"], ["bottom_top"], []],
                vectorize=True,
                output_dtypes=[float],
            )
            units = "K"
        else:
            raise ValueError(f"Unsupported diagnostic: {name}")
        if not isinstance(value, xr.DataArray):
            raise TypeError(f"Diagnostic did not return a labelled array: {name}")
        value = value.copy(deep=False)
        if units:
            value.attrs["units"] = units
        value.attrs["aggregation"] = aggregation
        value.attrs["definition_version"] = DEFINITION_VERSION
        if name in ["u", "v", "u10", "v10"]:
            value.attrs["wind_reference"] = (
                "Earth-relative eastward" if name.startswith("u") else "Earth-relative northward"
            )
        cache[name] = value
        return value

    for name in variables:
        if name == "inversion":
            result.update(inversion_layers(get_state().temperature, get_state().height_agl, search))
        elif name in ["mcape", "mcin", "lcl", "lfc"]:
            if "mcape" not in cache:
                if rip_library is None:
                    raise ValueError("MCAPE/MCIN requires an explicitly built NCAR RIP library")
                from darwin.native_diagnostics import mcape_mcin

                cache.update(
                    {str(key): value for key, value in mcape_mcin(dataset, rip_library).items()}
                )
            result[name] = cache[name]
            result[name].attrs["aggregation"] = "mean"
        else:
            result[name] = variable(name)
        if pressure_levels and name in PRESSURE_INTERPOLATED:
            result[f"{name}_press"] = interpolate_pressure(
                variable(name), get_state().pressure_pa, pressure_levels
            )
            result[f"{name}_press"].attrs = variable(name).attrs.copy()
    # Right-closed intervals: discard the initial boundary for every diagnostic.
    result = result.sel(time=dataset.time.isel(time=slice(1, None)))
    result["time_bounds"] = xr.DataArray(
        np.column_stack([dataset.time.values[:-1], dataset.time.values[1:]]),
        dims=("time", "bounds"),
        coords={"time": result.time},
    )
    result.time.attrs["bounds"] = "time_bounds"
    return result


def aggregate(
    dataset: xr.Dataset, start: str, end: str, frequency: str, cadence_minutes: int
) -> xr.Dataset:
    """Aggregate (start,end] against all expected timestamps, with per-cell completeness.

    Monthly and annual products are weighted directly by native records, avoiding
    unequal-month weighting. Flags retain absence vs invalid diagnostic profiles.
    """
    if frequency not in ["D", "MS", "YS"] or cadence_minutes <= 0:
        raise ValueError("Use D, MS or YS and a positive native cadence")
    if np.any(np.diff(dataset.time.values) <= np.timedelta64(0, "s")):
        raise ValueError("Duplicate/non-monotonic product timestamps")
    left, right = datetime.fromisoformat(start), datetime.fromisoformat(end)
    if right <= left:
        raise ValueError("Aggregation end must follow start")
    offset = pd.tseries.frequencies.to_offset(frequency)
    if not offset.is_on_offset(left) or not offset.is_on_offset(right):
        raise ValueError("Aggregation bounds must lie on frequency boundaries")
    expected = pd.date_range(left, right, freq=f"{cadence_minutes}min", inclusive="right")
    times = pd.DatetimeIndex(dataset.time.values)
    inside = times[(times > left) & (times <= right)]
    off_grid = inside.difference(expected)
    if len(off_grid):
        raise ValueError(
            f"{len(off_grid)} timestamps are off the {cadence_minutes}-minute native cadence "
            f"(first {off_grid[0].isoformat()}); resample explicitly before aggregating"
        )
    aligned = dataset.reindex(time=expected)
    boundaries = pd.date_range(left, right, freq=frequency)
    attrs = {k: v for k, v in dataset.attrs.items() if k not in FRAGMENT_IDENTITY_ATTRS}
    results = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
        block = aligned.sel(
            time=(aligned.time > np.datetime64(lo)) & (aligned.time <= np.datetime64(hi))
        )
        out = xr.Dataset(attrs=attrs)
        for name, value in block.data_vars.items():
            if name == "time_bounds" or "time" not in value.dims:
                continue
            aggregation = value.attrs.get("aggregation", "mean")
            conditional = name in INVERSION_DESCRIPTORS
            fraction = (
                block.inversion_profile_valid.fillna(0).mean("time")
                if conditional
                else value.notnull().mean("time")
            )
            reduced = (
                value.sum("time", skipna=False)
                if aggregation == "sum"
                else value.mean("time", skipna=conditional)
            )
            out[name] = reduced.where(fraction == 1)
            out[name].attrs = {
                **value.attrs,
                "cell_methods": f"time: {aggregation}",
                "completeness_policy": "require-all-expected-timestamps",
            }
            if conditional:
                out[name].attrs["aggregation"] = "conditional-mean-when-inversion-present"
            out[f"{name}_valid_fraction"] = fraction
            out[f"{name}_valid_fraction"].attrs["units"] = "1"
        out = out.expand_dims(time=[np.datetime64(lo)])
        out["time_bounds"] = (("time", "bounds"), [[np.datetime64(lo), np.datetime64(hi)]])
        results.append(out)
    if not results:
        raise ValueError("No complete aggregation intervals requested")
    result = xr.concat(results, dim="time")
    result.time.attrs["bounds"] = "time_bounds"
    return result


def native_coordinates(source: xr.Dataset) -> dict[str, xr.DataArray]:
    import pyproj  # noqa: PLC0415 - writer only; ``native`` extra

    if source.attrs.get("MAP_PROJ") != 3:
        raise ValueError("This GAR writer currently requires the native Mercator projection")
    proj_string = (
        f"+proj=merc +lat_ts={source.attrs['TRUELAT1']} +lon_0={source.attrs['STAND_LON']} "
        "+a=6370000 +b=6370000 +units=m +no_defs"
    )
    projection = pyproj.CRS.from_proj4(proj_string)
    lon, lat = source.XLONG.isel(time=0), source.XLAT.isel(time=0)
    transform = pyproj.Transformer.from_crs(projection.geodetic_crs, projection, always_xy=True)
    x, y = transform.transform(lon.values, lat.values)
    coords = {
        "x": xr.DataArray(
            np.median(x, axis=0),
            dims="west_east",
            attrs={"units": "m", "standard_name": "projection_x_coordinate"},
        ),
        "y": xr.DataArray(
            np.median(y, axis=1),
            dims="south_north",
            attrs={"units": "m", "standard_name": "projection_y_coordinate"},
        ),
        "lon": xr.DataArray(lon.values, dims=lon.dims, attrs={"units": "degrees_east"}),
        "lat": xr.DataArray(lat.values, dims=lat.dims, attrs={"units": "degrees_north"}),
        "crs": xr.DataArray(
            0,
            attrs={
                **projection.to_cf(),
                "spatial_ref": projection.to_wkt(),
                "pyproj_srs": proj_string,
            },
        ),
    }
    if "ZNU" in source:
        eta = source.ZNU.isel(time=0, drop=True) if "time" in source.ZNU.dims else source.ZNU
        coords["eta"] = xr.DataArray(
            eta.values,
            dims="bottom_top",
            attrs={"units": "1", "positive": "down", "long_name": "WRF mass-level eta coordinate"},
        )
    if "ZS" in source:
        soil = source.ZS.isel(time=0, drop=True) if "time" in source.ZS.dims else source.ZS
        coords["soil_depth"] = xr.DataArray(
            soil.values,
            dims="soil_layers_stag",
            attrs={"units": "m", "positive": "down", "long_name": "WRF soil-layer centre depth"},
        )
    return coords


def _render(  # noqa: PLR0913
    dataset: xr.Dataset,
    destination: Path,
    domain: int,
    label: str,
    frequency: str,
    source: xr.Dataset | None,
) -> list[tuple[Path, xr.Dataset, dict]]:
    """Plan every product of a set (path, dataset, encoding) without writing anything."""
    if domain not in DOMAIN_NAMES or frequency not in ["h", "d", "m", "y"]:
        raise ValueError("Unsupported GAR domain/frequency")
    if not label or any(char not in "0123456789-" for char in label):
        raise ValueError("Period label must be numeric ISO date or year")
    name = DOMAIN_NAMES[domain]
    coordinates = (
        native_coordinates(source)
        if source is not None
        else {k: dataset[k] for k in ["x", "y", "lon", "lat", "crs"] if k in dataset}
    )
    planned = []
    for variable, value in dataset.data_vars.items():
        variable = str(variable)
        if variable in ["time_bounds", "crs"] or variable.endswith("_valid_fraction"):
            continue
        main_variable = dataset.attrs.get("VARNAME")
        if variable in INVERSION_COMPANIONS and main_variable and main_variable != variable:
            continue
        category = (
            "3d_press"
            if "pressure" in value.dims
            else "3d_soil"
            if "soil_layers_stag" in value.dims
            else "3d_eta"
            if "bottom_top" in value.dims
            else "3d_inversion"
            if "inversion_layer" in value.dims
            else "2d"
        )
        path = (
            destination
            / name
            / frequency
            / category
            / f"GAR_{name}_{frequency}_{category}_{variable}_{label}.nc"
        )
        applicable = {
            key: coord for key, coord in coordinates.items() if set(coord.dims).issubset(value.dims)
        }
        product = value.to_dataset(name=variable).assign_coords(applicable)
        if variable in INVERSION_DESCRIPTORS:
            for companion in INVERSION_COMPANIONS:
                if companion in dataset:
                    product[companion] = dataset[companion]
        if "time_bounds" in dataset:
            product["time_bounds"] = dataset.time_bounds
        completeness = f"{variable}_valid_fraction"
        if completeness in dataset:
            product[completeness] = dataset[completeness]
        product.attrs = {
            **dataset.attrs,
            "DOMAIN": name,
            "VARNAME": variable,
            "PRODUCT_LEVEL": frequency,
            "time_zone": "UTC",
            "Conventions": "CF-1.10",
            "product_status": "candidate",
        }
        if "crs" in coordinates:
            product[variable].attrs["grid_mapping"] = "crs"
            product.attrs["pyproj_srs"] = coordinates["crs"].attrs["pyproj_srs"]
        for key, item in list(product.attrs.items()):
            if isinstance(item, (dict, list, bool)):
                product.attrs[key] = json.dumps(item)
        dims = [
            d
            for d in [
                "time",
                "bottom_top",
                "pressure",
                "soil_layers_stag",
                "inversion_layer",
                "south_north",
                "west_east",
            ]
            if d in product[variable].dims
        ]
        product[variable] = product[variable].transpose(*dims)
        encoding = {variable: {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32"}}
        if "time" in product:
            encoding["time"] = {
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
            }
            if "time_bounds" in product:
                encoding["time_bounds"] = encoding["time"].copy()
        planned.append((path, product, encoding))
    return planned


def write_products(  # noqa: PLR0913
    dataset: xr.Dataset,
    destination: Path,
    domain: int,
    label: str,
    frequency: str = "h",
    source: xr.Dataset | None = None,
    *,
    provenance: dict | None = None,
    retry: Literal["refuse", "verify-identical"] = "refuse",
) -> list[Path]:
    """Write one variable per NetCDF4 product candidate; explicit period label.

    Every product is published through :func:`darwin.publisher.immutable.staged_artifact`.
    With ``provenance`` (``{builder, sources, recipe?, parameters?}``, as
    :func:`derive_products` and :func:`aggregate_products` pass it) each product
    gets a ``darwin.artifact-identity/1`` record and can be bound by a handoff;
    without it products are unbound candidates. The whole set is planned and,
    under ``retry="refuse"``, every target checked before the first write, so an
    existing product or an invalid variable leaves nothing new behind. A crash
    mid-set is completed by rerunning with ``retry="verify-identical"``:
    products are written deterministically, so finished ones match byte for byte.
    """
    planned = _render(dataset, Path(destination), domain, label, frequency, source)
    if retry == "refuse":
        for path, _, _ in planned:
            for target in (path, identity_path(path)):
                if os.path.lexists(target):
                    raise FileExistsError(f"Product candidate already exists: {target}")
    for path, product, encoding in planned:
        with staged_artifact(path, provenance=provenance, retry=retry) as staged:
            product.to_netcdf(staged, engine="netcdf4", encoding=encoding)
    return [path for path, _, _ in planned]


def _software() -> dict[str, str]:
    try:
        darwin_version = version("darwin")
    except PackageNotFoundError:  # pragma: no cover - source tree without metadata
        darwin_version = "unknown"
    return {"product_contract": CONTRACT, "darwin_version": darwin_version}


def derive_products(  # noqa: PLR0913
    source: Path,
    destination: Path,
    domain: int,
    variables: list[str] | None = None,
    native_library: Path | None = None,
    pressure_levels: list[float] | None = None,
    label: str | None = None,
) -> list[Path]:
    """Read one WRF history file, derive diagnostics and write one candidate per variable.

    Same as ``python -m darwin.wrf_products derive``. Records the source byte
    identity (``source_sha256``), never its deployment path, and binds every
    product to it in an identity record. Run-layout checks such as CJC's
    retained-day rule belong to the executing system, not to this function.
    """
    source = Path(source)
    variables = variables or DEFAULT_VARIABLES
    with xr.open_dataset(source) as dataset:
        normalized = normalize_wrf(dataset)
        product = derive(
            normalized,
            variables,
            native_library,
            pressure_levels=pressure_levels,
        )
        digest = file_sha256(source)
        product.attrs["source_sha256"] = digest
        software = _software()
        product.attrs.update(software)
        label = label or str(product.time.values[0].astype("datetime64[D]"))
        provenance = {
            "builder": "darwin.wrf_products.derive_products",
            "sources": [{"id": source.name, "sha256": digest}],
            "recipe": {
                "operation": "derive",
                "definition_version": DEFINITION_VERSION,
                "variables": list(variables),
                "pressure_levels": pressure_levels,
                "native_library": native_library is not None,
            },
            "parameters": {"domain": domain, "label": label},
            "software": software,
        }
        return write_products(
            product, Path(destination), domain, label, source=normalized, provenance=provenance
        )


def aggregate_products(  # noqa: PLR0913
    sources: list[Path],
    destination: Path,
    domain: int,
    start: str,
    end: str,
    frequency: str,
    cadence_minutes: int,
) -> list[Path]:
    """Strictly aggregate native fragments of one variable and write the candidates.

    Same as ``python -m darwin.wrf_products aggregate``. The products name every
    fragment by exact sha256, in the given order, plus the aggregation recipe;
    attributes that differ between fragments (such as each fragment's own source
    identity) are dropped rather than inherited from the first.
    """
    sources = [Path(path) for path in sources]
    fragments = [{"id": path.name, "sha256": file_sha256(path)} for path in sources]
    recipe = {
        "operation": "aggregate",
        "start": start,
        "end": end,
        "frequency": frequency,
        "cadence_minutes": cadence_minutes,
    }
    opened = [xr.open_dataset(path) for path in sources]
    try:
        combined = xr.concat(
            opened,
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="equals",
            combine_attrs="drop_conflicts",
        ).sortby("time")
        reduced = aggregate(combined, start, end, frequency, cadence_minutes)
        coordinates = {k: combined[k] for k in ["x", "y", "lon", "lat", "crs"] if k in combined}
        reduced = reduced.assign_coords(coordinates)
        software = _software()
        reduced.attrs.update(software)
        reduced.attrs["sources"] = fragments
        reduced.attrs["recipe"] = recipe
        provenance = {
            "builder": "darwin.wrf_products.aggregate_products",
            "sources": fragments,
            "recipe": recipe,
            "parameters": {"domain": domain},
            "software": software,
        }
        return write_products(
            reduced,
            Path(destination),
            domain,
            start,
            PRODUCT_LEVELS[frequency],
            provenance=provenance,
        )
    finally:
        for dataset in opened:
            dataset.close()


def _entry(name: str, units: str | None, aggregation: str, **extra: object) -> dict:
    return {"name": name, "units": units, "aggregation": aggregation, **extra}


def inventory() -> dict:
    """Machine-readable diagnostic/aggregation inventory, shared by Python and CLI."""
    from darwin.native_diagnostics import BUILD_CONTRACT, SOURCE  # noqa: PLC0415

    provenance = json.loads((SOURCE / "provenance.json").read_text())
    accumulated = ["prcp_c", "prcp_nc", "prcp_fr", "snowfall", "graupel", "hail", "sfroff"]
    diagnostics = [
        _entry(name, "source units", "mean", kind="direct", source=field)
        for name, field in DIRECT.items()
    ]
    diagnostics += [
        _entry("u10", "m s-1", "mean", kind="earth-relative wind"),
        _entry("v10", "m s-1", "mean", kind="earth-relative wind"),
        _entry("ws10", "m s-1", "mean", kind="earth-relative wind speed"),
        _entry("u", "m s-1", "mean", kind="earth-relative wind"),
        _entry("v", "m s-1", "mean", kind="earth-relative wind"),
        _entry("ws", "m s-1", "mean", kind="earth-relative wind speed"),
        _entry("w", "m s-1", "mean", kind="destaggered"),
        _entry("theta", "K", "mean", kind="state"),
        _entry("temperature", "K", "mean", kind="state"),
        _entry("pressure", "hPa", "mean", kind="state"),
        _entry("geopotential", "m2 s-2", "mean", kind="state"),
        _entry("height_agl", "m", "mean", kind="state"),
        _entry("qliquid", "kg kg-1", "mean", kind="constituent sum"),
        _entry("qsolid", "kg kg-1", "mean", kind="constituent sum"),
        _entry("col_<x>", "kg m-2", "mean", kind="hybrid dry-mass water path"),
        _entry("flux_int_<q>_<u|v|w|hor>_2d", "kg m-1 s-1", "mean", kind="integrated flux"),
        _entry("flux_<q>_<u|v|w>_3d", "kg m-2 s-1", "mean", kind="dry-air-density flux"),
        _entry("prcp", "mm", "sum", kind="deaccumulated amount"),
        _entry("prcp_rate", "mm h-1", "mean", kind="deaccumulated rate"),
        *(_entry(name, "mm", "sum", kind="deaccumulated amount") for name in accumulated),
        _entry("udroff", "mm", "sum", kind="deaccumulated amount"),
        _entry("potevap", "mm", "sum", kind="signed deaccumulated amount"),
        _entry("et", "mm", "sum", kind="signed deaccumulated amount"),
        _entry("qfx", "kg m-2", "sum", kind="right-endpoint rectangle"),
        _entry("swup", "W m-2", "mean", kind="radiation"),
        _entry("lwup", "W m-2", "mean", kind="radiation"),
        _entry("lwdown", "W m-2", "mean", kind="radiation"),
        _entry("netrad", "W m-2", "mean", kind="radiation, positive downward"),
        _entry("low_cloud_fraction", "1", "mean", kind="max-overlap below 3 km AGL"),
        _entry("t2eta", "K", "mean", kind="2 m extrapolation"),
        _entry("t2pbl", "K", "mean", kind="2 m extrapolation"),
        _entry(
            "inversion",
            "m, K, K m-1, 1",
            "conditional-mean-when-inversion-present",
            kind="adjacent-positive-dT-dz-v1",
        ),
        _entry("mcape", "J kg-1", "mean", kind="native", native=True),
        _entry("mcin", "J kg-1", "mean", kind="native, positive magnitude", native=True),
        _entry("lcl", "m", "mean", kind="native, AGL", native=True),
        _entry("lfc", "m", "mean", kind="native, AGL", native=True),
        _entry("slp", "hPa", "mean", kind="native", native=True),
    ]
    return {
        "contract": CONTRACT,
        "definition_version": DEFINITION_VERSION,
        "default_variables": list(DEFAULT_VARIABLES),
        "pressure_interpolation": [
            "theta",
            "temperature",
            "qvapor",
            "qliquid",
            "qsolid",
            "u",
            "v",
            "w",
            "ws",
            "geopotential",
        ],
        "diagnostics": diagnostics,
        "time": "right-closed (t[i-1], t[i]]; first record is the boundary and is dropped",
        "aggregation": {
            "frequencies": ["D", "MS", "YS"],
            "completeness_policy": "require-all-expected-timestamps",
            "interval": "(start, end]",
        },
        "native": {
            "build_contract": BUILD_CONTRACT,
            "upstream": provenance["revision"],
            "release": provenance["release"],
            "files": {record["file"]: record["sha256"] for record in provenance["files"]},
        },
        "product_status": "candidate",
    }


def _error_code(error: Exception) -> str:
    if isinstance(error, ImportError):
        return "missing_dependency"
    return "destination_exists" if isinstance(error, FileExistsError) else "invalid_input"


def main(argv: list[str] | None = None) -> None:
    """``python -m darwin.wrf_products {inventory,derive,aggregate,build-native}``."""
    import argparse  # noqa: PLC0415
    import sys  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory")
    generate = commands.add_parser("derive")
    generate.add_argument("source", type=Path)
    generate.add_argument("--destination", type=Path, required=True)
    generate.add_argument("--domain", type=int, choices=[1, 2], required=True)
    generate.add_argument("--variables", nargs="+")
    generate.add_argument("--native-library", type=Path)
    generate.add_argument("--pressure-levels", type=float, nargs="+")
    generate.add_argument("--label")
    reduce = commands.add_parser("aggregate")
    reduce.add_argument("sources", type=Path, nargs="+")
    reduce.add_argument("--destination", type=Path, required=True)
    reduce.add_argument("--domain", type=int, choices=[1, 2], required=True)
    reduce.add_argument("--start", required=True)
    reduce.add_argument("--end", required=True)
    reduce.add_argument("--frequency", choices=["D", "MS", "YS"], required=True)
    reduce.add_argument("--cadence-minutes", type=int, required=True)
    build = commands.add_parser("build-native")
    build.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            report = inventory()
        elif args.command == "derive":
            paths = derive_products(
                args.source,
                args.destination,
                args.domain,
                args.variables,
                args.native_library,
                args.pressure_levels,
                args.label,
            )
            report = {"contract": CONTRACT, "products": [str(path) for path in paths]}
        elif args.command == "aggregate":
            paths = aggregate_products(
                args.sources,
                args.destination,
                args.domain,
                args.start,
                args.end,
                args.frequency,
                args.cadence_minutes,
            )
            report = {"contract": CONTRACT, "products": [str(path) for path in paths]}
        else:
            from darwin.native_diagnostics import build_native, verify_native  # noqa: PLC0415

            library = build_native(args.destination)
            report = {"library": str(library), "receipt": verify_native(library)}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
        if isinstance(error, ImportError):
            message = f"{message}; install the Darwin 'native' extra for this product writer"
        print(json.dumps({"code": _error_code(error), "message": str(message)}), file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
