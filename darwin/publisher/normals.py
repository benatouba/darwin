"""Monthly climate-normal computation.

Migrated from the verified ``gar-for-rob`` algorithm; formulas are
preserved verbatim while provenance is re-issued under
``darwin.publisher`` with source attributes retained.

Conventions (see spec):
- ``t2_max`` / ``t2_min``: mean of daily maximum/minimum 2 m temperature,
  converted from K to C and rounded to 2 decimals.
- ``prcp_sum``: monthly rate multiplied by days-in-month x 24 (leap
  February uses 696 hours), then averaged across valid years.
- Missing values are excluded per grid cell via valid-value counts.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import xarray as xr

PROCESSING_METHOD = (
    "t2: daily maximum and minimum from hourly 2m temperature, averaged by calendar month; "
    "prcp: monthly mean rate multiplied by calendar hours, averaged by calendar month"
)
CREATOR = "darwin.publisher"
DERIVED_PRODUCTS = ("t2_max", "t2_min", "prcp_sum")


@dataclass(frozen=True)
class NormalPeriod:
    """Inclusive reference period, e.g. 1991-2020."""

    start_year: int
    end_year: int

    def __post_init__(self) -> None:
        if self.start_year > self.end_year:
            raise ValueError(
                f"normal period starts after it ends: {self.start_year}-{self.end_year}"
            )

    @property
    def label(self) -> str:
        return f"{self.start_year}-01-01 through {self.end_year}-12-31"


@dataclass(frozen=True)
class YearInputs:
    """One year of validated source arrays for a single domain.

    ``temperature`` is the hourly (or 3-hourly) 2 m series in K;
    ``precipitation`` is the twelve monthly mean rates in mm h-1,
    following the gar-for-rob source layout.
    """

    temperature: xr.DataArray
    precipitation: xr.DataArray
    source_attributes: dict[str, object]
    temperature_attributes: dict[str, object]
    precipitation_attributes: dict[str, object]
    coordinates: dict[str, xr.DataArray]


@dataclass(frozen=True)
class AnnualStatistics:
    maximum_sum: xr.DataArray
    maximum_count: xr.DataArray
    minimum_sum: xr.DataArray
    minimum_count: xr.DataArray
    precipitation_sum: xr.DataArray
    precipitation_count: xr.DataArray
    source_attributes: dict[str, object]
    temperature_attributes: dict[str, object]
    precipitation_attributes: dict[str, object]
    coordinates: dict[str, xr.DataArray]


@dataclass(frozen=True)
class DomainNormals:
    temperature_maximum: xr.DataArray
    temperature_minimum: xr.DataArray
    precipitation_total: xr.DataArray
    coordinates: dict[str, xr.DataArray]
    source_attributes: dict[str, object]
    temperature_attributes: dict[str, object]
    precipitation_attributes: dict[str, object]


def calculate_year(inputs: YearInputs) -> AnnualStatistics:
    daily_maximum = inputs.temperature.resample(time="1D").max(skipna=True)
    daily_minimum = inputs.temperature.resample(time="1D").min(skipna=True)
    maximum_sum, maximum_count = _monthly_sum_and_count(daily_maximum)
    minimum_sum, minimum_count = _monthly_sum_and_count(daily_minimum)
    monthly_total = inputs.precipitation * inputs.precipitation.time.dt.days_in_month * 24
    precipitation_sum, precipitation_count = _monthly_sum_and_count(monthly_total)
    return AnnualStatistics(
        maximum_sum=maximum_sum,
        maximum_count=maximum_count,
        minimum_sum=minimum_sum,
        minimum_count=minimum_count,
        precipitation_sum=precipitation_sum,
        precipitation_count=precipitation_count,
        source_attributes=dict(inputs.source_attributes),
        temperature_attributes=dict(inputs.temperature_attributes),
        precipitation_attributes=dict(inputs.precipitation_attributes),
        coordinates=dict(inputs.coordinates),
    )


class NormalAccumulator:
    """Year-by-year accumulation; never holds all years in memory."""

    def __init__(self) -> None:
        self._maximum_sum: xr.DataArray | None = None
        self._maximum_count: xr.DataArray | None = None
        self._minimum_sum: xr.DataArray | None = None
        self._minimum_count: xr.DataArray | None = None
        self._precipitation_sum: xr.DataArray | None = None
        self._precipitation_count: xr.DataArray | None = None
        self._grid_reference: xr.DataArray | None = None
        self._coordinates: dict[str, xr.DataArray] | None = None
        self._source_attributes: dict[str, object] | None = None
        self._temperature_attributes: dict[str, object] | None = None
        self._precipitation_attributes: dict[str, object] | None = None

    def add(self, statistics: AnnualStatistics) -> None:
        if self._grid_reference is None:
            self._grid_reference = statistics.maximum_sum
            self._coordinates = dict(statistics.coordinates)
            self._source_attributes = dict(statistics.source_attributes)
            self._temperature_attributes = dict(statistics.temperature_attributes)
            self._precipitation_attributes = dict(statistics.precipitation_attributes)
        else:
            _validate_spatial_shape(self._grid_reference, statistics.maximum_sum, "t2")
            assert self._source_attributes is not None
            _merge_attributes(self._source_attributes, statistics.source_attributes)
        _validate_spatial_shape(statistics.maximum_sum, statistics.minimum_sum, "t2")
        _validate_spatial_shape(statistics.maximum_sum, statistics.precipitation_sum, "prcp")
        assert self._coordinates is not None
        statistics = _with_canonical_coordinates(statistics, self._coordinates)

        if self._maximum_sum is None:
            self._maximum_sum = statistics.maximum_sum
            self._maximum_count = statistics.maximum_count
            self._minimum_sum = statistics.minimum_sum
            self._minimum_count = statistics.minimum_count
            self._precipitation_sum = statistics.precipitation_sum
            self._precipitation_count = statistics.precipitation_count
            return

        assert self._maximum_sum is not None
        assert self._maximum_count is not None
        assert self._minimum_sum is not None
        assert self._minimum_count is not None
        assert self._precipitation_sum is not None
        assert self._precipitation_count is not None
        self._maximum_sum += statistics.maximum_sum
        self._maximum_count += statistics.maximum_count
        self._minimum_sum += statistics.minimum_sum
        self._minimum_count += statistics.minimum_count
        self._precipitation_sum += statistics.precipitation_sum
        self._precipitation_count += statistics.precipitation_count

    def normals(self) -> DomainNormals:
        assert self._maximum_sum is not None
        assert self._maximum_count is not None
        assert self._minimum_sum is not None
        assert self._minimum_count is not None
        assert self._precipitation_sum is not None
        assert self._precipitation_count is not None
        assert self._coordinates is not None
        assert self._source_attributes is not None
        assert self._temperature_attributes is not None
        assert self._precipitation_attributes is not None
        return DomainNormals(
            temperature_maximum=(self._maximum_sum / self._maximum_count - 273.15).round(2),
            temperature_minimum=(self._minimum_sum / self._minimum_count - 273.15).round(2),
            precipitation_total=self._precipitation_sum / self._precipitation_count,
            coordinates=self._coordinates,
            source_attributes=self._source_attributes,
            temperature_attributes=self._temperature_attributes,
            precipitation_attributes=self._precipitation_attributes,
        )


def derive_products(
    normals: DomainNormals, period: NormalPeriod
) -> tuple[dict[str, xr.DataArray], dict[str, object]]:
    """Attach layer semantics and provenance to the three normal layers.

    Returns the three monthly layers plus the global publication attributes.
    """
    publication_attributes = _publication_attributes(normals.source_attributes, period)
    layers = {
        "t2_max": normals.temperature_maximum.assign_attrs(
            _derived_attributes(
                normals.temperature_attributes,
                long_name="Mean daily maximum 2m Temperature",
                units="C",
                agg_method="MEAN",
            )
        ),
        "t2_min": normals.temperature_minimum.assign_attrs(
            _derived_attributes(
                normals.temperature_attributes,
                long_name="Mean daily minimum 2m Temperature",
                units="C",
                agg_method="MEAN",
            )
        ),
        "prcp_sum": normals.precipitation_total.assign_attrs(
            _derived_attributes(
                normals.precipitation_attributes,
                long_name="Mean total precipitation",
                units="mm",
                agg_method="MEAN",
            )
        ),
    }
    return layers, publication_attributes


def _monthly_sum_and_count(values: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    groups = values.groupby("time.month")
    return groups.sum(dim="time", skipna=True).load(), groups.count(dim="time").load()


def _with_canonical_coordinates(
    statistics: AnnualStatistics, coordinates: dict[str, xr.DataArray]
) -> AnnualStatistics:
    def canonicalize(values: xr.DataArray) -> xr.DataArray:
        return values.reset_coords(drop=True).assign_coords(coordinates)

    return AnnualStatistics(
        maximum_sum=canonicalize(statistics.maximum_sum),
        maximum_count=canonicalize(statistics.maximum_count),
        minimum_sum=canonicalize(statistics.minimum_sum),
        minimum_count=canonicalize(statistics.minimum_count),
        precipitation_sum=canonicalize(statistics.precipitation_sum),
        precipitation_count=canonicalize(statistics.precipitation_count),
        source_attributes=statistics.source_attributes,
        temperature_attributes=statistics.temperature_attributes,
        precipitation_attributes=statistics.precipitation_attributes,
        coordinates=statistics.coordinates,
    )


def _validate_spatial_shape(reference: xr.DataArray, candidate: xr.DataArray, product: str) -> None:
    reference_dimensions = tuple(d for d in reference.dims if d != "month")
    candidate_dimensions = tuple(d for d in candidate.dims if d != "month")
    if reference_dimensions != candidate_dimensions:
        raise ValueError(
            f"incompatible spatial dimensions for {product}: "
            f"expected {reference_dimensions}, got {candidate_dimensions}"
        )
    for dimension in reference_dimensions:
        if reference.sizes[dimension] != candidate.sizes[dimension]:
            raise ValueError(
                f"incompatible {dimension} size for {product}: "
                f"expected {reference.sizes[dimension]}, got {candidate.sizes[dimension]}"
            )


def _merge_attributes(existing: dict[str, object], incoming: dict[str, object]) -> None:
    for name, value in incoming.items():
        if name not in existing:
            existing[name] = value
        elif existing[name] != value:
            existing[f"source_{name}"] = value


def _derived_attributes(
    source_attributes: dict[str, object], **derived: object
) -> dict[str, object]:
    attributes = _utf8_safe_attributes(source_attributes)
    for name, value in derived.items():
        if name in attributes and attributes[name] != value:
            attributes[f"source_{name}"] = attributes[name]
        attributes[name] = value
    return attributes


def _publication_attributes(
    source_attributes: dict[str, object], period: NormalPeriod
) -> dict[str, object]:
    attributes = _utf8_safe_attributes(source_attributes)
    created_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    creation_entry = f"{created_at}: {CREATOR} created monthly GAR climate-normal products"
    if "history" in attributes:
        attributes["history"] = f"{attributes['history']}\n{creation_entry}"
    else:
        attributes["history"] = creation_entry

    _set_derived_global_attribute(attributes, "climate_normal_period", period.label)
    _set_derived_global_attribute(attributes, "processing_method", PROCESSING_METHOD)
    _set_derived_global_attribute(attributes, "created_at", created_at)
    _set_derived_global_attribute(attributes, "creator", CREATOR)
    _set_derived_global_attribute(
        attributes,
        "software_versions",
        json.dumps(
            {
                "darwin": _package_version("darwin"),
                "numpy": _package_version("numpy"),
                "python": ".".join(map(str, sys.version_info[:3])),
                "xarray": _package_version("xarray"),
            },
            sort_keys=True,
        ),
    )
    return attributes


def _set_derived_global_attribute(
    attributes: dict[str, object], name: str, value: object
) -> None:
    if name in attributes and attributes[name] != value:
        attributes[f"source_{name}"] = attributes[name]
    attributes[name] = value


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _utf8_safe_attributes(attributes: dict[str, object]) -> dict[str, object]:
    return {name: _utf8_safe_value(value) for name, value in attributes.items()}


def _utf8_safe_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        try:
            return value.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        except UnicodeEncodeError:
            return value.encode("utf-8", "replace").decode("utf-8")
    return value
