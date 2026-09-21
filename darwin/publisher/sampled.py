"""GAR station-sampled products in three methods (D6).

Each sampled series records its method and sampled grid coordinates, so
the dashboard can switch among nearest, prescribed-point and bilinear
series without touching source data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

#: Sampling methods with a published immutable series per station.
METHODS = ("nearest", "prescribed", "bilinear")


@dataclass(frozen=True)
class SampledSeries:
    """One immutable native-cadence GAR series sampled at a station."""

    method: str
    grid_x: int
    grid_y: int
    grid_lon: float
    grid_lat: float
    values: tuple[float, ...]


def sample(
    lons: Sequence[float],
    lats: Sequence[float],
    values: Sequence[Sequence[Sequence[float]]],
    station_lon: float,
    station_lat: float,
    method: str,
    prescribed_points: Mapping[tuple[float, float], tuple[int, int]] | None = None,
) -> SampledSeries:
    """Sample a native-cadence grid series at a station location.

    ``prescribed`` uses the approved prescribed-point register entry for
    the station coordinates; a missing entry raises LookupError so the
    option stays disabled instead of silently falling back to nearest.
    """
    if method not in METHODS:
        raise ValueError(f"unknown sampling method: {method!r}")
    if method == "prescribed":
        return _prescribed(
            lons, lats, values, station_lon, station_lat, prescribed_points or {}
        )
    if method == "bilinear":
        return _bilinear(lons, lats, values, station_lon, station_lat)
    grid_x = min(range(len(lons)), key=lambda x: abs(lons[x] - station_lon))
    grid_y = min(range(len(lats)), key=lambda y: abs(lats[y] - station_lat))
    return SampledSeries(
        method=method,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_lon=lons[grid_x],
        grid_lat=lats[grid_y],
        values=tuple(step[grid_y][grid_x] for step in values),
    )


def _prescribed(
    lons: Sequence[float],
    lats: Sequence[float],
    values: Sequence[Sequence[Sequence[float]]],
    station_lon: float,
    station_lat: float,
    prescribed_points: Mapping[tuple[float, float], tuple[int, int]],
) -> SampledSeries:
    """Sample the registered prescribed cell; missing entries stay disabled."""
    try:
        grid_x, grid_y = prescribed_points[(station_lon, station_lat)]
    except KeyError:
        raise LookupError(
            "no approved prescribed sampling point for this station: "
            "the option stays disabled, never a silent nearest fallback"
        ) from None
    return SampledSeries(
        method="prescribed",
        grid_x=grid_x,
        grid_y=grid_y,
        grid_lon=lons[grid_x],
        grid_lat=lats[grid_y],
        values=tuple(step[grid_y][grid_x] for step in values),
    )


def _bilinear(
    lons: Sequence[float],
    lats: Sequence[float],
    values: Sequence[Sequence[Sequence[float]]],
    station_lon: float,
    station_lat: float,
) -> SampledSeries:
    """Bilinear interpolation; the recorded cell is the enclosing lower corner."""
    x0 = max(0, min(len(lons) - 2, _lower(lons, station_lon)))
    y0 = max(0, min(len(lats) - 2, _lower(lats, station_lat)))
    x1, y1 = x0 + 1, y0 + 1
    fx = (station_lon - lons[x0]) / (lons[x1] - lons[x0])
    fy = (station_lat - lats[y0]) / (lats[y1] - lats[y0])
    series = tuple(
        values[t][y0][x0] * (1 - fx) * (1 - fy)
        + values[t][y0][x1] * fx * (1 - fy)
        + values[t][y1][x0] * (1 - fx) * fy
        + values[t][y1][x1] * fx * fy
        for t in range(len(values))
    )
    return SampledSeries(
        method="bilinear",
        grid_x=x0,
        grid_y=y0,
        grid_lon=lons[x0],
        grid_lat=lats[y0],
        values=series,
    )


def _lower(coords: Sequence[float], point: float) -> int:
    """Return the largest index with coord <= point (may be out of range)."""
    index = 0
    for i, coord in enumerate(coords):
        if coord <= point:
            index = i
    return index
