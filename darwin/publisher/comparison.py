"""Comparison engine and v1 variable specifications (D7).

Declared Comparison variable specifications execute over Comparison
intervals to produce Comparison results with completeness filtering,
exclusions and full statistics.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ComparisonSpec:
    """One declared GAR variable / AWS observation field pairing."""

    gar_variable: str
    aws_field: str
    units: str
    conversion: str
    aggregation: str


#: The nine v1 declared pairings (darwin#9 acceptance 1).
COMPARISON_SPECS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec("t2", "T", "C", "K to C", "mean"),
    ComparisonSpec("rh2", "RH", "%", "none", "mean"),
    ComparisonSpec("swdown", "SLR", "W m-2", "none", "mean"),
    ComparisonSpec(
        "wind10",
        "WS/WD",
        "m s-1/degrees",
        "grid to earth-relative",
        "vector",
    ),
    ComparisonSpec("psfc", "Pabs", "hPa", "Pa to hPa", "mean"),
    ComparisonSpec("prcp", "PCP_tot_bucket", "mm", "rate integrated to depth", "sum"),
    ComparisonSpec("prcp", "PCP_diff_radar", "mm", "rate integrated to depth", "sum"),
    ComparisonSpec("prcp", "PCP_acoustic", "mm", "rate integrated to depth", "sum"),
    ComparisonSpec("tslb", "ST", "C", "K to C", "mean"),
)

# The tslb pairing compares the uppermost layer with a visible
# unverified-vertical-comparability note; FOG, Vwc and WSmax stay
# observation-only with no GAR pairing.


@dataclass(frozen=True)
class ComparisonResult:
    """Paired values, exclusions and statistics for one spec over one interval.

    Sample variance and Pearson correlation are ``None`` (not defined)
    with fewer than two eligible pairs; every other scalar needs at
    least one pair.
    """

    gar_variable: str
    aws_field: str
    paired_count: int
    excluded_count: int
    gar_mean: float | None
    obs_mean: float | None
    gar_variance: float | None
    obs_variance: float | None
    bias: float | None
    mae: float | None
    rmse: float | None
    correlation: float | None
    cadence: str = "hourly"


#: Comparison cadences with agreed completeness rules.
CADENCES = ("hourly", "daily", "monthly", "annual")


def compare(
    gar_variable: str,
    aws_field: str,
    gar_values: Sequence[float],
    obs_values: Sequence[float],
    excluded_count: int = 0,
    cadence: str = "hourly",
) -> ComparisonResult:
    """Compare paired GAR and observation values for one variable spec."""
    if cadence not in CADENCES:
        raise ValueError(f"unknown comparison cadence: {cadence!r}")
    pairs = list(zip(gar_values, obs_values))
    count = len(pairs)
    if count == 0:
        return ComparisonResult(
            gar_variable, aws_field, 0, excluded_count,
            None, None, None, None, None, None, None, None, cadence,
        )
    gar_mean = statistics.fmean(g for g, _ in pairs)
    obs_mean = statistics.fmean(o for _, o in pairs)
    errors = [g - o for g, o in pairs]
    result_variance = count >= 2
    return ComparisonResult(
        gar_variable=gar_variable,
        aws_field=aws_field,
        paired_count=count,
        excluded_count=excluded_count,
        gar_mean=gar_mean,
        obs_mean=obs_mean,
        gar_variance=statistics.variance((g for g, _ in pairs)) if result_variance else None,
        obs_variance=statistics.variance((o for _, o in pairs)) if result_variance else None,
        bias=statistics.fmean(errors),
        mae=statistics.fmean(abs(e) for e in errors),
        rmse=math.sqrt(statistics.fmean(e * e for e in errors)),
        correlation=_pearson(pairs) if result_variance else None,
        cadence=cadence,
    )


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    """Pearson correlation, or None when either series has no spread."""
    gar = [g for g, _ in pairs]
    obs = [o for _, o in pairs]
    gar_dev = [g - statistics.fmean(gar) for g in gar]
    obs_dev = [o - statistics.fmean(obs) for o in obs]
    denom = math.sqrt(sum(g * g for g in gar_dev) * sum(o * o for o in obs_dev))
    if denom == 0.0:
        return None
    return sum(g * o for g, o in zip(gar_dev, obs_dev)) / denom


#: Minimum ten-minute readings per hour for a valid hourly observation value.
HOURLY_READINGS_REQUIRED = 5


def complete_hourly(readings: Sequence[float | None]) -> float | None:
    """Aggregate six ten-minute readings to an hourly mean.

    Returns None (excluded) with fewer than five valid readings.
    """
    valid = [r for r in readings if r is not None]
    if len(valid) < HOURLY_READINGS_REQUIRED:
        return None
    return statistics.fmean(valid)


#: Minimum share of eligible hourly values for a valid daily/monthly/annual value.
COARSER_COVERAGE_REQUIRED = 0.9


def complete_coarser(
    hourly_values: Sequence[float | None], expected: int
) -> float | None:
    """Aggregate hourly values; None (excluded) below 90% coverage."""
    valid = [v for v in hourly_values if v is not None]
    if not valid or len(valid) / expected < COARSER_COVERAGE_REQUIRED:
        return None
    return statistics.fmean(valid)


#: Wet-interval threshold in mm at the selected cadence.
WET_THRESHOLD_MM = 0.0


def precip_depth(hourly_rates_mm_per_h: Sequence[float]) -> tuple[float, ...]:
    """Integrate valid hourly GAR rates over one hour into depths in mm."""
    return tuple(rate * 1.0 for rate in hourly_rates_mm_per_h)


@dataclass(frozen=True)
class PrecipExtras:
    """Precipitation totals, wet-interval frequency and contingency table."""

    total_gar: float
    total_obs: float
    wet_frequency_gar: float
    wet_frequency_obs: float
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int


def precip_extras(
    gar_depths: Sequence[float], obs_depths: Sequence[float]
) -> PrecipExtras:
    """Totals, wet frequency and zero/non-zero contingency for paired depths."""
    gar_wet = [d > WET_THRESHOLD_MM for d in gar_depths]
    obs_wet = [d > WET_THRESHOLD_MM for d in obs_depths]
    count = len(gar_depths)
    return PrecipExtras(
        total_gar=sum(gar_depths),
        total_obs=sum(obs_depths),
        wet_frequency_gar=sum(gar_wet) / count,
        wet_frequency_obs=sum(obs_wet) / count,
        hits=sum(1 for g, o in zip(gar_wet, obs_wet) if g and o),
        misses=sum(1 for g, o in zip(gar_wet, obs_wet) if g and not o),
        false_alarms=sum(1 for g, o in zip(gar_wet, obs_wet) if not g and o),
        correct_negatives=sum(1 for g, o in zip(gar_wet, obs_wet) if not g and not o),
    )


#: Minimum GAR and AWS wind speed for an eligible direction pair.
CALM_THRESHOLD_M_S = 0.5


@dataclass(frozen=True)
class WindResult:
    """Circular direction diagnostics; calm pairs are excluded and counted."""

    paired_count: int
    excluded_calm: int
    gar_circ_mean: float | None
    obs_circ_mean: float | None
    signed_bias: float | None
    maae: float | None


def _circ_mean(degrees: Sequence[float]) -> float:
    """Circular mean in [0, 360)."""
    radians = [math.radians(d) for d in degrees]
    mean = math.degrees(
        math.atan2(
            statistics.fmean(math.sin(r) for r in radians),
            statistics.fmean(math.cos(r) for r in radians),
        )
    )
    # round() before the modulo so -epsilon maps to 0.0, never 360.0.
    return round(mean, 9) % 360.0


def _wrap_signed(diff: float) -> float:
    """Wrap an angular difference to (-180, 180]."""
    return (diff + 180.0) % 360.0 - 180.0


def circular_wind(
    gar_dir: Sequence[float],
    gar_speed: Sequence[float],
    obs_dir: Sequence[float],
    obs_speed: Sequence[float],
) -> WindResult:
    """Direction diagnostics over eligible pairs; calm pairs excluded."""
    eligible = [
        (g, o)
        for g, gs, o, os in zip(gar_dir, gar_speed, obs_dir, obs_speed)
        if gs >= CALM_THRESHOLD_M_S and os >= CALM_THRESHOLD_M_S
    ]
    excluded = len(gar_dir) - len(eligible)
    if not eligible:
        return WindResult(0, excluded, None, None, None, None)
    gar_angles = [g for g, _ in eligible]
    obs_angles = [o for _, o in eligible]
    diffs = [_wrap_signed(o - g) for g, o in eligible]
    return WindResult(
        paired_count=len(eligible),
        excluded_calm=excluded,
        gar_circ_mean=_circ_mean(gar_angles),
        obs_circ_mean=_circ_mean(obs_angles),
        signed_bias=statistics.fmean(diffs),
        maae=statistics.fmean(abs(d) for d in diffs),
    )
