"""DARWIN-derived native-hourly 2 m relative-humidity product.

Algorithm GAR-RH2M-MK05-LIQUID-1.0.0: relative humidity with respect to
liquid water, derived at each native timestamp from WRF Q2 (2 m
water-vapour mixing ratio), T2, and PSFC, before any aggregation:

    e = PSFC * Q2 / (0.622 + Q2)
    RH = 100 * e / esw(T2)

with Murphy & Koop (2005) saturation vapour pressure over liquid water.
Supersaturated values are retained with a flag; invalid inputs go
missing. Nothing is clipped.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from darwin.publisher.candidate import Candidate, run_manifest
from darwin.publisher.manifest import Manifest

if TYPE_CHECKING:
    from darwin.publisher.catalogue import Catalogue

RH_ALGORITHM_ID = "GAR-RH2M-MK05-LIQUID-1.0.0"
RH_REFERENCE_SURFACE = "liquid water"
RH_PRODUCT_ID = "GAR_d02km_h_2d_rh2"

#: Mass-ratio constant epsilon = Rd / Rv.
EPSILON = 0.622

FLAG_OK = 0
FLAG_SUPERSATURATED = 1
FLAG_INVALID = 2

RH_PROVENANCE_NOTE = (
    "DARWIN-derived 2 m relative humidity with respect to liquid water "
    f"({RH_ALGORITHM_ID}): e = PSFC*Q2/(0.622+Q2), RH = 100*e/esw(T2) with "
    "Murphy & Koop (2005) saturation vapour pressure, derived at each "
    "native timestamp before aggregation. Supersaturation is retained with "
    "a flag and invalid inputs go missing; values are never clipped. This "
    "differs from the WRF AFWA diagnostic, which uses a Tetens/Magnus "
    "approximation for saturation specific humidity and clamps its output "
    "to the 1-100 % range."
)


def saturation_vapor_pressure(t_kelvin: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over liquid water in Pa (Murphy & Koop 2005, Eq. 10)."""
    t = np.asarray(t_kelvin, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ln_t = np.log(t)
        tanh_arg = np.tanh(0.0415 * (t - 218.8))
        ln_esw = (
            54.842763
            - 6763.22 / t
            - 4.210 * ln_t
            + 0.000367 * t
            + tanh_arg
            * (
                53.878
                - 1331.22 / t
                - 9.44523 * ln_t
                + 0.014025 * t
            )
        )
        return np.exp(ln_esw)


def derive_rh2m(
    q2: np.ndarray, t2: np.ndarray, psfc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Derive RH in % and per-element quality flags.

    Flags: 0 ok, 1 supersaturated (value retained, not clipped),
    2 invalid input (NaN output). Invalid means missing/non-finite
    inputs, Q2 < 0, PSFC <= 0, or non-physical esw (<= 0 or >= PSFC).
    """
    q = np.asarray(q2, dtype=np.float64)
    t = np.asarray(t2, dtype=np.float64)
    p = np.asarray(psfc, dtype=np.float64)
    esw = saturation_vapor_pressure(t)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        vapour_pressure = p * q / (EPSILON + q)
        rh = 100.0 * vapour_pressure / esw
    invalid = (
        ~np.isfinite(q)
        | ~np.isfinite(t)
        | ~np.isfinite(p)
        | (q < 0.0)
        | (p <= 0.0)
        | ~np.isfinite(esw)
        | (esw <= 0.0)
        | (esw >= p)
    )
    rh = np.where(invalid, np.nan, rh)
    flags = np.where(invalid, FLAG_INVALID, FLAG_OK).astype(np.int8)
    supersaturated = ~invalid & np.isfinite(rh) & (rh > 100.0)
    flags = np.where(supersaturated, FLAG_SUPERSATURATED, flags).astype(np.int8)
    return rh, flags


def rh2_product_attrs() -> dict[str, str]:
    """Immutable metadata attached to the published rh2 GAR product."""
    return {
        "long_name": "2 m relative humidity with respect to liquid water",
        "units": "%",
        "algorithm_id": RH_ALGORITHM_ID,
        "reference_surface": RH_REFERENCE_SURFACE,
        "inputs": "Q2 water-vapour mixing ratio (kg kg-1); T2 (K); PSFC (Pa)",
        "provenance_note": RH_PROVENANCE_NOTE,
    }


def register_rh_candidate(manifest: Manifest, catalogue: Catalogue) -> Candidate:
    """Register the manifest's rh2 product as a candidate bound to its revision."""
    rh_specs = tuple(
        spec
        for spec in manifest.products
        if spec.kind == "rh2" or spec.product_id == RH_PRODUCT_ID
    )
    if not rh_specs:
        raise ValueError("manifest requests no rh2 product")
    scoped = replace(manifest, products=rh_specs)
    candidate = run_manifest(scoped, catalogue)
    if candidate.product_id != rh_specs[0].product_id:
        raise AssertionError("rh2 candidate registration mismatch")
    return candidate
