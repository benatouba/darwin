"""Tests for the DARWIN-derived native-hourly relative-humidity product.

Algorithm GAR-RH2M-MK05-LIQUID-1.0.0: RH with respect to liquid water
from WRF Q2 mixing ratio, T2, and PSFC, using Murphy & Koop (2005)
saturation vapour pressure. Derived at each native timestamp before
aggregation; supersaturation retained with a flag; invalid inputs go
missing without clipping.
"""

import math

import numpy as np
import pytest

from darwin.publisher import humidity
from darwin.publisher.catalogue import Catalogue
from darwin.publisher.manifest import load_manifest


def tetens_esw_pa(t_k: float) -> float:
    """Independent Murray/Tetens formulation, used only as a cross-check."""
    t_c = t_k - 273.15
    return 610.78 * math.exp(17.27 * t_c / (t_c + 237.3))


def test_algorithm_identity_is_versioned():
    assert humidity.RH_ALGORITHM_ID == "GAR-RH2M-MK05-LIQUID-1.0.0"
    assert humidity.RH_REFERENCE_SURFACE == "liquid water"


def test_saturation_vapour_pressure_positive_and_increasing():
    temps = np.array([253.15, 263.15, 273.15, 283.15, 293.15, 303.15])
    esw = humidity.saturation_vapor_pressure(temps)
    assert np.all(np.isfinite(esw))
    assert np.all(esw > 0)
    assert np.all(np.diff(esw) > 0)


def test_saturation_vapour_pressure_matches_independent_formulation():
    # Murphy-Koop and Tetens agree closely near 0 C and room temperature.
    for t_k in (273.15, 293.15):
        mk = float(humidity.saturation_vapor_pressure(np.array([t_k]))[0])
        assert mk == pytest.approx(tetens_esw_pa(t_k), rel=0.01)


def test_vapour_pressure_formula_uses_mixing_ratio():
    # e = PSFC * Q2 / (0.622 + Q2); hand-computed for q2 = 0.01, psfc = 1e5.
    q2 = np.array([0.01])
    t2 = np.array([293.15])
    psfc = np.array([100000.0])
    rh, flags = humidity.derive_rh2m(q2, t2, psfc)
    expected_e = 100000.0 * 0.01 / (0.622 + 0.01)
    expected_rh = 100.0 * expected_e / tetens_esw_pa(293.15)
    assert rh[0] == pytest.approx(expected_rh, rel=0.01)
    assert flags[0] == humidity.FLAG_OK


def test_supersaturation_retained_with_flag_not_clipped():
    # q2 = 0.01 at 0 C gives e >> esw: value must survive, flagged.
    rh, flags = humidity.derive_rh2m(
        np.array([0.01]), np.array([273.15]), np.array([100000.0])
    )
    assert np.isfinite(rh[0])
    assert rh[0] > 100.0
    assert flags[0] == humidity.FLAG_SUPERSATURATED


@pytest.mark.parametrize(
    "q2,t2,psfc",
    [
        (-0.001, 293.15, 100000.0),  # negative moisture
        (0.01, 293.15, 0.0),  # non-positive pressure
        (0.01, 293.15, -500.0),  # negative pressure
        (float("nan"), 293.15, 100000.0),  # missing moisture
        (0.01, float("nan"), 100000.0),  # missing temperature
        (0.01, 300.0, 500.0),  # esw(300 K) exceeds ambient pressure
    ],
)
def test_invalid_inputs_go_missing_with_flag(q2, t2, psfc):
    rh, flags = humidity.derive_rh2m(
        np.array([q2]), np.array([t2]), np.array([psfc])
    )
    assert np.isnan(rh[0])
    assert flags[0] == humidity.FLAG_INVALID


def test_provenance_distinguishes_afwa_diagnostic():
    note = humidity.RH_PROVENANCE_NOTE
    assert "AFWA" in note
    assert humidity.RH_ALGORITHM_ID in note
    assert "liquid water" in note
    assert "clamp" in note.lower() or "clip" in note.lower()


def test_product_attrs_carry_units_and_algorithm():
    attrs = humidity.rh2_product_attrs()
    assert attrs["units"] == "%"
    assert attrs["algorithm_id"] == humidity.RH_ALGORITHM_ID
    assert "long_name" in attrs


def test_manifest_run_registers_rh_candidate_bound_to_revision(tmp_path):
    manifest_text = (
        "name: rh-release\n"
        "release: v1-test\n"
        "products:\n"
        "  - product_id: GAR_d02km_h_2d_rh2\n"
        "    kind: rh2\n"
        "sources:\n"
        "  - path: /archive/q2_2020.nc\n"
        "    sha256: abc\n"
    )
    path = tmp_path / "manifest.yaml"
    path.write_text(manifest_text)
    manifest = load_manifest(path)
    catalogue = Catalogue()
    candidate = humidity.register_rh_candidate(manifest, catalogue)
    assert candidate.product_id == "GAR_d02km_h_2d_rh2"
    assert candidate.manifest_revision == manifest.revision
    with pytest.raises(LookupError):
        catalogue.current("GAR_d02km_h_2d_rh2")  # still a candidate
