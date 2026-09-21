"""Comparison engine and v1 variable specifications (D7, darwin#9)."""

import pytest

from darwin.publisher.comparison import (
    COMPARISON_SPECS,
    circular_wind,
    compare,
    complete_coarser,
    complete_hourly,
    precip_depth,
    precip_extras,
)


def test_nine_declared_pairings_with_units_and_conversions():
    assert len(COMPARISON_SPECS) == 9
    by_pair = {(spec.gar_variable, spec.aws_field): spec for spec in COMPARISON_SPECS}
    assert by_pair[("t2", "T")].conversion == "K to C"
    assert by_pair[("psfc", "Pabs")].conversion == "Pa to hPa"
    assert by_pair[("tslb", "ST")].conversion == "K to C"
    prcp_fields = sorted(
        spec.aws_field for spec in COMPARISON_SPECS if spec.gar_variable == "prcp"
    )
    assert prcp_fields == ["PCP_acoustic", "PCP_diff_radar", "PCP_tot_bucket"]
    assert by_pair[("wind10", "WS/WD")].aggregation == "vector"


def test_compare_reports_scalar_stats():
    result = compare(
        "t2", "T", gar_values=(20.0, 22.0, 24.0), obs_values=(21.0, 21.0, 25.0)
    )
    assert result.paired_count == 3
    assert result.excluded_count == 0
    assert result.gar_mean == pytest.approx(22.0)
    assert result.obs_mean == pytest.approx(22.333333, abs=1e-6)
    assert result.gar_variance == pytest.approx(4.0)
    assert result.obs_variance == pytest.approx(5.333333, abs=1e-6)
    assert result.bias == pytest.approx(-0.333333, abs=1e-6)
    assert result.mae == pytest.approx(1.0)
    assert result.rmse == pytest.approx(1.0)
    assert result.correlation == pytest.approx(0.866025, abs=1e-6)


def test_insufficient_pairs_carry_not_defined_markers():
    result = compare("t2", "T", gar_values=(20.0,), obs_values=(21.0,))
    assert result.paired_count == 1
    assert result.gar_mean == pytest.approx(20.0)
    assert result.bias == pytest.approx(-1.0)
    assert result.gar_variance is None
    assert result.obs_variance is None
    assert result.correlation is None


def test_hourly_completeness_needs_five_of_six_readings():
    assert complete_hourly((1.0, 2.0, 3.0, 4.0, 5.0, None)) == pytest.approx(3.0)
    assert complete_hourly((1.0, 2.0, 3.0, 4.0, None, None)) is None


def test_coarser_completeness_needs_ninety_percent_of_hourlies():
    assert complete_coarser(tuple(range(1, 22)) + (None, None, None), expected=24) is None
    assert complete_coarser(tuple(range(22)), expected=24) == pytest.approx(10.5)


def test_precip_depth_integration_and_extras():
    assert precip_depth((1.0, 0.0, 2.0)) == (1.0, 0.0, 2.0)
    extras = precip_extras(
        gar_depths=(1.0, 0.0, 2.0), obs_depths=(0.5, 0.0, 0.0)
    )
    assert extras.total_gar == pytest.approx(3.0)
    assert extras.total_obs == pytest.approx(0.5)
    assert extras.wet_frequency_gar == pytest.approx(2 / 3)
    assert extras.wet_frequency_obs == pytest.approx(1 / 3)
    assert (extras.hits, extras.misses) == (1, 1)
    assert (extras.false_alarms, extras.correct_negatives) == (0, 1)


def test_compare_rejects_unknown_cadence():
    with pytest.raises(ValueError):
        compare("t2", "T", gar_values=(20.0,), obs_values=(21.0,), cadence="decadal")


def test_compare_accepts_all_v1_cadences():
    for cadence in ("hourly", "daily", "monthly", "annual"):
        result = compare(
            "t2", "T", gar_values=(20.0,), obs_values=(21.0,), cadence=cadence
        )
        assert result.cadence == cadence


def test_wind_circular_diagnostics_exclude_calm_pairs():
    result = circular_wind(
        gar_dir=(10.0, 350.0, 90.0),
        gar_speed=(3.0, 3.0, 0.1),
        obs_dir=(20.0, 10.0, 90.0),
        obs_speed=(3.0, 3.0, 3.0),
    )
    assert result.paired_count == 2
    assert result.excluded_calm == 1
    assert result.gar_circ_mean == pytest.approx(0.0)
    assert result.obs_circ_mean == pytest.approx(15.0)
    assert result.signed_bias == pytest.approx(15.0)
    assert result.maae == pytest.approx(15.0)