"""Timestamp alignment is required before scientific comparison (Darwin #15)."""

from datetime import UTC, datetime, timedelta

import pytest

from darwin.publisher.comparison import compare
from darwin.publisher.series import TimeSeries, TimeValidationError

START = datetime(2025, 1, 1, tzinfo=UTC)


def series(values, hours, **kwargs):
    return TimeSeries(
        values=values,
        times=[START + timedelta(hours=h) for h in hours],
        cadence="hourly",
        **kwargs,
    )


def test_different_years_are_not_paired_by_position():
    with pytest.raises(TimeValidationError, match="no overlapping"):
        compare("t2", "T", series([20, 21], [0, 1]), series([20, 21], [8760, 8761]))


def test_gaps_and_shifted_axes_join_by_timestamp_and_report_exclusions():
    result = compare("t2", "T", series([10, 20, 40], [0, 1, 3]), series([19, 30, 38], [1, 2, 3]))
    assert result.paired_count == 2
    assert result.excluded_count == 2
    assert result.gar_mean == 30
    assert result.obs_mean == 28.5
    assert result.bias == 1.5


def test_timezone_offsets_are_normalized_before_pairing():
    local = TimeSeries([11], ["2024-12-31T18:00:00-06:00"], "hourly")
    assert compare("t2", "T", series([10], [0]), local).bias == -1


def test_invalid_values_and_completeness_masks_are_excluded():
    result = compare(
        "t2",
        "T",
        series([10, float("nan"), 30], [0, 1, 2]),
        series([11, 22, 33], [0, 1, 2], valid=[True, True, False]),
    )
    assert result.paired_count == 1
    assert result.excluded_count == 2


@pytest.mark.parametrize(
    "times",
    [
        ["2025-01-01T00:00:00"],
        ["2025-01-01T00:30:00+00:00"],
        ["not a timestamp"],
        [],
    ],
)
def test_missing_invalid_or_off_cadence_coordinates_are_rejected(times):
    with pytest.raises(TimeValidationError):
        compare("t2", "T", TimeSeries([10], times, "hourly"), series([11], [0]))


def test_duplicate_timestamps_are_not_silently_collapsed():
    with pytest.raises(TimeValidationError, match="strictly increasing"):
        compare("t2", "T", series([10, 20], [0, 0]), series([11], [0]))


def test_native_cadence_mismatch_requires_explicit_aggregation():
    with pytest.raises(TimeValidationError, match="cadence"):
        compare("t2", "T", series([10], [0]), TimeSeries([11], [START], "daily"))


def test_unknown_calendar_is_not_silently_reinterpreted():
    with pytest.raises(TimeValidationError, match="calendar"):
        compare("t2", "T", series([10], [0], calendar="360_day"), series([11], [0]))


def test_synthetic_demonstrations_are_not_eligible_for_comparison():
    with pytest.raises(TimeValidationError, match="synthetic"):
        compare("t2", "T", series([10], [0], synthetic=True), series([11], [0]))


def test_explicit_interval_bounds_must_match_declared_cadence():
    with pytest.raises(TimeValidationError, match="bounds"):
        compare(
            "t2",
            "T",
            series([10], [0], bounds=[(START, START + timedelta(days=1))]),
            series([11], [0]),
        )


def test_lengths_must_not_be_silently_truncated():
    with pytest.raises(TimeValidationError, match="length"):
        compare("t2", "T", series([10, 20], [0]), series([11], [0]))


def test_explicit_request_does_not_silently_change_native_cadence():
    with pytest.raises(TimeValidationError, match="requested cadence"):
        compare("t2", "T", series([10], [0]), series([11], [0]), cadence="daily")


def test_equal_clock_labels_with_nonoverlapping_offsets_are_rejected():
    local = TimeSeries([11], ["2025-01-01T00:00:00-06:00"], "hourly")
    with pytest.raises(TimeValidationError) as error:
        compare("t2", "T", series([10], [0]), local)
    assert error.value.code == "no_overlap"
