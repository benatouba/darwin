"""Station metadata parsing used by the comparison-store builders."""

import pytest

from darwin.publisher.comparison_series import _station_point as comparison_point
from darwin.publisher.station_series import _station_point as station_point


@pytest.mark.parametrize("reader", [comparison_point, station_point])
def test_extract_station_coordinates(tmp_path, reader):
    source = tmp_path / "station.xml"
    source.write_text(
        '<eml><dataset><coverage><geographicCoverage><boundingCoordinates>'
        '<westBoundingCoordinate>-90.326°</westBoundingCoordinate>'
        '<northBoundingCoordinate>-0.642398°</northBoundingCoordinate>'
        '</boundingCoordinates></geographicCoverage></coverage></dataset></eml>',
        encoding="utf-8",
    )
    assert reader(source) == (-90.326, -0.642398)


@pytest.mark.parametrize("reader", [comparison_point, station_point])
def test_missing_station_coordinates_are_rejected(tmp_path, reader):
    source = tmp_path / "station.xml"
    source.write_text("<eml><dataset /></eml>", encoding="utf-8")
    with pytest.raises(AssertionError, match="boundingCoordinates"):
        reader(source)
