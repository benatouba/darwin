"""GAR station-sampled products in three methods (D6, darwin#8)."""

import pytest

from darwin.publisher.sampled import sample


def _grid():
    # 2 timesteps, 2 rows (lat), 3 cols (lon); value = t*100 + y*10 + x.
    lons = (0.0, 1.0, 2.0)
    lats = (10.0, 11.0)
    values = (
        ((0.0, 1.0, 2.0), (10.0, 11.0, 12.0)),
        ((100.0, 101.0, 102.0), (110.0, 111.0, 112.0)),
    )
    return lons, lats, values


def test_nearest_records_method_and_sampled_coordinates():
    lons, lats, values = _grid()
    series = sample(lons, lats, values, station_lon=1.4, station_lat=10.6, method="nearest")
    assert series.method == "nearest"
    assert (series.grid_x, series.grid_y) == (1, 1)
    assert (series.grid_lon, series.grid_lat) == (1.0, 11.0)
    assert series.values == (11.0, 111.0)


def test_unknown_method_is_rejected():
    lons, lats, values = _grid()
    with pytest.raises(ValueError):
        sample(lons, lats, values, station_lon=1.0, station_lat=10.0, method="cubic")


def test_bilinear_interpolates_and_records_footprint():
    lons, lats, values = _grid()
    series = sample(lons, lats, values, station_lon=0.5, station_lat=10.5, method="bilinear")
    assert series.method == "bilinear"
    assert (series.grid_x, series.grid_y) == (0, 0)
    assert series.values == (5.5, 105.5)


def test_prescribed_uses_registered_cell():
    lons, lats, values = _grid()
    series = sample(
        lons,
        lats,
        values,
        station_lon=0.1,
        station_lat=10.1,
        method="prescribed",
        prescribed_points={(0.1, 10.1): (2, 1)},
    )
    assert series.method == "prescribed"
    assert (series.grid_x, series.grid_y) == (2, 1)
    assert series.values == (12.0, 112.0)


def test_prescribed_without_register_entry_is_disabled():
    lons, lats, values = _grid()
    with pytest.raises(LookupError):
        sample(
            lons,
            lats,
            values,
            station_lon=0.1,
            station_lat=10.1,
            method="prescribed",
            prescribed_points={},
        )
