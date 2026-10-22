"""RED: parameterized per-station series builder (spec #32).

Generalizes comparison_series.build_comparison_series (hardcoded
CerroCrocker/Jan-2020/2025) to any station/month/variable, plus a
temperature-means mode. All tests must FAIL until
darwin.publisher.station_series.build_station_series exists.
"""

import csv
import hashlib
from calendar import monthrange
from datetime import datetime
from xml.etree import ElementTree

import netCDF4 as nc
import numpy as np
import pytest
import zarr

from darwin.publisher.comparison import COARSER_COVERAGE_REQUIRED
from darwin.publisher.multiyear_store import (
    NATIVE_CRS,
    TIME_UNITS,
    _month_start_epoch_hours,
)

NX, NY = 245, 230
X0, DX = -244000.0, 2000.0
Y0, DY = -274001.375, 2000.0
STATION_LON = X0 + 10 * DX
STATION_LAT = Y0 + 20 * DY


def _native_netcdf(path, variable, units, year, months_since=False):
    """Real-shape synthetic GAR file on the exact native grid."""
    ds = nc.Dataset(str(path), "w")
    ds.createDimension("time", 12)
    ds.createDimension("south_north", NY)
    ds.createDimension("west_east", NX)
    xs = X0 + DX * np.arange(NX, dtype=np.float64)
    ys = Y0 + DY * np.arange(NY, dtype=np.float64)
    xv = ds.createVariable("west_east", "f8", ("west_east",))
    yv = ds.createVariable("south_north", "f8", ("south_north",))
    xv[:] = xs
    yv[:] = ys
    lon = ds.createVariable("lon", "f8", ("south_north", "west_east"))
    lat = ds.createVariable("lat", "f8", ("south_north", "west_east"))
    lon[:] = np.broadcast_to(xs, (NY, NX))
    lat[:] = np.broadcast_to(ys[:, None], (NY, NX))
    if months_since:
        time_units = f"months since {year}-01-01 00:00:00"
        values = np.arange(12, dtype=np.float64)
    else:
        time_units = f"days since {year}-01-01 00:00:00"
        starts = [datetime(year, m, 1) for m in range(1, 13)]
        values = np.array([(s - starts[0]).days for s in starts], dtype=np.float64)
    tv = ds.createVariable("time", "f8", ("time",))
    tv.units = time_units
    tv[:] = values
    vv = ds.createVariable(variable, "f8", ("time", "south_north", "west_east"))
    vv.units = units
    vv[:] = np.reshape(100.0 + np.arange(12, dtype=np.float64), (12, 1, 1))
    ds.setncattr("pyproj_srs", NATIVE_CRS)
    ds.close()


def _meta_xml(path, lon=STATION_LON, lat=STATION_LAT):
    root = ElementTree.fromstring(
        "<eml><dataset><coverage><geographicCoverage><boundingCoordinates>"
        f"<westBoundingCoordinate>{lon}°</westBoundingCoordinate>"
        f"<eastBoundingCoordinate>{lon}°</eastBoundingCoordinate>"
        f"<northBoundingCoordinate>{lat}°</northBoundingCoordinate>"
        f"<southBoundingCoordinate>{lat}°</southBoundingCoordinate>"
        "</boundingCoordinates></geographicCoverage></coverage></dataset></eml>"
    )
    ElementTree.ElementTree(root).write(str(path), encoding="utf-8")


def _write_csv(path, header, rows):
    with open(str(path), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _hour_rows(year, month, day, hour, values):
    return [
        (f"{year}-{month:02d}-{day:02d} {hour:02d}:{m:02d}:00", v)
        for m, v in zip(range(0, 60, 10), values)
    ]


def test_prcp_mode_layout_and_tip_sums(tmp_path):
    from darwin.publisher.station_series import build_station_series

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    out = tmp_path / "out"
    # February 2023: 28 days, proves month-indexing (monthly[1], not [0]).
    _native_netcdf(repo / "GAR_d02km_m_2d_prcp_2023.nc", "prcp", "mm h-1", 2023)
    _meta_xml(repo / "data" / "ST_meta.xml")
    rows = []
    for day in range(1, 29):
        for hour in range(24):
            if day == 3 and hour == 5:
                # Rain event: cumulative within the hour, then reset.
                rows += [(s, v) for s, v in _hour_rows(2023, 2, day, hour,
                          [0.0, 0.2, 0.6, 1.2, 2.0, 3.0])]
            elif day == 10:
                # Only 3 readings: hour invalid (< 6 required).
                rows += [(s, v) for s, v in _hour_rows(2023, 2, day, hour,
                          [0.0, 0.0, 0.0])[:3]]
            else:
                rows += [(s, 0.0) for s, _ in _hour_rows(2023, 2, day, hour,
                          [0.0] * 6)]
    _write_csv(repo / "data" / "ST_data.csv", ["datetime", "PCP_tot_bucket"], rows)
    gar_bytes = (repo / "GAR_d02km_m_2d_prcp_2023.nc").read_bytes()
    obs_bytes = (repo / "data" / "ST_data.csv").read_bytes()

    result = build_station_series(
        repo, out,
        gar_source="GAR_d02km_m_2d_prcp_2023.nc", gar_variable="prcp",
        obs_csv="data/ST_data.csv", obs_meta="data/ST_meta.xml",
        obs_year=2023, obs_month=2,
        gar_store_name="GAR-ST-sampled_2023.zarr",
        obs_store_name="AWS-ST_2023-02.zarr",
        mode="sum", obs_field="PCP_tot_bucket",
        gar_sha256=hashlib.sha256(gar_bytes).hexdigest(),
        obs_sha256=hashlib.sha256(obs_bytes).hexdigest(),
    )

    gar = zarr.open_group(str(result["gar_sampled"]))
    assert len(gar["monthly"][:]) == 12
    assert gar["monthly"].attrs["units"] == "mm h-1"
    # Constants disaggregate monthly[month - 1], not monthly[0].
    assert float(gar["hourly"][0]) == pytest.approx(101.0)
    assert float(gar["daily"][0]) == pytest.approx(101.0 * 24.0)
    assert list(gar["time"][:]) == _month_start_epoch_hours(2023)
    assert gar["time"].attrs["units"] == TIME_UNITS
    assert gar.attrs["grid_x"] == 10 and gar.attrs["grid_y"] == 20
    assert gar.attrs["method"] == "nearest"

    obs = zarr.open_group(str(result["obs"]))
    assert obs.attrs["year"] == 2023 and obs.attrs["month"] == 2
    assert obs["hourly"].attrs["units"] == "mm"
    # Rain hour: positive diffs 0.2+0.4+0.6+0.8+1.0 = 3.0.
    assert float(obs["hourly"][(3 - 1) * 24 + 5]) == pytest.approx(3.0)
    assert bool(obs["hourly_valid"][(3 - 1) * 24 + 5]) is True
    # Day 10 has only 3 readings/hour: invalid.
    assert bool(obs["hourly_valid"][(10 - 1) * 24]) is False
    assert bool(obs["daily_valid"][9]) is False
    assert float(obs["daily"][2]) == pytest.approx(3.0)


def test_tips_carry_across_hour_boundary():
    from darwin.publisher.station_series import _derive_tips

    base = datetime(2024, 10, 1, 10, 0, 0)
    readings = [
        (base.replace(minute=0), 5.0),
        (base.replace(minute=30), 5.4),
        (base.replace(hour=11, minute=0), 5.6),
        (base.replace(hour=11, minute=30), 5.8),
    ]
    tips = _derive_tips(readings)
    assert [round(t, 6) for _, t in tips] == [0.0, 0.4, 0.2, 0.2]


def test_tips_reset_on_bucket_emptying():
    from darwin.publisher.station_series import _derive_tips

    base = datetime(2024, 10, 1, 10, 0, 0)
    readings = [
        (base.replace(minute=0), 5.0),
        (base.replace(minute=30), 5.4),
        (base.replace(hour=11, minute=0), 0.0),
        (base.replace(hour=11, minute=30), 0.2),
    ]
    tips = _derive_tips(readings)
    assert [round(t, 6) for _, t in tips] == [0.0, 0.4, 0.0, 0.2]


def test_temperature_mode_kelvin_means(tmp_path):
    from darwin.publisher.station_series import build_station_series

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    out = tmp_path / "out"
    # months-since axis (t2_2025-style) must normalize like days-since.
    _native_netcdf(repo / "GAR_d02km_m_2d_t2_2025.nc", "t2", "k", 2025,
                   months_since=True)
    _meta_xml(repo / "data" / "ST_meta.xml")
    rows = []
    for day in range(1, 31):
        for hour in range(24):
            if day == 5 and hour < 2:
                continue  # 2 missing hours: day still valid (22/24 >= 0.9).
            if day == 6 and hour < 3:
                continue  # 3 missing hours: day invalid (21/24 < 0.9).
            rows.append((f"2025-04-{day:02d} {hour:02d}:00:00", "20.0"))
    _write_csv(repo / "data" / "ST_data.csv", ["datetime", "T"], rows)
    gar_bytes = (repo / "GAR_d02km_m_2d_t2_2025.nc").read_bytes()
    obs_bytes = (repo / "data" / "ST_data.csv").read_bytes()

    result = build_station_series(
        repo, out,
        gar_source="GAR_d02km_m_2d_t2_2025.nc", gar_variable="t2",
        obs_csv="data/ST_data.csv", obs_meta="data/ST_meta.xml",
        obs_year=2025, obs_month=4,
        gar_store_name="GAR-ST-sampled-t2_2025.zarr",
        obs_store_name="AWS-ST-T_2025-04.zarr",
        mode="mean", obs_field="T",
        gar_sha256=hashlib.sha256(gar_bytes).hexdigest(),
        obs_sha256=hashlib.sha256(obs_bytes).hexdigest(),
    )

    gar = zarr.open_group(str(result["gar_sampled"]))
    assert gar["monthly"].attrs["units"] == "K"
    assert float(gar["hourly"][0]) == pytest.approx(103.0)
    assert list(gar["time"][:]) == _month_start_epoch_hours(2025)

    obs = zarr.open_group(str(result["obs"]))
    assert obs["hourly"].attrs["units"] == "K"
    assert float(obs["hourly"][0]) == pytest.approx(293.15)
    assert float(obs["daily"][0]) == pytest.approx(293.15)
    assert bool(obs["daily_valid"][4]) is True
    assert bool(obs["daily_valid"][5]) is False


def test_foreign_grid_rejected(tmp_path):
    from darwin.publisher.station_series import build_station_series

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    _native_netcdf(repo / "GAR_d02km_m_2d_prcp_2023.nc", "prcp", "mm h-1", 2023)
    with nc.Dataset(str(repo / "GAR_d02km_m_2d_prcp_2023.nc"), "r+") as ds:
        xs = ds.variables["west_east"][:]
        ds.variables["west_east"][:] = xs + 1000.0
    _meta_xml(repo / "data" / "ST_meta.xml")
    _write_csv(repo / "data" / "ST_data.csv",
               ["datetime", "PCP_tot_bucket"],
               [("2023-02-01 00:00:00", "0.0")])
    with pytest.raises(ValueError, match="native grid"):
        build_station_series(
            repo, tmp_path / "out",
            gar_source="GAR_d02km_m_2d_prcp_2023.nc", gar_variable="prcp",
            obs_csv="data/ST_data.csv", obs_meta="data/ST_meta.xml",
            obs_year=2023, obs_month=2,
            gar_store_name="g.zarr", obs_store_name="o.zarr",
            mode="sum", obs_field="PCP_tot_bucket",
            gar_sha256=hashlib.sha256(
                (repo / "GAR_d02km_m_2d_prcp_2023.nc").read_bytes()).hexdigest(),
            obs_sha256=hashlib.sha256(
                (repo / "data" / "ST_data.csv").read_bytes()).hexdigest(),
        )


def test_missing_crs_falls_back_to_native(tmp_path):
    from darwin.publisher.station_series import build_station_series

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    _native_netcdf(repo / "GAR_d02km_m_2d_t2_2025.nc", "t2", "k", 2025)
    with nc.Dataset(str(repo / "GAR_d02km_m_2d_t2_2025.nc"), "r+") as ds:
        delattr(ds, "pyproj_srs")
    _meta_xml(repo / "data" / "ST_meta.xml")
    _write_csv(repo / "data" / "ST_data.csv", ["datetime", "T"],
               [("2025-04-01 00:00:00", "20.0")] * 6)
    result = build_station_series(
        repo, tmp_path / "out",
        gar_source="GAR_d02km_m_2d_t2_2025.nc", gar_variable="t2",
        obs_csv="data/ST_data.csv", obs_meta="data/ST_meta.xml",
        obs_year=2025, obs_month=4,
        gar_store_name="g.zarr", obs_store_name="o.zarr",
        mode="mean", obs_field="T",
        gar_sha256=hashlib.sha256(
            (repo / "GAR_d02km_m_2d_t2_2025.nc").read_bytes()).hexdigest(),
        obs_sha256=hashlib.sha256(
            (repo / "data" / "ST_data.csv").read_bytes()).hexdigest(),
    )
    gar = zarr.open_group(str(result["gar_sampled"]))
    assert gar.attrs["crs"] == NATIVE_CRS


def test_tampered_source_rejected(tmp_path):
    from darwin.publisher.station_series import build_station_series

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    _native_netcdf(repo / "GAR_d02km_m_2d_prcp_2023.nc", "prcp", "mm h-1", 2023)
    _meta_xml(repo / "data" / "ST_meta.xml")
    _write_csv(repo / "data" / "ST_data.csv",
               ["datetime", "PCP_tot_bucket"],
               [("2023-02-01 00:00:00", "0.0")])
    with pytest.raises(ValueError, match="sha256"):
        build_station_series(
            repo, tmp_path / "out",
            gar_source="GAR_d02km_m_2d_prcp_2023.nc", gar_variable="prcp",
            obs_csv="data/ST_data.csv", obs_meta="data/ST_meta.xml",
            obs_year=2023, obs_month=2,
            gar_store_name="g.zarr", obs_store_name="o.zarr",
            mode="sum", obs_field="PCP_tot_bucket",
            gar_sha256="0" * 64,
            obs_sha256="0" * 64,
        )


def test_coverage_rule_uses_shared_threshold():
    import math

    min_valid_hours = math.ceil(24 * COARSER_COVERAGE_REQUIRED)
    assert min_valid_hours / 24 >= COARSER_COVERAGE_REQUIRED
    assert (min_valid_hours - 1) / 24 < COARSER_COVERAGE_REQUIRED
