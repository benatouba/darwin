"""S11 (#31): builders emit the canonical time axis.

Canonical axis: int64 hours since 1970-01-01 00:00:00 UTC, so future
hourly data fits without another migration. Month starts stay exact.
"""

import hashlib
from datetime import datetime
from pathlib import Path

import zarr

from darwin.publisher import multiyear_store, temperature_store

EPOCH = datetime(1970, 1, 1)
CANONICAL_UNITS = "hours since 1970-01-01 00:00:00"


def _expected_hours(year: int) -> list[int]:
    return [
        int((datetime(year, month, 1) - EPOCH).total_seconds() // 3600)
        for month in range(1, 13)
    ]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_year_store_writes_epoch_hour_axis(tmp_path, gar_fixtures):
    """Year stores carry int64 epoch hours, not float day offsets."""
    source = gar_fixtures / "GAR_d02km_m_2d_prcp_2020.nc"
    path = multiyear_store.build_year_store(
        source,
        variable="prcp",
        year=2020,
        expected_sha256=_sha(source),
        out_dir=tmp_path,
    )
    group = zarr.open_group(str(path), mode="r")
    assert group["time"].dtype == "int64"
    assert [int(v) for v in group["time"][:]] == _expected_hours(2020)
    assert group.attrs["time_units"] == CANONICAL_UNITS


def test_build_year_store_rejects_tampered_time_axis(tmp_path, gar_fixtures):
    """A shifted source time axis fails the build instead of publishing."""
    import netCDF4 as nc

    source = tmp_path / "prcp-shifted.nc"
    import shutil

    shutil.copyfile(gar_fixtures / "GAR_d02km_m_2d_prcp_2020.nc", source)
    with nc.Dataset(str(source), "r+") as dataset:
        time = dataset.variables["time"][:]
        dataset.variables["time"][:] = time + 1.0
    import pytest

    with pytest.raises(ValueError):
        multiyear_store.build_year_store(
            source,
            variable="prcp",
            year=2020,
            expected_sha256=_sha(source),
            out_dir=tmp_path / "out",
        )


def test_temperature_store_writes_epoch_hour_axis(tmp_path, gar_fixtures):
    """The temperature seed store carries the canonical axis too."""
    source = gar_fixtures / "GAR_d02km_m_2d_t2_2020.nc"
    path = temperature_store.build_temperature_store(source, tmp_path)
    group = zarr.open_group(str(path), mode="r")
    assert group["time"].dtype == "int64"
    assert [int(v) for v in group["time"][:]] == _expected_hours(2020)[:1]
    assert group.attrs["time_units"] == CANONICAL_UNITS


def test_smoke_slice_builder_writes_epoch_hour_axis(tmp_path, analysis_sources):
    """The prcp slice builder normalizes its verbatim source axis."""
    import shutil

    from darwin.publisher.analysis_stores import (
        MASK_SOURCE,
        SLICE_SOURCE,
        build_smoke_analysis_stores,
    )

    shutil.copyfile(analysis_sources / SLICE_SOURCE, tmp_path / SLICE_SOURCE)
    shutil.copyfile(analysis_sources / MASK_SOURCE, tmp_path / MASK_SOURCE)
    out = build_smoke_analysis_stores(tmp_path, tmp_path / "out")
    group = zarr.open_group(str(out["slice"]), mode="r")
    assert group["time"].dtype == "int64"
    assert [int(v) for v in group["time"][:]] == _expected_hours(2020)
    assert group.attrs["time_units"] == CANONICAL_UNITS
