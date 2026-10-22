"""S12 (#31): the temperature seed store must refuse a non-native grid."""

import hashlib
import shutil

import netCDF4 as nc
import pytest

from darwin.publisher import temperature_store


def _shifted_grid_source(source, dest):
    """Copy the reviewed source with west_east shifted by half a cell."""
    shutil.copyfile(source, dest)
    with nc.Dataset(str(dest), "r+") as dataset:
        xs = dataset.variables["west_east"][:]
        dataset.variables["west_east"][:] = xs + 1000.0


def test_temperature_store_rejects_shifted_grid(tmp_path, monkeypatch):
    source = tmp_path / "t2-shifted.nc"
    _shifted_grid_source("/tmp/opencode/gar-monthly/GAR_d02km_m_2d_t2_2020.nc", source)
    monkeypatch.setattr(
        temperature_store,
        "SOURCE_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="native grid"):
        temperature_store.build_temperature_store(source, tmp_path / "out")
