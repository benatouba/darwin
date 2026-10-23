"""Shared fixtures; downloaded inputs are explicit, verified, and portable."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.fetch_test_fixtures import fixture_directory, load_sources, verify

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def gar_fixtures():
    """Fail with setup instructions rather than silently skipping integration tests."""
    directory = fixture_directory().resolve()
    try:
        for source in load_sources():
            verify(directory / source["path"], source)
    except ValueError as error:
        pytest.fail(f"{error}. Run: python scripts/fetch_test_fixtures.py", pytrace=False)
    return directory


@pytest.fixture(scope="session")
def analysis_sources(gar_fixtures, tmp_path_factory):
    """Reproduce the historical mask's CRS annotation without modifying downloads.

    The public static file omits pyproj_srs. The historical smoke input added
    it from the precipitation grid; permit that annotation only when both
    native axes agree within 1 cm (source float32 rounding on a 2 km grid).
    Values and coordinates remain untouched.
    """
    import netCDF4 as nc

    root = tmp_path_factory.mktemp("analysis-inputs")
    source = gar_fixtures / "GAR_d02km_m_2d_prcp_2020.nc"
    (root / source.name).symlink_to(source)
    mask = root / "GAR_d02km_static_landmask.nc"
    shutil.copyfile(gar_fixtures / mask.name, mask)
    with nc.Dataset(str(source)) as grid, nc.Dataset(str(mask), "r+") as target:
        for axis in ("west_east", "south_north"):
            np.testing.assert_allclose(
                target.variables[axis][:], grid.variables[axis][:], rtol=0, atol=0.01
            )
        target.pyproj_srs = str(grid.pyproj_srs)
    return root


@pytest.fixture(scope="session")
def comparison_sources(gar_fixtures, tmp_path_factory):
    """Present the builder's input layout without touching checkout-local data."""
    root = tmp_path_factory.mktemp("comparison-inputs")
    source = gar_fixtures / "GAR_d02km_m_2d_prcp_2020.nc"
    (root / source.name).symlink_to(source)
    (root / "data").mkdir()
    shutil.copyfile(
        FIXTURES / "cerro-crocker-2025-01.csv",
        root / "data" / "11_AWS-PCerroCrocker_data.csv",
    )
    shutil.copyfile(
        FIXTURES / "cerro-crocker.xml",
        root / "data" / "11_AWS-PCerroCrocker_meta.xml",
    )
    return root


@pytest.fixture(scope="session")
def trend_fixtures(tmp_path_factory):
    """Small synthetic grids: all seasons, nondegenerate trends, and land masking."""
    root = tmp_path_factory.mktemp("trend-inputs")
    time = pd.date_range("1980-01-01", "2023-12-01", freq="MS")
    step = np.arange(len(time), dtype=float)
    signal = step * 0.001 + np.sin(step / 7) * 0.03 + np.cos(step / 19) * 0.02
    coords = {"time": time, "south_north": [0, 1], "west_east": [0, 1, 2]}
    for variable, baseline, units in (("prcp", 0.1, "mm h-1"), ("et", 0.08, "mm h-1"), ("t2", 290.0, "K")):
        values = np.broadcast_to((baseline + signal)[:, None, None], (len(time), 2, 3)).copy()
        dataset = xr.Dataset(
            {variable: (("time", "south_north", "west_east"), values, {"units": units})},
            coords=coords, attrs={"VARNAME": variable},
        )
        dataset.to_netcdf(root / f"MM_d02km_m_2d_{variable}.nc")
        if variable == "prcp":
            dataset.resample(time="YS").mean(keep_attrs=True).to_netcdf(root / "MM_d02km_y_2d_prcp.nc")
    for variable, values in (
        ("landmask", [[1, 1, 0], [1, 0, 1]]),
        ("hgt", [[100.0, 500.0, 0.0], [800.0, 0.0, 200.0]]),
    ):
        xr.Dataset(
            {variable: (("south_north", "west_east"), values)},
            coords={"south_north": [0, 1], "west_east": [0, 1, 2]},
        ).to_netcdf(root / f"MM_d02km_static_{variable}.nc")
    return root

AGGREGATION_MANIFEST_TEXT = """\
name: d02km-aggregation-contract
release: v1
products:
  - product_id: GAR_d02km_m_2d_prcp_2020-01
    kind: slice
    aggregations:
      - method: sum
        interval: daily
        missing_data: require-complete
      - method: mean
        interval: monthly
        missing_data: exclude-missing
  - product_id: GAR_d02km_static_landmask
    kind: reference
"""


@pytest.fixture
def aggregation_manifest_path(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text(AGGREGATION_MANIFEST_TEXT)
    return path
