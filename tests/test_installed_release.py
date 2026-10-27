"""An installed Darwin release runs from an empty directory with runtime deps only.

No checkout, sibling project or research/native/visualization stack is needed.
Builds and installs offline from the local uv cache; never downloads during pytest.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = {"jsonschema", "netcdf4", "numpy", "pyyaml", "xarray", "zarr"}
EXTRAS = {"research", "native", "visualization", "all"}
# Representative packages that must never be runtime prerequisites.
OPTIONAL = {
    "visualization": {"matplotlib", "seaborn", "cartopy", "pyqt6"},
    "native": {"cdo", "rasterio", "salem", "geopandas"},
    "research": {"metpy", "statsmodels", "dask", "cdsapi"},
}
SUPPORTED_MODULES = (
    "darwin.operations",
    "darwin.comparison",
    "darwin.simulation_output",
    "darwin.observation_qc",
    "darwin.publisher.manifest",
    "darwin.publisher.smoke",
    "darwin.publisher.immutable",
    "darwin.publisher.station_series",
    "darwin.publisher.analysis_stores",
)
FORBIDDEN_IMPORTS = ("matplotlib", "salem", "cartopy", "metpy", "statsmodels", "seaborn", "cdo")


def uv(*arguments, **kwargs):
    executable = shutil.which("uv")
    if executable is None:
        pytest.fail("uv is required to build/install the release (devenv provides it)")
    env = {**os.environ, "UV_OFFLINE": "1", "UV_NO_CONFIG": "1"}
    env.pop("VIRTUAL_ENV", None)
    return subprocess.run(
        [executable, *arguments], check=True, capture_output=True, text=True, env=env, **kwargs
    )


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    uv("build", "--wheel", "--out-dir", str(out), str(ROOT))
    (built,) = out.glob("darwin-*.whl")
    return built


def requirements(wheel_path):
    with zipfile.ZipFile(wheel_path) as archive:
        (name,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(archive.read(name).decode())
    return metadata, [Requirement(value) for value in metadata.get_all("Requires-Dist")]


def test_runtime_requirements_exclude_research_native_and_visualization(wheel):
    metadata, declared = requirements(wheel)
    unconditional = {r.name.lower() for r in declared if r.marker is None}
    assert unconditional == RUNTIME
    assert set(metadata.get_all("Provides-Extra")) >= EXTRAS
    for extra, packages in OPTIONAL.items():
        in_extra = {
            r.name.lower()
            for r in declared
            if r.marker is not None and r.marker.evaluate({"extra": extra})
        }
        assert packages <= in_extra, extra


def test_installed_release_runs_from_an_empty_directory(wheel, tmp_path):
    environment = tmp_path / "venv"
    empty = tmp_path / "empty"
    empty.mkdir()
    uv("venv", "--python", sys.executable, str(environment), cwd=empty)
    python = environment / "bin" / "python"
    uv("pip", "install", "--python", str(python), str(wheel), cwd=empty)
    script = f"""
import importlib, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np, xarray as xr
for module in {SUPPORTED_MODULES!r}:
    importlib.import_module(module)
import darwin
assert Path(darwin.__file__).is_relative_to(sys.prefix), darwin.__file__
for name in {FORBIDDEN_IMPORTS!r}:
    assert importlib.util.find_spec(name) is None, name
    assert name not in sys.modules, name
from darwin.operations import execute, inventory, make_recipe
source = Path("native.nc")
xr.Dataset(
    {{"prcp": ("time", [1.0, 2.0])}},
    coords={{"time": np.array(["2020-01-01T00", "2020-01-01T01"], dtype="datetime64[ns]")}},
).to_netcdf(source)
select = {{"operation": "select", "parameters": {{"variables": ["prcp"]}}}}
recipe = make_recipe({{"native": source}}, [select])
report = execute(recipe, {{"native": str(source)}}, "python-product")
Path("recipe.json").write_text(json.dumps(recipe))
Path("locations.json").write_text(json.dumps({{"native": str(source)}}))
cli = subprocess.run(
    [sys.executable, "-m", "darwin.operations", "recipe.json",
     "--locations", "locations.json", "--output", "cli-product"],
    check=True, capture_output=True, text=True,
)
assert json.loads(cli.stdout) == report, (cli.stdout, report)
listed = subprocess.run(
    [sys.executable, "-m", "darwin.operations", "--inventory"],
    check=True, capture_output=True, text=True,
)
assert json.loads(listed.stdout) == inventory()
print(json.dumps(report))
"""
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(("PYTHON", "VIRTUAL"))
    }
    env.update(HOME=str(tmp_path), MPLBACKEND="Agg")
    completed = subprocess.run(
        [str(python), "-I", "-c", script],
        cwd=empty,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["scientific_status"] == "unassessed"
    assert report["execution_completeness"] == "complete"
