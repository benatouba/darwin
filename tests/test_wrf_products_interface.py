"""Python/CLI equivalence of ``darwin.wrf_products`` on the copied CJC golden input."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
import xarray as xr

from darwin.wrf_products import (
    CONTRACT,
    aggregate_products,
    derive_products,
    inventory,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "cjc-product-golden-v1"
VARIABLES = ["t2", "prcp", "prcp_rate", "temperature", "inversion"]


def cli(*arguments, check=True):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "darwin.wrf_products", *map(str, arguments)],
        capture_output=True,
        text=True,
        check=check,
        env=env,
    )


def products(root: Path) -> dict[str, xr.Dataset]:
    found = {}
    for path in sorted(root.rglob("*.nc")):
        with xr.open_dataset(path) as dataset:
            found[str(path.relative_to(root))] = dataset.load()
    return found


def assert_same_products(left: Path, right: Path) -> None:
    first, second = products(left), products(right)
    assert first
    assert first.keys() == second.keys()
    for name, dataset in first.items():
        xr.testing.assert_identical(dataset, second[name])


def test_inventory_is_shared_by_python_and_cli():
    listed = json.loads(cli("inventory").stdout)
    assert listed == inventory()
    assert listed["contract"] == CONTRACT
    assert listed["definition_version"] == "gar-v2-python-1"
    names = {entry["name"] for entry in listed["diagnostics"]}
    assert {"prcp", "mcape", "slp", "inversion", "col_<x>"} <= names


def test_derive_python_and_cli_write_identical_products(tmp_path):
    source = GOLDEN / "input.nc"
    python = derive_products(source, tmp_path / "python", 2, VARIABLES, pressure_levels=[700.0])
    completed = cli(
        "derive", source, "--destination", tmp_path / "cli", "--domain", 2,
        "--variables", *VARIABLES, "--pressure-levels", 700,
    )  # fmt: skip
    report = json.loads(completed.stdout)
    assert report["contract"] == CONTRACT
    assert [Path(p).name for p in report["products"]] == [p.name for p in python]
    assert_same_products(tmp_path / "python", tmp_path / "cli")
    with xr.open_dataset(python[0]) as product:
        assert product.attrs["product_status"] == "candidate"
        assert product.attrs["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert "source_path" not in product.attrs
        assert product.attrs["product_contract"] == CONTRACT
        assert product.attrs["darwin_version"] == version("darwin")
        assert product.attrs["DOMAIN"] == "d02km"


def test_aggregate_python_and_cli_write_identical_products(tmp_path):
    native = derive_products(GOLDEN / "input.nc", tmp_path / "native", 2, ["prcp"])
    python = aggregate_products(native, tmp_path / "python", 2, "2024-01-01", "2024-01-02", "D", 60)
    completed = cli(
        "aggregate", *native, "--destination", tmp_path / "cli", "--domain", 2,
        "--start", "2024-01-01", "--end", "2024-01-02", "--frequency", "D",
        "--cadence-minutes", 60,
    )  # fmt: skip
    assert [Path(p).name for p in json.loads(completed.stdout)["products"]] == [
        p.name for p in python
    ]
    assert_same_products(tmp_path / "python", tmp_path / "cli")
    with xr.open_dataset(python[0]) as daily:
        assert daily.prcp.attrs["cell_methods"] == "time: sum"
        assert daily.prcp_valid_fraction.dims == daily.prcp.dims
        assert daily.attrs["product_contract"] == CONTRACT


def test_cli_and_python_report_the_same_structured_error(tmp_path):
    with pytest.raises(ValueError, match="Unsupported diagnostic: nope") as raised:
        derive_products(GOLDEN / "input.nc", tmp_path / "python", 2, ["nope"])
    completed = cli(
        "derive", GOLDEN / "input.nc", "--destination", tmp_path / "cli", "--domain", 2,
        "--variables", "nope", check=False,
    )  # fmt: skip
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert json.loads(completed.stderr) == {"code": "invalid_input", "message": str(raised.value)}
    assert not (tmp_path / "cli").exists() or not list((tmp_path / "cli").rglob("*.nc"))


def test_native_build_cli_matches_python_receipt(tmp_path):
    from darwin.native_diagnostics import verify_native

    if shutil.which("gfortran") is None:
        pytest.fail("gfortran is required for native tests (devenv provides it)")
    report = json.loads(cli("build-native", "--destination", tmp_path / "native").stdout)
    library = Path(report["library"])
    assert report["receipt"] == verify_native(library)
    assert report["receipt"]["contract"] == "darwin.native-build/1"
