"""MCAPE/MCIN and SLP using the exact versioned NCAR RIP/WRF scientific kernels.

Migrated from CJC ``cjc.workflow.cape`` without changing the kernel sources, the
C binding (``interface.f90``, byte-identical, so ``interface_sha256`` is shared),
the column packing/masking or the output attributes. Part of the ``native``
extra: building needs ``gfortran`` and ``ldd`` and is always explicit; importing
this module never compiles, downloads or loads a library. No Slurm or CJC
installation is involved. See docs/wrf-products.md.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from darwin.wrf_diagnostics import state

SOURCE = Path(__file__).parent / "native" / "rip"
LIBRARY_NAME = "libdarwin_rip.so"
BUILD_CONTRACT = "darwin.native-build/1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_identity(dependencies: dict[str, dict]) -> dict[str, dict[str, str]]:
    """Reduce loader resolution to soname and content; the found path is provenance only."""
    return {
        name: {soname: entry["sha256"] for soname, entry in libraries.items()}
        for name, libraries in dependencies.items()
    }


def dynamic_dependencies(executables: dict[str, Path]) -> dict[str, dict]:
    """Capture actual loader resolution, including symlink-facing paths and targets."""
    results = {}
    for name, executable in executables.items():
        process = subprocess.run(  # noqa: S603
            ["ldd", str(executable)],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        output = process.stdout + process.stderr
        if process.returncode:
            raise RuntimeError(f"Cannot inspect dynamic dependencies for {executable}: {output}")
        libraries = {}
        for line in output.splitlines():
            if "not found" in line:
                raise RuntimeError(f"Unresolved native dependency: {line}")
            if "=>" in line:
                soname, rest = line.split("=>", 1)
                candidate = rest.strip().split()[0]
            else:
                candidate = line.strip().split()[0] if line.strip() else ""
                soname = Path(candidate).name
            if candidate.startswith("/"):
                path = Path(candidate)
                libraries[soname.strip()] = {
                    "loader_path": str(path),
                    "target": str(path.resolve()),
                    "sha256": sha256(path),
                }
        results[name] = libraries
    return results


def verify_native(library: Path) -> dict:
    """Return the build receipt after checking sources, table, library and runtime."""
    receipt = json.loads((library.parent / "build.json").read_text())
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    for record in provenance["files"]:
        if sha256(SOURCE / record["file"]) != record["sha256"]:
            raise ValueError("Pinned NCAR source/lookup-table hash mismatch")
    if (
        receipt.get("contract") != BUILD_CONTRACT
        or sha256(library) != receipt["library_sha256"]
        or receipt["upstream"] != provenance["revision"]
        or receipt["interface_sha256"] != sha256(SOURCE / "interface.f90")
    ):
        raise ValueError("Native diagnostic library does not match current pinned source")
    if runtime_identity(receipt.get("dependencies") or {}) != runtime_identity(
        dynamic_dependencies({"rip": library})
    ):
        raise ValueError("Native diagnostic runtime-library resolution changed")
    return receipt


def load_native(library: Path) -> ctypes.CDLL:
    verify_native(library)
    return ctypes.CDLL(str(library.resolve()))


def build_native(destination: Path) -> Path:
    """Explicit native build; never compile or download at import time."""
    destination.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    for record in provenance["files"]:
        if sha256(SOURCE / record["file"]) != record["sha256"]:
            raise ValueError("NCAR source hash mismatch")
    target = destination.resolve() / LIBRARY_NAME
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        output = Path(temporary) / LIBRARY_NAME
        subprocess.run(
            [
                "gfortran",
                "-shared",
                "-fPIC",
                "-O2",
                "-frecursive",
                "-ffree-line-length-none",
                str(SOURCE / "wrf_constants.f90"),
                str(SOURCE / "rip_cape.f90"),
                str(SOURCE / "wrf_user.f90"),
                str(SOURCE / "interface.f90"),
                "-o",
                str(output),
            ],
            cwd=temporary,
            check=True,
        )
        output.replace(target)
    receipt = {
        "contract": BUILD_CONTRACT,
        "upstream": provenance["revision"],
        "library_sha256": sha256(target),
        "interface_sha256": sha256(SOURCE / "interface.f90"),
        "compiler": subprocess.run(
            ["gfortran", "--version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0],
        "dependencies": dynamic_dependencies({"rip": target}),
    }
    (destination / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return target


def mcape_mcin(dataset: xr.Dataset, library: Path) -> xr.Dataset:
    """Output one MCAPE/MCIN value per horizontal cell and timestamp, J kg-1.

    RIP selects maximum theta-e below 3 km AGL and a ~500 m pressure-weighted
    parcel. MCIN is a positive magnitude, masked for MCAPE<100. No qualifying
    equilibrium level means missing, not zero. The NCAR algorithm can use model
    top as EL. Entire invalid profiles are withheld from the native routine.
    """
    dll = load_native(library)
    kernel = dll.cjc_mcape
    pointer = ctypes.POINTER(ctypes.c_double)
    kernel.argtypes = (
        [ctypes.c_int] * 3
        + [pointer] * 10
        + [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    )
    kernel.restype = None
    atmosphere = state(dataset)
    fields = [
        atmosphere.pressure_pa / 100,
        atmosphere.temperature,
        dataset.QVAPOR,
        atmosphere.height_msl,
    ]
    outputs = []
    for it in range(dataset.sizes["time"]):
        # Each native column has ascending pressure: top first, surface last.
        arrays = [
            np.asfortranarray(
                field.isel(time=it)
                .transpose("west_east", "south_north", "bottom_top")
                .values[:, :, ::-1],
                dtype=np.float64,
            )
            for field in fields
        ]
        terrain = np.asfortranarray(
            dataset.HGT.isel(time=it).transpose("west_east", "south_north").values, dtype=np.float64
        )
        psfc = np.asfortranarray(
            (dataset.PSFC.isel(time=it) / 100).transpose("west_east", "south_north").values,
            dtype=np.float64,
        )
        nx, ny, nz = arrays[0].shape
        if nz < 3:
            raise ValueError("CAPE requires at least three vertical mass levels")
        valid = np.isfinite(terrain) & np.isfinite(psfc) & (psfc > 0)
        valid &= np.all(arrays[0] > 0, axis=2) & np.all(arrays[1] > 0, axis=2)
        valid &= np.all(arrays[2] >= 0, axis=2)
        valid &= np.all(np.diff(arrays[0], axis=2) > 0, axis=2)
        valid &= np.all(np.diff(arrays[3], axis=2) < 0, axis=2)
        valid &= np.any((arrays[3] - terrain[:, :, None]) < 3000, axis=2)
        valid &= np.all(arrays[0] < psfc[:, :, None], axis=2)
        for array in arrays:
            valid &= np.all(np.isfinite(array), axis=2)
        # Pack only valid columns into one batch; Fortran does not check missing values.
        result = [np.full((nx, ny), np.nan) for _ in range(4)]
        count = int(valid.sum())
        if count:
            inputs = [np.asfortranarray(a[valid][:, None, :]) for a in arrays]
            inputs.extend(
                [
                    np.asfortranarray(terrain[valid][:, None]),
                    np.asfortranarray(psfc[valid][:, None]),
                ]
            )
            native = [np.zeros((count, 1), dtype=np.float64, order="F") for _ in range(4)]
            error = ctypes.c_int()
            table = str(SOURCE / "psadilookup.dat").encode()
            kernel(
                count,
                1,
                nz,
                *(a.ctypes.data_as(pointer) for a in inputs + native),
                table,
                len(table),
                ctypes.byref(error),
            )
            if error.value:
                raise ValueError(f"NCAR RIP diagnostic failed with code {error.value}")
            for index, value in enumerate(native):
                scalar = value[:, 0]
                result[index][valid] = np.where(np.abs(scalar) >= 1e19, np.nan, scalar)
        outputs.append(result)
    data = np.asarray(outputs)
    variables = {}
    for i, name in enumerate(["mcape", "mcin", "lcl", "lfc"]):
        variables[name] = xr.DataArray(
            data[:, i].transpose(0, 2, 1),
            dims=("time", "south_north", "west_east"),
            coords={"time": dataset.time},
            attrs={
                "units": "J kg-1" if i < 2 else "m",
                "definition": "NCAR-RIP-cape_2d-v1.4.2",
                "parcel": "max-theta-e-lowest-3km-500m-pressure-weighted",
                "sign_convention": "positive magnitude" if i == 1 else "nonnegative",
                "height_reference": "AGL" if i >= 2 else "not applicable",
                "native_library_sha256": sha256(library),
                "lookup_sha256": sha256(SOURCE / "psadilookup.dat"),
            },
        )
    return xr.Dataset(variables)


def sea_level_pressure(dataset: xr.Dataset, library: Path) -> xr.DataArray:
    """NCAR WRF sea-level pressure diagnostic, hPa; retain upstream definition."""
    dll = load_native(library)
    kernel = dll.cjc_slp
    pointer = ctypes.POINTER(ctypes.c_double)
    kernel.argtypes = [ctypes.c_int] * 3 + [pointer] * 5 + [ctypes.POINTER(ctypes.c_int)]
    kernel.restype = None
    atmosphere = state(dataset)
    result = []
    for it in range(dataset.sizes["time"]):
        arrays = [
            np.asfortranarray(
                v.isel(time=it).transpose("west_east", "south_north", "bottom_top").values,
                dtype=np.float64,
            )
            for v in [
                atmosphere.height_msl,
                atmosphere.temperature,
                atmosphere.pressure_pa,
                dataset.QVAPOR,
            ]
        ]
        if any(not np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("SLP requires finite complete profiles")
        nx, ny, nz = arrays[0].shape
        out = np.zeros((nx, ny), dtype=np.float64, order="F")
        error = ctypes.c_int()
        kernel(
            nx, ny, nz, *(a.ctypes.data_as(pointer) for a in [*arrays, out]), ctypes.byref(error)
        )
        if error.value:
            raise ValueError(f"NCAR sea-level pressure calculation failed: {error.value}")
        result.append(out.T)
    return xr.DataArray(
        np.asarray(result),
        dims=("time", "south_north", "west_east"),
        coords={"time": dataset.time},
        attrs={"units": "hPa", "definition": "NCAR-WRF-SLP-v1.4.2"},
    )
