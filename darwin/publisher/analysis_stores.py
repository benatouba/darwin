"""Smoke analysis stores (T4): real bundled Zarrs from pinned NetCDF bytes.

Reads the two source files pinned by the smoke manifest
(``GAR_d02km_m_2d_prcp_2020.nc``, ``GAR_d02km_static_landmask.nc``) and
writes full-domain Zarr seed stores with verbatim values: the 12-month
prcp field on native WRF-Mercator coordinates with its UTC time axis,
and the binary static land mask on the same grid. The dashboard bundles
these stores as seed data; the published ``s3://`` uris stay the
long-term locations.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from darwin.publisher.immutable import staged_artifact

SLICE_SOURCE = "GAR_d02km_m_2d_prcp_2020.nc"
MASK_SOURCE = "GAR_d02km_static_landmask.nc"
SLICE_STORE = "GAR_d02km_m_2d_prcp_2020-01.zarr"
MASK_STORE = "GAR_d02km_static_landmask.zarr"


def build_smoke_analysis_stores(repo_root: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Build the slice + mask Zarr stores, returning their paths by key."""
    import netCDF4 as nc
    import numpy as np
    import zarr

    from darwin.publisher.multiyear_store import TIME_UNITS, _normalize_time

    repo_root, out_dir = Path(repo_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def provenance(name: str, parameters: dict) -> dict:
        digest = hashlib.sha256((repo_root / name).read_bytes()).hexdigest()
        return {
            "builder": "darwin.publisher.analysis_stores.build_smoke_analysis_stores",
            "sources": [{"id": name, "sha256": digest}],
            "parameters": parameters,
        }

    slice_provenance = provenance(SLICE_SOURCE, {"variable": "prcp"})
    mask_provenance = provenance(MASK_SOURCE, {"variable": "landmask", "time_index": 0})
    source = nc.Dataset(str(repo_root / SLICE_SOURCE))
    try:
        time_var = source.variables["time"]
        time_values = _normalize_time(
            str(time_var.units), np.asarray(time_var[:]), 2020
        )
        time_units = TIME_UNITS
        xs = [float(v) for v in source.variables["west_east"][:].tolist()]
        ys = [float(v) for v in source.variables["south_north"][:].tolist()]
        prcp = source.variables["prcp"][:]
        crs = str(source.pyproj_srs)
    finally:
        source.close()
    slice_path = out_dir / SLICE_STORE
    with staged_artifact(slice_path, provenance=slice_provenance) as staged:
        group = zarr.open_group(str(staged), mode="w-")
        group.create_array("prcp", data=np.asarray(prcp), chunks=(1,) + prcp.shape[1:])
        group.create_array("x", data=np.asarray(xs))
        group.create_array("y", data=np.asarray(ys))
        group.create_array("time", data=np.asarray(time_values, dtype=np.int64))
        group.attrs["time_units"] = time_units
        group.attrs["crs"] = crs

    mask_source = nc.Dataset(str(repo_root / MASK_SOURCE))
    try:
        landmask = mask_source.variables["landmask"][0]
        mask_crs = str(mask_source.pyproj_srs)
    finally:
        mask_source.close()
    mask_path = out_dir / MASK_STORE
    with staged_artifact(mask_path, provenance=mask_provenance) as staged:
        mask_group = zarr.open_group(str(staged), mode="w-")
        mask_group.create_array("landmask", data=np.asarray(landmask), chunks=landmask.shape)
        mask_group.create_array("x", data=np.asarray(xs))
        mask_group.create_array("y", data=np.asarray(ys))
        mask_group.attrs["crs"] = mask_crs

    return {"slice": slice_path, "mask": mask_path}
