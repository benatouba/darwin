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

from pathlib import Path

SLICE_SOURCE = "GAR_d02km_m_2d_prcp_2020.nc"
MASK_SOURCE = "GAR_d02km_static_landmask.nc"
SLICE_STORE = "GAR_d02km_m_2d_prcp_2020-01.zarr"
MASK_STORE = "GAR_d02km_static_landmask.zarr"


def build_smoke_analysis_stores(repo_root: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """Build the slice + mask Zarr stores, returning their paths by key."""
    import netCDF4 as nc
    import numpy as np
    import zarr

    repo_root, out_dir = Path(repo_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = nc.Dataset(str(repo_root / SLICE_SOURCE))
    try:
        time_var = source.variables["time"]
        time_units = str(time_var.units)
        time_values = [float(v) for v in time_var[:].tolist()]
        xs = [float(v) for v in source.variables["west_east"][:].tolist()]
        ys = [float(v) for v in source.variables["south_north"][:].tolist()]
        prcp = source.variables["prcp"][:]
        crs = str(source.pyproj_srs)
    finally:
        source.close()
    slice_path = out_dir / SLICE_STORE
    group = zarr.open_group(str(slice_path), mode="w")
    group.create_array("prcp", data=np.asarray(prcp), chunks=(1,) + prcp.shape[1:])
    group.create_array("x", data=np.asarray(xs))
    group.create_array("y", data=np.asarray(ys))
    group.create_array("time", data=np.asarray(time_values))
    group.attrs["time_units"] = time_units
    group.attrs["crs"] = crs

    mask_source = nc.Dataset(str(repo_root / MASK_SOURCE))
    try:
        landmask = mask_source.variables["landmask"][0]
        mask_crs = str(mask_source.pyproj_srs)
    finally:
        mask_source.close()
    mask_path = out_dir / MASK_STORE
    mask_group = zarr.open_group(str(mask_path), mode="w")
    mask_group.create_array("landmask", data=np.asarray(landmask), chunks=landmask.shape)
    mask_group.create_array("x", data=np.asarray(xs))
    mask_group.create_array("y", data=np.asarray(ys))
    mask_group.attrs["crs"] = mask_crs

    return {"slice": slice_path, "mask": mask_path}
