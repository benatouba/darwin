"""Immutable consolidated Zarr analysis products in object storage.

The publisher writes one consolidated Zarr v3 store per product version,
chunked for native-projection spatiotemporal reads (one chunk per native
timestep slab). NetCDF remains the download exchange format. Map styling
and fixed colour ranges live on the catalogue record and inside the store
so tile renderers never guess presentation semantics.

``zarr`` is imported lazily so ``darwin.publisher`` stays dependency-light;
only the analysis-publication path needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from darwin.publisher.candidate import Candidate
from darwin.publisher.catalogue import Catalogue
from darwin.publisher.manifest import Manifest

if TYPE_CHECKING:
    import xarray as xr

ANALYSIS_FORMAT = "zarr-v3-consolidated"


@dataclass(frozen=True)
class StorageLayout:
    """Immutable versioned addresses for one product version."""

    product_id: str
    version: str
    analysis_uri: str
    download_uri: str


def storage_layout(root: str | Path, product_id: str, version: str) -> StorageLayout:
    """Build the immutable object addresses for a product version."""
    base = Path(root) / product_id / version
    return StorageLayout(
        product_id=product_id,
        version=version,
        analysis_uri=str(base / "analysis.zarr"),
        download_uri=str(base / "download.nc"),
    )


def publish_analysis_product(
    manifest: Manifest,
    dataset: xr.Dataset,
    layout: StorageLayout,
    variable: str,
    style: dict[str, str],
    color_range: tuple[float, float],
    catalogue: Catalogue,
) -> dict[str, Any]:
    """Write the consolidated analysis store and register the version.

    The dataset is provided by the caller (derived by the normal, humidity
    or sampling paths) — this function never touches the source archive.
    """
    try:
        import zarr  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "zarr is required to publish analysis products"
        ) from exc

    # One chunk per native timestep slab, declared via encoding so no
    # distributed chunk manager is needed on the publication path.
    dataset = dataset.copy()
    for var_name in list(dataset.data_vars):
        data_array = dataset[var_name]
        if data_array.dims and data_array.dims[0] == "time":
            data_array.encoding["chunks"] = (1,) + tuple(data_array.shape[1:])
    dataset.attrs["map_palette"] = style["palette"]
    dataset.attrs["map_transform"] = style["transform"]
    dataset.attrs["map_color_range"] = list(color_range)
    dataset.to_zarr(
        layout.analysis_uri, zarr_format=3, consolidated=True, mode="w"
    )

    candidate = Candidate(
        candidate_id=f"{layout.product_id}@{layout.version}",
        product_id=layout.product_id,
        version=layout.version,
        manifest_revision=manifest.revision,
        source_entries=tuple(
            (entry.path, entry.sha256) for entry in manifest.sources
        ),
    )
    catalogue.register(candidate)
    return {
        "product_id": layout.product_id,
        "version": layout.version,
        "variable": variable,
        "format": ANALYSIS_FORMAT,
        "manifest_revision": manifest.revision,
        "style": dict(style),
        "color_range": color_range,
        "analysis_uri": layout.analysis_uri,
        "download_uri": layout.download_uri,
    }


def read_analysis_slice(
    layout: StorageLayout, variable: str, time_index: int, y: slice, x: slice
) -> Any:
    """Read one native-projection spatiotemporal slice without full load."""
    import xarray as xr

    store = xr.open_zarr(layout.analysis_uri)
    try:
        return store[variable].isel(time=time_index, south_north=y, west_east=x).values
    finally:
        store.close()


def write_download_netcdf(record: dict[str, Any], dataset: Any) -> str:
    """Write the NetCDF download exchange for a published analysis record."""
    dataset = dataset.copy()
    dataset.attrs["manifest_revision"] = record["manifest_revision"]
    dataset.attrs["analysis_format"] = record["format"]
    path = record["download_uri"]
    dataset.to_netcdf(path)
    return path
