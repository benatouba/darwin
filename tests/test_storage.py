"""D8 RED: immutable consolidated Zarr analysis products.

Acceptance: chunking for native-projection spatiotemporal reads, NetCDF as
download exchange, catalogue records exposing versions, map-styling metadata
and fixed colour ranges.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from darwin.publisher.catalogue import Catalogue
from darwin.publisher.manifest import Manifest, ProductSpec, SourceEntry
from darwin.publisher.storage import (
    ANALYSIS_FORMAT,
    publish_analysis_product,
    read_analysis_slice,
    storage_layout,
    write_download_netcdf,
)


def _manifest() -> Manifest:
    return Manifest(
        name="analysis-release",
        release="v1",
        sources=(SourceEntry(path="/archive/GAR_d02km_h_2d_t2_2020.nc", sha256="abc"),),
        products=(ProductSpec(product_id="GAR_d02km_h_2d_t2", kind="analysis"),),
        revision="rev-1",
    )


def _sample() -> xr.Dataset:
    time = np.array(["2020-01-01T00", "2020-01-01T01"], dtype="datetime64[h]")
    data = np.arange(2 * 3 * 4, dtype="float32").reshape(2, 3, 4)
    return xr.Dataset(
        {"t2": (("time", "south_north", "west_east"), data, {"units": "K"})},
        coords={"time": time},
        attrs={"climate_normal_period": "n/a"},
    )


def test_storage_layout_is_immutable_versioned_address(tmp_path: Path) -> None:
    layout = storage_layout(
        root=tmp_path, product_id="GAR_d02km_h_2d_t2", version="2020-v1"
    )
    assert layout.analysis_uri.endswith("GAR_d02km_h_2d_t2/2020-v1/analysis.zarr")
    assert layout.download_uri.endswith("GAR_d02km_h_2d_t2/2020-v1/download.nc")


def test_publish_roundtrip_preserves_values_and_metadata(tmp_path: Path) -> None:
    layout = storage_layout(
        root=tmp_path, product_id="GAR_d02km_h_2d_t2", version="2020-v1"
    )
    record = publish_analysis_product(
        manifest=_manifest(),
        dataset=_sample(),
        layout=layout,
        variable="t2",
        style={"palette": "viridis", "transform": "linear"},
        color_range=(200.0, 320.0),
        catalogue=Catalogue(),
    )
    assert record["format"] == ANALYSIS_FORMAT == "zarr-v3-consolidated"
    assert record["manifest_revision"] == "rev-1"
    assert record["color_range"] == (200.0, 320.0)
    assert record["style"]["palette"] == "viridis"
    assert Path(layout.analysis_uri).is_dir()

    reread = xr.open_zarr(layout.analysis_uri)
    np.testing.assert_array_equal(reread["t2"].values, _sample()["t2"].values)
    assert reread["t2"].attrs["units"] == "K"
    reread.close()


def test_spatiotemporal_slice_read_without_full_load(tmp_path: Path) -> None:
    layout = storage_layout(
        root=tmp_path, product_id="GAR_d02km_h_2d_t2", version="2020-v1"
    )
    publish_analysis_product(
        manifest=_manifest(),
        dataset=_sample(),
        layout=layout,
        variable="t2",
        style={"palette": "viridis", "transform": "linear"},
        color_range=(200.0, 320.0),
        catalogue=Catalogue(),
    )
    cell = read_analysis_slice(
        layout, variable="t2", time_index=1, y=slice(0, 2), x=slice(1, 3)
    )
    assert cell.shape == (2, 2)
    assert float(cell[0, 0]) == float(_sample()["t2"].values[1, 0, 1])


def test_netcdf_download_exchange_keeps_units_and_provenance(tmp_path: Path) -> None:
    layout = storage_layout(
        root=tmp_path, product_id="GAR_d02km_h_2d_t2", version="2020-v1"
    )
    record = publish_analysis_product(
        manifest=_manifest(),
        dataset=_sample(),
        layout=layout,
        variable="t2",
        style={"palette": "viridis", "transform": "linear"},
        color_range=(200.0, 320.0),
        catalogue=Catalogue(),
    )
    path = write_download_netcdf(record, _sample())
    assert Path(path).is_file()
    with xr.open_dataset(path) as ds:
        assert ds["t2"].attrs["units"] == "K"
        assert ds.attrs["manifest_revision"] == "rev-1"


def test_catalogue_record_exposes_versions_and_map_style(tmp_path: Path) -> None:
    catalogue = Catalogue()
    layout = storage_layout(
        root=tmp_path, product_id="GAR_d02km_h_2d_t2", version="2020-v1"
    )
    record = publish_analysis_product(
        manifest=_manifest(),
        dataset=_sample(),
        layout=layout,
        variable="t2",
        style={"palette": "viridis", "transform": "linear"},
        color_range=(200.0, 320.0),
        catalogue=catalogue,
    )
    # The D1 approval gate: a registered candidate becomes current only
    # after explicit publication (admin scientific approval).
    catalogue.publish("GAR_d02km_h_2d_t2@2020-v1")
    current = catalogue.current("GAR_d02km_h_2d_t2")
    assert current.version == record["version"] == "2020-v1"
    assert current.manifest_revision == "rev-1"
    # Map styling lives on the record and inside the store, readable by tiles.
    assert record["style"] == {"palette": "viridis", "transform": "linear"}
    assert record["color_range"] == (200.0, 320.0)
    reread = xr.open_zarr(layout.analysis_uri)
    assert reread.attrs["map_palette"] == "viridis"
    assert reread.attrs["map_transform"] == "linear"
    assert tuple(reread.attrs["map_color_range"]) == (200.0, 320.0)
    reread.close()


def test_analysis_store_never_touches_source_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import darwin.publisher.storage as storage

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("source archive must never be scanned")

    monkeypatch.setattr(storage, "open_dataset", _forbidden, raising=False)
    layout = storage_layout(
        root=tmp_path, product_id="GAR_d02km_h_2d_t2", version="2020-v1"
    )
    publish_analysis_product(
        manifest=_manifest(),
        dataset=_sample(),
        layout=layout,
        variable="t2",
        style={"palette": "viridis", "transform": "linear"},
        color_range=(200.0, 320.0),
        catalogue=Catalogue(),
    )
