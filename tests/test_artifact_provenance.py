"""Every production builder binds its exact output bytes to recipe and source provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from darwin.publisher.catalogue import Catalogue
from darwin.publisher.immutable import ArtifactConflict, identity_path, verify_artifact
from darwin.publisher.manifest import Manifest, ProductSpec, SourceEntry

PRCP = "GAR_d02km_m_2d_prcp_2020.nc"
MASK = "GAR_d02km_static_landmask.nc"
T2 = "GAR_d02km_m_2d_t2_2020.nc"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digests(record: dict) -> dict[str, str]:
    return {source["id"]: source["sha256"] for source in record["provenance"]["sources"]}


def test_year_store_identity_pins_source_and_parameters(tmp_path, gar_fixtures):
    from darwin.publisher.multiyear_store import build_year_store

    source = gar_fixtures / PRCP
    path = build_year_store(
        source, variable="prcp", year=2020, expected_sha256=_sha(source), out_dir=tmp_path
    )
    record = verify_artifact(path)
    assert _source_digests(record) == {PRCP: _sha(source)}
    assert record["provenance"]["builder"] == "darwin.publisher.multiyear_store.build_year_store"
    assert record["provenance"]["parameters"] == {"variable": "prcp", "year": 2020}
    assert record["provenance"]["artifact_class"] == "derived"


def test_temperature_store_identity_pins_reviewed_source(tmp_path, gar_fixtures):
    from darwin.publisher.temperature_store import SOURCE_SHA256, build_temperature_store

    path = build_temperature_store(gar_fixtures / T2, tmp_path)
    assert _source_digests(verify_artifact(path)) == {T2: SOURCE_SHA256}


def test_smoke_analysis_stores_pin_each_input(tmp_path, analysis_sources):
    from darwin.publisher.analysis_stores import build_smoke_analysis_stores

    out = build_smoke_analysis_stores(analysis_sources, tmp_path)
    assert _source_digests(verify_artifact(out["slice"])) == {
        PRCP: _sha(analysis_sources / PRCP)
    }
    assert _source_digests(verify_artifact(out["mask"])) == {
        MASK: _sha(analysis_sources / MASK)
    }


def _manifest() -> Manifest:
    return Manifest(
        name="analysis-release",
        release="v1",
        sources=(SourceEntry(path="/archive/GAR_d02km_h_2d_t2_2020.nc", sha256="c" * 64),),
        products=(ProductSpec(product_id="GAR_d02km_h_2d_t2", kind="analysis"),),
        revision="d" * 64,
    )


def _sample() -> xr.Dataset:
    data = np.arange(2 * 3 * 4, dtype="float32").reshape(2, 3, 4)
    return xr.Dataset(
        {"t2": (("time", "south_north", "west_east"), data, {"units": "K"})},
        coords={"time": np.array(["2020-01-01T00", "2020-01-01T01"], dtype="datetime64[h]")},
    )


def test_analysis_and_download_products_are_bound_to_manifest(tmp_path):
    from darwin.publisher.storage import (
        publish_analysis_product,
        storage_layout,
        write_download_netcdf,
    )

    layout = storage_layout(tmp_path, "GAR_d02km_h_2d_t2", "2020-v1")
    record = publish_analysis_product(
        _manifest(), _sample(), layout, "t2",
        {"palette": "viridis", "transform": "linear"}, (200.0, 320.0), Catalogue(),
    )
    analysis = verify_artifact(layout.analysis_uri)
    assert record["analysis_sha256"] == analysis["artifact"]["sha256"]
    assert _source_digests(analysis) == {"/archive/GAR_d02km_h_2d_t2_2020.nc": "c" * 64}
    assert analysis["provenance"]["recipe"] == {"manifest_revision": "d" * 64}

    download = verify_artifact(write_download_netcdf(record, _sample()))
    assert download["artifact"]["kind"] == "file"
    assert download["provenance"]["recipe"] == {"manifest_revision": "d" * 64}
    assert download["provenance"]["parameters"]["analysis_sha256"] == record["analysis_sha256"]


def test_analysis_product_refuses_unpinned_manifest_sources(tmp_path):
    from darwin.publisher.storage import publish_analysis_product, storage_layout

    manifest = Manifest(
        name="x", release="v1", sources=(SourceEntry(path="/a.nc", sha256="abc"),),
        products=(), revision="d" * 64,
    )
    layout = storage_layout(tmp_path, "p", "v1")
    with pytest.raises(ValueError, match="sha256"):
        publish_analysis_product(
            manifest, _sample(), layout, "t2",
            {"palette": "viridis", "transform": "linear"}, (0.0, 1.0), Catalogue(),
        )
    assert not Path(layout.analysis_uri).exists()


def test_smoke_handoff_is_identity_bound_and_never_overwritten(tmp_path):
    from darwin.publisher.manifest import load_manifest
    from darwin.publisher.smoke import build_smoke_release

    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "darwin" / "publisher" / "manifests" / "smoke-jan2020-prcp.yaml"
    )
    manifest = load_manifest(manifest_path)
    out = tmp_path / "handoff.json"
    build_smoke_release(manifest_path, out, created_at="2026-09-22T00:00:00Z")
    original = out.read_bytes()
    record = verify_artifact(out)
    assert record["provenance"]["recipe"] == {"manifest_revision": manifest.revision}
    assert _source_digests(record) == {
        f"release-manifest:{manifest.name}": manifest.revision,
        **{entry.path: entry.sha256 for entry in manifest.sources},
    }
    with pytest.raises(ArtifactConflict):
        build_smoke_release(manifest_path, out, created_at="2026-09-23T00:00:00Z")
    assert out.read_bytes() == original
    assert json.loads(identity_path(out).read_text())["artifact"]["sha256"] == (
        record["artifact"]["sha256"]
    )
