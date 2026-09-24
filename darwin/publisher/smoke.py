"""Smoke-release builder (T1): reviewed manifest in, published handoff out.

Loads one reviewed release manifest, registers one Product candidate per
requested product, publishes every candidate, and writes the handoff
JSON the dashboard catalogue ingests: product/version identifiers plus
full per-product provenance (manifest revision, creator, creation time,
citation, normal period/method where applicable, the
model/observation pairing for Comparison variable specifications) and
the publisher-declared map presentation (palette, transform, fixed
colour range, analysis-store uris) and the permitted time-span
aggregations (method, target interval, missing-data semantics).
`created_at` stamps build time, so rebuilds share the manifest revision
but diverge in stamp.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from darwin.publisher.candidate import run_manifest
from darwin.publisher.catalogue import Catalogue
from darwin.publisher.manifest import load_manifest

CREATOR = "darwin.publisher"


def _utc_now() -> str:
    """Return the current UTC time in trailing-Z ISO-8601 form."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_smoke_release(
    manifest_path: str | Path,
    out_path: str | Path,
    *,
    creator: str = CREATOR,
    created_at: str | None = None,
) -> dict:
    """Build the smoke release and write its handoff JSON.

    Every requested product is registered as an immutable candidate and
    published, so the handoff only ever contains ``published`` versions
    bound to the exact reviewed manifest revision.
    """
    stamp = created_at or _utc_now()
    manifest = load_manifest(manifest_path)
    catalogue = Catalogue()
    run_manifest(manifest, catalogue)
    specs = {spec.product_id: spec for spec in manifest.products}
    products = []
    for candidate_id in catalogue.candidate_ids():
        published = catalogue.publish(candidate_id)
        spec = specs[published.product_id]
        products.append(
            {
                "product_id": published.product_id,
                "version": published.version,
                "candidate_id": published.candidate_id,
                "kind": spec.kind,
                "status": "published",
                "manifest_revision": manifest.revision,
                "creator": creator,
                "created_at": stamp,
                "citation": spec.citation,
                "normal_period": spec.normal_period,
                "method": spec.method,
                "station_id": spec.station_id,
                "model_product": spec.model_product,
                "model_variable": spec.model_variable,
                "observation_product": spec.observation_product,
                "observation_field": spec.observation_field,
                "variable": spec.variable,
                "domain": spec.domain,
                "format": spec.format,
                "palette": spec.palette,
                "transform": spec.transform,
                "color_range": list(spec.color_range),
                "uris": dict(spec.uris),
                "aggregations": [asdict(agg) for agg in spec.aggregations],
            }
        )
    handoff = {
        "release": manifest.release,
        "manifest": manifest.name,
        "manifest_revision": manifest.revision,
        "creator": creator,
        "created_at": stamp,
        "products": products,
    }
    Path(out_path).write_text(json.dumps(handoff, indent=2) + "\n")
    return handoff


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the Jan-2020 prcp smoke release.")
    parser.add_argument("manifest", help="reviewed release manifest YAML")
    parser.add_argument("handoff", help="handoff JSON path to write")
    args = parser.parse_args()
    result = build_smoke_release(args.manifest, args.handoff)
    print(f"release {result['release']}: {len(result['products'])} products published")
