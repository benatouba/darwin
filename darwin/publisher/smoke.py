"""Smoke-release builder (T1): reviewed manifest in, legacy handoff out.

Loads one reviewed release manifest, prepares one Product candidate per
requested product (nothing is published) and writes the legacy handoff
JSON the dashboard catalogue imports: product/version identifiers plus
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

from darwin.publisher.candidate import Candidate, plan_candidates
from darwin.publisher.immutable import staged_artifact
from darwin.publisher.manifest import Manifest, ProductSpec, load_manifest

CREATOR = "darwin.publisher"


def _utc_now() -> str:
    """Return the current UTC time in trailing-Z ISO-8601 form."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def legacy_product_entry(
    spec: ProductSpec,
    candidate: Candidate,
    manifest: Manifest,
    *,
    creator: str,
    created_at: str,
) -> dict:
    """Return the legacy (unversioned) handoff fields of one product candidate.

    ``darwin.product-handoff/2`` products carry every one of these fields
    except ``status``, so importers of the legacy format keep working.
    """
    return {
        "product_id": candidate.product_id,
        "version": candidate.version,
        "candidate_id": candidate.candidate_id,
        "kind": spec.kind,
        "manifest_revision": manifest.revision,
        "creator": creator,
        "created_at": created_at,
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
        "units": spec.units,
        "palette": spec.palette,
        "transform": spec.transform,
        "color_range": list(spec.color_range),
        "uris": dict(spec.uris),
        "aggregations": [asdict(agg) for agg in spec.aggregations],
    }


def build_smoke_release(
    manifest_path: str | Path,
    out_path: str | Path,
    *,
    creator: str = CREATOR,
    created_at: str | None = None,
) -> dict:
    """Build the smoke release and write its legacy (unversioned) handoff JSON.

    Migration (darwin#18): this builder used to publish every candidate in a
    throwaway in-memory catalogue and label each product ``published``. That
    was never a dashboard publication or scientific acceptance. It now only
    prepares candidates; each product carries ``status: "candidate"`` (a
    dashboard publication decision is still required) and no acceptance
    claim. Prefer :func:`darwin.handoff.prepare_handoff`
    (``darwin.product-handoff/2``), which binds exact artifacts.
    """
    stamp = created_at or _utc_now()
    manifest = load_manifest(manifest_path)
    specs = {spec.product_id: spec for spec in manifest.products}
    products = [
        {
            **legacy_product_entry(
                specs[candidate.product_id],
                candidate,
                manifest,
                creator=creator,
                created_at=stamp,
            ),
            "status": "candidate",
        }
        for candidate in plan_candidates(manifest)
    ]
    handoff = {
        "release": manifest.release,
        "manifest": manifest.name,
        "manifest_revision": manifest.revision,
        "creator": creator,
        "created_at": stamp,
        "products": products,
    }
    provenance = {
        "builder": "darwin.publisher.smoke.build_smoke_release",
        "recipe": {"manifest_revision": manifest.revision},
        "sources": [
            {"id": f"release-manifest:{manifest.name}", "sha256": manifest.revision},
            *({"id": entry.path, "sha256": entry.sha256} for entry in manifest.sources),
        ],
        "parameters": {"creator": creator, "created_at": stamp},
    }
    with staged_artifact(out_path, provenance=provenance) as staged:
        staged.write_text(json.dumps(handoff, indent=2) + "\n")
    return handoff


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the Jan-2020 prcp smoke release.")
    parser.add_argument("manifest", help="reviewed release manifest YAML")
    parser.add_argument("handoff", help="handoff JSON path to write")
    args = parser.parse_args()
    result = build_smoke_release(args.manifest, args.handoff)
    print(f"release {result['release']}: {len(result['products'])} candidates prepared")
