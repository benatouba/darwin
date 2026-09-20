"""Immutable product candidates produced from a manifest run."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from darwin.publisher.manifest import Manifest

if TYPE_CHECKING:
    from darwin.publisher.catalogue import Catalogue


@dataclass(frozen=True)
class Candidate:
    """One immutable product-candidate version.

    `manifest_revision` binds the candidate to the exact reviewed
    manifest bytes that produced it. `source_entries` carries the
    (path, sha256) pairs from the background manifest — the publisher
    never touches the source archive itself.
    """

    candidate_id: str
    product_id: str
    version: str
    manifest_revision: str
    source_entries: tuple[tuple[str, str], ...]
    status: str = "candidate"


def run_manifest(manifest: Manifest, catalogue: Catalogue) -> Candidate:
    """Register one candidate per requested product, without scanning sources."""
    registered: Candidate | None = None
    for spec in manifest.products:
        version = catalogue.next_version(spec.product_id, manifest.release)
        digest = hashlib.sha256(
            f"{manifest.revision}|{spec.product_id}|{version}".encode("utf-8")
        ).hexdigest()[:16]
        candidate = Candidate(
            candidate_id=f"{spec.product_id}@{version}#{digest}",
            product_id=spec.product_id,
            version=version,
            manifest_revision=manifest.revision,
            source_entries=tuple((s.path, s.sha256) for s in manifest.sources),
        )
        catalogue.register(candidate)
        if registered is None:
            registered = candidate
    if registered is None:
        raise ValueError("manifest requests no products")
    return registered
