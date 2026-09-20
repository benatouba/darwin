"""Reviewed release manifests: the only publisher input."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SourceEntry:
    """One source-data reference from the background manifest."""

    path: str
    sha256: str


@dataclass(frozen=True)
class ProductSpec:
    """One product requested by the manifest."""

    product_id: str
    kind: str


@dataclass(frozen=True)
class Manifest:
    """A reviewed, version-controlled release manifest.

    `revision` is the sha256 of the exact manifest bytes, so every
    candidate records precisely which reviewed input produced it.
    """

    name: str
    release: str
    sources: tuple[SourceEntry, ...]
    products: tuple[ProductSpec, ...]
    revision: str = field(compare=False)


def load_manifest(path: str | Path) -> Manifest:
    """Parse a YAML release manifest and bind its content revision."""
    raw_path = Path(path)
    content = raw_path.read_bytes()
    data = yaml.safe_load(content.decode("utf-8"))
    return Manifest(
        name=data["name"],
        release=str(data.get("release", "v1")),
        sources=tuple(
            SourceEntry(path=entry["path"], sha256=entry["sha256"])
            for entry in data.get("sources", [])
        ),
        products=tuple(
            ProductSpec(product_id=item["product_id"], kind=item.get("kind", ""))
            for item in data.get("products", [])
        ),
        revision=hashlib.sha256(content).hexdigest(),
    )
