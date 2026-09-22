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
    """One product requested by the manifest, with its reviewed provenance.

    `citation` should be set for every product; `normal_period`
    and `method` apply to Normal products, `station_id` to AWS
    observation products, the `model_*`/`observation_*` pairing
    to Comparison variable specifications, `variable`/`domain`/`format`
    to the catalogue identity of every product, and `palette`,
    `transform`, `color_range` plus the `uris` analysis store to the
    publisher-declared map presentation of every product.
    """

    product_id: str
    kind: str
    citation: str = ""
    normal_period: str = ""
    method: str = ""
    station_id: str = ""
    model_product: str = ""
    model_variable: str = ""
    observation_product: str = ""
    observation_field: str = ""
    variable: str = ""
    domain: str = ""
    format: str = ""
    palette: str = ""
    transform: str = ""
    color_range: tuple[float, float] = ()
    uris: dict[str, str] = field(default_factory=dict)


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
            ProductSpec(
                product_id=item["product_id"],
                kind=item.get("kind", ""),
                citation=item.get("citation", ""),
                normal_period=item.get("normal_period", ""),
                method=item.get("method", ""),
                station_id=item.get("station_id", ""),
                model_product=item.get("model_product", ""),
                model_variable=item.get("model_variable", ""),
                observation_product=item.get("observation_product", ""),
                observation_field=item.get("observation_field", ""),
                variable=item.get("variable", ""),
                domain=item.get("domain", ""),
                format=item.get("format", ""),
                palette=item.get("palette", ""),
                transform=item.get("transform", ""),
                color_range=tuple(item.get("color_range", ())),
                uris=dict(item.get("uris", {})),
            )
            for item in data.get("products", [])
        ),
        revision=hashlib.sha256(content).hexdigest(),
    )
