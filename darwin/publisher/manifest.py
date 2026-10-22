"""Reviewed release manifests: the only publisher input."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

AGGREGATION_METHODS = ("mean", "min", "max", "sum", "component_mean")
AGGREGATION_INTERVALS = ("daily", "monthly", "annual")
AGGREGATION_MISSING_DATA = ("exclude-missing", "require-complete")

_AGGREGATION_VOCABULARIES = {
    "method": AGGREGATION_METHODS,
    "interval": AGGREGATION_INTERVALS,
    "missing_data": AGGREGATION_MISSING_DATA,
}
_AGGREGATION_KEYS = frozenset(_AGGREGATION_VOCABULARIES)


@dataclass(frozen=True)
class SourceEntry:
    """One source-data reference from the background manifest."""

    path: str
    sha256: str


@dataclass(frozen=True)
class AggregationSpec:
    """One permitted time-span aggregation for a product.

    The publisher declares which aggregations a product allows; a
    consumer must never aggregate outside this set. `interval` is the
    target interval the aggregated output covers.
    """

    method: str
    interval: str
    missing_data: str


@dataclass(frozen=True)
class ProductSpec:
    """One product requested by the manifest, with its reviewed provenance.

    `citation` should be set for every product; `normal_period`
    and `method` apply to Normal products, `station_id` to AWS
    observation products, the `model_*`/`observation_*` pairing
    to Comparison variable specifications, `variable`/`domain`/`format`
    to the catalogue identity of every product, and `palette`,
    `transform`, `color_range` plus the `uris` analysis store to the
    publisher-declared map presentation of every product, and
    `aggregations` to the permitted time-span aggregations of a
    product (empty tuple when the product permits none).
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
    units: str = ""
    palette: str = ""
    transform: str = ""
    color_range: tuple[float, float] = ()
    uris: dict[str, str] = field(default_factory=dict)
    aggregations: tuple[AggregationSpec, ...] = ()
    version: str = ""
    """Explicit version-label pin for a corrective re-publication.

    Empty (the default) auto-assigns the next label for the release via
    ``Catalogue.next_version``. A reviewed corrective manifest may pin a
    later label (e.g. ``smoke1.2``) so the corrected candidate supersedes
    the earlier publication instead of colliding with it.
    """


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


def _parse_aggregations(product_id: str, raw: object) -> tuple[AggregationSpec, ...]:
    """Parse the permitted aggregations of one product, or fail loudly.

    A reviewed manifest may only declare combinations of the closed
    vocabularies above: `method` values how the span combines, `interval`
    is the target interval of the output, and `missing_data` states
    whether missing inputs are dropped (`exclude-missing`) or refuse the
    aggregation (`require-complete`). An absent or null declaration means
    the product permits no time-span aggregation; any other non-list is
    rejected.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"product {product_id}: aggregations must be a list")
    specs = []
    for entry in raw:
        if not isinstance(entry, dict) or set(entry) != _AGGREGATION_KEYS:
            raise ValueError(
                f"product {product_id}: each aggregation needs exactly "
                "method, interval and missing_data"
            )
        spec = AggregationSpec(
            method=entry["method"],
            interval=entry["interval"],
            missing_data=entry["missing_data"],
        )
        for name, allowed in _AGGREGATION_VOCABULARIES.items():
            value = getattr(spec, name)
            if value not in allowed:
                raise ValueError(
                    f"product {product_id}: unknown aggregation {name} {value!r}"
                )
        specs.append(spec)
    return tuple(specs)


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
                units=item.get("units", ""),
                palette=item.get("palette", ""),
                transform=item.get("transform", ""),
                color_range=tuple(item.get("color_range", ())),
                uris=dict(item.get("uris", {})),
                aggregations=_parse_aggregations(
                    item["product_id"], item.get("aggregations")
                ),
                version=str(item.get("version", "") or ""),
            )
            for item in data.get("products", [])
        ),
        revision=hashlib.sha256(content).hexdigest(),
    )
