"""Observation-source register and AWS observation-product importer (D4).

Implements the import side of the publisher boundary: a reviewed,
version-controlled register entry plus a downloaded processed artifact produce
an immutable observation version. Raw entries are never imported, exact
duplicate fields collapse to one audited canonical field, unexpected but
physically plausible values are retained with warnings, and structurally
invalid products are rejected.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import yaml

#: Version of the plausibility envelopes applied at import.
ENVELOPE_VERSION = "1.0.0"

#: Headers recognised as the observation timestamp column.
TIMESTAMP_CANDIDATES = frozenset({"", "timestamp", "time", "datetime"})

#: Warning-only plausibility envelopes: field -> (low, high).
PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "T": (-80.0, 60.0),
    "RH": (0.0, 100.0),
    "SLR": (0.0, 1400.0),
    "WS": (0.0, 90.0),
    "WSmax": (0.0, 120.0),
    "WD": (0.0, 360.0),
    "Pabs": (800.0, 1100.0),
}


class StructuralError(ValueError):
    """Raised when an observation product is structurally invalid."""


@dataclass(frozen=True)
class SourceRegisterEntry:
    """One reviewed Observation-source register entry.

    Only ``processed`` entries are importable; raw entries are refused so the
    publisher can never ingest a raw station product.
    """

    url: str
    expected_product_id: str
    kind: str = "processed"
    active: bool = True


@dataclass(frozen=True)
class ObservationProduct:
    """An immutable imported AWS observation-product version."""

    product_id: str
    station_id: str
    version: int
    raw_sha256: str
    fields: tuple[str, ...]
    values: dict[str, tuple[str, ...]] = field(default_factory=dict)
    audit: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def checksum_of(artifact_text: str) -> str:
    """Return the sha256 checksum of a downloaded source artifact."""
    return hashlib.sha256(artifact_text.encode("utf-8")).hexdigest()


def load_register(path: str | Path) -> list[SourceRegisterEntry]:
    """Load a version-controlled YAML Observation-source register.

    The register is a top-level YAML list of entries.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [
        SourceRegisterEntry(
            url=item["url"],
            expected_product_id=item["expected_product_id"],
            kind=item.get("kind", "processed"),
            active=item.get("active", True),
        )
        for item in raw
    ]


def _timestamp_index(header: list[str]) -> int:
    for position, name in enumerate(header):
        if name.strip().lower() in TIMESTAMP_CANDIDATES:
            return position
    raise StructuralError("no unambiguous timestamp column")


def import_product(
    entry: SourceRegisterEntry, artifact_text: str, station_id: str
) -> ObservationProduct:
    """Import one processed artifact into an immutable observation version."""
    if entry.kind != "processed":
        raise StructuralError(
            f"refuses to import raw entry {entry.expected_product_id!r}"
        )
    rows = list(csv.reader(artifact_text.splitlines()))
    if len(rows) < 2:
        raise StructuralError("no unambiguous timestamp column")
    header = [name.strip() for name in rows[0]]
    time_col = _timestamp_index(header)
    data = rows[1:]
    if not data or not any(row[time_col].strip() for row in data if len(row) > time_col):
        raise StructuralError("timestamps cannot be parsed")

    positions: dict[str, list[int]] = {}
    for position, name in enumerate(header):
        if position == time_col or not name:
            continue
        positions.setdefault(name, []).append(position)

    audit: list[str] = []
    canonical: dict[str, int] = {}
    for name, cols in positions.items():
        if len(cols) == 1:
            canonical[name] = cols[0]
            continue
        first, rest = cols[0], cols[1:]
        for other in rest:
            for row in data:
                left = row[first].strip() if len(row) > first else ""
                right = row[other].strip() if len(row) > other else ""
                if left != right:
                    raise StructuralError(
                        f"conflicting duplicate values for field {name!r}"
                    )
        canonical[name] = first
        audit.append(
            f"normalized duplicate field {name!r}: "
            f"{len(cols)} identical columns collapsed to one"
        )

    values = {
        name: tuple(row[pos].strip() if len(row) > pos else "" for row in data)
        for name, pos in canonical.items()
    }

    warnings: list[str] = []
    for name, bounds in PLAUSIBILITY.items():
        if name not in values:
            continue
        low, high = bounds
        for raw_value in values[name]:
            if raw_value in ("", "NA", "NaN"):
                continue
            try:
                number = float(raw_value)
            except ValueError:
                continue
            if not low <= number <= high:
                warnings.append(
                    f"{name}={raw_value} outside plausibility envelope "
                    f"[{low}, {high}] (envelope {ENVELOPE_VERSION}); value retained"
                )
                break

    return ObservationProduct(
        product_id=entry.expected_product_id,
        station_id=station_id,
        version=1,
        raw_sha256=checksum_of(artifact_text),
        fields=tuple(canonical),
        values=values,
        audit=tuple(audit),
        warnings=tuple(warnings),
    )


def import_only_if_changed(
    entry: SourceRegisterEntry,
    artifact_text: str,
    store: Mapping[str, str],
    station_id: str,
) -> Optional[ObservationProduct]:
    """Import only active entries whose artifact checksum is new.

    ``store`` maps expected product IDs to already-imported raw checksums.
    Returns the new observation version, or ``None`` when the entry is
    inactive (skipped) or its checksum was already imported (unchanged).
    """
    if not entry.active:
        return None
    if store.get(entry.expected_product_id) == checksum_of(artifact_text):
        return None
    return import_product(entry, artifact_text, station_id=station_id)


__all__ = [
    "ENVELOPE_VERSION",
    "ObservationProduct",
    "SourceRegisterEntry",
    "StructuralError",
    "checksum_of",
    "import_only_if_changed",
    "import_product",
    "load_register",
]
