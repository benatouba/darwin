"""Station location records with approval gate (D5).

A new or changed station identity, coordinates, elevation, exposure or
sensor setup creates a new time-bounded record — records are never
overwritten. Products tied to unapproved records stay
researcher-invisible until an admin approves the record.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class StationLocationRecord:
    """One canonical, time-bounded station location record (pending approval)."""

    station_id: str
    display_name: str
    lon: float
    lat: float
    elevation_m: float
    exposure: str
    sensor_heights: tuple[tuple[str, float], ...]
    valid_from: str
    valid_to: str | None = None
    approved: bool = False
    approved_by: str | None = None
    approved_at: str | None = None


class StationRegistry:
    """Registry of station location records with an admin approval gate."""

    def __init__(self) -> None:
        self._records: dict[str, list[StationLocationRecord]] = {}

    def register(self, **fields) -> StationLocationRecord:
        """Register a station; a re-registration end-dates the open record.

        A new or changed identity, coordinates, elevation, exposure or
        sensor setup creates a new time-bounded record — the previous
        record is end-dated, never overwritten — and the new record
        enters pending approval, researcher-invisible.
        """
        record = StationLocationRecord(**fields)
        chain = self._records.setdefault(record.station_id, [])
        if chain and chain[-1].valid_to is None:
            previous = chain[-1]
            chain[-1] = replace(previous, valid_to=record.valid_from)
        chain.append(record)
        return record

    def history(self, station_id: str) -> tuple[StationLocationRecord, ...]:
        """Return every time-bounded record for a station, oldest first."""
        return tuple(self._records[station_id])

    def approve(self, station_id: str, approved_by: str, approved_at: str) -> StationLocationRecord:
        """Approve the open record for a station, making it researcher-visible.

        The approval stores who approved and when; the previously
        registered object is left untouched and the chain entry is
        replaced, never mutated in place.
        """
        chain = self._records[station_id]
        record = chain[-1]
        approved_record = replace(
            record, approved=True, approved_by=approved_by, approved_at=approved_at
        )
        chain[-1] = approved_record
        return approved_record

    def visible_stations(self) -> tuple[StationLocationRecord, ...]:
        """Return the latest approved record per station.

        Pending records stay researcher-invisible; superseded end-dated
        records are hidden once a newer record exists, except that the
        most recently approved record remains visible while its
        successor is still pending approval.
        """
        visible: list[StationLocationRecord] = []
        for chain in self._records.values():
            for record in reversed(chain):
                if record.approved:
                    visible.append(record)
                    break
        return tuple(visible)
