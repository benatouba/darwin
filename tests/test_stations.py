"""Station location records with approval gate (D5, darwin#7)."""

from darwin.publisher.stations import StationRegistry


def _minas_rojas(**overrides):
    fields = {
        "station_id": "30",
        "display_name": "Minas Rojas",
        "lon": -90.3673,
        "lat": -0.618625,
        "elevation_m": 619.0,
        "exposure": "open ridge",
        "sensor_heights": (("T", 2.0), ("WS", 10.0)),
        "valid_from": "2022-03-24",
    }
    fields.update(overrides)
    return fields


def test_new_station_is_pending_and_researcher_invisible():
    registry = StationRegistry()
    record = registry.register(**_minas_rojas())
    assert record.approved is False
    assert registry.visible_stations() == ()
    registry.approve("30", approved_by="admin", approved_at="2026-01-01")
    assert [r.station_id for r in registry.visible_stations()] == ["30"]


def test_changed_station_creates_new_record_and_never_overwrites():
    registry = StationRegistry()
    registry.register(**_minas_rojas())
    registry.approve("30", approved_by="admin", approved_at="2026-01-01")
    moved = registry.register(
        **_minas_rojas(elevation_m=625.0, valid_from="2024-06-01")
    )
    assert moved.approved is False
    assert moved.elevation_m == 625.0
    history = registry.history("30")
    assert len(history) == 2
    assert history[0].elevation_m == 619.0
    assert history[0].valid_to == "2024-06-01"
    assert history[1].valid_to is None
    # The approved historic record stays visible; the pending change does not.
    assert [r.elevation_m for r in registry.visible_stations()] == [619.0]


def test_approve_does_not_mutate_the_registered_record_and_stores_provenance():
    registry = StationRegistry()
    record = registry.register(**_minas_rojas())
    registry.approve("30", approved_by="admin", approved_at="2026-01-01")
    assert record.approved is False
    stored = registry.history("30")[0]
    assert stored.approved is True
    assert stored.approved_by == "admin"
    assert stored.approved_at == "2026-01-01"


def test_only_the_latest_approved_record_is_visible():
    registry = StationRegistry()
    registry.register(**_minas_rojas())
    registry.approve("30", approved_by="admin", approved_at="2026-01-01")
    registry.register(**_minas_rojas(elevation_m=625.0, valid_from="2024-06-01"))
    registry.approve("30", approved_by="admin", approved_at="2026-02-01")
    assert [r.elevation_m for r in registry.visible_stations()] == [625.0]
