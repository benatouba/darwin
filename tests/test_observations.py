"""RED tests for D4: observation-source register and importer (darwin#6).

Acceptance:
- Register entries record URL, expected processed-product identity and
  activation status; raw entries are never imported.
- Exact duplicate source fields normalize to one audited canonical field;
  warning-only plausibility envelopes apply; structurally invalid products
  are rejected per the agreed rules.
"""

import pytest

from darwin.publisher.observations import (
    StructuralError,
    import_product,
    import_only_if_changed,
    load_register,
)


def _processed_entry(**overrides):
    from darwin.publisher.observations import SourceRegisterEntry

    base = {
        "url": "https://example.invalid/catalogue/30",
        "expected_product_id": "aws-p-minas-rojas-30",
        "kind": "processed",
        "active": True,
    }
    base.update(overrides)
    return SourceRegisterEntry(**base)


MINAS_HEADER = "timestamp,T,RH,WS,WSmax,WD,Pabs,PCP_diff_radar,PCP_acoustic,FOG,ST,Vwc,WSmax\n"
MINAS_ROW = "2022-03-24T00:00:00,16.5,80,4.5,4.5,180,1010,0.0,0.0,0,15.0,100,4.5\n"


def test_raw_register_entry_is_never_imported(tmp_path):
    entry = _processed_entry(kind="raw")
    with pytest.raises(StructuralError):
        import_product(entry, MINAS_HEADER + MINAS_ROW, station_id="30")


def test_inactive_entry_is_skipped():
    entry = _processed_entry(active=False)
    assert import_only_if_changed(entry, MINAS_HEADER + MINAS_ROW, {}, station_id="30") is None


def test_unchanged_checksum_produces_no_new_version():
    entry = _processed_entry()
    content = MINAS_HEADER + MINAS_ROW
    first = import_only_if_changed(entry, content, {}, station_id="30")
    assert first is not None and first.version == 1
    store = {entry.expected_product_id: first.raw_sha256}
    assert import_only_if_changed(entry, content, store, station_id="30") is None


def test_exact_duplicate_columns_normalize_to_one_canonical_field_with_audit():
    entry = _processed_entry()
    product = import_product(entry, MINAS_HEADER + MINAS_ROW, station_id="30")
    assert "WSmax" in product.fields
    assert product.fields.count("WSmax") == 1
    assert any("WSmax" in note and "duplicate" in note for note in product.audit)


def test_conflicting_duplicates_for_same_timestamp_and_field_are_rejected():
    entry = _processed_entry()
    header = "timestamp,T,WSmax,WSmax\n"
    row = "2022-03-24T00:00:00,16.5,4.5,9.9\n"
    with pytest.raises(StructuralError):
        import_product(entry, header + row, station_id="30")


def test_missing_timestamp_column_is_rejected():
    entry = _processed_entry()
    with pytest.raises(StructuralError):
        import_product(entry, "T,RH\n16.5,80\n", station_id="30")


def test_unknown_extra_columns_are_retained_as_provenance():
    entry = _processed_entry()
    content = "timestamp,T,mystery_sensor\n2022-03-24T00:00:00,16.5,42\n"
    product = import_product(entry, content, station_id="30")
    assert "mystery_sensor" in product.fields


def test_implausible_value_is_retained_with_warning():
    entry = _processed_entry()
    content = "timestamp,T\n2022-03-24T00:00:00,999.0\n"
    product = import_product(entry, content, station_id="30")
    assert "T" in product.fields
    assert any("T" in warning for warning in product.warnings)


def test_register_loader_reads_version_controlled_yaml(tmp_path):
    path = tmp_path / "observation-sources.yaml"
    path.write_text(
        "- url: https://example.invalid/catalogue/30\n"
        "  expected_product_id: aws-p-minas-rojas-30\n"
        "  kind: processed\n"
        "  active: true\n",
        encoding="utf-8",
    )
    (entry,) = load_register(path)
    assert entry.url == "https://example.invalid/catalogue/30"
    assert entry.kind == "processed"
    assert entry.active is True
