# CJC simulation-output inspection

Darwin supports **`cjc.simulation-output/1`** through
`darwin.simulation_output`, without CJC, Slurm, a sibling checkout or a server.
This adapter inspects declarations and optional artifact byte identities. It is
not yet a native WRF record reader or an analysis-ready dataset importer.

```python
from darwin.simulation_output import load_manifest, inspect_manifest

document = load_manifest("manifest.json")
report = inspect_manifest(document, locations={"opaque-artifact-id": "/resolved/native.nc"})
```

The equivalent CLI uses a JSON object mapping IDs to resolved paths:

```sh
python -m darwin.simulation_output manifest.json --locations locations.json
```

Use `validate_manifest(document)` for an independent checked copy and
`contract_schema()` (or CLI `--schema`) for the supported structural schema.
Unknown contract versions fail with `unsupported_contract`; malformed structures,
coverage or references fail with `invalid_manifest`. Missing and checksum-invalid
artifacts produce `missing_artifact` and `corrupt_artifact`. The CLI prints JSON
errors to stderr and exits 2. Python raises `SimulationOutputError` with `.code`.
No automatic version substitution, path guessing or missing-value repair occurs.

Each view owns `(UTC midnight, following midnight]` for one domain, with explicit
native record indices and producing segments. Complete and partial views are
distinct; a fully absent domain is a partial view with all expected times missing.
This convention applies only to this new contract, not existing archive products.

Locations are supplied by the deployment/storage adapter, not embedded in manifests.
Without them, inspection reports `payload_verification=not_checked`. With them,
all declared payload byte counts and SHA-256 digests are checked by streaming.
Source files must already be sealed; checksums do not prevent later modifications.

Every report preserves the limits of this operation:

- `native_record_metadata=not_checked`: the manifest's time/grid/domain/units are
  declarations, not yet verified against NetCDF records.
- `referenced_receipts=not_authenticated`: lineage/build/evidence references have
  not been resolved and authenticated.
- `scientific_status=not_assessed`: neither a complete declaration nor a valid
  checksum grants acceptance, publication readiness or dashboard publication.

CJC owns this contract and fixture. Darwin bundles its structural schema snapshot
in `darwin/cjc-output-v1.schema.json`; tests use the producer-owned fixture snapshot
in `tests/data/cjc-output-v1.fixture.json`. Updates require producer/consumer
compatibility checks, not independent edits or imports of CJC's private classes.
The producer compatibility runner also exercises fresh complete/partial/absent
views and missing/corrupt/unsupported/false-complete failures through the installed
Darwin CLI. Native validation and scientific pilot acceptance remain future work.
