# Supported dataset operations — Darwin 1.2

The public entry points are `darwin.operations.make_recipe`, `validate_recipe`,
`source_identity`, `execute`, and `python -m darwin.operations`. They use the same
implementation and the packaged producer-owned `operations-v1.schema.json`.
`OperationError.code` is the structured error boundary. Comparison remains the
separate supported `darwin.comparison` interface (`native-intervals-v1`).

```python
from darwin.operations import make_recipe, execute

locations = {"model-output-version-1": "/adapter/resolved/native.nc"}
recipe = make_recipe(locations, [
    {"operation": "select", "parameters": {"variables": ["prcp"]}},
])
result = execute(recipe, locations, "/new/immutable/product-version-1")
```

Serialize the recipe and deployment-local ID/path map separately:

```sh
python -m darwin.operations recipe.json --locations local-paths.json --output new-product
```

The atomic output directory contains `data.nc`, the exact `recipe.json`, and
`result.json` with SHA-256/size of the data and SHA-256 of the recipe. Success
means execution complete, **scientifically unassessed**, and not published.
Existing destinations, including identical retries, are never replaced. A new
version/destination is required. Inputs are checked before and after execution.

## Version-1 semantics

| Operation | Parameters and scope |
| --- | --- |
| `select` | Explicit data-variable names; unknown variables fail |
| `cut` | `dimension`, ISO `start`, ISO `end`; standard decoded datetime64 coordinates only |
| `crop` | `dimension`, numeric `start`, numeric `end`; 1-D increasing coordinates only |
| `integrate` | `dimension`, `variables`, `method: sum-discrete-amounts`; sum amounts, retain units, propagate missing values, record `cell_methods`; NOT rate/time integration |
| `assemble` | `dimension`, positive `cadence_seconds`; ordered source files, exact inventories, static fields/grids, metadata, units, calendars and cadence |

Selections are half-open `[start,end)`. Empty selections fail. Multi-file inputs
must be assembled before selections. Assembly never sorts, fills, deduplicates,
interpolates, averages or converts units. Incompatible global attributes and
encoded time units fail too: reconciliation requires a separately supported,
explicit transformation. Existing files must provide actual native coordinates.
The recipe pins bytes and software; relocating unchanged bytes is allowed.

## Explicit implementation gaps

This interface currently reads CF-style NetCDF through xarray/netCDF4. It is not
a native WRF `Times` decoder or a CJC manifest-to-native-data adapter (#13).
Curvilinear geographic crops, time-bound/accumulation-aware rate integrals,
resampling and nonstandard-calendar cuts are not implemented. Comparison and
specialized existing publisher builders are not silently redirected here.

Darwin's historical package still has broad plotting/research dependencies; the
optional-dependency split remains outstanding. PyYAML is now declared directly
because the publisher imports it. No sibling project, Slurm or dashboard server
is needed to call the interface.

The Linux local-filesystem atomic writer fails closed without `renameat2` support;
it is not an object-store transaction or an fsync durability guarantee. Each
legacy station/model artifact commits independently. Hidden crash staging may
be cleaned only after confirming no live writer owns it. Existing immutable
artifacts/recipes are never rewritten; choose new versions for migrated outputs.
Historical smoke handoffs and publisher lifecycle APIs retain their legacy
meaning and are not evidence of scientific acceptance or dashboard publication.

## Immutable artifact writes

### Write-path inventory

Every production write in `darwin/publisher` and `darwin.operations` goes through
`darwin.publisher.immutable.staged_artifact` (stage beside the destination, publish
with no-replace rename):

| Writer | Artifact | Identity record sources |
| --- | --- | --- |
| `multiyear_store.build_year_store` | `<product>.zarr` | reviewed year-file sha256 |
| `temperature_store.build_temperature_store` | t2 seed `.zarr` | pinned `SOURCE_SHA256` |
| `analysis_stores.build_smoke_analysis_stores` | prcp slice, landmask `.zarr` | sha256 of the bytes read |
| `station_series.build_station_series` (and the `comparison_series` wrapper) | model and observation `.zarr` | GAR file; observation CSV + metadata XML |
| `storage.publish_analysis_product` | `analysis.zarr` | manifest sources; recipe = manifest revision |
| `storage.write_download_netcdf` | `download.nc` | manifest sources; parameters pin the analysis-store sha256 |
| `smoke.build_smoke_release` | handoff JSON | release manifest (revision) + manifest sources |
| `operations.execute` | bundle directory | `result.json` binds `data.nc` and `recipe.json` (no sidecar) |

Out of scope, not published products: matplotlib figures in research/plotting
modules (`plots.py`, `core.py`, `accessors.py`, `plot_*.py`, `geogrid_simulator.py`)
and one-off maintenance scripts (`create_sh.py`, `convert.py`,
`change_gar_names.py`). They must not be used to produce dashboard products.
If they ever are, migrate them to `staged_artifact` first.

### Identity records

Publishing with `provenance=` writes `<name>.identity.json` (contract
`darwin.artifact-identity/1`) beside the artifact. It records the artifact kind,
size, and the sha256 of every file. The artifact `sha256` covers the
`<relative path>\0<size>\0<file sha256>\n` lines in sorted path order; a single
file uses path `.`. The record also holds the builder, any recipe and parameters,
and the sources' ids and sha256s. Every record has `artifact_class: derived` and
`scientific_status: unassessed`. Provenance must name at least one source with a
lowercase sha256. A destination equal to, inside, or containing a source path is
refused, so derived products can't overwrite original simulation output or be
confused with it. Links and special files are refused. Records have no timestamps.
`verify_artifact(path)` re-hashes the bytes and returns the record. A missing
record, a different name or any byte difference raises `ArtifactIntegrityError`.
Each historical version verifies on its own.

### Retry, interruption and concurrency

- Default `retry="refuse"`: an existing artifact or identity record is refused
  before any work (`ArtifactConflict`, a `FileExistsError`). Builders that run
  from a release manifest use this default. The smoke handoff embeds a build
  timestamp, so a rebuild into the same path always conflicts.
- `retry="verify-identical"`: the writer stages its full output and then tries
  to publish. If the target already exists, the retry counts as idempotent
  success only when every file and the identity record are byte-identical (a full
  byte comparison, never size/mtime/hash shortcuts). Any difference raises
  `ArtifactConflict` and leaves existing bytes and records untouched. Equal bytes
  with different provenance are a conflict, not a match.
- Interruption before publication leaves only a hidden
  `.<name>.staging-*` directory and never a partial final artifact. The artifact
  is published before its identity record. A crash between the two leaves an
  *unbound* artifact that `verify_artifact` rejects. A `verify-identical` retry
  with the same inputs completes the record. A differing retry conflicts.
- Concurrent writers: exactly one rename wins. Losers get `ArtifactConflict`, or
  succeed under `verify-identical` only when their bytes are identical.

### Migration plan for existing stores

Nothing here rewrites, relabels, re-hashes in place, or adds files inside
existing published stores. Existing artifacts written before identity records
existed, including the dashboard's bundled seed stores, the smoke handoffs and the
`s3://` analysis uris, stay where they are with their current bytes. They are
*unbound legacy versions*:

1. **Inventory (read-only).** For each existing store, record its path, the
   `artifact_digest` output and, for Zarr, `multiyear_store.hash_store`. Record
   the source identities it already claims (`source_sha256`, `source_gar_sha256`,
   the manifest revision). Keep this inventory outside the stores, in the
   consuming project's release evidence.
2. **Rebuild as new versions.** Run the current builder from the same reviewed,
   checksum-pinned sources into a **new versioned destination**, never the old
   path, so it gets an identity record. Compare data payloads with `hash_store`.
   Equal payloads show the legacy store's data is reproducible. A difference is a
   finding to review, never something to fix by overwriting.
3. **Switch consumers by version.** Point the dashboard catalogue at the new,
   identity-bound version through its normal publication and current-version
   process. Darwin does not publish or change pointers. Legacy versions stay
   addressable as superseded history.
4. **Retire only by owner decision.** Deleting or archiving a legacy store is a
   data-owner/dashboard publication decision with its own evidence, outside these
   builders.

Writing an identity record next to a legacy artifact afterwards is not supported:
Darwin can't prove which builder and sources produced bytes it didn't stage.
Object-store targets still need a conditional-write adapter (put-if-absent plus a
checksum) before these guarantees reach remote storage.
