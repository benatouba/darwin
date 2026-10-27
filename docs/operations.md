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
