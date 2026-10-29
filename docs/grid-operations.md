# Native published-grid operations

`darwin.grid_operations` is a supported Python interface with the matching CLI:

```sh
python -m darwin.grid_operations recipe.json --locations locations.json
```

`make_recipe(sources, operation, scientific_status=labels)` pins the recipe;
`replay(recipe, locations)` executes it. Source descriptors contain a local `path`,
`variable`, declared `units`, `product_id`, and exact `version`. Locations are deliberately excluded
from recipes: replay receives `{source_id: staged_directory}` separately. The CLI
prints the same result as Python, encoding arrays as lists and missing values as
JSON null. It reports `{code,message}` on stderr with exit status 2 on refusal.
Python raises `GridError` with the same code. Encoding NetCDF/GeoTIFF and HTTP
access decisions remain consumer responsibilities.

Results include a `darwin.grid-operation-result/1` receipt, verified by
`validate_result(result)`. It binds the recipe (sorted-key, two-space-indented JSON
plus newline) and science result (all fields except receipt, sorted-key compact
JSON, arrays as lists and nonfinite values as null), each by SHA-256. It records
execution completeness and derived status `unassessed`. A delivery encoder may
add a separate hash of its file bytes; that is not the science-result digest.

## Contract

Recipes declare `darwin.grid-operations/1`, definition `dashboard-native-grid/1`,
all source identities, one operation, exact software, and reported scientific
source labels. This explicitly named compatibility definition preserves existing
dashboard computations; it is **not a reviewed study recipe**, an acceptance
policy, or evidence that legacy timestamps are scientifically suitable.

Sources are immutable local Zarr directories, including metadata. SHA-256 hashes
each sorted relative file name and its bytes, framed with their byte lengths;
`size_bytes` counts file bytes. Replay verifies before and after reading. This is
stronger than the old payload-only publication hash (which excludes attributes).
The implementation file, Darwin/Python/NumPy/Zarr/numcodecs versions must match.
Do not rewrite metadata, regenerate or reconsolidate a published source for replay.

All operations explicitly name `source`, `bbox` (`xmin,ymin,xmax,ymax`), optional
`polygon` (one finite exterior ring, no holes) and optional native `level` index.
Only regular increasing native x/y axes are supported. No projection conversion,
curvilinear geographic crop, resampling, reconciliation or time fabrication occurs.
Polygon clipping is Sutherland-Hodgman plus shoelace area. Native coordinates are
preserved verbatim. Four-dimensional variables use `(time,level,y,x)`; physical
level coordinates/attributes/bounds are retained, never inferred from index.
Every time-dependent operation, including summaries and individual layers, requires
a matching nonempty numeric time axis with finite, unique, increasing values.
Missing or malformed native times are refused before any result or receipt is created.

| `name` | Explicit parameters and definition |
| --- | --- |
| `subset` | Full native time series, closed cell-center bbox; polygon retains every cell with positive footprint intersection, masking others as NaN. |
| `layer` | Native `timestep` index; defaults to zero only for non-vertical time series. Static `(y,x)` variables reject a timestep. Vertical layers require one explicit level and timestep. Same bbox/mask as subset. |
| `aggregate` | Inclusive `start,end`, `method`, and `missing_data` (`require-complete` or `propagate`). Mean uses NumPy arithmetic mean, not nanmean. `depth_sum` multiplies the published rate by native epoch-hour differences; the final selected store-end step repeats the previous selected duration. A one-step store-end window has no duration and is refused. This preserves the legacy definition, not a new interval policy. `component_mean` names an exact `partner`: mean earth-relative u10/v10 first, then hypot and `(degrees(atan2(-u,-v))+360)%360`. Coordinates, times and time units must match exactly, not just shapes. Require-complete counts NaNs inside the polygon for both components. |
| `regional_summary` | Explicit `mode`: whole-domain or land-only; the latter requires an exact `mask` source with matching coordinates and static values >0.5. Cell footprints start at x/y and extend by dx/dy (not centered footprints). Rectangle overlap or polygon fraction weights cells. Any cell with any non-finite native timestep is excluded entirely. Series sums retain legacy x-major/y-minor order; period mean/variance use math.fsum and population variance. Ten-bin distributions use the full series, including its final endpoint. |

Unknown fields/operations and incompatible sources are structured refusals. No
implicit conversion or source substitution is performed. Source acceptance labels
are caller-reported provenance, not independently verified acceptance evidence.
Derived acceptance is always `unassessed`. Execution completeness is complete only
if every supplied source explicitly reports complete. Any partial source keeps the
result partial; otherwise unknown source completeness stays unreported.
Consumers remain responsible for access and binding catalogue labels to versions.

Depth sums require declared `prcp` units `mm h-1`; wind requires component units
`m s-1`. Output units are recorded (`mm` for depth sums), not inherited from a rate
attribute. Optional explicit `declared_intervals` on aggregations reproduces the
legacy interval gate: daily 23..25 hours, monthly 672..744 hours, annual a January
start and a multiple of twelve monthly steps. At least one named interval must
match. This gate is also exposed as `window_satisfies_interval`. It never resamples
or creates a time axis. The dashboard pins all matching publisher declarations;
standalone operations without that constraint are explicit native-index reductions.

Golden tests cover each definition, exact source/metadata/software mutation,
multi-source alignment, missing-data rules, level metadata and Python/CLI equality.
