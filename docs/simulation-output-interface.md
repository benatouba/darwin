# CJC simulation output and existing archives

Darwin consumes CJC's versioned simulation-output views through
`darwin.simulation_output` and assembles them through `darwin.operations`. Neither
needs CJC, `darwin-wrf-setups`, Slurm, a sibling checkout or a server. Darwin never
imports CJC: the schemas and fixtures it tests against are copies of the
producer-owned files.

| Contract | Darwin support |
| --- | --- |
| `cjc.simulation-output/1` | Declarations and optional byte identity only (inspection). |
| `cjc.simulation-output/2` | Declarations, bytes, native WRF records, completion records and strict assembly. |
| anything else | `unsupported_contract` |

## Inspecting a view

```python
from darwin.simulation_output import load_manifest, inspect_manifest

view = load_manifest("view.json")
report = inspect_manifest(
    view,
    locations={"artifact-id": "/resolved/wrfout_d02_...nc"},  # optional
    completions={"segment-id": "/resolved/run-root/s1-complete.json"},  # optional, v2
)
```

```sh
python -m darwin.simulation_output view.json --locations artifacts.json --completions completions.json
python -m darwin.simulation_output --schema [--contract cjc.simulation-output/2]
```

Locations are flat JSON objects mapping opaque IDs to paths. A storage or
deployment adapter supplies them; manifests never contain paths. The CLI prints
`{"code", "message"}` to stderr and exits with status 2. Python raises
`SimulationOutputError` with `.code`. Nothing guesses paths, substitutes
versions or repairs missing values.

For v2, `inspect_manifest` (and `verify_native`) checks these in order:

1. **Declarations.** The bundled producer schema plus the v2 rules. The calendar
   day axis is right-closed `(start, end]`. Status must be derived consistently:
   `complete`, `partial`, `failed` or `missing`. Run-root, segment and artifact
   references must agree. Failed segments are never data. The grid digest and
   `GRID_ID` must match the domain. Failures: `invalid_manifest`.
2. **Bytes.** Each artifact's size and SHA-256 is checked by streaming
   (`missing_artifact`, `corrupt_artifact`).
3. **Native records.** `darwin.wrf_history.read_wrf_history` decodes the WRF
   `Times` character array and reads `XTIME`, the projection attributes, the mass
   dimensions and the variable `units`/dimensions/`stagger`. It reports
   disagreement as `native_records`, `native_grid`, `native_variable` or
   `calendar_mismatch`. `XTIME` offsets and any `XTIME:calendar` are checked
   under the declared calendar. Files that are not WRF history raise
   `native_structure` or `native_unreadable`.
4. **Completion records (optional).** Each completed segment's record is read in
   its own run root. Its bytes and its `fingerprint` must equal that root's
   `bundle_fingerprint` (`missing_completion`, `changed_completion`,
   `fingerprint_mismatch`). Recovery roots are checked where they are. Nothing
   is copied or rewritten.

Reports always carry `scientific_status=not_assessed`. `native_record_metadata`
and `referenced_receipts` say what was actually checked. Completion records are
digest-checked, not authenticated receipts.

## Assembling views: `assemble_simulation_output`

```python
from darwin.operations import execute, make_recipe
from darwin.simulation_output import artifact_ids, assemble_operation

views = [day1, day2]  # chronological, adjacent v2 views
sources = {key: artifact_locations[key] for key in artifact_ids(views)}
recipe = make_recipe(sources, [
    assemble_operation(views, ["T2", "U"]),
    {"operation": "cut", "parameters": {"dimension": "Time", "start": "...", "end": "..."}},
])
report = execute(recipe, sources, "/new/immutable/product-version")
```

This step must come first. The recipe embeds the full view manifests, and its
sources must be exactly the views' artifacts, with the declared sizes and digests
(`invalid_recipe`). `check_views(views, variables)` runs these checks without
opening any native file, at recipe creation and again at replay. The views must:

- be distinct (`view_conflict`) and come from the same simulation and domain
  (`incompatible_assembly`);
- use the same calendar (`incompatible_calendar`) and cadence
  (`incompatible_cadence`);
- keep the same meaning for every shared run root, segment and artifact ID
  (`run_root_conflict`, `segment_conflict`, `artifact_conflict`);
- among the views with data, declare an identical grid/projection
  (`incompatible_grid`), units (`incompatible_units`) and dimensions/staggering
  (`incompatible_variable`) for each requested variable, and declare every
  requested variable (`variable`);
- be in chronological order (`time_order`), with no two views covering the same
  day (`time_overlap`) and no day missing between them (`time_gap`);
- contain at least one record (`empty_selection`) and no missing record between
  covered records (`time_gap`).

At execution, each view is verified natively as above, and only the selected
records are read. Values keep their native dtype, dimensions (including
staggered ones) and variable attributes. There is no masking, scaling or unit
change.

The dataset has a `Time` dimension coordinate. For the `standard` calendar it is
decoded `datetime64`, written as proleptic Gregorian seconds since the first
view's start. For `noleap` it stays encoded as integer seconds with
`calendar: noleap`, and `cut` refuses that non-`datetime64` axis. Per-record
coordinates keep the native `Times` string and the `source_artifact`,
`source_record_index`, `source_segment` and `source_run_root`. Global attributes
record the source contract, simulation, domain, calendar, cadence, the full
grid, the views with their canonical-JSON digests, `source_coverage_status`,
`source_missing_times` and `scientific_status=unassessed`.

`result.json` gains `source_coverage`, which holds `status`
(`complete` only if every view is complete), `covered_times`, `missing_times`
and each view's `id`, `sha256`, `status` and `reason`. Leading and trailing
missing records, such as a partial last day followed by failed or missing days,
are reported this way and never filled. Partial output stays partial.

Conversion, interpolation, regridding, averaging, gap filling and overlap or
metadata reconciliation are not operations of version 1. A recipe that names one
is refused (`unsupported_operation`; see [operations.md](operations.md)), so
any inconsistency above stops assembly instead of being approximated.

## Existing archives (no CJC, no setups)

Existing datasets do not need a CJC manifest. CF-NetCDF archive files are
sources of `darwin.operations` directly: `make_recipe({id: path}, operations)`
pins their bytes, and `select`/`cut`/`crop`/`integrate`/`assemble` apply. That
interface needs only the Darwin wheel and its runtime dependencies, which
`tests/test_installed_release.py` proves for an installed wheel (together with
v2 inspection and assembly from the copied fixtures). Native WRF history files
without a view can be inspected with `darwin.wrf_history.read_wrf_history` and
`check_calendar`. Assembling them into a product needs a v2 view, because Darwin
does not derive coverage or segment ownership from file names or directories.

## Fixtures and compatibility

| Darwin file | Producer origin (CJC `integration/project-integration`) |
| --- | --- |
| `darwin/cjc-output-v1.schema.json` | `DailyView.model_json_schema()` without titles |
| `darwin/cjc-output-v2.schema.json` | `DailyViewV2.model_json_schema()` without titles (5983323) |
| `tests/data/cjc-output-v1.fixture.json` | v1 transport fixture |
| `tests/data/cjc-simulation-output-v2/` | byte-for-byte copy of `cjc/workflow/fixtures/simulation-output-v2/` at 5983323 |

`tests/test_simulation_output_v2.py` pins the producer's published payload
digests. Updating a copy requires a producer/consumer compatibility check, not an
independent edit. CJC's `tools/check_output_consumer.py` still exercises only v1
through Darwin's CLI (`--schema` without `--contract` still exports v1). A v2
consumer runner belongs to CJC.

## Remaining gaps

- The fixtures are synthetic WRF-like NETCDF3 files. Darwin's reader has not
  been run against retained real `wrfout` output: frames per file, real `XTIME`
  attributes, compiled calendars and lat/lon fields.
- Only the `standard` (proleptic Gregorian) and `noleap` calendars are
  supported. 360-day and other calendars are refused. Assembly of a `noleap`
  view set is implemented but has no native fixture yet.
- Lat/lon (`XLAT`/`XLONG`) and other static fields are not yet carried into
  assembled products. Geographic crops of curvilinear WRF grids are not
  supported.
- Completion and evidence references are digests, not authenticated receipts.
- Assembly produces a scientifically unassessed product. Acceptance and
  publication are separate.
