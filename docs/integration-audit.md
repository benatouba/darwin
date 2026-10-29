# Darwin integration audit — 2026-10-28

Scope: Darwin issues #12–#18, including their downloaded comments, audited against
the implementation at `16fc62a` plus this audit's fixes. This is not a deployment,
scientific acceptance, or an assertion that all four projects have migrated.

## Producer implementation and executable evidence

| Issue | Implemented boundary and tests | Remaining limit |
| --- | --- | --- |
| #12 | `darwin.operations`: one Python/CLI implementation, exact-byte/source/software recipes, explicit operation inventory and errors. `test_operations.py` checks equivalence, replay, no-clobber and unsupported operations. This audit adds the missing producer result schema, synthetic recipe/result shape fixtures, `validate_result`, and installed-wheel coverage for `darwin.comparison_policy`. `test_installed_release.py` installs runtime-only dependencies and runs outside the checkout. | General worker operations listed below are not implemented. Consumers must pin a newly built wheel after algorithm bytes change. |
| #13 | `darwin.simulation_output` validates producer CJC v2 manifests and native records; `assemble_simulation_output` exposes strict assembly through operations. `test_simulation_output_v2.py` and `test_simulation_output_assembly.py` cover split roots, partial coverage, incompatible definitions, grids, calendars, units and source tampering. CF-NetCDF archives remain an independent source interface. | Retained real WRF history validation is separate from synthetic producer fixtures. No implicit conversion, interpolation, fill or overlap reconciliation. |
| #14 | `wrf_products`, `wrf_diagnostics`, `native_diagnostics`; inventory in `cjc-science-inventory.md`. Independent CJC-produced golden files and copied reference examples in `test_wrf_products_golden.py`/`test_wrf_products_reference.py`; Python/CLI and native build provenance in `test_wrf_products_interface.py`. | CJC caller adoption/deprecation is owned by CJC. Software parity does not approve real profiles, units or projections; setups #10/#12 retain those gates. |
| #15 | Native monthly model series, authoritative timestamp/bounds/timezone and checksum metadata; explicit observation timezone and completeness. `test_series_alignment.py`, `test_station_series.py`, `test_time_axis.py` cover timestamp joins, shifted/disjoint years, gaps and incompatible cadences. `station-series-time-contract.md` inventories affected smoke products and immutable replacements. | Existing model builder emits monthly; observation builder emits hourly/daily. They do not yet yield a comparable native-cadence pair. No fabricated disaggregation or approved real comparison. |
| #16 | `comparison_policy` binds exact stores/station/sampling, requested timeframe, policy bytes and setups requirements; returns declared scalar/distribution/precipitation metrics and undefined reasons. `test_comparison_policy.py` covers independent examples, completeness/exclusions, source/policy binding, CLI/replay and signed dataset acceptance. | Annual cadence and circular station diagnostics are explicitly refused. Real reviewed policies/requirements and wind-store contracts are not invented. Dashboard migration remains consumer-owned. |
| #17 | `publisher.immutable` stages on the destination filesystem and uses no-replace publication. Identity records bind bytes, recipe and sources. `test_immutable.py` and `test_artifact_provenance.py` cover races, interruption, identical/conflicting retries, source protection and historical versions. Writer inventory/migration is in `operations.md`. | Linux local-filesystem guarantee only, not remote/object-store atomicity, multi-artifact transactions or fsync durability. Legacy unbound stores remain unchanged. |
| #18 | `acceptance`, `handoff`, `study_contracts`, `decision_signature`: versioned schemas/fixtures, standalone preparation and portable exact dataset/requirements/evidence/authority binding. `test_acceptance.py`, `test_handoff.py`, `test_handoff_fixtures.py`, `test_comparison_policy.py` include tampering and signed decisions. | Real authority keys, authorization, requirements review and scientific decisions remain human gates. Darwin never publishes into the dashboard catalogue. |

## Defects reproduced and fixed here

Failing regressions established that CF assembly accepted empty/numeric/string
singleton axes, and `integrate` emitted zero for an empty amount axis. Assembly's
float `total_seconds()` also discarded a one-nanosecond timestamp gap and accepted
it as exact hourly cadence. Nonempty decoded axes and exact duration comparisons
now fail closed. Valid singleton standard and 360-day calendars remain supported.
The operation result previously had a version string but no producer-owned schema;
it now has a packaged schema and public validator, without changing report fields.

The legacy `darwin.comparison` interface also echoed a store's self-declared
`accepted` status without validating an assessment. A regression reproduced that
promotion. Legacy source `scientific_status` is now always `unassessed`; raw
metadata is preserved as `store_declared_status`. Only `comparison_policy` with
verified signed dataset assessments can report validated source acceptance. This
changes legacy recipe software identities and adds a source metadata field;
consumers must rebuild/re-pin the wheel and regenerate recipes, not edit old ones.

The runtime-only wheel check also exposed missing `libz.so.1` for the manylinux
NumPy wheel's Fortran runtime in this Nix worktree. `devenv.nix` now explicitly
provides zlib; no wheel, cache or scientific dependency was patched in place.

## Dashboard general worker migration boundary

Update, 2026-10-29: the limitations inventoried below describe the original audit
at `e85f028`, not the final worker-local interface. The subsequent
[`darwin.grid_operations`](grid-operations.md) compatibility migration implements
native Zarr identities, index/level selection, polygons, regional statistics,
temporal reductions and vector wind using the preserved dashboard definitions.
Its standalone Python/CLI recipes and result receipts do not grant scientific
acceptance. All time-dependent requests now refuse missing, mismatched, nonfinite,
duplicate or decreasing native time coordinates before producing results.

Current public operations support CF-NetCDF variable select, half-open coordinate
cut/crop, discrete-amount sums and strict assembly, plus native WRF assembly from
CJC manifests. This is **not** the dashboard's complete worker science API:

* Zarr source identity/opening needs a directory byte-identity contract and explicit
  source-format declaration. Passing Zarr to a NetCDF reader is not migration.
* Index-based selection needs explicit dimension/index semantics; coordinate crop
  is not an equivalent replacement for `isel` on an index request.
* Polygon masking needs a pinned geometry/CRS, boundary convention and geographic
  grid handling. A rectangular coordinate crop is not a polygon mask.
* Regional statistics need an explicit mask/weighting, missing-value and reduction
  policy. An unweighted sum cannot be silently substituted for an area statistic.
* Temporal means and rate integrals need explicit interval bounds, units,
  completeness, calendar and weighting rules. Discrete-amount summation does not
  integrate a rate. `wrf_products.aggregate` preserves the migrated CJC definitions
  only; it is not an arbitrary dashboard temporal reduction API.
* Vector wind needs the declared frame, rotation metadata and direction/averaging
  convention. Existing native diagnostics or scalar reductions are not substitutes.

Index selection and a declared discrete arithmetic mean are comparatively small
future interface additions once their consumer contract is agreed. The full list
must not be implemented through private/sibling-source imports, a broad arbitrary
expression API, or scientific defaults invented during integration. Keep these
requests refused/blocked rather than label the worker migration complete. Exact
existing worker semantics may be supplied for a parity migration separately.

## Verification

Use `env -u UV_PROJECT_ENVIRONMENT devenv shell -- uv run --frozen pytest -q`
from this worktree, with `--basetemp` in its ext4 scratch directory when needed.
Unsetting the inherited variable prevents reusing a dashboard environment.
No published product, original source, cluster job, deployment or approval was
created or changed by this audit.

Final validation including the legacy-status regression: **561 passed**, 12 warnings
(Zarr consolidated-metadata, historical empty-slice and CF time-bound warnings).
An initial fresh-worktree run had 14 missing-cache setup errors; only the reviewed
GAR fixture cache was copied from the existing Darwin checkout and pytest then
verified its pinned sizes/checksums. Changed Python files pass Ruff E/F/I checks
and formatting; `git diff --check` passes. No tests are skipped to hide
unavailable inputs.

Post-migration verification: **607 passed**, 12 warnings, including 20 regression
cases for malformed native time axes across summary/subset/layer/aggregation.
Changed grid Python files pass Ruff E/F/I and formatting. Producer artifacts must
be rebuilt from the final committed source before consumer validation.
