# Station-series time contract, version 2

Implementation for [Darwin #15](https://github.com/benatouba/darwin/issues/15).
This is a compatibility change, not scientific approval of any existing product.
The [integration specification](integration-spec.md) remains authoritative for
ownership and acceptance. Existing manifests, releases and published stores are
left unchanged; their historical presence is not evidence of scientific validity.

## Public comparison interface

The local Darwin **1.1.0** release candidate exposes `darwin.comparison`, contract
version 1. Consumers use `compare_stores` or `replay`, not publisher internals.
`compare_stores(model_path, observations_path, model_variable="t2",
observation_field="T", cadence="hourly", policy="native-intervals-v1")` returns
statistics, distributions, precipitation extras when applicable, and a reproducible
recipe. t2/T requires K on both sides; prcp/PCP_tot_bucket requires native interval
depths in mm. All other conversions/aggregations require explicit future operations.
Optional `start`/`end` select whole intervals inside a half-open UTC-aware window.

`replay(recipe, model_path, observations_path)` checks exact software and source
identities, including store metadata, masks and coordinates. Paths are deployment
inputs, not recipe identities. The matching CLI is:

```sh
python -m darwin.comparison --model MODEL.zarr --observations OBS.zarr --recipe recipe.json
```

The named policy preserves explicit upstream masks; it grants no scientific
acceptance. Source status remains in the recipe. Comparing mutable stores fails
if their identity changes while reading. Recipes bind the exact implementation;
a new software release needs explicit re-evaluation, not silent replay fallback.

Packaging now includes the Python modules in built wheels (the old configuration
produced metadata-only wheels). This candidate has not been published or deployed;
consumer tests must install its built wheel and verify the interface imports.

## Native cadence and provenance

`build_station_series` validates source hashes, actual NetCDF time coordinates,
calendar, units and any declared interval bounds before writing new stores.
Monthly GAR input produces **monthly values only**. No hourly or daily model
values are reconstructed by repeating monthly values. A missing, shifted or
unsupported source time axis fails rather than being inferred from a filename.
The current monthly builder requires twelve calendar months in one year.

Observation CSV timestamps must contain explicit offsets, or the caller must
provide `obs_timezone` from reviewed source metadata. A station location alone
does not establish the logger's timezone. Duplicate or unordered readings fail.
The current builder rejects months crossing a daylight-saving offset transition;
it does not silently impose 24-hour local days on such input.

Both stores record `station` (`{id, lon, lat}`; `id` is the caller-supplied
reviewed `station_id`, recorded as `null` rather than inferred). Model stores
record `sampling` (method, CRS, grid cell indices and cell coordinates).
Observation `source_coverage.window` states the half-open aggregation month in UTC.

Stores carry `series_schema_version: 2`, `cadences`, `calendar`, UTC coordinate
`timezone`, `aggregation_timezone`, `source_coverage`, `synthetic: false`, source
identity/checksum metadata and `scientific_status: unassessed`.

| Series | Values | Coordinates | Bounds | Completeness mask |
| --- | --- | --- | --- | --- |
| Monthly model | `monthly` | `time` | `time_bounds` | finite values |
| Hourly observations | `hourly` | `hourly_time` | `hourly_time_bounds` | `hourly_valid` |
| Daily observations | `daily` | `daily_time` | `daily_time_bounds` | `daily_valid` |

Coordinates identify interval starts. Bounds have shape `(N, 2)` and describe
half-open calendar intervals. Coordinate and bounds arrays declare matching
epoch units: model hours, observation seconds, since `1970-01-01 00:00:00` UTC.
Model coverage records start/end-exclusive bounds. Observation coverage records
actual first/last source timestamps and reading counts, not a claim that all
intervening readings exist. Missing observation intervals retain invalid masks.
Aggregation rules and thresholds are recorded in the observation metadata.

Existing observation aggregation rules and scalar comparison formulas remain
unchanged. Preserving them is not a review of their scientific suitability.
The legacy Cerro Crocker wrapper preserves its increment-summing convention;
the general precipitation builder retains its cumulative-counter convention.

## Comparison interface

```python
from darwin.publisher.comparison import compare
from darwin.publisher.series import load_series

model = load_series("reviewed-model-hourly.zarr", "hourly")
observations = load_series("reviewed-observations.zarr", "hourly")
result = compare("t2", "T", model, observations)
```

The model path above must genuinely contain an hourly model series. A monthly
station store cannot satisfy that request. The current monthly builder cannot
produce it; obtain native finer-cadence data or explicitly aggregate observations
under a reviewed policy before making a same-cadence comparison.

For in-memory callers, use `TimeSeries(values=..., times=..., cadence=...)` with
timezone-aware timestamps. Optional bounds and completeness masks are validated.
Bare value arrays are rejected by `compare`; equal length does not imply temporal
alignment. Low-level numeric kernels are not timestamp-aware comparison entrypoints.

Comparisons validate both entire series, then join exact UTC interval starts and
matching interval ends. Cadence must agree; duplicate timestamps, unsupported
calendars, synthetic series and incompatible bounds fail. Different years or
otherwise disjoint time axes raise `TimeValidationError` with code `no_overlap`.
Gaps are not filled. Invalid/nonfinite values and unmatched represented intervals
are excluded and counted; absent intervals are not invented. If overlapping
intervals all fail completeness checks, existing empty-result behavior applies.
Errors expose a stable `.code` plus an explanatory message.

There is no implicit monthly observation aggregation, disaggregation, calendar
conversion, unit conversion or interpolation in this interface. Gregorian
variants are supported for their compatible modern dates; mixed-calendar dates
before 1583 and non-Gregorian calendars fail explicitly.

## Historical products requiring reassessment

These are an inventory of potentially affected artifacts, **not** a permission to
rewrite their bytes or an assertion that a valid replacement already exists.

* `darwin/publisher/manifests/smoke-jan2020-prcp.yaml` and
  `releases/smoke-jan2020-prcp.json` (`smoke1.1`): inspect
  `AWS-PCerroCrocker_2020-01`, `AWS-CerroCrocker_2025-01`,
  `GAR-CerroCrocker-sampled_2020` and `CMP_prcp_CerroCrocker_2020-01`.
  The comparison specification combines model 2020 with observation 2025 and
  describes uniform monthly disaggregation. Neither is valid native hourly
  skill evidence. Audit actual input dates even where a product label says 2020.
* `darwin/publisher/manifests/stations-compare-v1.yaml` (`DRAFT`, not reviewed)
  and `releases/stations-compare-v1.json` (`stations1`): reassess all sampled
  model, observation and comparison triples listed below. Same-year labels do
  not repair model hourly/daily values fabricated from monthly input.
* The dashboard's `backend/data/smoke-jan2020-prcp.json` and
  `backend/data/stations-compare-v1.json` are separate catalogue copies
  (`smoke1.1` and `stations1.1`). Audit their exact identities independently;
  similar names do not establish identical bytes or approval.

| Station | Variable | Observation month |
| --- | --- | --- |
| SantaRosa | t2 | 2022-12 |
| CerroCrocker | t2 | 2024-02 |
| SierraNegra | t2 | 2023-01 |
| CuevaSucre | t2 | 2023-03 |
| LaGalapaguera | t2 | 2023-06 |
| ElJunco | t2 | 2023-03 |
| Baquerizo | t2 | 2023-06 |
| Villamil | t2, prcp | 2023-05 |
| Ayora | t2, prcp | 2023-05 |
| Militar | t2 | 2023-05 |

## Migration and consumer blocker

1. Preserve old stores, release records and checksums. Audit exact source time
   coordinates, logger timezone and existing synthetic/disaggregation history.
2. Obtain a reviewed timezone declaration where source timestamps are naive;
   missing metadata blocks rebuilding, rather than justifying a UTC guess.
3. Choose genuinely overlapping source periods and native matching cadence, or
   define a reviewed explicit aggregation policy. Use new product/version IDs
   and new destinations. Builders refuse existing destinations; `load_series`
   rejects legacy stores rather than relabelling their schema in place.
4. Record new provenance and assess scientific acceptance separately. Publishing
   or changing catalogue pointers requires its own authorized workflow.

The dashboard's current `backend/app/comparison.py` is an independent legacy
implementation: it can pair hourly/daily arrays positionally and monthly values
by month number without checking the year. This Darwin change does **not** repair
or validate that path. Migration to the supported Darwin interface is tracked in
[dashboard #36](https://github.com/benatouba/darwin-dashboard/issues/36).
Policy-bound comparisons are implemented by `darwin.comparison_policy`
([comparison-policy.md](comparison-policy.md), Darwin #16); broader supported
Python/CLI packaging is tracked in
[Darwin #12](https://github.com/benatouba/darwin/issues/12).
