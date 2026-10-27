# Policy-bound station comparisons

Implementation for [Darwin #16](https://github.com/benatouba/darwin/issues/16),
building on the [station-series time contract](station-series-time-contract.md)
(#15). It adds an interface; it approves no policy, threshold or dataset.

The version-1 `darwin.comparison` interface (`native-intervals-v1`, recipe
`schema_version: 1`) is unchanged and remains what the dashboard currently
consumes. It is a separate policy, not a default for this interface.

## Interface

```python
from darwin.comparison_policy import compare_with_policy, replay_with_policy, store_identity

result = compare_with_policy(
    request,                       # dict, see below
    model_path="model.zarr",       # deployment-resolved local stores
    observations_path="obs.zarr",
    policy=policy_bytes,           # darwin.comparison-policy/1, exact bytes
    requirements=requirements_bytes,  # darwin-wrf-setups.study-requirements/1, exact bytes
    assessments=None,              # optional acceptance-assessment/1 records per source
    allow_draft=False,
    allow_synthetic=False,
)
replay_with_policy(result["recipe"], model_path=..., observations_path=...,
                   policy=policy_bytes, requirements=requirements_bytes)
```

The CLI uses the same functions and prints the result JSON, or a structured
`{"code", "message", "details"}` error on stderr with exit status 2:

```sh
python -m darwin.comparison_policy --model M.zarr --observations O.zarr \
  --policy policy.json --requirements requirements.json --request request.json
python -m darwin.comparison_policy ... --recipe recipe.json   # exact replay
```

`--allow-draft`, `--test-mode` (accept `synthetic:` fixtures) and repeated
`--assessment model=bundle.json` (bundle: `{"assessment": path, "evidence":
{id: path}, "prior": path|null}`, relative to the bundle) mirror the keywords.

### Request

| Field | Meaning |
| --- | --- |
| `model`, `observations` | Exact store identity `{id, version, sha256}`; `sha256` is `store_identity(path)`. `latest`/`current`/`HEAD` are refused. |
| `station` | `{id, lon, lat}`; must equal the `station` attribute of both stores, and `id` must be listed in the requirements' `stations`. |
| `sampling` | Must equal the model store's `sampling` attribute (method, CRS, grid cell). |
| `model_variable`, `observation_field` | A pairing declared by the policy. |
| `cadence` | `hourly`, `daily` or `monthly`, declared by the policy and natively present in both stores. |
| `start`, `end` | Required half-open, offset-aware timeframe on interval boundaries of the observation aggregation timezone. |

## Policy and requirements binding

A `darwin.comparison-policy/1` document (`darwin/comparison-policy-v1.schema.json`)
declares `id`, `revision`, `pairings` (with the requirements variable each one
serves, units, kind, and `wet_threshold_mm` for precipitation), `cadences`,
`completeness` (`minimum_paired_count`, `minimum_paired_fraction`), `exclusions`
(half-open UTC intervals with reasons), `metrics` and `histogram_bins`.
Darwin supplies no default threshold: every completeness rule, exclusion and wet
threshold comes from the policy bytes.

A policy is usable only when setups-owned requirements pin it: the requirements'
`operation_recipe` must equal `{id: policy.id, version: policy.revision,
sha256: sha256(policy bytes)}`. A different revision or an edited byte is refused
(`policy_not_bound`), so separately reviewed policies stay separate. Each
store's recorded upstream source checksum (`source_gar_sha256`/`source_sha256`)
must appear in the requirements' `sources`; the policy pairing's
`requirement_variable` must appear in its `variables`.

`review_status: draft` requirements are refused unless `allow_draft=True`, and
results are then labelled `review_status: "draft"`. Neither a draft document nor
a caller's (for example a UI's) selection of a policy grants approval.
`synthetic:` identities are refused outside explicit test mode.

Darwin vendors setups' `study-requirements/1` and `acceptance-assessment/1`
schemas byte-identically (`darwin/setups-*-v1.schema.json`) and validates them in
`darwin.study_contracts`. The copied producer fixtures and their source commit
are in `tests/fixtures/setups-contracts/`; Darwin never imports setups code.

## Alignment, completeness and exclusions

The requested timeframe is divided into calendar intervals of the cadence. Each
interval is, in order: excluded by a policy exclusion (`policy:<reason>`),
excluded because the model or observation value is absent or invalid
(`model_absent`, `model_invalid`, `observations_absent`,
`observations_invalid`), or paired. Represented intervals must have the declared
bounds. No interpolation, gap filling, aggregation or unit conversion occurs.
The completeness fraction is paired / (requested − policy-excluded). Below the
policy minimums the comparison fails with `insufficient_completeness` and the
counts in `details`. A store with no interval in the timeframe fails `no_overlap`.

## Metrics and undefined semantics

Only declared metrics are returned: `paired_count`, `means`, `sample_variance`
(n − 1), `bias` (model − observations), `mae`, `rmse`, Pearson `correlation`,
`histogram` (shared equal-width edges over both series, last bin closed), `ecdf`
(right-continuous, at each distinct value) and, for precipitation pairings,
`precipitation` (totals, wet frequency above the policy threshold and the
contingency table with observations as truth).

Scalar metrics are `{"value", "defined", "reason"}`. Undefined values are
`null` with a reason: `no_pairs`, `fewer_than_two_pairs` (variance,
correlation) or `zero_variance` (correlation). An undefined histogram reports
`zero_range` or `no_pairs`.

Unsupported here, and refused rather than approximated:

* `annual` cadence (`unsupported_cadence`): no station store carries annual
  series and no reviewed annual completeness rule exists.
* `circular_direction` diagnostics (`unsupported_metric`): no station-store
  contract carries paired wind direction/speed series. The kernel in
  `darwin.publisher.comparison.circular_wind` is not wired to stores.
* Monthly comparisons need monthly series in both stores; current observation
  stores hold hourly/daily only, and current model stores monthly only, so no
  existing builder output pair is comparable at one native cadence yet.

## Provenance

`result["recipe"]` (`darwin.comparison-recipe/2`) records the request, the
policy and requirements identities (with review status), validated assessment
identities, the draft/test-mode flags and the exact software digest. Replay
refuses any changed store byte, policy, requirements or implementation.

`scientific_status` of the comparison is always `unassessed`. Each source's
`scientific_status` comes only from a supplied acceptance-assessment record that
validates against the same requirements bytes and the requested dataset
identity; a store's own `scientific_status` attribute is reported as
`store_declared_status` and never promoted.

## Open gates

Real reviewed requirements, policies and thresholds, real-data scientific
validation, and the dashboard migration (darwin-dashboard#36) are separate,
human-authorized steps.
