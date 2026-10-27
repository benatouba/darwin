# Product handoff and portable acceptance

Darwin prepares products. It doesn't publish them. Three things stay separate:

| Dimension | Values | Owner | Where it is recorded |
| --- | --- | --- | --- |
| Execution completeness | `complete`, `partial`, `null` (not reported) | producer of the run (CJC or Darwin) | handoff `execution_completeness` + `execution_evidence` |
| Scientific status | `accepted`, `rejected`, `unassessed` | a reviewer or an authorized policy named by setups requirements | acceptance-assessment records, cited by the handoff |
| Publication, access, current pointer | dashboard decision | dashboard catalogue | dashboard only, never in a Darwin handoff |

## Contracts

| Contract | Owner | Schema | Python | CLI |
| --- | --- | --- | --- | --- |
| `darwin.product-handoff/2` | Darwin | `darwin/contracts/product-handoff-v2.schema.json` | `darwin.handoff.prepare_handoff`, `validate_handoff` | `python -m darwin.handoff prepare/validate` |
| `darwin.assessment-evidence/1` | Darwin | `darwin/contracts/assessment-evidence-v1.schema.json` | `darwin.acceptance.run_checks` | none |
| `darwin-wrf-setups.acceptance-assessment/1` | setups | vendored: `darwin/contracts/setups/` | `darwin.acceptance.record_assessment`, `validate_assessment` | none |
| `darwin-wrf-setups.study-requirements/1` | setups | vendored: `darwin/contracts/setups/` | `darwin.acceptance.inspect_requirements` | none |

The setups schemas are copied verbatim from darwin-wrf-setups `981ba71`. The
setups producer fixtures are copied under `tests/data/setups`. Darwin never
imports setups code. `tests/test_acceptance.py` checks that Darwin's validator
returns the producer's expected outcome for every setups fixture case.

## Handoff v2

Each product in a handoff:

- **`lifecycle: "prepared"`.** A handoff never carries `status`, `access` or
  `current_version`. The schema refuses those fields.
- **`artifacts`.** Each published artifact's `name`, `kind`, `sha256`, `size_bytes`
  and the sha256 of its `darwin.artifact-identity/1` record. They are
  re-verified byte for byte when the handoff is prepared.
- **`dataset`.** `{id: product_id, version, sha256}`. The `sha256` is the sha256
  of the compact, key-sorted JSON of `artifacts`. Acceptance records bind this
  exact identity, so they can't transfer to another version or to changed bytes.
- **`sources` and `recipe`.** The manifest's pinned sources and the exact
  release-manifest revision.
- **`scientific_status` and `acceptance`.** The full validated assessment
  history, oldest first. Each record supersedes the one before it. The status
  is the status of the latest record, or `unassessed` when there is none.
- **Legacy fields.** Every field of the unversioned legacy format except
  `status`: `candidate_id`, presentation, aggregations, provenance and the rest.

### Compatibility

The dashboard's existing importer ignores unknown keys and doesn't read `status`,
so it still imports a v2 handoff. A v2-aware consumer must:

1. Dispatch on `contract`. A document without one is legacy. Refuse any other
   unknown value.
2. Run the same checks as `validate_handoff`.
3. Validate each cited acceptance record's bytes against the product's `dataset`.

It must not treat `lifecycle` or `scientific_status` as a publication decision.

Synthetic fixtures (`synthetic:` identities or `synthetic: true`) are refused
unless `allow_synthetic=True`. The CLI flag is `--test-mode`.

### Consumer fixtures

`darwin/contracts/fixtures/product-handoff-v2/` ships in the wheel.
`cases.json` lists each handoff, its artifact root, whether it needs test mode,
and the expected outcome or error code:

- `handoff-unassessed.json`: both products unassessed, one with `complete`
  execution.
- `handoff-assessed.json`: `synthetic:prcp-slice` was rejected, then accepted
  on reassessment. Its execution is `partial`, which shows that status and
  completeness are independent. The cited records, evidence receipts and
  requirements are under `acceptance/`.
- `legacy-v1.json`: the same manifest in the legacy format.
- `negative/`: an unknown contract, a publication claim, a status no record
  backs, a tampered artifact digest and a dynamic version.

Regenerate the fixtures with `python tests/build_handoff_fixtures.py <new dir>`
and review the diff. The fixtures are synthetic. They are not scientific
acceptance, reviewed requirements, a real authority or a publication.

## Checks and decisions

`run_checks(dataset, requirements_bytes, inputs=...)` runs the check named by
each criterion of the exact requirements revision. It returns one evidence
receipt per criterion with a `pass` or `fail` outcome, plus the Darwin version
and check implementation sha256. A receipt has no status and no authority. An
unknown check name raises `unsupported_check`. Darwin has no reviewed
scientific check bound to a requirements revision yet. Only the fixture-only
`synthetic.test-only` check exists, and it runs only in test mode. Real checks
are added only together with their scientific review. Darwin never invents a
threshold.

`record_assessment(...)` writes a setups acceptance-assessment record:

- `accepted` and `rejected` require an `authority` that the exact requirements
  list as authorized. Darwin never infers one. Check outcomes and execution
  success never set a status.
- Only `reviewed` requirements can carry a decision.
- `accepted` needs evidence for every criterion.
- A reassessment passes the exact prior record bytes. The new record
  supersedes the prior one by sha256 and keeps every prior evidence entry. The
  prior bytes are never changed.

## Migration of the legacy auto-publication

Before darwin#18, `build_smoke_release` registered candidates in a throwaway
in-memory `Catalogue`, called `publish` on each one and wrote
`status: "published"`. That was never a dashboard publication or a scientific
acceptance. The builder now prepares candidates without any catalogue
(`plan_candidates` assigns the same version labels and candidate ids) and writes
`status: "candidate"`. Handoffs already written with `published` keep their
bytes and mean only "prepared". No historical approval or acceptance is inferred
from them. Any version imported from them stays `unassessed` until a real
assessment record exists.

`Catalogue` and `lifecycle` remain a legacy in-memory model for existing
storage builders and tests. They are not a publication authority.

Still open:

- Real reviewed requirements, checks and authorities (setups#39, darwin#15 and
  darwin#16).
- Authenticating an authority beyond binding its authorization identity.
- Dashboard consumption of v2 and of the setups assessment records, replacing
  its provisional acceptance reference (dashboard#18). The dashboard keys
  imports by `manifest_revision`, so a new handoff that adds assessments for the
  same manifest needs its own import identity there.
