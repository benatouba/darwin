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
| `darwin.assessment-evidence/1` | Darwin | `darwin/contracts/assessment-evidence-v1.schema.json` | `darwin.acceptance.run_checks` | `python -m darwin.acceptance run-checks` |
| `darwin-wrf-setups.acceptance-assessment/1` | setups | vendored: `darwin/setups-acceptance-assessment-v1.schema.json` | `darwin.acceptance.record_assessment`; validation in `darwin.study_contracts.validate_assessment` | `python -m darwin.acceptance record`; validated within `python -m darwin.handoff validate --assessment` |
| `darwin-wrf-setups.study-requirements/1` | setups | vendored: `darwin/setups-study-requirements-v1.schema.json` | `darwin.study_contracts.inspect_requirements` | none |

`darwin.study_contracts` is the single consumer-side validator for the setups
contracts. Its copied producer fixtures and conformance tests live in
`tests/fixtures/setups-contracts` and `tests/test_study_contracts.py`. Darwin
never imports setups code.

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
- **`sources` and `recipe`.** The manifest's pinned sources and the release
  manifest as the recipe (`version` is its release label, `sha256` the sha256
  of its exact bytes). Every artifact's identity record may only name sources
  whose sha256 the manifest pins, otherwise preparation fails with
  `source_mismatch`.
- **`scientific_status` and `acceptance`.** The full validated assessment
  history, oldest first. Each record supersedes the one before it. The status
  is the status of the latest record, or `unassessed` when there is none.
  A reference is a claim: `validate_handoff` re-derives every reference from
  the exact record bytes (`assessments=`, CLI `--assessment PRODUCT=BUNDLE`)
  and refuses a forged or altered one (`acceptance_mismatch`) or one without
  record bytes (`acceptance_unverified`). `verify_acceptance=False`
  (`--no-verify-acceptance`) only checks the chain's internal consistency and
  is for consumers that validate every cited record themselves.
- **Legacy fields.** Every field of the unversioned legacy format except
  `status`: `candidate_id`, presentation, aggregations, provenance and the rest.

### Compatibility

The dashboard's existing importer ignores unknown keys and doesn't read `status`,
so it still imports a v2 handoff. A v2-aware consumer must:

1. Dispatch on `contract`. A document without one is legacy. Refuse any other
   unknown value.
2. Run the same checks as `validate_handoff`.
3. Validate each cited acceptance record's bytes against the product's `dataset`
   (`validate_handoff(..., assessments=...)` does this; with
   `verify_acceptance=False` the consumer must do it itself).

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
  completeness are independent. The cited records, evidence receipts,
  authorization artifact and requirements are under `acceptance/`, with one
  `load_assessment_bundle` file per record (`bundle-*.json`) that cases name
  under `acceptance`.
- `legacy-v1.json`: the same manifest in the legacy format.
- `negative/`: an unknown contract, a publication claim, a status no record
  backs, a tampered artifact digest, a dynamic version and a forged acceptance
  reference (`forged-reference.json`). The `acceptance-without-record-bytes`
  case shows that an assessed handoff alone is `acceptance_unverified`.

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
  list as authorized, plus the exact bytes of the authorization artifact whose
  sha256 the requirements pin for it (`authorization=`, CLI
  `--authorization`). Copying the authority entry out of the requirements is
  therefore not enough. Darwin never infers an authority. Check outcomes and
  execution success never set a status.
- Cited Darwin receipts must report the record's dataset, requirements and
  criterion. `accepted` needs a passing receipt for every criterion whose
  cited evidence is all Darwin receipts; a failed attempt may stay cited next
  to a passing rerun. Evidence of other types is checked by exact bytes only.
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
- Authenticating an authority beyond binding its authorization bytes. Darwin
  checks that a decision cites the exact authorization artifact the reviewed
  requirements pin, but holding those bytes does not prove that the reviewer
  or policy issued the decision. Signing or otherwise attesting who records a
  decision is a human/deployment gate (reviewed requirements, controlled
  storage of authorization artifacts and of records); Darwin does not provide
  a PKI. Consumers must not present `accepted` as authenticated beyond that.
- Dashboard consumption of v2 and of the setups assessment records, replacing
  its provisional acceptance reference (dashboard#18). The dashboard keys
  imports by `manifest_revision`, so a new handoff that adds assessments for the
  same manifest needs its own import identity there.
