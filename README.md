# Analysis tools for the [DARWIN](https://vhrz669.hrz.uni-marburg.de/darwin/) project and GAR data

The approved [integration specification](docs/integration-spec.md) defines Darwin's
analysis, reproducible-recipe and publication-preparation role, its Python/CLI
interfaces, and current migration gaps. The legacy tools below are implementation
context, not permission to mutate original source data or bypass versioning.

Station-series callers must follow the [version-2 time contract](docs/station-series-time-contract.md):
native cadence, explicit timestamp provenance and interval joins replace monthly
repetition and positional comparison. The document inventories legacy products
requiring reassessment and the outstanding dashboard migration.

The GAR dataset is modelled on a WRF Mercator grid with 2 km grid spacing. The
[salem](https://salem.readthedocs.io/en/stable/index.html) package provides methods for easy access to the data that are
used in this project.

If there are any issues with metadata, inhibiting correct data access via
[salem](https://salem.readthedocs.io/en/stable/index.html), the `darwin/change_gar_names.py` script can be used to fix
the metadata.

```python
python darwin/change_gar_names.py -h # for help
```

The `darwin/core.py` module contains the main classes and methods for data access and analysis. The `darwin/defaults.py`
module contains default values for the project.

There are scripts specifically to cater to the
[darwin-dashboard](https://gitlab.klima.tu-berlin.de/schmidt/darwin-dashboard) project. The script
`darwin/combine_to_measurement_period.py` is used to combine the GAR data to the measurement period of the DARWIN
measurement network.

BEWARE: This project is a WIP.

## Installation and optional dependencies

Installing a Darwin release wheel without extras provides only what the
supported interface needs: `darwin.operations`, `darwin.comparison`,
`darwin.comparison_policy`, `darwin.study_contracts`,
`darwin.simulation_output`, `darwin.observation_qc` and the publisher. The
historical research, mapping and plotting modules need extras
(for example `uv add "darwin[research,visualization] @ <wheel>"`):

| Extra | Contents | Needed by |
| --- | --- | --- |
| `research` | metpy, pint, statsmodels, dask, cdsapi, pandas, ... | `core`, `utils`, `accessors`, download/analysis scripts |
| `native` | cdo, rasterio, rioxarray, salem, geopandas, pyproj, shapely, elevation | reprojection, maps, raster conversion |
| `visualization` | matplotlib, seaborn, cartopy, contextily, PyQt6, ... | `plot*`, `accessors`, map scripts |
| `all` | all of the above | development; the `test` group installs it |

`tests/test_installed_release.py` builds the wheel and installs it, runtime
dependencies only, into a fresh environment from an empty directory (offline,
from the uv cache), then runs the Python interface and CLI there.
See [`docs/operations.md`](docs/operations.md) for the supported operation inventory
and [`docs/product-handoff.md`](docs/product-handoff.md) for the product handoff and
portable acceptance contracts.

## Signed acceptance decisions

The source of truth for a dataset is the WRF setup the researcher chose. Its
reviewed setups `study-requirements/1` define the acceptance criteria and the
authorities (reviewer or policy) that may decide, and pin for each authority the
Ed25519 OpenSSH public keys (`signing_keys`) allowed to sign its decisions. An
`accepted` or `rejected` `darwin-wrf-setups.acceptance-assessment/1` record is
valid only with a detached SSHSIG over its exact bytes, under the namespace
`darwin-acceptance`, by one of those keys. `unassessed` records are never signed.

1. Record the decision. `darwin.acceptance.record_assessment(...)` (CLI
   `python -m darwin.acceptance record ...`) validates every binding and returns
   the **unsigned** record bytes. Darwin never holds a reviewer's private key.
2. The authority signs the exact file, without editing or re-formatting it:
   `ssh-keygen -Y sign -f ~/.ssh/<pinned key> -n darwin-acceptance record.json`
   writes `record.json.sig`. Keep both files together.
3. Consumers pass the signature with the record. In Python it is
   `validate_assessment(..., signature=sig_bytes)`; in an assessment bundle
   (`darwin.study_contracts.load_assessment_bundle`) it is the `"signature"`
   path, used by `python -m darwin.acceptance validate --bundle`,
   `python -m darwin.handoff ... --assessment PRODUCT=BUNDLE` and
   `python -m darwin.comparison_policy ... --assessment NAME=BUNDLE`.

Verification (`darwin.decision_signature`, a copy of the setups verifier kept in
parity by the copied setups fixtures) parses the SSHSIG, requires the
`darwin-acceptance` namespace and a key pinned for the record's own authority,
and checks the signature over the exact bytes. It needs the `cryptography`
runtime dependency, not `ssh-keygen`. Refusals: `signature_unavailable`,
`signature_invalid`, `signature_namespace_mismatch`, `signer_not_authorized`
(including a key pinned for another authority), `signature_mismatch` and
`unexpected_signature` (a signed unassessed record). See the setups README for
the reviewer how-to.

A decision binds a product **dataset**, not a store:

> setup → reviewed requirements → signed acceptance → dataset → compared store

The dataset identity is `darwin.handoff.dataset_identity(product_id, version,
artifacts)` over the producer's handoff artifact list. A comparison reports a
source as accepted only when its signed record validates against that dataset and
the compared store is one of its artifacts (see
[`docs/comparison-policy.md`](docs/comparison-policy.md)).

## Tests on a fresh checkout

Install [uv](https://docs.astral.sh/uv/) and Python 3.14, then run:

```sh
uv sync --frozen --group test --no-group dev
uv run --frozen --no-sync python scripts/fetch_test_fixtures.py
MPLBACKEND=Agg uv run --frozen --no-sync python -m pytest -q
```

The downloader fetches only three checksum-pinned GAR NetCDF files, about
17 MB total, into the ignored `tests/.cache/gar/` directory. It enforces
8 MiB per-file and 20 MiB total limits, bounds the response stream even if
the server misreports its size, and removes incomplete downloads. Cached
files are always verified; corrupt or changed upstream bytes fail rather
than silently changing scientific test inputs. No hourly or 3D files are
downloaded. Salem comes from the locked public package, not a sibling checkout.

Set `DARWIN_TEST_FIXTURES` to use another cache location. To check a prepared
cache offline, run `python scripts/fetch_test_fixtures.py --check`.
Pytest itself never downloads test fixtures and fails with setup instructions
if they are missing. Small processed station samples are checked in;
multi-year plotting data is generated in temporary directories. See
[`tests/fixtures/README.md`](tests/fixtures/README.md) for input provenance.
Time-series/accessor imports also work outside the checkout and do not load
Salem's map sample-data downloader; mapping support is loaded on demand.

GitLab CI runs the same full test command headlessly, caches only downloaded
fixtures and uv packages, and publishes JUnit results. Generated data and
test environments must not be committed.
