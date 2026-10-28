# WRF product diagnostics and strict aggregation — `darwin.wrf-products/1`

Darwin owns the scientific implementation formerly in CJC (`cjc.workflow.products`,
`diagnostics`, `cape`, `rip/`), migrated for benatouba/darwin#14 **without changing
any definition**: definition version `gar-v2-python-1`, the same constants, interval
conventions, masking, attributes and NCAR kernel sources. The full baseline (every
variable, unit, aggregation rule, kernel/table hash and input assumption) is in
[cjc-science-inventory.md](cjc-science-inventory.md).

## Interface

| Python | CLI (`python -m darwin.wrf_products ...`) |
| --- | --- |
| `inventory()` | `inventory` |
| `derive_products(source, destination, domain, variables=None, native_library=None, pressure_levels=None, label=None)` | `derive SOURCE --destination D --domain {1,2} [--variables ...] [--native-library L] [--pressure-levels ...] [--label L]` |
| `aggregate_products(sources, destination, domain, start, end, frequency, cadence_minutes)` | `aggregate SOURCES... --destination D --domain {1,2} --start --end --frequency {D,MS,YS} --cadence-minutes N` |
| `darwin.native_diagnostics.build_native(destination)` / `verify_native(library)` | `build-native --destination D` |

Lower-level, in-memory building blocks (used by CJC's compatibility path):
`normalize_wrf`, `derive`, `aggregate`, `write_products` in `darwin.wrf_products`,
the definitions in `darwin.wrf_diagnostics`, and `mcape_mcin`/`sea_level_pressure`
in `darwin.native_diagnostics`.

CLI and Python run the same functions. The CLI prints one JSON report on stdout
(`{"contract": "darwin.wrf-products/1", "products": [...]}`; `build-native` prints
`{"library", "receipt"}`). On failure it exits 2, prints nothing on stdout and
writes `{"code", "message"}` to stderr: `destination_exists` (no-clobber refusal),
`missing_dependency` (for example pyproj without the `native` extra) or
`invalid_input` with the same message as the Python exception.

Products are one-variable NetCDF4 **candidates** (`product_status = candidate`);
nothing here accepts, approves or publishes data. `derive_products` records the
source byte identity (`source_sha256`), `product_contract` and `darwin_version`,
never a deployment path. Both `derive_products` and `aggregate_products` publish
every product with a `darwin.artifact-identity/1` record (see
[operations.md](operations.md#immutable-artifact-writes)), so products can be
bound by a handoff. An aggregate names every fragment (`sources` attribute and
identity record, in order) and its recipe (`recipe`); attributes that differ
between fragments, including each fragment's own `source_sha256`, are dropped
instead of inherited from the first. `aggregate` refuses in-window timestamps
off the declared native cadence instead of dropping them.

## Runtime and native extra

The diagnostics and aggregation need only the runtime dependencies (numpy,
xarray, pandas, netCDF4). Writing candidates with native Mercator coordinates
needs `pyproj` (`darwin[native]`). MCAPE/MCIN/LCL/LFC and SLP need an explicitly
built kernel: `build-native` compiles the packaged, hash-pinned NCAR wrf-python
v1.4.2 sources with `gfortran` and writes `build.json`
(`darwin.native-build/1`: upstream revision, library/interface hashes, compiler,
ldd runtime identities). Every load re-verifies sources, lookup table, library and
runtime resolution. Importing never compiles, downloads or loads anything. No
Slurm, CJC installation or sibling checkout is required; the devenv provides
`gfortran`/`ldd`.

`interface.f90` is byte-identical to CJC's (C symbols `cjc_mcape`, `cjc_slp`), so
`interface_sha256` is comparable across both builds. `native_library_sha256` is
build-specific and differs between a CJC and a Darwin build of the same sources.

## What stays in CJC

Run-layout and execution checks (the retained midnight-to-midnight day rule,
bundles, task status, Slurm, HPC build commands) and the `source_path` attribute
written by `cjc products`. CJC's `products`, `aggregate` and `build-rip` commands
keep working; `--science-backend darwin` (contract `cjc.science-backend/1`)
delegates the calculation to an installed Darwin. See CJC's
`docs/science-migration.md`.

## Evidence

- `tests/test_wrf_products_golden.py`: Darwin reproduces CJC-produced reference
  files (`tests/fixtures/cjc-product-golden-v1/`, copied from CJC
  `cjc/workflow/fixtures/product-golden-v1/`, contract `cjc.product-golden/1`) for
  47 diagnostics plus pressure levels, daily strict aggregation with a missing
  cell, an absent timestamp, and native MCAPE/MCIN/LCL/LFC/SLP including a withheld
  invalid column and native provenance attributes. Tolerance: rtol 1e-12.
- `tests/test_wrf_products_reference.py`: CJC's own reference tests with their
  literal worked-example values, copied unchanged apart from imports.
- `tests/test_wrf_products_interface.py`: Python/CLI equivalence for inventory,
  derive, aggregate, structured errors and native build receipts.
- `tests/test_installed_release.py`: the modules and packaged kernel sources work
  from an installed wheel without extras.

This is **software parity on synthetic inputs**. Real-data validation (units in
the selected WRF build, real-profile MCAPE/MCIN/SLP, projection metadata, pressure
interpolation) and scientific acceptance remain separate human gates, tracked by
darwin-wrf-setups#10/#12; nothing here satisfies them.
