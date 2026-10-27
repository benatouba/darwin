# Inventory of CJC scientific operations (migration baseline)

Status: inventory for benatouba/darwin#14, taken from `cjc` integration branch
`integration/project-integration` at `bcbb8e4` (`cjc/workflow/products.py`,
`diagnostics.py`, `cape.py`, `rip/`, `cli.py`). It records what CJC computes **before**
the implementation moved to Darwin, so that the move can be checked against it. It
changes no scientific definition. The Darwin implementation is described in
[wrf-products.md](wrf-products.md).

Definition version of every derived field: `gar-v2-python-1` (attribute
`definition_version`; dataset attribute `diagnostic_version`). Every product
dataset carries `product_status = "candidate"`; nothing here is acceptance or
publication.

## Entry points (CJC, unchanged)

| CJC command | Python | Role |
| --- | --- | --- |
| `cjc products SOURCE --domain {1,2} --destination D [--variables ...] [--rip-library L] [--pressure-levels ...]` | `normalize_wrf` → `derive` → `write_products` | Native-cadence product candidates for one retained WRF day |
| `cjc aggregate SOURCES... --domain --start --end --frequency {D,MS,YS} --cadence-minutes N --destination D` | `xr.concat` → `aggregate` → `write_products` | Strict time aggregates of one variable |
| `cjc build-rip --destination D` | `cape.build_rip` | Explicit gfortran build of the pinned NCAR kernel + `build.json` receipt |

Execution-specific checks that stay in CJC: `cli.retained_day_cadence` (input must be
a retained midnight-to-midnight day — 9 d01 or 25 d02 records — not a 36 h spinup
run), bundle/task layout, Slurm scheduling, HPC build commands.

## Input assumptions

- **Timestamps** come from WRF `Times` (char `YYYY-MM-DD_HH:MM:SS`) or an existing
  `time` coordinate, read as `datetime64[s]`; never synthesized. `normalize_wrf`
  requires ≥2 strictly increasing records, including the initial boundary, and
  renames `Time` → `time`.
- **Right-closed intervals**: `derive` uses the first record only as the
  deaccumulation boundary and then drops it for *every* output; `time_bounds` is
  `[t[i-1], t[i]]`, labelled by the interval end.
- **Accumulations** are from one model run: a monotonic field decreasing by more
  than 1e-6 raises `Accumulation reset`; small negative noise is clipped to 0.
- **Vertical**: mass levels `bottom_top` ascending in height; `P+PB` Pa,
  `T+300` K potential temperature, `(PH+PHB)/g` interface heights (destaggered by
  positional mean), AGL = MSL − `HGT`. Constants: g = 9.81, Rd = 287.0,
  cp = 1004.5 (Poisson exponent Rd/cp).
- **Hybrid coordinate** water paths require `DNW, MU, MUB, C1H, C2H`.
- **Grid**: writer requires native Mercator (`MAP_PROJ == 3`), sphere radius
  6 370 000 m, `TRUELAT1`, `STAND_LON`, `XLAT/XLONG`.
- Unknown variables or missing source fields fail visibly (`ValueError`/`KeyError`).

## Diagnostics (`derive`)

`aggregation` is the attribute later used by `aggregate` (`sum` or `mean`).

| Name(s) | Definition | Units | Aggregation |
| --- | --- | --- | --- |
| direct: `t2 q2 psfc tsk sst pblh ust snowh snow canwat swdown albedo emiss grdflx hfx lh smois sh2o smcrel tslb qvapor cldfra sr vegfra lai lu_index landmask hgt cosalpha sinalpha f mapfac_m isltyp ivgtyp` | WRF field of the upper-case name, unchanged | WRF units attribute (none added) | mean |
| `u10 v10` | Earth-relative: u·cosα − v·sinα, v·cosα + u·sinα (`COSALPHA/SINALPHA`) | m s-1 | mean |
| `ws10` | hypot(u10, v10) of Earth-relative components | m s-1 | mean |
| `u v ws` | as above after destaggering `U`/`V` | m s-1 | mean |
| `w` | destaggered `W` | m s-1 | mean |
| `theta` | `T + 300` | K | mean |
| `temperature` | θ·(p/1e5)^(Rd/cp) | K | mean |
| `pressure` | (`P+PB`)/100 | hPa | mean |
| `geopotential` | mass-level MSL height × 9.81 | m2 s-2 | mean |
| `height_agl` | mass-level MSL height − `HGT` | m | mean |
| `qliquid` / `qsolid` | sum of available `QCLOUD,QRAIN` / `QICE,QSNOW,QGRAUP`; attr `constituents`; none available → error | kg kg-1 | mean |
| `col_<x>` | Σ x·(−DNW)·(C1H·(MU+MUB)+C2H)/g over `bottom_top`, NaN propagates (hybrid dry layer mass) | kg m-2 | mean |
| `flux_int_<q>_<u\|v\|w>_2d` | col(q·wind component) | kg m-1 s-1 | mean |
| `flux_int_<q>_hor_2d` | hypot of integrated u and v fluxes | kg m-1 s-1 | mean |
| `flux_<q>_<u\|v\|w>_3d` | ρd·q·wind, ρd = p/(287·T·(1+QVAPOR/0.622)) | kg m-2 s-1 | mean |
| `prcp` | Δ(`RAINC+RAINNC`) | mm | sum |
| `prcp_rate` | Δ(`RAINC+RAINNC`)/interval hours | mm h-1 | mean |
| `prcp_c prcp_nc snowfall graupel hail sfroff udroff` | Δ of `RAINC RAINNC SNOWNC GRAUPELNC HAILNC SFROFF UDROFF` | mm | sum |
| `prcp_fr` | Δ`SNOWNC` (resolved frozen; graupel/hail not added again) | mm | sum |
| `potevap` | Δ(`POTEVP`·1000), signed (not clipped) | mm | sum |
| `et` | Δ(`ACLHF`/2.5e6), signed | mm | sum |
| `qfx` | `QFX`·interval seconds at right endpoint (rectangle), signed | kg m-2 | sum |
| `swup` | `SWDOWN`·`ALBEDO` | W m-2 | mean |
| `lwup` | 5.670374419e-8·ε·TSK⁴ + (1−ε)·`GLW` | W m-2 | mean |
| `lwdown` | `GLW` | W m-2 | mean |
| `netrad` | SWDOWN + GLW − swup − lwup (positive downward) | W m-2 | mean |
| `low_cloud_fraction` | max over levels of `CLDFRA` where AGL < 3000 m else 0; NaN propagates | 1 | mean |
| `t2eta` / `t2pbl` | linear fit T(z) over lowest two levels / levels below PBLH (≥2, else lowest two), evaluated at 2 m AGL; non-finite profile → NaN | K | mean |
| `inversion` | adjacent mass-level intervals with dT>0 inside AGL search (200, 3000) m: `inversion_base/top` (m), `_delta_t` (K), `_gradient` (K m-1), `inversion_count` (1, NaN for invalid profile), `inversion_profile_valid` (i1), `inversion_interval_present` (1, NaN where invalid). Definition `adjacent-positive-dT-dz-v1`; non-increasing height → error | as listed | conditional mean / mean |
| `mcape mcin lcl lfc` | native NCAR RIP (below) | J kg-1 / m | mean |
| `slp` | native NCAR WRF `dcomputeseaprs` | hPa | mean |
| `<x>_press` for `theta temperature qvapor qliquid qsolid u v w ws geopotential` | linear-in-pressure interpolation to requested hPa levels; NaN below ground/above top; non-monotonic or non-finite column → NaN; coordinate `pressure` (hPa, positive down) | as source | as source |

Attributes on all derived fields: `units` (where defined), `aggregation`,
`definition_version`; winds add `wind_reference`; fluxes add `definition`;
`prcp_fr`, `qfx`, `low_cloud_fraction` add `definition`/`integration`.

## Aggregation (`aggregate`)

- Frequencies `D`, `MS`, `YS` over `(start, end]`; bounds must lie on the frequency
  boundary; positive native cadence in minutes; duplicate/non-monotonic times fail.
- The source is reindexed to **all expected** timestamps at the declared cadence:
  absent timestamps become missing.
- Per cell: `fraction` = fraction of non-missing expected values; result kept only
  where `fraction == 1` (`completeness_policy = require-all-expected-timestamps`);
  `<name>_valid_fraction` (units 1) written for every variable.
- `sum` fields: `sum(skipna=False)`; `mean` fields: `mean(skipna=False)`.
- Inversion descriptors (`inversion_base/top/delta_t/gradient`): coverage from
  `inversion_profile_valid`, value is a mean over present intervals
  (`skipna=True`), aggregation `conditional-mean-when-inversion-present`.
  Physical absence (no inversion) is not missingness.
- Monthly/annual means are weighted by native records directly.
- Output time = interval start, `time_bounds` = `[lo, hi]`, `cell_methods = time: <agg>`.

## Native kernels and tables

| Item | Identity |
| --- | --- |
| Upstream | NCAR wrf-python v1.4.2, commit `124a8336529af6397fe150e14bd436d923122cdd`, Apache-2.0 |
| `wrf_constants.f90` | sha256 `48ddf57db41a0894344091a2ee81c32cd1946e4bf5ab2346a574c352b470d81b` |
| `rip_cape.f90` | sha256 `fa58761962ce740ca36f5950f211b7e246f953f51107ce1c5c941076f6c48fd0` |
| `wrf_user.f90` | sha256 `42df9fe7ddcbfb4e87a25b0b2e5a343e27bdd4601cd946ebed396a1339b8930a` |
| `psadilookup.dat` (pseudo-adiabat table) | sha256 `021b6a2de2724f3cd91050d29d3fc4ecb209688ae5a05bc3b2cfb7b09e590ee1` |
| `interface.f90` (custom C binding only: `cjc_mcape`, `cjc_slp`) | CJC-authored, hashed into the build receipt (`interface_sha256`) |
| Build | `gfortran -shared -fPIC -O2 -frecursive -ffree-line-length-none`; receipt `build.json` with `upstream`, `library_sha256`, `interface_sha256`, `compiler`, `dependencies` (ldd soname → sha256) |
| Load check | source/table hashes, library hash, upstream revision, interface hash and runtime library resolution must all match the receipt |

**MCAPE/MCIN** (`dcapecalc2d`, attrs `definition = NCAR-RIP-cape_2d-v1.4.2`,
`parcel = max-theta-e-lowest-3km-500m-pressure-weighted`): J kg-1, MCIN positive
magnitude (masked by NCAR for MCAPE < 100), no equilibrium level → missing.
`lcl`/`lfc` in m AGL. Inputs per column in ascending pressure (top first):
p (hPa), T (K), QVAPOR, MSL height, terrain, PSFC (hPa). Columns are withheld
(missing output) unless finite, p>0, T>0, q≥0, strictly monotonic p and height,
some level < 3000 m AGL, all p < PSFC; ≥3 levels required. |value| ≥ 1e19 → missing.
Provenance attrs: `native_library_sha256`, `lookup_sha256`, `sign_convention`,
`height_reference`.

**SLP** (`dcomputeseaprs`, `definition = NCAR-WRF-SLP-v1.4.2`): hPa; any
non-finite profile value fails the call (no masking).

## Writer (`write_products`)

One variable per NetCDF4 file
`<dest>/<d10km|d02km>/<h|d|m|y>/<2d|3d_eta|3d_press|3d_soil|3d_inversion>/GAR_<dom>_<freq>_<cat>_<var>_<label>.nc`,
float32 zlib 4 with shuffle, `seconds since 1970-01-01` proleptic Gregorian times,
CF-1.10, `time_zone = UTC`, Mercator `crs` grid mapping with `pyproj_srs`, `x/y` as
row/column medians, `eta`/`soil_depth` where present, inversion companions and
`*_valid_fraction` carried; written to a temporary file and published with a
no-clobber hard link (existing destination → `FileExistsError`).

## Provenance contracts

- Dataset attrs: source attrs + `diagnostic_version`, `product_status`; CLI adds
  `source_path` (deployment path, CJC only).
- Field attrs as above; native fields add library/table hashes.
- Build receipt as above. No recipe/source-byte identity is recorded by CJC;
  that is added by Darwin's operation recipes, not by this layer.

## Reference tests in CJC at inventory time

`tests/workflow/test_diagnostics.py`, `test_products.py`, `test_review_regressions.py`,
`test_writer.py`, `test_cape.py` (real gfortran build of the pinned kernel). Their
literal expected values are copied into Darwin's `tests/test_wrf_products_reference.py`;
CJC-produced reference outputs are the `cjc.product-golden/1` fixtures.
