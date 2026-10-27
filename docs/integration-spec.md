# Darwin ownership and integration specification

Status: **approved target design**, 2026-10-25, agreement revision 1.
The [four-project agreement](../../darwin-wrf-setups/docs/project-integration-spec.md)
records the shared decisions. This document specifies Darwin's role, not a claim
that the target interface or migrations are complete.

## Owned interface

Darwin owns reusable scientific data operations: output reading, observations and
station metadata, sampling, diagnostics, comparisons, cut/crop/select/integrate/
assemble operations, transformations, scientific checks and publication-ready
products. Study policies and versioned acceptance requirements belong to
`darwin-wrf-setups`; Darwin implements and validates supported algorithms/parameters.
Different named scientific policies are explicit inputs, not hidden UI defaults.

Expose a supported Python interface and matching CLI over the same implementation.
The dashboard backend and CJC adapters use this interface, not private modules.
Recipes must be executable without the dashboard. No mandatory Darwin server or
implicit Slurm dependency is part of this design.

Consume CJC's versioned simulation-output manifest rather than its internal task
layout. Also support documented source-data interfaces for existing archives;
analysis of an existing dataset must not require CJC or setups to be installed.
Darwin owns its analysis-recipe and product/publication-handoff contracts, schemas
and fixtures. Consumers test those contracts; production uses declared compatible
releases, not sibling source checkouts.

## Scientific and data guarantees

Keep original data immutable. Derived products carry exact source identities and
machine-readable recipes: selections, operation order/parameters, software versions,
scientific status and acceptance references. Published versions identify immutable
artifacts with checksums. Storage adapters resolve locations without conflating
different versions.

Assembly validates grids, units, variable definitions, times/calendars, gaps and
overlaps. Conversion, interpolation, averaging, gap filling and reconciliation
must be explicit operations. Unsupported versions/operations, missing coordinates
or unavailable inputs produce actionable structured errors; synthetic timestamps,
silent approximation and automatic version substitution are not repairs.

Darwin runs scientific checks and prepares evidence against a named requirements
revision. Acceptance is granted by a reviewer or explicitly authorized policy and
recorded portably against exact datasets, requirements, evidence and authority.
Execution success alone is not acceptance.

Darwin prepares validated products and publication-ready handoffs, independently
of the dashboard. The dashboard catalogue owns publication/access and current
version pointers. Darwin preparation does not silently approve or publish data
into that catalogue. Scientific acceptance and publication remain separate.

## Execution and migration

The dashboard owns asynchronous user requests, access checks, cancellation,
progress, download delivery and expiry. Darwin performs their data operations.
An explicit adapter may ask CJC to execute an operation on the cluster; storage,
transfer and credentials belong to deployment adapters. Shared paths between
dashboard and cluster are not assumed.

Current gaps include duplicated comparison calculations in the dashboard,
CJC product diagnostics/aggregation/native kernels (now migrated to
`darwin.wrf_products`, see [wrf-products.md](wrf-products.md); CJC keeps a
versioned compatibility path), overwrite-capable stores, broad packaging dependencies and smoke series
that are not evidence of timestamp-aligned native-hourly comparisons. Preserve
existing work and plan migrations separately. `CONTEXT.md`'s product-candidate
terminology must not collapse acceptance and publication into one state.
The handoff, evidence and acceptance contracts and the migration of the former
smoke auto-publication are specified in [`product-handoff.md`](product-handoff.md).

Future acceptance requires Python/CLI equivalence tests, dashboard consumption of
the same operations, producer/consumer schema fixtures, reproducible exact-version
recipes, immutable artifacts and explicit failure tests. Scientific validity of
real inputs must be demonstrated separately from smoke tests.
