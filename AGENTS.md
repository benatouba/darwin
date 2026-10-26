# Darwin agent instructions

## Ownership and integration

Before changing data operations, recipes, publisher behavior, acceptance or
cross-project integration, read `docs/integration-spec.md`; use `CONTEXT.md` for
existing domain vocabulary. Darwin owns reusable scientific operations through a
supported Python interface and matching CLI. Study policy belongs to setups,
cluster execution to CJC, and user requests/publication/access to the dashboard.

Preserve original data and exact source versions. Derived products carry
reproducible recipes and scientific status. Keep scientific acceptance separate
from publication preparation and dashboard publication authority. Version and test
producer-owned contracts; keep deployment/Slurm details behind explicit adapters.
Document migration gaps without treating smoke behavior as scientific acceptance.
