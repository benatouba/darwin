# Copied darwin-wrf-setups contract fixtures

Verbatim copies of producer-owned files from `darwin-wrf-setups`
`integration/project-integration` at commit
`981ba715381c351fd4e30dbeae9bb48262e18d01` (`contracts/` prefix dropped).
Darwin never imports setups; these bytes are consumer conformance fixtures.
All `synthetic:` identities are test-only records: they are not reviewed
requirements, reviewers, policies or scientific acceptance.

The two schemas are also vendored, byte-identical, as
`darwin/setups-*-v1.schema.json` for runtime validation. Refresh both copies
together from a newer producer commit; never edit them in place.
