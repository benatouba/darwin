# Copied darwin-wrf-setups producer fixtures

Consumer copies of setups-owned contracts, taken verbatim from
`darwin-wrf-setups` commit `981ba715381c351fd4e30dbeae9bb48262e18d01`
(`contracts/`). Darwin never imports setups code; `tests/test_acceptance.py`
checks that Darwin's own validator agrees with the producer's expected
outcomes in `fixtures/acceptance-assessment-v1/manifest.json`.

Every identity uses the `synthetic:` namespace. These files are not scientific
acceptance, reviewed requirements or a real reviewer/policy authorization.
Refresh them only by copying a newer producer revision, never by editing.
