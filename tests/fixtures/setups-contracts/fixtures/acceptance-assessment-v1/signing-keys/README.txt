TEST-ONLY synthetic SSH signing keys for the acceptance-assessment fixtures.

Only public keys and signatures are stored. Synthetic signing keys are derived in
memory from public labels so the fixture signatures are reproducible, never exported
as private-key files. They are pinned only by synthetic
requirements that validators refuse outside test mode. Never use them as real keys,
never pin them in real reviewed requirements, and never add them to an SSH agent,
authorized_keys or allowed_signers file.

TEST-ONLY-reviewer  pinned for synthetic:reviewer
TEST-ONLY-policy    pinned for synthetic:policy
TEST-ONLY-unpinned  pinned for no authority (negative cases)
