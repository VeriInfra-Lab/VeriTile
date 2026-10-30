# Checked direct domain

The current domain and proof boundaries are documented in `README.md` and
`NUMERICS.md`. The original Python file is preserved. The repair selects FP32
X/Y, FP16 W and independently signed int32/int64 metadata. Every original
buffer is contiguous and nonlazy; aligned storage-offset views are admitted.
Y backing is distinct from X, W and all three metadata buffers. Read/read
aliases, both add modes, arbitrary validated overlaps, K=0 and zero-work
branches are retained.

The executable wrapper and `Contract.lean` specify the checks. All three
metadata roles are inspected before launch, including zero-column calls;
zero batches return first. Unused empty/sentinel segment fields and length
suffixes remain ignored. Active ranges must fit both independent input/output
row counts. Serial launch order defines overlapping updates, and actual
lengths determine coverage independently of `max_seq_length`.

Historical snapshot-host proofs remain available under `Proofs`. They describe
the earlier candidate and are not premises of the current direct public
contracts. Current source identities are in `source_pin.json`.
