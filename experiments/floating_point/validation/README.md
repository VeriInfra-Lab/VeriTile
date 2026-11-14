# Two-gates validation receipt

[two-gates.json](two-gates.json) records a full run, its exact source hashes,
settings, backend, CPU checks and numerical outcomes. The inventory contains
115 runnable configurations and 53 unsupported precision combinations. The three
weighted FP32 configurations use the shared sample bank and full restored peak
budget.

The receipt is compact metadata. A complete measurement bundle contains source
snapshots, all part statistics, input/output fingerprints, PTX and required raw
audits outside Git. Its manifest and independent CPU replay digests are recorded
in the receipt. The [experiment guide](../README.md#cpu-replay-evidence-and-resume)
describes how to generate and independently import a bundle.

Evidence applies to its recorded source identities and configuration. CI runs
CPU regressions; the full GPU budget is invoked explicitly. Numerical rejections,
unsupported precision combinations and execution errors have distinct statuses.
Lean admission requires evidence matching the exact rule instance used by the proof.
