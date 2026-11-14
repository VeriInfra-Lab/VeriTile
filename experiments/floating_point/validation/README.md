# Two-gates validation receipt

[two-gates.json](two-gates.json) records a full run of the tracked current entry,
its exact source hashes, settings, backend, CPU checks and numerical outcomes.
All 115 runnable configurations were executed afresh; 53 unsupported precision
combinations retained their reasons. The 112 unchanged samplers reproduced every
saved part moment, count and error peak bit for bit. The three weighted FP32
configurations used the shared bank and full restored peak budget.

The checked-in receipt is compact metadata. The complete bundle retains source
snapshots, all part statistics, input/output fingerprints, PTX and mandatory raw
audits outside Git. Its manifest and independent local replay digests are in the
receipt. Use the [experiment guide](../README.md#cpu-replay-evidence-and-resume)
to regenerate and independently import a complete bundle.

Future source changes do not inherit this receipt's results automatically.
The CI job runs CPU regressions only; it does not run the full GPU budget.
Numerical REJECT outcomes and untested combinations are never relabeled PASS.
Lean admission data was not changed by this run.
