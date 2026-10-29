# Source-only audit after legacy public-proof cleanup

`bash bench/audit_tritonbench_g.sh --source-only` passed for all 178 imported source views. The receipt records exit code 0 and no changed inputs among 3,563 captured files. Exact captured files, hashes and the terminal log are included here.

This audit checks the current source comparisons, coverage reviews, proof-gap manifest and completion inventory after retiring superseded public proof entries across 37 ports. It does not add mathematical coverage or independently accepted tasks.

The receipt helper uses a generic scope string mentioning Python regressions. The actual recorded command for this snapshot was the source-only corpus audit; no separate Python unit-test suite, Lean elaboration, native compilation or independent acceptance was run by this receipt. The separate [successful aggregate and public-entry checks](../legacy-public-cleanup-root/README.md) cover Lean elaboration and the public attributes.
