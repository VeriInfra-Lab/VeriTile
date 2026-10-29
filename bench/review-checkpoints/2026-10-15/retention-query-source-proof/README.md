# Retention DQ source proof and deferred numerical experiments

This extension adds the complete original fp32 DQ helper and sequential
DQ/barrier/DK/DV helper expansion after caller index specialization. Both
public contracts use an independent input-memory recurrence and specify the
complete final memory. Aliases remain allowed: the second helper reads the
first helper's stored memory.

The contracts remain backend-parametric. They add no software bit-closed task
and make no GPU claim. The preceding classification remains 58 software,
125 backend-dependent and one repaired candidate, with 35/6 for the original
41 tasks. This extension adds two public contracts and six Lean modules.
Host NK/NV sums, launch/grid composition and native semantics remain open.

The numerical handoff is [the unified queue](../../../tritonbench_g/deferred_numerical_experiments.md),
with [seven detailed Retention obligations](../../../tritonbench_g/parallel_retention_attention/NUMERICAL_ASSUMPTIONS.json).
Five existing numerical rule profiles still need exact binding. Existing
reports, conditional model use, refuted assumptions and not-run experiments
are kept distinct. No GPU experiment is performed by the reproduction below.

## Reproduce

From the repository root, with the local Lean toolchain available:

```sh
python3 -B bench/review-checkpoints/2026-10-15/retention-query-source-proof/check.py lean_build --output /tmp/retention-query-validation
python3 -B bench/review-checkpoints/2026-10-15/retention-query-source-proof/check.py axioms --output /tmp/retention-query-validation
python3 -B bench/review-checkpoints/2026-10-15/retention-query-source-proof/check.py python_tests --output /tmp/retention-query-validation
python3 -B bench/review-checkpoints/2026-10-15/retention-query-source-proof/check.py spec_sheet --output /tmp/retention-query-validation
python3 -B bench/deferred_numerical_experiments.py --check
python3 -B bench/fp_value_extensions.py --check
```

Pass `--toolchain /path/to/lean/bin` to `check.py` for a different installation.
The commands retain exit codes and log hashes in the output directory's
`validation.json`, preserving the registered evidence in this package. Failed
development attempts retain their own logs; only the latest successful checks
can be registered. The named axiom audit covers all 20 new proof declarations.
This is affected-module validation, not a fresh compilation of every corpus
module or a successful full official comparator run.

Final integration checks passed: 130 Python regression tests, the current
value registry and deferred queue checks, and all 23 unchanged historical
source-comparison blocks for 178 selected source views. The current catalog
contains 184 tasks, 893 public declarations and 3,840 registered modules, with
no source-review binding gaps. [Metadata validation](metadata-validation.json)
records these results and the retained replay receipt and log.

The import cleanup moves the old ParallelAttention real proof dependency
from `FPSources.lean` to `FPShape.lean`, where its source-shape bridge actually
uses it. All existing declarations in those two files are byte-for-byte
unchanged after removing import lines. Retention's primitive source proofs
therefore no longer rebuild that unrelated real theorem stack.

`register.py` binds the successful validation records and source hashes in
`receipt.json`, extends the evidence index and the current value registry,
and writes a catalog supplement. Frozen preceding reviews are preserved in
their existing packages. The review records the current agent's source and
scope audit; it does not claim a separate human or independent agent review.
