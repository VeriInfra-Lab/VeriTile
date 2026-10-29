# Two Real correctness formula surfaces

This is a focused follow-up to the three-task spot check. Only `reversed_cumsum`
and `softmax_triton2` are changed. It does not certify a fresh 183-task proof replay.

- `reversed_cumsum/Correct.lean` now uses `IO ⊨ ...` and directly displays the
  finite ordinary suffix sum. The old `IOComplete.implements` result is reused
  by definitional equality. The existing input/output separation, configuration,
  selected program, source execution, safety, output and frame scope are retained.
- `softmax_triton2/Correct.lean` now displays a sum over exactly `Fin n` and
  explicitly requires `n ≤ B`. The current wrapper already chooses a covering
  block, proved by `SoftmaxTriton1.blockSize_bounds`. The prior more general
  masked-window result remains `RealProof.io_correct`. Strict Mathlib proofs
  connect its padded sum to the logical-column sum and `Contract.reference`.
- `AlgorithmProgramIO` gains the same scoped `⊨` notation used by the examples;
  its judgment definition and execution semantics are unchanged.

Targeted compilation of the shared notation, the changed Real helpers, both
Correct entries and both unchanged FPEquiv entries passed in an isolated
workspace build generation. A fresh joint import passed. Main statement-surface,
non-circularity and standard-base axiom checks passed; the three new Real helper
lemmas were also explicitly checked for their standard-base axiom footprint.

The current role/IO review manifests refresh only these two reader pins and
retain all other task rows. Previous affected rows and source hashes are saved
in [audit.json](audit.json); source snapshots and build receipts live under
`.veritile-cache/correct-surface-fix-20261022/`. Kernel sources, Python guards,
contracts, IO records and FP equivalence sources are unchanged. The two tasks
remain `verified`, with no new numerical assumption or experiment. Existing
whole-task totals and all historical acceptance evidence remain retained.
