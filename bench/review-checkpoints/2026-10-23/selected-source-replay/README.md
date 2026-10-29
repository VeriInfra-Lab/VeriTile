# Selected source replay

The 183 selected tasks now build from tracked source: 551 public modules plus
all their transitive imports. Both the development checkout and an isolated
checkout passed `lake build` (16,629 jobs, including reused external packages).
The final source commit is recorded in `selected-build.json`.

The isolated checkout began without a project `.lake/build` or `.veritile-cache`.
Pinned external Lean/Mathlib artifacts were reused. After each source repair,
the corresponding Git commit was checked out and Lake rebuilt affected modules
in the same isolated directory; no project artifacts were copied from the
development checkout. `clean-build-attempts.json` retains that history, including
failed and interrupted attempts. This is not a claim that external dependencies
were rebuilt or that every intermediate attempt succeeded.

The current proof receipts are `selected-build.json`, `joint-checks.json`, and
`all-selected.json`. Per-task checks import the selected public entries together
and audit each recorded public headline with `#axiomsClean`; the combined check
imports all 551 entries into one Lean environment and audits the same headlines.
These checks detect import conflicts and disallowed axioms. They do not replace
semantic review of the mathematical target or prove hardware conformance.

## Source repairs

- Reconstruct the selected KV copy/INT8/INT4 proof chain and its named logical IO;
  see [the focused review](../kv-reconstruction/README.md).
- Resolve legacy source modules through Lake and unfold the actual readable
  quotations in existing execution, erasure and safety proofs. Correct explicit
  imports and lexical collisions introduced by shared DSL syntax.
- Preserve signed INT8 narrowing in rowwise and transpose quantization. The
  shared integer-stage executor now applies destination widths to pointer and
  block-pointer writes as well as direct region writes. Prove the unchanged
  public byte observation and test pointer narrowing, region selection and mask.
- Model the source `tl.minimum` as its own primitive in INT8 quantization, with
  the Real interpretation bound to `min`; do not assume a comparison/select law.
- Give the two matmul Real projections an explicit real-number algebra and prove
  the reached primitive laws. Preserve the separate FP interpretation.
- Preserve the two-bit extraction through signed-byte casts. Prove segmented
  launch coverage symbolically, and prove full state equality between matrix
  state writes and a memory fold to avoid eager expansion of 32,768 writes.

The public mathematical targets, source FP pairs, scope conditions and numerical
rule requirements remain explicit. `review.json` lists the integration changes
and limitations. Existing tracked semantic reviews are retained; this pass is a
direct source-integration review and replay, not a new independent second review
of all 183 tasks.

## Reproduction

Run from the repository root with the pinned Lean toolchain and dependencies:

```sh
python3 bench/review-checkpoints/2026-10-23/selected-source-replay/build-selected.py /tmp/selected-build
python3 bench/review-checkpoints/2026-10-23/selected-source-replay/replay-joint.py
lake env lean bench/tests/FPIntegerStageIO.lean
python3 bench/check_main_formula_review.py
python3 bench/check_main_entry_roles.py
python3 bench/check_io_organization.py
python3 bench/task_progress.py --check
```

The scripts generate proof receipts; they do not silently renew semantic-review
pins. Current metadata renewal was performed only after the actual checks.
Historical cache/tmp receipts were not recovered: their old hashes and selected
metadata are archived in `superseded-proof-evidence.json.gz` and
`superseded-current-summary.json`. Historical review sections still document
those earlier runs; `current_source_replay` identifies the present receipts.

The official comparator is unavailable (`comparator.log`); no comparator pass is
claimed. No GPU or new numerical experiment ran. `matmul_leakyrelu` retains its
existing numerical deferral. Seven optional historical KV ABI/raw/host/Tensor/
backing modules remain unavailable and outside the selected main proof scope.
