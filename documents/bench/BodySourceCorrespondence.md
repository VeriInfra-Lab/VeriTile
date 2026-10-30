# Triton function-body correspondence

The selected corpus is **not yet fully migrated** to pasted Python bodies.
This acceptance scope is separate from the previously recorded Real and FP
proof results. An old proof, a successful build, or an unchanged inventory
does not establish correspondence with a Python JIT body.

The current machine-readable inventory is
[`validation/source-correspondence.json`](../../validation/source-correspondence.json).
It discovers all **536 task directories** directly under the four benchmark
collections, including the `tasks/` layout of the two imported collections.
The proof-selection ledger supplies reference paths, not inventory membership.
Missing kernels or stale selections are errors; missing Python references remain
explicitly pending. Equal task names in different collections have distinct path IDs.

Each task inventories every retained task-local Python file as well as external
references explicitly named by its pair manifest. This includes dynamically loaded
bodies and candidate files absent from that manifest, without executing Python.
The selected wrapper must still be anchored to an actual JIT reference before
that task can pass. `cross_entropy_ops` and `fast_ce_loss` now expose the five
JIT functions in their loaded files while retaining their unresolved-entry status.

At this snapshot there are 183 TritonBench tasks, 19 worked-example groups,
65 bug-detection challenges and 269 pretraining challenges. The latter two
collections have no registered independent Python references for this comparison;
their declared backend goals do not certify native correspondence.

There are 328 literal JIT functions in the selected primary Python files and
412 across the inventoried Python files. Nine pairs are checked: six argmax pairs and two complete matmul bodies in
legacy AST mode, plus the dequantize body in body-copy mode. Of the
selected primary functions, 322 remain uncovered; across the complete Python
inventory, 403 remain uncovered. The shifted argmax file's unchanged stage-2
function is explicitly paired with the shared original Lean stage. Both Python
bodies are checked independently. Argmax has complete legacy AST coverage;
it still does not have body-copy completion.
The matmul entries retain the complete standalone-dot/add bodies and uint64
casts separately from the existing compiler-selected accumulator-dot graphs.
Their public proof entries explicitly retain the compiler-graph scope; the
source comparison does not establish FP equivalence between those graphs.

These numbers describe source-check coverage, not the number of incorrect
algorithms or invalid Lean theorems.

The TritonBench file-level screen finds type directives in 141 task files and antiquotes
in 161 task files and all 19 example groups. In 44 primary files, Python has
explicit int64 casts that are not visible in a direct Lean quotation. This
last count is a warning screen: imported source, specialization and semantic
connections require review. Unflagged files are not certified.

## Checking and completion

```sh
# Check all registered pairs and reject changes to the recorded inventory.
python3 -m scripts.bench.check_body_corpus --check

# Full source acceptance: currently exits 1 because pending entries remain.
python3 -m scripts.bench.check_body_corpus --check --require-complete
```

The CI job is named **Inventory drift and correspondence regressions**. It
runs on `main` and `fp-examples`, checks the recorded inventory and tests the
checker. Its green status is not a full-corpus body-copy certificate. The
explicit `complete` field remains false and full acceptance fails.

The statuses describe mechanical comparison coverage. They are not a count of
semantically faithful translations: a missing pair is not a demonstrated
mismatch, and legacy syntax adaptations or explicitly proved semantic connections
must be assessed separately by reading the actual source and connection proofs.
The checker does not certify that the public theorem reaches the checked body.

Regenerate the current inventory only after inspecting the source changes:

```sh
python3 -m scripts.bench.check_body_corpus --write
```

An actual migration must also replay the affected Lean proofs and retain the
independent mathematical specification and public FP comparison. This script
does not establish proof reachability, interpreter correctness, implicit type
promotion correctness, or hardware behavior.

## Per-function migration requirements

1. Anchor the actual selected Python source, including each stage. Do not
   substitute a comment, a simplified kernel, a generated Python mirror, or
   a function with only the same mathematical formula.
2. Keep the Python function body in `triton { ... }`. Keep variable names,
   assignments, casts, calls, branches, masks and operation order. Supply
   pointer storage, runtime scalar types and constexpr names outside the body
   with `with_triton_signature`. Bind runtime argument values at the call.
3. Add a `jit-correspondence.json` pair for every selected JIT function and
   separately declared Python candidate. Body mode requires
   `signature_definition` and a reviewed `expected_signature`. It permits no
   metadata directives or Lean antiquotes inside the body. Tuple/list and
   positional/keyword expression changes are not normalized in body mode.
4. Connect the public Real and FP theorems to the migrated actual program.
   Prove complete program equality or an explicit execution connection;
   rewriting a proof to refer to a detached old helper is insufficient.
5. Rebuild affected public entries, inspect axiom and FP assumption output,
   and run the official comparator when available. Record unavailable checks
   explicitly. Then update the inventory and relevant review records.

Legacy AST mode preserves its documented brace and shape adaptations. A full
legacy manifest still has status `legacy_ast_only`, not body-copy completion.
A manifest covering one function cannot certify another stage in the file.
An unchanged stage in distinct original/candidate files may explicitly reuse
the same Lean definition; each distinct Python function is still checked.
Any broken registered pair is a hard error even when that task is pending.

## Worked examples

All 19 worked-example groups currently have Lean-authored sources and no
independent Python reference registered for this scope. Their existing proof
results remain scoped to those sources. None is marked Python body-copy
complete. Several also use Lean-specific `:=`, `tl.for`, `tl.toReal`,
`dtype=` load extensions or `.round_to`; these must be reviewed as language
and semantic differences, not removed by a blind text substitution.

When a real source is identified, register the path in the optional
`examples/source-references.json` mapping, for example
`{"ExampleName": "path/to/actual_source.py"}`, and add the example's pair
manifest. A source newly authored for an example should disclose its
provenance; it must not be presented as an independent historical original.
Unknown example names and missing source files fail the gate.
The same collection-level mapping is supported for `bug_detection`,
`pretrain_kernels` and `tritonbench_g`; it supplies references without admitting
new numerical rules or solving a task's source/backend connection.

The current frontend supports the demonstrated straight-line body. Unsupported
Python control flow and unresolved store aliases must be implemented in the
shared frontend and checked against its semantics before those kernels are
marked migrated. Retaining a pending entry is preferable to omitting a cast,
changing a branch, or claiming Python syntax support that is not implemented.
