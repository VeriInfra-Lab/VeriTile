# Triton function-body correspondence

The selected corpus is **not yet fully migrated** to pasted Python bodies.
This acceptance scope is separate from the previously recorded Real and FP
proof results. An old proof, a successful build, or an unchanged inventory
does not establish correspondence with a Python JIT body.

The current machine-readable inventory is
[`source-correspondence.json`](source-correspondence.json). It covers every
selected revision in `tritonbench_g/task_progress.json` and every
`examples/*/Kernels.lean`, including entries without a source correspondence
manifest. Each task lists every literal `@triton.jit` function found in its
selected Python file, uncovered functions, checked pairs and source hashes.
Authored example counterparts are explicitly distinguished from selected task
revisions by `source_provenance` and a separate status.
Files loaded indirectly remain unresolved until explicitly anchored; this is
visible for `cross_entropy_ops` and `fast_ce_loss`.

At this snapshot there are 183 tasks, 19 worked-example groups and 328 literal
JIT functions in the selected Python files. Five of those functions have
checked pairs: three argmax stages in legacy AST mode, dequantize and cache
transform in body-copy mode. Argmax also checks two candidate functions in a
separate file, and cache transform checks its quotient/remainder candidate.
The other 323 selected functions are uncovered by pair manifests.
These numbers describe source-check coverage, not the number of incorrect
algorithms or invalid Lean theorems.

The file-level screen finds type directives in 140 task files and antiquotes
in 160 task files and 16 example groups. In 44 primary files, Python has
explicit int64 casts that are not visible in a direct Lean quotation. This
last count is a warning screen: imported source, specialization and semantic
connections require review. Unflagged files are not certified.

## Checking and completion

```sh
# Check all registered pairs and reject changes to the recorded inventory.
python3 -m bench.check_body_corpus --check

# Full source acceptance: currently exits 1 because pending entries remain.
python3 -m bench.check_body_corpus --check --require-complete
```

The CI job is named **Inventory drift and correspondence regressions**. It
runs on `main` and `fp-examples`, checks the recorded inventory and tests the
checker. Its green status is not a full-corpus body-copy certificate. The
explicit `complete` field remains false and full acceptance fails.

Regenerate the current inventory only after inspecting the source changes:

```sh
python3 -m bench.check_body_corpus --write
```

An actual migration must also replay the affected Lean proofs and retain the
independent mathematical specification and public FP comparison. This script
does not establish proof reachability, interpreter correctness, implicit type
promotion correctness, or hardware behavior.

## Per-function migration requirements

1. Anchor the actual selected Python source, including each stage. Do not
   substitute a comment, a simplified kernel, a generated Python mirror, or
   a function with only the same mathematical formula.
   Lean-authored examples may publish a native counterpart with explicit
   provenance; that establishes usability of the new example interface, not
   correspondence to an independent historical Python original. Such a
   counterpart cannot certify a selected benchmark task's original source.
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
Any broken registered pair is a hard error even when that task is pending.

## Worked examples

All 19 groups began with Lean-authored sources. `VectorAdd`, `FlatVectorAdd`
and `TritonBenchVectorAddition` now have newly authored native counterparts,
checked body-copy entry points and full old/new program equalities. They have
status `authored_counterpart_body_copy_checked`, not an independent historical
source certificate. Their sizes remain constexpr as in the existing examples;
this does not change a selected task's runtime ABI. The other 16 groups remain
pending. Several still use Lean-specific `:=`, `tl.for`, `tl.toReal`,
`dtype=` load extensions or `.round_to`; these must be reviewed as language
and semantic differences, not removed by a blind text substitution.

When a real source is identified, register the path in the optional
`examples/source-references.json` mapping, for example
`{"ExampleName": "path/to/actual_source.py"}`, and add the example's pair
manifest. A newly authored counterpart uses an object with `python`,
`provenance: "authored_native_counterpart"` and a nonempty `note`; it must
not be presented as an independent historical original.
Unknown example names and missing source files fail the gate.

The current frontend supports the demonstrated straight-line body. Unsupported
Python control flow and unresolved store aliases must be implemented in the
shared frontend and checked against its semantics before those kernels are
marked migrated. Retaining a pending entry is preferable to omitting a cast,
changing a branch, or claiming Python syntax support that is not implemented.
