# Triton function-body correspondence

The selected corpus is **not yet fully migrated** to pasted Python bodies.
This acceptance scope is separate from the previously recorded Real and FP
proof results. An old proof, a successful build, or an unchanged inventory
does not establish correspondence with a Python JIT body.

The current machine-readable inventory is
[`validation/source-correspondence.json`](../../validation/source-correspondence.json).
It covers 517 tasks and 19 worked-example groups: 183 TritonBench tasks,
65 bug-detection tasks and 269 pretrain tasks. Each entry records every
retained Python JIT, registered pairs, source provenance and source hashes.
The 334 imported bug-detection/pretrain tasks remain outside this migration's
explicit 177-task plus 19-example scope.

The active scope and separate validation stages are recorded in
[`validation/body-migration-progress.json`](../../validation/body-migration-progress.json).
All 19 examples now have registered Python bodies. Thirty of the 177 target
TritonBench tasks have complete body-copy manifests; six previously aligned
TritonBench cases are retained as regressions. The remaining 147 target tasks
still require body migration. Indirectly loaded functions remain unresolved
until explicitly anchored, including cross_entropy_ops and fast_ce_loss;
triton_argmax remains in legacy AST mode.

Body correspondence, public Real/FP proofs, retained-module builds and official
proof replay are separate checks. A successful historical replay certifies only
its immutable source commit and recorded dependency closure. Later changes to
the shared frontend or semantics require fresh replay before affected cases can
be marked complete. The progress record preserves that distinction.

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
Any broken registered pair is a hard error even when that task is pending.

## Worked examples

All 19 worked-example groups have registered Python references and complete
pasted-body manifests. Sources reconstructed from independent mathematical
specifications are clearly identified in
[`bench/examples/SOURCE_PROVENANCE.md`](../../bench/examples/SOURCE_PROVENANCE.md);
they are not described as recovered historical Python originals.
The original and candidate bodies preserve casts, masks, branches, shapes and
operation order. Parameter/storage metadata remains outside each quotation.

The example references are registered in
[`bench/examples/source-references.json`](../../bench/examples/source-references.json).
Unknown example names and missing source files fail the gate. The separate
local record includes 94 successful offline Triton compilations across these
19 groups; it does not claim GPU execution or proof acceptance. Public proofs
and all retained helper modules must be revalidated after shared semantic changes.

Unsupported Python syntax and unresolved bindings fail closed. Such cases
remain pending until the shared frontend and the complete source-to-proof
connection are implemented and verified.

## Shared control-flow sources

Spinning and StreamK use the shared `source_program triton` frontend for their
complete control flow. Expressions retain tokens, syntax kinds and operands;
loops, calls, branches, casts and atomic expressions remain explicit. The typed
adapter must decode each entire source body successfully. `Proofs/SourceEntry.lean`
connects the original and candidate source bodies directly to their executable
programs through those checked decoding witnesses. StreamK retains all eight
helper/stage bodies, helper argument bindings and explicit FP16 conversion.
The previous complete quotation fixtures are retained in Git history. The
existing SC, capacity, execution, frame and numerical scopes remain explicit.
This connection is separate from Python body-copy coverage and CUDA/compiler
refinement. Unsupported syntax continues to fail closed.
