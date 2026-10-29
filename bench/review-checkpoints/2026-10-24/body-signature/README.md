# Pasted Triton body with an external call signature

The selected `dequantize_rowwise` function body can be copied unchanged into
`triton { ... }`. Its nine statements, including both int64 casts, are text-equal
to Python after dedenting and pass the full AST comparison. There are no
`tl.region`, `tl.param`, Lean antiquotes or manually inserted output casts in
the body.

The generic `DSL/CallSignature.lean` frontend obtains pointer storage types,
runtime scalar precision/carriers and constexpr names from a separate static
call/verification signature. It inserts Python's implicit store conversion and
rejects missing metadata, duplicate bindings, unsupported aliases and unresolved
store destinations. It does not infer arbitrary Tensor types from Python text.
The call adapter supplies runtime argument expressions outside the source body;
the regression proves that different supplied values reach the actual store.

The Real proof executes the new call and connects its full memory to the
established reference, then proves successful execution, safety, the independent
`scale * signed_weight / 127` formula and every exterior cell. The FP proof
compares complete new original/candidate calls directly, using only signed
integer multiplication commutativity; all typed memory agrees and both runs
succeed. Floating operation order and conversions remain in the source model.

Validation passes:

- Selected Real/FP public entries, required source bridges and optional Tensor
  entry; public axiom checks use only the standard base and FP assumptions print
  `none`.
- Nine Python regression tests, including literal body equality, cast removal,
  metadata/antiquote rejection, wrong scalar/output precision and missing unused
  parameters.
- Both Lean frontend regression files, including implicit FP16 masked storage,
  signed input typing, runtime scalar values and fail-closed pointer/store cases.
- Both current argmax public entries after exposing shared store helpers and
  strengthening alias validation.
- A separate Git checkout began with no project artifacts or `.veritile-cache`;
  initial and final builds pass using pinned external Lean/Mathlib artifacts.
  The final store guard was checked out from Git and incrementally rebuilt in
  that same isolated directory. No project artifacts were copied.
- Formula, reader-role, IO-organization and task-ledger guards pass.

`review.json` records the exact scope and source commits. `source-closure.json`
pins all 192 local imported modules; all are tracked in Git. Build and proof
receipts are retained beside them.

This is the first straight-line example using the pasted-body interface. Other
operation/control-flow spellings keep the existing DSL support boundary. It
preserves the project's mathematical-index/primitive-model boundaries and makes
no new compiler/ISA/host refinement claim. The official comparator is unavailable.
The optional Torch CPU contract checker was not rerun because system Python has
no Torch; Python implementation/wrapper sources are unchanged. No GPU run or
new numerical experiment is claimed.

The refreshed 183-task inventory has **2 AST-checked tasks, 44 other tasks missing
explicit int64 casts, and 137 uncertified tasks**. Only `dequantize_rowwise` has
been migrated to this pasted-body interface; argmax retains its earlier
normalized source correspondence check. The other tasks are not silently
recertified.

Reproduce from the repository root:

```bash
python3 -m bench.jit_correspondence bench/repairs/tritonbench_g/dequantize_rowwise/jit-correspondence.json
python3 -m unittest bench.tests.test_jit_correspondence
lake build bench.repairs.tritonbench_g.dequantize_rowwise.Correct \
  bench.repairs.tritonbench_g.dequantize_rowwise.FPEquiv bench.tests.CallSignature bench.tests.SourceDType
python3 bench/check_main_formula_review.py
python3 bench/check_main_entry_roles.py
python3 bench/check_io_organization.py
python3 bench/task_progress.py --check
```
