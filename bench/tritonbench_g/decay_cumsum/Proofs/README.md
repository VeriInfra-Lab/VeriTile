# decay_cumsum

Current repaired Python implementation: [decay_cumsum.py](../decay_cumsum.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

decay_cumsum_kernel_real for each forward, prepare and backward main/tail Program.

The three original `@triton.jit` bodies are copied statement for statement in
[Sources/Body.lean](../Kernels.lean), with pointer dtypes, runtime arguments,
constexprs and the module literal `inv_ln2` bound outside `triton { ... }`.
All source int64 casts, forward and reverse loop bounds, the unmasked final-gate
load, pointer updates and the distinction between plain dq/dk stores and explicit
casts remain visible. `Selected.lean` and `Mathematical_Kernels.lean` only select
invocation profiles of these same bodies.

The existing candidate commutes three integer products in the feature masks;
[its Python source](../decay_cumsum.py) and
[candidate provenance](../../../../validation/tasks/decay_cumsum/candidate-provenance.json) record that exact change.
The public entries retain their input conditions, execution scope and numerical
assumptions. Proof helpers decompose the actual source and connect successful
execution, every output, access safety and the full exterior frame to independent
mathematics. Validation remains pending until exact-current independent replay.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
