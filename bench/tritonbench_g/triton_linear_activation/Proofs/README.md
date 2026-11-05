# triton_linear_activation

Current repaired Python implementation: [triton_linear_activation.py](../triton_linear_activation.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Complete dense K sum plus optional bias, every activation branch, and optional saved preactivation.

The complete original `kernel_fma` body and all four original JIT helper
bodies are pasted in [Sources/Body.lean](../Kernels.lean). The caller keeps its
27 top-level statements, every original branch, descending loop, mask, pointer
update and plain store. Helpers keep their docstrings, nested calls and return
expressions. `SourceKernels.lean` and `CandidateKernels.lean` only bind invocation
arguments to that shared source; they contain no separate full implementation.

The module bindings preserve the Python `math.sqrt(../2.0)` and
`math.sqrt(2.0 / math.pi)` expression trees. As in the existing public contracts,
Real interpretations use mathematical source tags and the native model binds
the actual rounded FP32 words separately; no equality of host floating sqrt
with mathematical Real sqrt is asserted. The actual helper intrinsic is
`tl.extra.cuda.libdevice.tanh`, including its nested fast-GELU call.

[The candidate Python](../triton_linear_activation.py) retains
only the existing conditional `GROUP_M == 8` width-index shift; all other
statements and all helpers are unchanged. The independently checked hashes and
transformation are in [candidate-provenance.json](../../../../validation/tasks/triton_linear_activation/candidate-provenance.json).

The public entries retain their input conditions, complete execution and output
scope, safety, exterior memory frame and numerical assumptions. This migration
remains pending until the current source and proofs pass independent replay.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
