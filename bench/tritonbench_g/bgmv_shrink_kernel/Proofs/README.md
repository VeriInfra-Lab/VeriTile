# bgmv_shrink_kernel

Current repaired Python implementation: [bgmv_shrink_kernel.py](../bgmv_shrink_kernel.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Initial Y plus scaling times the complete logical K gathered GEMV; signed index -1 preserves initial Y. Existing toFloat/ofFloat wrappers retain the declared channel/storage interpretation.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` copies the complete repaired `_bgmv_shrink_kernel` body, including
all 15 top-level statements and every nested statement. Runtime stride bindings,
FP16/BF16 input storage, signed32/signed64 metadata storage and output dtype
metadata stay outside the quotation. `Proofs/MetadataStorage.lean` proves the
complete signed32 and signed64 kernel instances identical after the source
widening, for every other call argument. Both accepted metadata widths thus
connect to the same full execution, safety and output proofs.
The original early return, signed64 loop bounds, `accumulator *= scaling`,
`SPLIT_K == 1` branch, masks and atomic operation are retained verbatim.

`bgmv_shrink_kernel_scalar_left.py` is a new Python realization of the existing
scalar-left comparison graph. It changes only the final multiplication operands.
The caller frontend supplies Triton's implicit atomic operand conversion using
the destination pointer dtype, without adding casts to the pasted source.
The complete-program FP comparison retains the existing strict software FP32
multiplication certificate, all atomic orders, outputs and exterior frames.
It introduces no numerical experiment or hardware-conformance claim.

The migration stays pending until independent replay accepts its exact current
source/import closure. Body matching or successful compilation is insufficient.
