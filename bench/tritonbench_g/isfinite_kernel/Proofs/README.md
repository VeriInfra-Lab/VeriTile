# isfinite_kernel

Current repaired Python implementation: [isfinite_kernel.py](../isfinite_kernel.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

isfinite_real_correctness: every observed Real output of this masked tile is true.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` quotes the complete `isfinite_func` return body and both complete
strided caller bodies from `isfinite_kernel.py`. Call signatures, signed byte
strides and signed pointer bases stay outside each body. Both explicit stride
casts, the loaded pointer-element cast and the output pointer-element cast are
retained. The original CUDA libdevice import aliases and every `ImportError`
fallback are checked against the Python module; the model selects the declared
CUDA classifier port for the corresponding input precision.

Signed bases and displacements are added before the single memory-index
conversion, preserving negative strides. The FP32 and FP64 source/candidate
programs are checked against independent full typed programs, including the
implicit Boolean byte-storage conversion. Its identity on a Boolean result is
proved and used by complete execution and access-safety proofs. The public FP
comparison retains arbitrary classifier interpretations, NaNs, inactive-load
padding, masks and the full memory frame, without numerical rewrite rules.
The independent Real specification remains that every observed Real is finite.

Source checks, retained-module builds and independent proof-replay acceptance
are recorded separately; a successful build alone does not complete the task.
