# rms_matmul_rbe

Repaired Python source: [rms_matmul_rbe.py](rms_matmul_rbe.py).
Existing candidate: [rms_matmul_rbe_index_commuted.py](rms_matmul_rbe_index_commuted.py).
The candidate commutes only the final signed-integer row multiplication.

[Kernels.lean](Kernels.lean) contains the complete pasted callee and three-call
Q/K/V parent for each source. The same callee is instantiated with the caller's
pointer storage; no precision-specific rewritten bodies are maintained.
Every integer cast, constexpr branch, load mask, tuple split, arithmetic order
and implicit output conversion remains in the pasted source.

Caller metadata supplies pointer storage and runtime integer arguments.
Immutable `THETA`/`EPS` constexpr values use the explicit FP32 scalar
materialization interface; its realization remains an IO obligation.
`ParentArguments` in [IO.lean](IO.lean) records the actual common parent inputs
and per-projection weights, outputs and strides. The older generalized
three-phase interface retains its full scope, including output aliases.

[Correct.lean](Correct.lean) displays independent full-K normalized projection
and adjacent-pair rotary formulas, with full grid outputs, access safety and
exterior memory preservation. [FPEquiv.lean](FPEquiv.lean) compares the original
and existing candidate over their complete schedules without experimental
numerical rewrites. Both entries include the actual pasted parent.

[Proof helpers](Proofs/) connect the pasted parent, signed loops, precision
conversions, byte memory, tail masks and public IO to these statements.
The mathematical interpretation does not claim IEEE or hardware identity.

[Correspondence manifest](jit-correspondence.json) checks all four original and
candidate JIT bodies. [Validation records](../../../validation/) separately
track local checks and exact-current independent replay acceptance. Body
comparison and successful compilation alone do not establish completion.
