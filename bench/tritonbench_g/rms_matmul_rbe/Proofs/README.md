# rms_matmul_rbe

[rms_matmul_rbe.py](../rms_matmul_rbe.py) contains the repaired program and the
candidate that commutes the final signed-integer row multiplication. Both host
entry points share the same input checks and launch configuration.

[Kernels.lean](../Kernels.lean) contains the complete pasted callee and three-call
Q/K/V parent for each source. The same callee is instantiated with the caller's
pointer storage; no precision-specific rewritten bodies are maintained.
Every integer cast, constexpr branch, load mask, tuple split, arithmetic order
and implicit output conversion remains in the pasted source.

Caller metadata supplies pointer storage and runtime integer arguments.
Immutable `THETA`/`EPS` constexpr values use the explicit FP32 scalar
materialization interface; its realization remains an IO obligation.
`ParentArguments` in [IO.lean](../IO.lean) records the actual common parent inputs
and per-projection weights, outputs and strides. The older generalized
three-phase interface retains its full scope, including output aliases.
[Contract.lean](../Contract.lean) states the conditions checked by the host.

[RealCorrect.lean](../RealCorrect.lean) displays independent full-K normalized projection
and adjacent-pair rotary formulas, with full grid outputs, access safety and
exterior memory preservation. [FPEquiv.lean](../FPEquiv.lean) compares the original
and existing candidate over their complete schedules without experimental
numerical rewrites. Both entries include the actual pasted parent.

The proof helpers connect the pasted parent, signed loops, precision
conversions, byte memory, tail masks and public IO to these statements.
The mathematical interpretation does not claim IEEE or hardware identity.

The correspondence checker discovers all four JIT pairs from the Python and
Lean source files. [Validation records](../../../../validation/) track local
checks and independent proof replay.
