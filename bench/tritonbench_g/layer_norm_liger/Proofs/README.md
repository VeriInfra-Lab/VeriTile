# layer_norm_liger

Current repaired Python implementation: [layer_norm_liger.py](../layer_norm_liger.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Independent finite population statistics, DX expression and exact original Ico row sums for DW/DB.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The four quotations in `Sources/Forward.lean` and `Sources/Backward.lean`
copy the two repaired Python bodies and the two existing bodies in
`layer_norm_liger_optimized.py` statement for statement. The correspondence
manifest checks all four separately. Call signatures supply pointer dtypes,
integer scalar arguments and epsilon outside the quoted function bodies.
The original signed casts, tuple-shaped constructors, signed row loop,
implicit integer denominators and `.to(../dtype)` output casts are retained.
The candidate alone explicitly converts its hoisted denominator to FP32.

The public Real specifications cover forward Y/mean/rstd and backward
DX/partial DW/partial DB over the actual row interval, with the existing
separation, stride and width conditions. FP equivalence compares each complete
source against its candidate, including safety and exterior memory frames.
The optional software model uses the concrete integer-to-FP32 rounder;
source decimal literals and rsqrt remain explicit primitive bindings.
This is a software model and makes no GPU implementation claim.
