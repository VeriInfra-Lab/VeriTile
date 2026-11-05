# rms_rbe_matmul

Current repaired Python implementation: [rms_rbe_matmul.py](../rms_rbe_matmul.py).
The [candidate](../rms_rbe_matmul.py) changes only the final output-row
index multiplication from `pid_m * BLOCK_SIZE_M` to `BLOCK_SIZE_M * pid_m`.
The other two JIT functions and all host input checks remain identical.

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) ·
[Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

The complete RMS callee is shared verbatim with `rms_matmul_rbe` through its
pasted `SourceProcedure`. Its call interface retains the original literal
`EPS` and `THETA` constexpr ports. The local rotary caller and frequency helper
are complete pasted Python bodies. [source correspondence inventory](../../../../validation/source-correspondence.json)
checks all three original and all three candidate JIT bodies.

The main Real theorem covers the original complete rectangular grid,
RMS-normalized projection across all logical K entries, optional adjacent-pair
rotation, typed storage, execution, safety and exterior memory frame.
The FP comparison retains the original failure-aware full-state index
transformation. Runtime integer casts use integer-conversion ports; literal
constexprs use their literal ports. No equality between arbitrary conversion,
cast and literal operations is assumed.

The separate rotary entry uses the shared dense 2-by-1024 interface. Its source
keeps the original quotient/remainder Boolean ceiling expression. The bridge
must prove its discrete equivalence to the shared rotary grid and transport
complete Real execution and access safety, including odd and empty shapes.
The helper's local integer `DIM` remains integral in index arithmetic and uses
FP32 literal materialization beside its explicit FP32 tensor operand.

[Proof helpers](.) contain the source connections and mathematical proofs.
[Validation records](../../../../validation) distinguish source correspondence,
local proof checks and independent replay acceptance. A successful compilation
or correspondence check alone does not mark this case complete.
