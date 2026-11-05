# fast_rms_layernorm

[Python source](../fast_rms_layernorm.py) · [copied Lean bodies](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

All four production JIT bodies are copied directly: regular forward, Gemma forward, input gradient and deterministic increasing-row weight gradient. The three existing integer row-address candidates have matching Python bodies in [fast_rms_layernorm.py](../fast_rms_layernorm.py). Pointer storage and runtime/constexpr arguments are supplied outside the quotations. The selected repaired host interface uses FP32 staging and the same runtime epsilon ABI.

The source keeps its int64 program-id widening, signed weight-gradient loop, masks, ordered reductions, saved reciprocal RMS, source dtype queries and consecutive casts. The optional chain parser parenthesizes only consecutive casts; independent typed-program and signed-index regressions check that both conversions and the integer carrier survive lowering. Implicit stores do not add explicit FP32 cast instructions.

Real correctness covers complete Y and saved r, followed by complete DW and in-place DX in regular/Gemma modes. The saved-r condition and empty-row initial-DW condition remain public. Execution, source access safety, flat allocation transport, complete grid composition and exterior frames are proved from the actual bodies. The FP reference retains actual FP32 loads and integer-to-float conversions; it does not invent store casts. Equivalence preserves the existing complete typed-memory and schedule domain and uses no numerical rewrite assumptions.

[Proof helpers](.) contain mathematical specifications and execution connections. [Validation progress](../../../../validation/body-migration-progress.json) keeps local checks and independent replay acceptance separate; body copying and compilation alone do not mark the task complete.
