# fused_layernorm_triton

[Repaired Python](../fused_layernorm_triton.py) · [Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

`Kernels.lean` copies both complete Python JIT bodies: the repaired original and its existing scalar-view-hoisted candidate, with 26 and 29 top-level statements. Both statistics and normalization loops, INT64 casts, unused `xmask`, broadcasts, masks, helper calls and `libdevice.rsqrt` remain in the quotations. Pointer storage, runtime arguments, `XBLOCK = 1` and the ordered Welford tree are supplied outside the copied bodies. X, weight, bias and statistics use FP32 storage; Y uses BF16 storage.

The original normalization loop retains its two-dimensional tiles. The candidate hoists three scalar views and uses one-dimensional tiles. Proof helpers project the actual source statements and prove their execution and memory effects; they do not maintain another kernel implementation. The frontend preserves FP32 broadcast precision and treats integer shape views as pure index remaps.

`kernel_real` proves the independent all-D mean, variance, reciprocal square root of `variance + 1e-5`, and affine normalization over named logical tensors. Its public domain includes positive D, complete row coverage, fresh output regions, the existing 1024/2048 block configurations and a complete ordered reduction tree. Masked final chunks are covered.

`whole_fp` compares both complete grids, including successful execution, access safety, typed Mean/Rstd/Y observations and both exterior memory frames. It holds for the shared ordered floating interpretation and uses no experimental FP rules. The ordered reference retains actual FP32 load/cast boundaries, signed integer-to-float conversion and the distinct libdevice intrinsic. Native index representability remains a separate explicit contract.

[Proof helpers](.) · [source correspondence inventory](../../../../validation/source-correspondence.json) · [validation records](../../../../validation). Local proof checks and exact-source independent replay acceptance are recorded separately; compilation alone does not complete a task.
