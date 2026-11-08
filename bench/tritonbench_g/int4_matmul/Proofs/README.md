# int4_matmul

The repaired [Python source](../int4_matmul.py) and [stride-folding candidate](../int4_matmul.py) are copied into [Kernels.lean](../Kernels.lean). Each pasted body has 25 top-level Python statements, including its docstring. Pointer dtypes, runtime scalar bindings and constexpr parameters are supplied outside the quotations. The source casts, packed nibble operations, scale and zero-point loads, dynamic `tl.cdiv` loop, and store/atomic split branches are retained.

[Correct.lean](../RealCorrect.lean) proves the complete grid implements the independent logical formula `sum_k A[row,k] * (Q[k,col] - Z[group(../k),col]) * scale[group(../k),col]` under the public geometry, matrix-domain and schedule conditions. The proof evaluates the source loop bound, transports the actual loop body to the fixed-K8 accumulation model, and retains access safety and the exterior frame.

[FPEquiv.lean](../FPEquiv.lean) compares the complete original and stride-folding programs. The candidate precombines the integer block/split stride product; it retains FP32 operations and each output address's split accumulation order. This structural equivalence uses no numerical relaxation assumptions.

[Contract](../Contract.lean) · [IO](../IO.lean) · [Proof helpers](.). [Validation records](../../../../validation/body-migration-progress.json) distinguish local source/proof checks from current independent replay acceptance. Compilation or source comparison alone does not mark this task complete.
