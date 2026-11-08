# matmul_triton_autotune

[The repaired Python source](../matmul_triton_autotune.py) and its `leaky_relu` helper are copied into [Kernels.lean](../Kernels.lean). The complete kernel body has 28 top-level Python statements, including its docstring; the helper has two. Caller dtypes, runtime arguments and constexpr values remain outside these quotations.

The copied source keeps all signed int64 casts, the dynamic `tl.cdiv` loop, safe address recomputation, the three-argument dot with its live accumulator, the conditional helper call and FP16 output conversion. The old Lean implementation that advanced input pointers has been removed. The loop count comes from the source dimensions and block size rather than a separately supplied `numKBlocks` field.

[The integer-index candidate](../matmul_triton_autotune.py) is a newly introduced source derived from that repair. It commutes the grouped tile-count product and imports the unchanged original activation helper. Both variants retain the CUDA/HIP configurations and the same floating operations. No performance result is claimed.

[Correct.lean](../RealCorrect.lean) states the independent matrix-product and optional Leaky ReLU formula over logical matrices. [FPEquiv.lean](../FPEquiv.lean) is the complete-program comparison entry. The copied source is connected to the independent live-carry formula for all five retained input profiles. The complete selected grid, source safety, frame and retained native helpers have been rechecked locally, together with the dependent `triton_matmul` task. Integer grouping bounds follow from the existing launch grid; no public input restriction was added. The complete FP comparison uses no numerical relaxation assumption. Fresh independent comparator replay is still required; see the [current validation status](../../../../validation/body-migration-progress.json). Source comparison and successful compilation do not certify proof completion.

[Contract](../Contract.lean) · [IO](../IO.lean) · [Proof helpers](.).
