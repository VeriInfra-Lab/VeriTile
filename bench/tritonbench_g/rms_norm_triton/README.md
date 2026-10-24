# RMSNorm source and reusable proofs

The original benchmark Python is preserved unchanged. The task's selected
implementation and public correctness/native-equivalence endpoints are in
`bench/repairs/tritonbench_g/rms_norm_triton`; see that directory's README for
its checked contract and kernel-only proof boundary.

`Kernels.lean` is the one readable FP32 typed DSL shared by the original and
checked repair. `Contract.lean` contains the independent RMSNorm finite-sum
formula and the ordinary Tensor/copy/storage vocabulary. `Correct.lean` and
`FPEquiv.lean` are reusable successful-execution helpers, not the selected
public specification endpoints.

`Proofs/WholeTile`, `WholeGrid`, `WholeMath`, `KernelBoundary`, `KernelBinding`
and `KernelIndex` prove the source run, logical readback, complete storage
frame and actual finite pointer correspondence. `Observed`, `ObservedMath`
and `NativeKernel` connect the source graph to an independent lane-specific
native reduction/FMA recurrence, preserving runtime epsilon and the singleton
FMA case. Primitive numerical behavior and compiler binding are explicit
model boundaries; no numerical experiment is claimed.

Historical typed families, address-commutation equivalence and slice contracts
are preserved in `Proofs/LegacyKernels`, `LegacyCorrect` and `LegacyFPEquiv`.
The older runtime-outcome helpers remain available through `WholeHost`,
`WholePublic`, `WholeIndex` and `ObservedHost`, outside the selected public
kernel proof dependency chain.
