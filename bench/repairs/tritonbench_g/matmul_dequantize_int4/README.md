Optional Tensor/native applications, host results and audit readers are in `Misc/`; canonical `Correct.lean` and `FPEquiv.lean` do not re-export these optional results. Declaration names and proof scopes are preserved.

Current FPEquiv pair (2026-10-21): `FPEquiv.group_extent_equiv` compares the
actual original graph with a concrete group-extent specialization, written as
`original actual c i ≡[[]] groupExtent actual c i`. The candidate changes
`GROUP_SIZE_M * num_pid_n` to the precomputed
`GROUP_SIZE_M * ceil(N / BLOCK_SIZE_N)`. Its complete callable source is
`matmul_dequantize_int4_group_extent.py`; N and the group extent are specialized
with `tl.constexpr`, without adding any accepted-value restriction. It removes
a runtime scalar multiplication from grouped PID mapping. No measured speedup
is claimed. All 17 autotunes, strided output layouts, masks, both group branches
and the independent quantizer remain unchanged.

`Kernels.lean` displays complete ordinary original-source and compiled-source
candidate quotations. `Candidate.prelude_binding`, `source_tail` and
`compiled_tail` connect their exact common prefix and unchanged remainder.
`Proofs/GroupExtent.lean` proves complete source execution and trace-safety
equality for any arithmetic backend. The native pair replaces just the
prelude invocation in the full existing graph, then retains every K-block's
FP16 FMA, live-carry MMA, casts, stores and order. Exact state equality lifts
through the full original grid, including all outputs, safe source accesses
and exterior typed-cell frames. The existing four local primitive bindings
remain explicit in the main pair; no new numerical rewrite or experiment is
introduced. The independent Real formula and optional single-source native
and Tensor results retain their full original scope.

# matmul_dequantize_int4

`Correct.realIO` declares the complete A[M,K], Q[K/8,N], S[K/group,N], Z[K/group,N/8] input windows and C[M,N] output window. The channels retain FP16 A/S/C and signed-int32 Q/Z; their offsets are the original positive-stride views. `Correct.correct` uses shared `ProgramIO.ImplementsOn`: actual source access safety, successful complete grid execution, the independent GPTQ matrix product, and every exterior typed cell. The signed-int32 condition applies only to finite logical Q/Z arrays, not arbitrary unobserved memory. All 17 tuning configurations, both group branches, K loops, masks, precision and layout conditions are preserved.

`FPEquiv.nativeIO` in the optional `Misc/NativeCorrect.lean` exposes the already accepted compiled graph through shared Structural invocations: source prelude, each original loop prefix, one explicitly FP16 FMA, live-carry MMA suffix, and output epilogue. `IONativeAdapter.grid_run` connects this phase schedule to the exact existing `Native.grid` for every initial raw state under the same four local primitive obligations. `nativeFormula` is the independent global-coordinate FP16/FMA/fixed-K16 recurrence. It retains NaNs, infinities, subnormals and signed zeros; no floating rewrite or native-equals-Real claim is added. `IONative.physical_source` connects the same schedule to the existing full byte publication and exterior byte frame.

The separate PyTorch quantizer remains fully proved in `Proofs/Quantizer`, `QuantReal`, `QuantFP`, `QuantHost` and `QuantRaw`. `Proofs/RawAPI.lean` retains its independent formulas, every Q/S/Z array, typed publication and full byte publication/frame as ordinary frontend theorems, together with the old raw matrix theorems. The quantizer returns **FP32 S**, whereas the matrix API accepts **FP16 S**. The unchanged Python implements two independent APIs and supplies no intervening cast; these results are not claimed to be a directly composable typed pipeline. No conversion implementation or new conversion proof is introduced by this format pass.

`Misc/TensorBridge.lean` retains all checked Tensor applications, actual storage identities, physical readback and frames. Its supporting `KernelTensor` uses `RawAPI`; neither the readers nor proof helpers import `TensorBridge`. `Misc/KernelAudit.lean` checks ordinary kernel/helper results; `Misc/Audit.lean` additionally checks Tensor applications. Main statements contain only public IO, logical domains, independent targets and the existing primitive interfaces.

Original source quotations, Python guards, accepted precision and numerical/compiler/Torch boundaries are retained. The candidate changes only the documented pure integer group extent. There are no new numerical assumptions or experiments. The new typed-FMA interpreter node preserves the FP16 operation label; its operand conversion, broadcast and safety behavior are program/model facts, not deferred numerical experiments. Historical numerical/GPU validation and independent replay statuses remain as recorded in the task ledger.
