# fast_layernorm

Optional proof entries are now collected under `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean), [Misc/HostCorrect.lean](Misc/HostCorrect.lean), [Misc/HostFPEquiv.lean](Misc/HostFPEquiv.lean). Their declaration names, statements, proof bodies and recorded scopes are retained; they are imported separately from the primary entries.

The main `FPEquiv.kernel_equivalent` is a two-source shared
`ProgramIO.Equivalent` certificate for each of the three established per-program
IO scopes (`IOPair.Stage`). It requires both source executions to be safe and
successful, equal active typed output cells, and both complete exterior frames.
Forward and input-gradient use the historical X row-index commutation on the
current repaired bias-aware sources. The affine candidate replaces
`row*N + where(active, columns, 0)` by `row*N + columns`, removing the redundant
column selection in every row iteration while retaining both active load masks.
This is a direct address-computation optimization; performance has not been
measured. No floating operation or reduction order changes. Full candidate
source is readable in `Kernels.lean`; all rows, separate output counts, optional
bias and empty/tail cases are retained with no numerical experiment rule.

`Correct.lean` has one shared `ProgramIO.Implements` specification for each of
the forward, input-gradient and affine-gradient kernels. `IO.lean` names the
canonical source, channels, row/block addresses, active input/output masks and
single-program launch. The independent Real formulas describe normalization,
saved inverse/mean, input gradients and sums over every affine loop row.

`Misc/NativeCorrect.lean` uses shared `ProgramIO.Observes` for the same complete source
outputs under the existing FP32 primitive, reduction and compiler models.
Arbitrary symbolic input terms remain admitted; output observations preserve
each region/address-specific compiler schedule. These are source-to-native
observation results, not new floating-point transformation laws. No numerical
experiment or additional numerical assumption was introduced.

All three specifications prove actual source access safety, successful execution,
typed active outputs and the exact exterior frame. Vector inputs require backing
only on active lanes. The affine loop still reads both scalar statistics on every
row, including blocks with no active columns. Separate `wc` and `bc` masks and
optional bias are preserved, including output counts above `N`. Real forward and
backward retain `N ≤ B`; native results retain their broader scope without that
condition. `N = 0` is not excluded.

`Proofs/KernelSafety.lean` checks the real source execution path, including the
complete affine row loop. `Proofs/IOImplements.lean` and `Proofs/IOAffine.lean`
connect those certificates and the existing `KernelOutputs` completion results
to shared IO. `Proofs/RawCorrect.lean` and `Proofs/RawNative.lean` retain the former
whole-memory statements with their original names as ordinary theorems; the
latter also retains epsilon conversion and compiler-binding specializations.

`Misc/TensorBridge.lean` remains unchanged and retains all physical-pointer, checked
Tensor, output-storage and native-byte applications. It shares the existing
source-completion foundation with the public IO results. Original sources,
Python, `Contract.lean`, scalar ABI, source precision and numerical boundaries
are retained; the three readable candidate sources are added to `Kernels.lean`.

`Proofs/SourcePair.lean` connects the exact single index rewrite in the complete
forward/backward sources. `Proofs/AffineDirect.lean` and
`Proofs/AffineDirectSafety.lean` prove every optimized affine loop iteration,
termination, final memory and actual masked-access safety. The resulting final
memory equals the original complete heap, including inactive output lanes;
`Proofs/IOPair.lean` transfers output equality and both exterior frames into the
shared successful two-program judgment. All floating payloads are arbitrary.
