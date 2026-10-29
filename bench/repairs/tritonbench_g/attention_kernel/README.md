# attention_kernel

`Correct.kernel_correct` states that the complete original source grid implements the independent finite attention formula in `Target.mathematical`. The ordinary theorem `FPEquiv.kernel_native`, retained in `Misc/NativeCorrect.lean`, states the corresponding native word observation against `Target.native`. Both use the shared `ProgramIO.Implements` relation and retain the original machine/address closure conclusion, arbitrary complete grid permutation and empty behavior.

`IO.lean` lists Q/K/V as `[B,H,N,D]`, bias as `[B,H,N,128]` and caller output as `[B,H,N,D]`, with the original per-buffer strides and selected FP16 or BF16 storage tags. Input aliases remain allowed through simultaneous input binding; the original output separation condition remains explicit. There is no private scratch. Each invocation executes the original quoted source with the original fresh registers, program IDs and grid dimensions.

The mathematical formula uses exactly N keys and D features, both relative-bias windows and the source coefficient `1.44269504`. The native target retains actual scalar casts, FP32 online state, source-ordered reductions and dot operations, final storage conversion, runtime scale bits, compiler plan and physical lane owners. The instruction model is the existing scalar/conversion/MMA boundary. There is no added floating rewrite, finite-word restriction, numerical assumption or native-to-Real equality claim.

`Proofs/IO*Safety.lean` proves access safety along the actual entry, dynamic loop successors, final store and complete launch schedule. `Proofs/IOFormulaBinding.lean` and `IOMemoryBinding.lean` connect only the finite logical input windows to the independent source-ordered target. `IORealFormula.lean` reuses the existing softmax proof; its proof-only completion of unused memory is shown irrelevant by finite input congruence. `IOResults.lean` composes this with the existing `RawKernel.complete` execution/output/frame proof.

`Misc/KernelResults.lean` retains the former raw-memory and empty-grid results. `Misc/TensorBridge.lean` retains the complete Tensor, physical allocation, exterior/backing frame, read stability and numerical transport statements through the same execution foundation. Public entries do not import the bridge, and proof helpers do not import the public entries. The Real headline and native auxiliary theorem retain their statement-surface and standard axiom audits.

Python guards, quoted kernels, input contracts and historical evidence are unchanged. This format revision adds targeted Lean checks only; independent native replay and GPU experiments remain at their recorded status.

## Complete implementation pair (2026-10-21)

`FPEquiv.query_early_equiv` compares the actual complete original source grid
with the complete `queryEarly` candidate in `Kernels.lean`. In both FP16 and
BF16 branches, the unchanged `qk_scale`, Q load and scaled-Q narrowing statements
move directly after `Q_block_ptr`, ahead of independent K/V pointer construction
and FP32 carry initialization. This issues the Q load earlier to permit overlap
with independent prologue work. No speedup has been measured. The complete
Python candidate is `attention_kernel_query_early.py`; its whole-module AST
check verifies only these three statements move. The two original source
definitions remain identical to their preserved versions.

`Proofs/Pair.lean` proves equal final prologue states, equivalent actual access
safety, unchanged whole-kernel suffix execution, each invocation and the whole
caller-selected grid order. The shared IO judgment includes successful execution
and source safety for both implementations, every active typed output cell and
both complete raw exterior frames. Original arbitrary strides, input aliases,
output separation, empty grids, both original storage branches, all raw input
words and source precision remain admitted. The theorem is generic over the
original scalar/dot interpretation and encode/observer; its original native
instruction and lane-layout instantiations are included without a finite-word
restriction. The experimental rule set is empty.

The historical division-to-reciprocal candidate was examined first. Its local
`DIV-MUL-RCP` rule requires finite numerator/denominator and a nonzero denominator;
those properties do not hold over the retained all-word input domain. That
unguarded candidate is not used as an all-word equivalence or treated as a
missing experiment for the present exact scheduling pair. The original native
observations remain ordinary theorems in `Misc/NativeCorrect.lean`.

The independent `Correct.lean` attention formula, complete input domain,
precision, launch coverage, machine/address closure and frames remain intact.
Optional Native/Tensor applications stay in `Misc/`, outside the main import
closure. No GPU run, new compiler/ISA conformance task or numerical experiment
is introduced by this scheduling proof.
