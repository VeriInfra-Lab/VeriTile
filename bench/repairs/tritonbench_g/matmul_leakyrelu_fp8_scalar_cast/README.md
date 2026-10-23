# Minimal scalar-cast candidate

This sibling candidate changes exactly nine runtime scalar/stride calls from
`x.to(tl.int64)` to `tl.cast(x, tl.int64)`. It preserves the canonical input
validation, sole BM128/BN256/BK64/GROUP8/stages2/warps8 configuration, default
TF32 FP32-input profile, live FP32 accumulator, source-left slope, FP16 output,
empty/K=0 branches, and actual JIT specialization. No unit inputs are excluded.
The original canonical source and its compile-defect packet remain frozen.

Current source SHA256:
`185ae6684de33dfff4c28b33c11f824338e585f3a283712bf4f32f2d76829457`.

## Evidence

- `check_candidate.py`: eight successful actual-specialized offline SM90
  compiles, including the original four admitted failures and tail/unit-K
  controls. `_pack_args` receives arguments in actual JIT `arg_names` order;
  every specialized constant is checked against the actual wrapper arguments.
  Unit strides remain constexpr. Three empty/K=0 controls do not launch.
- `check_source.py`: 43 complete CPU Triton source interpreter executions,
  actual FP16 results and complete input-backing byte checks. Cases include
  empty/K=0, activation and fallback strings, K tails, row/column/group tails,
  transpose, broadcast, storage offsets, aliasing and trailing storage bytes.
  Twelve existing input rejection cases and five ordinary failure controls
  also pass. These are not GPU TF32/WGMMA numerical experiments.
- Evidence lives under
  `.veritile-cache/kernel-evidence/matmul_leakyrelu_fp8_scalar_cast/`.
  `preservation-check.json` records 116 frozen files unchanged and the exact
  nine-line source delta.

## Proof scope

`Contract.lean` binds actual shared FP32 Tensor inputs and actual FP16 output
metadata, indices and storage intervals. Both input symbolic roles read from
one common physical reader, so aliases agree without an input separation
premise. Representable descriptor/storage bounds are ordinary Torch ABI facts,
not added Python guards. No host-success, allocator, compiler-success or
whole-result assumption appears in this contract.

The `TensorBridge.correct` and `TensorBridge.fp_equiv` specifications execute the complete shared
typed source interpreter and prove the independent Real finite-K formula or selected native
recurrence, physical FP16 bytes and 16-bit reconstruction, and complement/full
input-backing frame. The older `_event` lemmas remain internal intermediate
results. The native instruction primitive has fixed K8 arity; each BK64 uses
eight ordered K8 steps with its live accumulator. Scalar operations and
instruction implementations remain local model boundaries. The Real cast is
identity only in the Real interpretation.

`Kernels.lean` contains the shared readable typed quotation. The current shared
DSL's natural-index inference accepts the method spelling of the local
signed64 cast; Python uses the function spelling that repairs actual constexpr
compilation. Their local cast meaning is the same on the representable domain;
this does not assert that the original Python method call compiles. AST equality
alone is **not** the source-execution proof.

The direct shared typed-interpreter proof spans `Execution`, `ExecutionContext`,
`ExecutionBody`, `Geometry`, `TypedProgram`, `TypedReference`, `SourceGrid` and
`SourceComputation`. It derives successful prelude, every loop iteration and
epilogue, masked-store readback and frame, the complete legal grouped grid,
every finite autotuner trial and final launch, and the unchanged empty/K=0
branches. `Execution.source_event` proves the quotation-to-event connection;
no caller source/event equality or completed-result premise is accepted.

`Backend.lean` recreates the small dot/comparison backend against the current
shared Algebra record, avoiding expansion of an older frozen backend cache.
The shared quote represents FP32 algorithm values with `.real` carrier cells
under an explicit FP32 precision channel. Loads use `fp32Load` and dot uses
`some .fp32`; this representation does not change the actual FP32 input ABI,
four-byte input addressing, live FP32 carry, or FP16 output.

`Misc/Safety.lean` derives all formed clamped addresses and the exact meaning of the
nine signed64 scalar casts from ordinary descriptor bounds. `Misc/Audit.lean` checks
the public results and source/grid/replay lemmas against the standard Lean
axioms and checks the contract, execution, projection and local scalar laws
for circular references to the desired results. The delivery packet pins the
current source/import closure and successful focused build artifacts.

Core sources, Lake configuration, ledger and canonical selection are not
modified by this candidate.

`Correct.kernel_correct` and the optional `FPEquiv.kernel_equiv` theorem in
`Misc/NativeCorrect.lean` use the genuine shared
`ProgramIO.Implements` judgment. `IO.lean` lists the two strided logical input
matrices, the FP16 output matrix, the literal A/B/C quoted source regions, no
private scratch, and the entire original grouped PID schedule. The Real target
is the independent finite-K product with optional Leaky ReLU. The native target
preserves eight ordered K8 operations per BK64 block, the live FP32 carry,
activation and final FP16 conversion. The Real entry contains one specification;
the native result is an ordinary auxiliary theorem. `FPEquiv.lean` now proves
the selected-source implementation pair described below.

`Proofs/SourceSafety.lean` checks the actual reached A/B load masks and final
store. `GridSafety.lean` follows the full dynamic loop with the existing
Context/HasAcc invariants. `IOImplements.lean` derives those access bounds from
the finite logical IO windows and composes exact output cells and the raw frame.
`ReadProjection.lean`, `RegionProjection.lean` and `GridProjection.lean` reuse the
existing program-execution and physical source-step results. Output tag
coverage is proved before the decoded projection; successful execution and a
decoded equality alone are not substituted for exact FP16 output cells.

The proof's temporary storage labels 0/1/2 are internal reader labels only.
`ReadProjection.source_same` is reflexive source equality, and
`ReadProjection.physical_binding` explicitly maps both logical input channels
from the same original caller Memory with their actual offsets and strides.
A/B may share storage; no new separation premise is added. Input decoded values
may have any FP32 word, and inactive/exterior raw cells remain unrestricted.
Original buffer geometry, capacities, output separation and numerical primitive
boundaries are preserved.

`Proofs/KernelRealResult.lean` and `KernelNativeResult.lean` retain the former
complete physical-grid statements and the separate ordinary zero-fill results.
Native kernel IO retains the original positive-K domain; the full Tensor bridge
continues to cover actual zero/empty branches and autotuner trials.
`TensorBridge.correct` and `TensorBridge.fp_equiv` are unchanged: actual output
bytes, FP16 reconstruction, full input-backing frames and returned Tensor scope
remain connected through `Execution.grid_source` and the existing
computation/replay correspondence. Kernel-buffer records contain no object
identity, device or Tensor API flags. No Python, quotation, contract, dtype,
runtime ABI, numerical assumption or experiment was changed by IO unification.

## Selected source pair (2026-10-21)

`matmul_leakyrelu_fp8_right_slope.py` applies the existing activation candidate's
`0.01 * x` to `x * 0.01` transformation to this complete safe-index selected
source. This canonicalizes the scalar scale as the right operand for elementwise
scale/fusion matching. No speed improvement is claimed without measurements.
The original and candidate Python ASTs differ only at that multiply, and both
complete ordinary source quotations are readable in `Kernels.lean`. The nine
scalar-cast repair calls, all validation, the grouped grid, strides, K tails,
FP32 input/live-dot precision, FP16 output, activation branches, configuration
and empty/K=0 wrapper paths are unchanged.

`FPEquiv.source_pair_equiv` is the actual two-program
`original ≡[[]] rightSlope` result in the shared complete `ProgramIO` syntax.
Under the original input binding and IO bounds it concludes both actual trace
safety judgments, successful executions, equal complete output cells, and both
exact exterior memory frames. `Proofs/PairIO.lean` supplies these obligations
without any additional numerical formula law. `Proofs/Pair.lean` proves the local rewrite
through complete kernel execution, every listed launch, constructive successful
termination and identical reached access-safety conditions. The successful
final states are equal in their entirety, including every typed memory cell.
`Pair.implements_transfer` transports logical output cells and the exact raw
frame to the candidate for any independently established original target.
`Proofs/PairComputation.lean` also proves equality through the existing physical
adapter, every finite autotuner trial/final replay, and unchanged empty/K=0
no-launch branches. These connections impose no whole-result equality premise.

The only changed numerical operation uses the shared strict
`FP.ScalarOpsLaws.mul_comm` certificate under explicit `FP.Config`, for every
FP32 word including NaNs, infinities and signed zeros. `Pair.fp32Algebra`
fixes exactly that primitive to `FP.mul`; all unchanged instruction, comparison,
literal and conversion implementations remain explicit and identical. This is
strict selected software-model equality, with zero experimental rules and no
hardware-conformance claim. `#print_fp_assumptions` reports no experimental
atoms; the direct strict theorem dependency is recorded separately because this
source execution proof uses the structural derivation branch.

The previous single-source native/software results remain in
`Misc/NativeCorrect.lean` as ordinary auxiliary theorems, retaining their names
and complete domains. The independent mathematical `Correct` target is unchanged.

Optional native observations, Tensor/host applications and reader/audit entries
are in `Misc/`. Import these modules explicitly when needed; the primary
`Correct.lean` and `FPEquiv.lean` entries do not re-export optional results.
