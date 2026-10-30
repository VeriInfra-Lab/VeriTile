This separate v2 candidate preserves the frozen first candidate and its
independent NEEDS_REVISION review. It closes the two reported domain gaps.

The Python wrapper retains the five positive-power-of-two/K-minimum checks and
adds three preallocation checks: M*K, K*N and M*N must each be at most 1048576.
This is Triton's complete block-tensor frontend limit, not a device resource
guard. The JIT body and the wrapper after the eight checks are unchanged.
The original Python and historical conditional proofs remain preserved.

`KernelOnly/Contract.lean` proves that the executable checks match the public
static domain. Positive dimensions plus the tile limits imply that every
dimension is at most 1048576, so all previously separate signed32 dimension
conditions follow from the contract.

`KernelOnly/Kernels.lean` reuses the complete readable shared source from v1.
`KernelOnly/Correct.lean` exposes `kernel_math`, the independent Real matrix
product over raw initial typed memory. `KernelOnly/Misc/NativeCorrect.lean` retains the ordinary theorem
`kernel_native`, the independent fixed-instruction result and full cell frame
over raw shape, flags and initial memory. The full checked Tensor applications
remain as `TensorBridge.kernel_real` and `TensorBridge.kernel_fp_equiv` in
optional `KernelOnly/Misc/TensorBridge.lean`. It does not require ObservedSchedule or
an extra Dimensions premise. The exact source operands, finite pointer loads,
fixed K16 first/carried instructions, FP32 carry, output encoding, byte stores,
actual Tensor results and input/exterior frames reuse the proved v1 refinement.

The selected fixed-instruction profile and instruction-local correspondence
remain explicit native realization boundaries. The twelve recorded offline
compiler profiles are retained as `TensorBridge.observed_profile`, a separate
corollary in optional `KernelOnly/Misc/TensorBridge.lean`.
Neither static checks nor that corollary assert universal successful compilation,
device capacity, or GPU instruction conformance. There is no whole-dot,
whole-lowering, successful-output, allocator or returned-object premise.

Validation: four new Lean modules; nine preallocation rejections; all eight flag
branches; and three exact-limit wrapper captures. The unchanged kernel reuses
16 actual nonempty interpreter calls and 12 real-binder offline compilations.
No GPU run, numerical transformation or new primitive experiment was performed.

## Public matrix IO (2026-10-20)

`KernelOnly/Correct.lean` and `KernelOnly/Misc/NativeCorrect.lean` declare typed
matrix IO for the complete shared source. The channels retain two FP16 input
matrices, the actual transpose/stride selection, and the selected FP16 or FP32
output. The independent matrix sum and native fixed-K16 recurrence take logical
input matrices rather than reading an operational state in the public formula.

The shared `FP.Structural.ProgramIO.Implements` judgment proves all actual
block-pointer accesses are within the declared channel capacities, the whole
source executes, every output satisfies the independent formula, and every
other typed cell is unchanged. `KernelOnly/Proofs/SourceSafety.lean` checks the
actual three block-pointer constructions, two unmasked loads, dot/conversion and
store using the source evaluator. `KernelOnly/Proofs/PublicIO.lean` connects that
safety certificate and the existing source execution proof to matrix IO.

The former operational statements are ordinary theorems in
`KernelOnly/Proofs/RealExecution.lean` and `NativeExecution.lean`. Tensor bridge
statements, shape/configuration domain, source precision, all instruction
profiles and the local first/carry/conversion correspondence remain unchanged.
This adds no dtype variant, numerical experiment or compiler claim. The regional
IO theorem does not itself assert a new flat-pointer relocation theorem; the
existing physical Tensor applications remain separate.

## Actual two-program equivalence (2026-10-21)

`KernelOnly/FPEquiv.lean` now compares the complete original with
`kernelDirectOutput`. No distinct historical kernel existed: v1 and v2 use the
same JIT body. The new directly related candidate removes the redundant
`.to(tl.float32)` after `tl.dot(a,b)` in the FP32 output branch. The dot already
has FP16 operands and an FP32 result. The FP16 output branch transparently
reuses the original complete source and retains its encoding conversion.
This removes one source conversion in the FP32 branch; no measured speedup is
claimed. All three block pointers, both complete input loads, dot options,
transpose strides, stores and the one-program schedule remain unchanged.

`FPEquiv.kernel_equivalent rest I profile h f` is an actual
`original ≡[[]] candidate` through the shared `ProgramIO.Equivalent` judgment.
It proves both source safety conclusions, both successful complete executions,
equality of every active typed output and both exterior frames. It has the same
unrestricted raw `Shape`/`Flags` scope as `Correct.kernel_math` and the retained
native IO theorem, including all eight transpose/output branches and every input
word. The existing static wrapper contract and optional Tensor results retain
their original scope.

The exact transformation uses the retained native model: FP32-to-FP32 conversion
is identity, and `sourceDot I profile` applies the same fixed K16 first/carry
instructions regardless of the inherited outer computation context. Operand
and result types, profile, instruction order and FP32 carry are unchanged.
`Proofs/SourcePair.lean` proves complete execution equality under that model and
trace-safety equivalence for the actual sources. It does not assert the
transformation for arbitrary dot interpretations. No relaxed numerical rule or
experimental assumption is used in this pair.

`kernel_native` retains its original name as an ordinary single-source theorem
in `KernelOnly/Misc/NativeCorrect.lean`. Its instruction-reference correspondence
and all previous optional compiler/hardware boundaries remain separate. Main
`Correct.lean`, `FPEquiv.lean` and their proof helpers do not import `Misc/`.
