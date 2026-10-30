# Guarded fused recurrent HGRN repair

The original backward kernel ignores the cotangent of the returned final state.
For example, a loss depending only on that state produces zero gradients instead
of propagating through the recurrence. The repaired backward kernel starts its
carry from that cotangent. It uses a forward loop counter to visit time in reverse
and forms the previous-output pointer only when the time index is positive.

The original forward kernel, all 12 autotune configurations (BD 32/64/128 and
1/2/4/8 warps), default fusion, output allocations, saved tensor identities, and
outer initial-state `detach()` are retained. Backward still returns `dx`, `dg`,
`None`, `None`. Missing output cotangents use one `zeros_like` allocation; present
cotangents are passed directly. There are no added layout copies, snapshots, or
device switches. The original corpus Python file is unchanged.

## Supported contract

Inputs and present cotangents are ordinary contiguous FP32 CUDA tensors on the
same device, which must already be current. Input shapes are `[B,H,T,D]`; the
optional initial/final-state tensors use `[B,H,D]`. B, H, and D are positive;
T may be zero. Aligned contiguous views with nonzero storage offsets are allowed.
Readonly inputs and incoming cotangents may alias, including an incoming output
cotangent that aliases the saved forward output. The ordinary allocations for
forward outputs and backward gradients are fresh at their respective phases.

Executable guards validate the actual storage extent and pointer alignment,
CUDA grid limits, signed32 source intermediates, and u64 formed pointers,
including padded lanes and forward loop endpoints. The accepted domain covers
all original tile candidates. Noncontiguous cotangents are rejected; supporting
those layouts is an optional future extension.

## Kernel-only checked results

`Correct.lean` proves complete arbitrary-Real forward/backward kernel IO and
the independently defined loss derivative with the initial state held detached.
`Misc/TensorBridge.lean` retains the actual Tensor-coordinate results. `Misc/NativeCorrect.lean` retains ordinary
exact FP32 word observation theorems under the explicit scalar instruction and
compiler-schedule models. Neither public theorem has an assumed execution
success, whole-output equation, saved-output equality, or allocator callback.
The previous wrapper endpoints remain in `Proofs/HistoricalCorrect.lean` and
`Proofs/HistoricalFPEquiv.lean` outside the default public import closure.

`Kernels.lean` quotes the complete, unedited forward/backward bodies from
`fused_recurrent_hgrn_repaired.py` and the existing stride-folded candidate
`fused_recurrent_hgrn_stride_folded.py`. The candidate shares the same backward
body; `jit-correspondence.json` checks the two original bodies and candidate
forward body. The correspondence regression also checks the candidate backward
body against the shared Lean definition. Call metadata
is outside the quotations. The tuple assignment, `_` loop counter, `do` pointer,
`elif` branch, load/store casts and optional-state branches remain in source order. The reused
loop and grid proofs establish execution, all outputs, and each phase's full
tagged-memory frame. `Proofs/KernelTensor.lean` ties source reads and results to
actual FP32 Tensor pointers and logical coordinates, retaining nonzero storage
offsets. Guard-derived bounds cover finite-width arithmetic, formed pointers,
active storage, every grid owner, writer uniqueness and physical separation.
Forward outputs saved for backward are derived from execution. Cotangent
observations use their later kernel entry time and may alias saved outputs.

The Native boundary has three local raw-word instruction laws (addRN, mulRN,
fmaRN) against concrete RNE FP32 software operations with gradual underflow and
canonical quiet NaN. Source graphs retain seed+0 and the forward-store/backward-
load rounding boundary. Compiler schedules depend on source topology, not word
values. `Kernel.profileSchedule` records the observed short-sequence contraction
and looped reverse cases. Instantiating these explicit primitive/compiler
models is separate from proving kernel equality; no GPU numerical experiment
is required or claimed. The Real interpretation accepts arbitrary Real input
values, independently of the raw-word model.

This delivery does not introduce host allocation, return-object, device-switch
or unified final-byteheap obligations. Ordinary Tensor metadata and coordinate
laws remain the input representation boundary. Existing runtime grid-product
compatibility remains outside this kernel-only scope.

## Validation

`python bench/repairs/tritonbench_g/fused_recurrent_hgrn/check_contract.py` passes
150 complete CPU forward/backward paths across all three tile sizes and 11
prelaunch guard rejections. The test interprets the repaired Python kernel AST
with real CPU backing storage and mocked CUDA launches. It covers empty time,
partial tiles, optional seed/final output, absent cotangents, the ignored-final-
cotangent defect, caller bytes, nonzero storage offsets, overlapping readonly input views,
cotangents aliasing saved outputs, and saved/returned identities. It checks that the
forward kernel AST is unchanged and that the repair retains detachment and
default fusion without adding contiguous copies. This is not GPU execution.

The public Lean roots and their helpers compile with standard-base axiom audits.

`check_kernel_profiles.py` compiled 56 actual wrapper/JIT-binder SM80 profiles
covering all 12 original autotune candidates and T=0/1/2/3/4 optional branches.
Its `--loop-controls` mode compiled eight further T=17/1024 profiles. PTX confirms
forward FMA, retained seed+0, short reverse-carry contraction, and independent
add/multiply in the looped reverse case. These are compiler observations, not
native numerical experiments. See `kernel-evidence/delivery.json` for pinned
sources, compiler artifacts, import closure, and fresh Lean build receipts.
The T=3, BD=128, warps=1 backward candidate also retains its loop; the schedule
selection follows the compiled specialization rather than sequence length alone.

`Correct.lean` now exposes a named `kernelIO`, the independent recurrence
`formula`, the scalar loss `objective`, and one `Spec.Real` specification. The
shared `ProgramIO.Implements` relation connects the complete forward/backward
source schedule to all saved states, both gradients, the optional final state,
access safety, successful execution and the exact exterior frame. The derivative
uses the same `dx`/`dg` functions as the logical output tuple. It is not an
execution-result assumption.

`IO.lean` declares all five input channels and four output channels, contiguous
windows, optional h0/ht/dht masks, actual source identity, and the two complete
2D grids with independently selected block sizes. Zero time and optional-state
branches remain covered. Source precision stays FP32; the Structural interpreter's
erased Real cell tags are documented explicitly.

`Misc/NativeCorrect.lean` has one ordinary native observation theorem using shared
`ProgramIO.Observes`. Arbitrary input graph terms remain admitted. The observer
maps only output payloads to words, preserving dtype tags, input binding, source
execution and the frame. The same three existing local scalar laws and the
value-independent compiler schedule remain explicit; no new numerical law or
experiment is introduced.

`Proofs/SourceSafety.lean` checks the actual loads/stores in both source bodies;
`Proofs/GridSafety.lean` follows actual successors through both time loops and
complete grids. `Proofs/IOImplements.lean` reuses `RawKernel.complete` for the
computed values and phase frames, connecting only the finite active input windows
to the existing total-reader proof. The former raw public endpoints, including
both separate phase frames and the computed-gradient derivative, are retained
in `Proofs/KernelRealResult.lean` and `Proofs/KernelNativeResult.lean`.

`Misc/TensorBridge.lean` is unchanged and retains the full accepted Tensor results,
including metadata checks, coordinate bindings, checked safety/separation and
native readback, from the same identified raw execution foundation. Supporting
proofs do not import reader entries. Reader/Tensor audits live in `Misc/Audit.lean`;
`Proofs/KernelMisc/Audit.lean` audits the shared foundations. Python, Kernels, Contract,
precision, source scope and numerical-validation status are unchanged.

## Implementation pair (2026-10-21)

The missing pair is now supplied by `stride_folded_equiv`, described below.
The retained source-to-native result remains optional in `Misc/NativeCorrect.lean`.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `Misc`:

- `Misc/Audit.lean`
- `Misc/NativeCorrect.lean`
- `Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.

## Complete stride folding pair (2026-10-21)

`stride_folded_equiv` compares the complete original two-phase program with
`Pair.strideFolded`. The complete ordinary `forwardStrideFolded` quotation folds
`T*D`, two constexpr parameters, before multiplying by the runtime head ID in
`p_x`, `p_g` and `p_o`. This reduces the source runtime address multiplications;
no measured performance improvement is claimed. The Python candidate is
`fused_recurrent_hgrn_stride_folded.py`, with a whole-module AST check that pins
these three changes. The actual backward kernel is reused unchanged.

The proof uses natural-number associativity on the three pointer values, proves
full forward state equality and access-safety equivalence, and composes every
forward and backward launch in the original phase order. It retains arbitrary
positive independent forward/backward block widths, padded lanes, T=0, empty
shapes, all optional initial/final/cotangent flags, every saved state and both
gradients, optional final state and both exterior frames. Every FP32 instruction,
cast and recurrent order is unchanged. The independent Real recurrence and
Mathlib loss derivative remain intact. The main experimental rule set is empty;
existing optional native local conformance boundaries are unchanged.
