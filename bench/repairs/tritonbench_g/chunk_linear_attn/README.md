`chunk_linear_attn` proves the four selected source kernels against independent
causal-attention and gradient targets. The selected program is the existing
`chunk_linear_attn_repaired.py`; this revision changes neither that Python file,
its guards, nor the original benchmark. It retains the repair's explicit zero
padding, terminal-state cotangent and initial-state gradient. This is a
kernel-only endpoint; allocation, dispatch and autograd implementation are not
new proof obligations.

For each batch/head the independent Real target is

- `S[t] = H0 + sum(u <= t, k[u] outer v[u])`;
- `O[t] = scale * q[t]^T S[t]`;
- `HT = H0 + sum(u < T, k[u] outer v[u])`.

`RealTarget` and `RealVJP` define these formulas and the complete VJP in Q/K/V
and optional H0, including the output and terminal-state cotangents. `normalize`
is the source's compatibility no-op; both Boolean values are covered.

The unchanged input contract admits canonical dense CUDA FP32 Q/K `[B,H,T,K]`
and V `[B,H,T,V]`, positive B/H/T, and K/V at least 16. Optional H0 is contiguous
`[B,H,K,V]`; devices agree and view flags are resolved. A supplied scale is a
finite builtin number in FP32 range. The default is Python's binary64
`K ** -0.5`, followed by its FP32 scalar ABI conversion; negative zero and
underflow are retained as actual argument words. Backward cotangents may be
strided, including zero-stride sum-loss views. Their existing `.contiguous()`
calls remain; the kernel contract binds the resulting dense buffers. A missing
output cotangent supplies the source's zero buffer.

`Contract.InputContract` and `Contract.PointerContract` name the existing
executable guards, with `check_iff` and `pointers_iff` correspondence theorems.
`Proofs/Admission` derives the old structural contract from those guards and
ordinary Tensor API facts. Existing shape, launch-axis, formed-offset, alignment
and actual-pointer checks remain unchanged. `KernelPhysical.address_conformance`
derives finite launch axes, signed32 offsets and signed63 byte pointer room.
The retained `KernelAddresses` and `AddressBounds` proofs cover all ten source
pointer orientations, their affine components, masked lanes and transposes.
The generic runtime grid-product issue remains deferred globally.

`Kernels.lean` is the shared readable source for `forwardH`, `forwardO`,
`backwardH` and `backwardQKV`. The actual settings are BT64, forward BK/BV caps64,
backward BV cap32, FP32 loads/casts/accumulators/stores, and `allow_tf32=False`.
The combined `NT*K` backward-history masks retain their exceptional-word
behavior. The final dV library sum selects a finite operation tree whose leaves
cover the key partitions; it never supplies a computed answer as a premise.

The public kernel endpoints use the shared `ReductionProgramIO.Implements`
judgment. `IOLayout` declares six finite input channels, six Real result
channels, the optional endpoint masks and three complete scratch buffers.
The native interface observes all nine result/intermediate channels, retaining
the old complete-memory result. The interpreter uses an erased `.real` cell
carrier; source precision remains FP32 loads, casts, accumulation and stores.

The mathematical specification and auxiliary native theorem are:

- `Correct.kernel_correct`: successful execution of all four shared sources,
  the independent Real outputs and complete VJP, optional state results, and
  the complete exterior memory frame, through visible finite IO.
- `FPEquiv.kernel_native` in `Misc/NativeCorrect.lean` (ordinary theorem): successful
  source execution under the retained
  FP32/FMA interpretation, the independent native coordinate-table target,
  and the complete exterior frame, through finite raw-word channels.

`IO.stages` contains the actual four source grids and the final finite FP32
sum stage. The shared relation proves access safety on the real execution
trace, successful execution, output bindings and frame; none of these is a
numerical assumption. `Proofs/IOSafety` derives every source capacity from the
declared finite windows, including the original combined `NT*K` history mask.
`Proofs/IORun` reuses existing stage proofs for arbitrary memory/local state.
`Mathematical` gives the independent Real formula and full VJP; `NativeFormula`
gives the original coordinate reference on six finite input arrays. The
`FiniteReadLocality`, `FiniteReferenceTables`, `FiniteReferenceResult` and
`IONative` helpers prove that every observed native result depends only on
those finite active inputs. No global input pin, new storage-tag guard or
restriction on exceptional words is added.

Both endpoints admit every complete permutation of programs within each grid;
the four launches retain source order. `Proofs/KernelSchedule` derives this
invariance from unique ownership, frozen reads and existing source proofs.
`TensorBridge.kernel_correct` and `kernel_native` retain the actual Tensor
input/output readback, pointer conformance, physical bytes, input backing and
exterior storage frames. Their ordinary buffer descriptors and valid Tensor
bindings supply no output formula or successful-computation assumption.
Raw strided cotangents retain their admitted layouts. The bridge imports
`Proofs/RealKernelResult` and `Proofs/NativeKernelResult`, preserving the exact
previous raw source endpoints, plus the unchanged physical execution
foundation; it has no dependency on either public reader.

`Proofs/KernelReference` defines an independent logical coordinate table, using
local arithmetic recurrences but no source executor, launch controller or
scatter writer. `ReferenceState` and `ReferenceOutput` connect existing readback
and frame proofs to those tables. `KernelExecution` connects all four shared
quotations to the tables; `KernelReal`, `KernelTensor`, `KernelPhysical` and
`KernelResults` provide the mathematical, actual Tensor and physical frame
connections. Existing helper modules remain in place because other tasks import
them. The exact old public files are retained as `Misc/LegacyCorrect` and
`Misc/LegacyFPEquiv`; their original theorem names remain exported. Their
historical host claims and conditional numeric interfaces are not new endpoints.

The native target and source interpretation use the same retained scalar/FMA
profile (`NativeTerms.lowerAt`, `Misc/ScalarBinding`, `NativeInstructions.ISA`). It
retains load/cast boundaries and binds the scale to its actual FP32 word without
confusing it with literal zero. It does not assert Real arithmetic is exact
FP32 arithmetic. IEEE primitive interpretation, Tensor representation, library
tree selection and correspondence of the chosen instruction profile to CUDA
are explicit model/API boundaries. In particular, a bounded opcode inventory
does not prove every compiler instruction or instruction ordering. There are
no numerical rewrite assumptions or deferred numerical experiments in these
new kernel endpoints.

Evidence lives in `.veritile-cache/kernel-evidence/chunk_linear_attn/`:

- Complete normalized AST comparison matches all four Python/Lean quotations
  (18/22/20/36 statements), including reverse-loop and pointer syntax mappings.
- The actual installed Triton JIT binder and packer accept all 12 specializations
  in three complete four-kernel cases: minimal unseeded, seeded T/K/V tails,
  and a single-value-block terminal-gradient case with negative-zero scale.
  SM80 offline compilation succeeds for all 12; compiler artifacts are pinned.
- A bounded inventory of those artifacts finds FP32/FMA operations and no
  MMA/TF32/FP16 instructions. Dead one-chunk state recurrences legitimately
  disappear. This is not a full compiler-refinement result.
- The unchanged `check_contract.py` passes 45 rejected and 16 admitted host
  cases and 24 exact CPU autograd controls, including tails, terminal/initial
  gradients and zero-stride cotangents. Its CPU fixture replaces CUDA shell
  handling; these controls are not GPU executions.

The frozen delivery records targeted Lean builds, axiom/noncircularity audits,
source and dependency hashes, compiler/CPU provenance and review status. GPU
execution was unavailable. Independent review/replay is reported separately;
no GPU or independent-replay success is inferred from local builds.

## Raw kernel and Tensor interfaces

The main entries take `KernelInterface.Config`: dimensions, a Real scale,
optional-state flags and finite library addition trees, with no Tensor,
metadata, device or profile label. Native execution additionally takes the
actual scale ABI word and the unchanged scalar/FMA interpretation. Both entries
execute all four source grids under any permitted per-stage schedule, give
independent full output targets, and expose finite IO output/scratch windows with their exterior memory
frame. The exact earlier source-store complement also remains available in
`RealKernelResult` / `NativeKernelResult` and the physical bridge. The Real target retains the complete
causal formula, VJP and optional initial/final-state components.

`TensorBridge.kernel_correct` and `kernel_native` retain the exact prior Tensor
applications, pointer conformance, input readback, physical output bytes and
exterior backing frames. They use the same existing KernelExecution and
KernelSchedule results. `KernelInterface.run_correspondence` proves that reduction profile/device
provenance does not affect source execution; only its certified operation tree
is a kernel premise.

The main pair is `FPEquiv.whole_equiv`: `original ≡[[]] hoisted`. The actual
candidate is `chunk_linear_attn_hoisted.py`, with a complete ordinary quotation
`forwardOHoisted` in `Candidate.lean`. It computes
`qk_head = i_bh * s_qk_h` once before the output-stage key loop and reuses it in
both Q/K block-pointer bases. This reduces two repeated integer multiplications
per key iteration to one shared multiplication per output program. No measured
speedup is claimed. The other three complete source kernels, all floating
operations and materialization/fusion boundaries, masks, optional endpoints,
launch geometry, the final finite dV sum, and the complete host module are
unchanged. `check_hoisted_source.py` verifies the complete Python-module delta
and the complete candidate quotation.

Historical arithmetic-order candidates were inspected first. In the retained
native interpretation, `profile.mul` is arbitrary and add-dot ordering can
select a different fused-dot seed; the historical scale-first/dot-left proofs
therefore remain historical local results, without a new all-word native
certificate claimed for those variants. The address CSE candidate preserves
the original entire numerical interpretation and raw-word domain directly.

`Proofs/PairOutputAddress` checks the two changed pointer assignments and the
unchanged continuation, including the hoisted-register invariant.
`PairOutputLoop` and `PairOutput` connect the actual candidate's complete
entry/body/finish to the same independent output memory recurrence as the
selected source. `PairOutputAccess` transfers the existing source access proof
through the exact pointer-prefix equality.
`PairOutputSafety` proves actual source access safety with the hoisted register
invariant. `PairIO` proves successful execution and safety of every complete
candidate grid and the final existing library sum. The minimal task-local
`ProgramPair` view fixes the original `IO.native` metadata, input encoding,
actual runtime scale-word binding and all nine output observers. It uses only
the shared `runReductionStages` and `ReductionScheduleSafe` judgments, and its
structural relation proves both successful runs, all observable channel
values, equality of the complete resulting memory and both exterior frames.
Every allowed within-grid permutation, arbitrary initial local/undefined
state, all FP32 input words and every original scalar/FMA profile are covered.
No finite-value guard, experimental rule or new numerical/hardware obligation
is added. Main import closures contain no `Misc` dependency.

The independent Mathlib Real theorem `Correct.kernel_correct` and its complete
VJP are unchanged. `kernel_native` remains an auxiliary ordinary theorem under
its original name in `Misc/NativeCorrect.lean`; existing Tensor/physical/host
applications retain their original scope and model boundaries.
