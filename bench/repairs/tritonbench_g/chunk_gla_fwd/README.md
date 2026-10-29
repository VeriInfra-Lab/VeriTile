The public entries follow the finite IO format:

- [Correct.lean](Correct.lean) contains the two short Real specifications,
  `producer_kernel_correctness` and `consumer_kernel_correctness`, using the
  shared `AlgorithmFillProgramIO.Implements` judgment.
- [FPEquiv.lean](FPEquiv.lean) compares the five complete kernel-stage sources
  with their existing candidates. [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean)
  retains `producer_native_observation` and `consumer_native_observation`,
  using shared `FillProgramIO.Implements` with the explicit instruction ISA,
  reduction profile and runtime scale word.
- [RealIO.lean](RealIO.lean) and [NativeIO.lean](NativeIO.lean) declare the
  actual sources, complete schedules, finite channels/windows/masks, dtypes,
  output interpretation and private scratch. Their mathematical/native targets
  are independent of execution and are defined over pure finite input spaces.
- `Proofs/` contains execution, actual source-access safety, finite input
  locality, output types, frames and physical-memory connections. The exact
  historical raw declarations, with their original names, are retained in
  [Misc/RawReal.lean](Misc/RawReal.lean) and
  [Misc/RawFP.lean](Misc/RawFP.lean).

Both IO judgments prove success, every declared output and the exact exterior
cell frame. Finite buffer bounds imply safety of the actual source/fill trace;
they are not assumed source-safety certificates. `Correct.lean` has two
specifications for the distinct producer and standalone consumer calls;
`FPEquiv.lean` has five complete kernel-stage pair specifications.

[Misc/TensorBridge.lean](Misc/TensorBridge.lean) retains the complete
`TensorBridge.kernel_correct` and `TensorBridge.kernel_equiv` applications,
including returned Tensor coordinates and FP32/device/shape metadata.
It also retains `wrapper_correct`, `wrapper_fp32`, `successful_values` and
`rejected`, with their `words`/`Values` definitions. These applications use
the same source proofs; their supported domain and observations are preserved.

This repair keeps the two original public calls separate:

- `chunk_fwd_intra_gated_gk_fn(q, k, g, scale, BT)` returns FP32 attention with shape `[B,H,T,BT]`.
- `chunk_fwd_o_gated_gk_fn(q, v, g_cumsum, A, h, BT, scale)` returns FP32 output with shape `[B,H,T,V]`. Attention and history are independent caller inputs. History has shape `[B,H,ceil(T/BT)*K,V]`.

Inputs use ordinary contiguous CUDA FP32 storage on one device, with resolved conjugate/negative view flags. Storage-offset views and harmless singleton strides accepted by `is_contiguous()` are supported directly. No input copies were added. `K` is positive; empty batch/head/time or value outputs return before setup. `BT` is a power of two from 16 through 1024, respecting Triton's `2**20` tile-element limit. Scale is a finite builtin scalar in FP32 range, converted to the actual FP32 argument by the runtime ABI; negative zero and tiny scales are retained.

The independent Real attention formula is

`A[b,h,t,j] = scale * sum_k q[b,h,t,k] * k[b,h,u,k] * exp(g[b,h,t,k] - g[b,h,u,k])`,

where `u = floor(t/BT)*BT+j` and `j <= t % BT`; every other rectangular cell is zero. The standalone output formula is the gated query/history contraction plus the causal contraction of supplied attention with values. `Proofs/Logical.lean` and `OutputGrid.formula` define these independently of source execution. `RealWhole.producer` constructs the selected direct or split/merge source branch and proves every returned attention cell and finite frame. `RealWhole.consumer` constructs the complete consumer source grid, all output cells, and its frame. The restored `WholeReal` theorems additionally prove composition of the two calls with shared gates/history.

The repair initializes attention and large-K scratch to zero, specifies zero padding on boundary loads, and removes unproved alignment hints. It preserves all arithmetic, causal masks, the runtime K branch at 256, casts, reductions, TF32 defaults, the final IEEE dot, and all four autotune warp configurations. Signed-32 arithmetic, CUDA grid axes, actual FP32 pointer alignment and byte ranges are checked before launch. `Admission.admit` constructs the proof-carrying dimensions and input contract from these checks. `FiniteSource` bounds the masked as well as active affine addresses for query/key/gate, attention, scratch, history and output. Allocations and launches use the input owner's device and restore the caller's current device.

The historical `NativeRuntime` specializes the restored source/grid/physical-memory proofs to its fixed graph profile. That profile does not describe every currently emitted reduction layout; the current kernel headline uses the corrected `KernelRuntime` path described below. Both paths retain pending dot/accumulator fusion, ordered scalar FMA chains, fixed TF32 instruction lanes, exp lowering, conversions and stores. The source scale position uses a distinct tag and is bound to the actual `BitVec 32` scalar argument. Padding zero remains a separate literal. Both public APIs have their own source-completion, returned-value, input-preservation and physical-frame theorems. There is no split-dot-plus-add versus fused-dot equality premise.

The numerical boundary has seven local interfaces: bit leaves, literals,
scalar binary operations, scalar unary operations, casts, scalar FMA, and one
fixed SM80 `m16n8k8` TF32 instruction result lane. The finite IO observation is
valid for its explicitly supplied ISA; the retained physical-frame theorem
keeps its original raw-bit identity premise. Graph induction derives numerical
transport from these local interfaces. There is no variable-sized dot, full
reduction, completed-kernel, or Tensor-result numerical port, and FP32 execution
is not identified with the Real formula.

The current native path is `KernelProfile` → `KernelRuntime` → `KernelNative`.
It corrects a historical reduction-layout mismatch: aligned K=64 uses four
contiguous register features and sixteen lanes, whereas the older
`NativeReduction.plan` used thirty-two lanes and additional warps. The new
finite `Profile` records the observed warp count and vector width; all 108
combinations of BK powers 1–256, four warp choices, and one/two/four contiguous
features have proved coverage without duplicate leaves. Register folds,
descending XOR shuffles, shared-memory moves, and local FMA contraction retain
their order. This metadata describes a local reduction layout; it contains no
program execution or numerical result. Pointer/stride specialization can change
the selected layout, so the profile must describe the emitted specialization.
The older `NativeRuntime` and wrapper results are retained with their historical
profile rather than promoted to the new entry.

`KernelReal.consumerFormula` is a closed sum over the logical K features and
causal time coordinates, including arbitrary caller-supplied attention/history.
`KernelReal.complete` derives it from the shared source for arbitrary Real
input memory. `KernelNative.complete` derives every FP32 word from the explicit
local instruction graph, preserves read-only inputs, and frames every physical
word outside the actual output/scratch allocation. The direct and empty producer
branches also preserve the unused scratch allocation. Tensor coordinate records
contain only metadata and contiguous indexing laws at the actual data pointer.

`CheckedHost.wrapper_complete` covers admission failure, allocation failure, output-address rejection, empty return, setup failure, successful constructed source execution, returned allocation identity, owner-device restoration, input preservation and physical frame. Allocator identity/freshness and successful setup configuration are ordinary runtime API facts. The native host error model exposes only caller-visible state: failed setup may have written private allocations before unwinding. The successful physical source theorem retains the producer's actual zero/inter/direct or zero/inter/zero-scratch/split/merge ordering through `FPHost.producerExecute`.

Validation: `python bench/repairs/tritonbench_g/chunk_gla_fwd/check_contract.py` compares all five source ASTs, checks 48 rejected and 8 admitted host controls, and runs 10 exact CPU interpreter controls covering both K branches, time/feature/value tails, and independently supplied attention. CUDA execution was unavailable. Original benchmark Python is unchanged.

Current validation additionally compares all five complete Python bodies with
their readable shared quotations using `check_source.py`. The delivery directory
records actual default-binder compilation, TTIR/TTGIR/LLVM/PTX artifacts, current
v2 Lean receipts, and recursive source/interface pins. Compiler controls use
metadata-only Tensor facades with no tensor allocation or GPU execution. Device
resource availability is an external applicability boundary: for example,
BT=256 can require 263168 bytes of shared memory. This observation neither adds
a wrapper guard nor narrows the contract. Representative compilation is not an
exhaustive compiler or device-resource proof.

The Real proof preserves its established attention/scratch initialization
before the inter stage. The native schedule preserves the original
zero-attention/inter/direct or zero-attention/inter/zero-scratch/split/merge
order, including empty launch skips and the full ordinary grids. Both retain
all five source kernels and the standalone consumer. Source-free geometry
partitions native attention into two disjoint, exhaustive output channels:
Structural `.real` cells under the FP32 profile and explicit `.fp32` cells.
Both carry and observe FP32 words; this distinction does not assert Real
arithmetic and does not retag memory between source invocations.

[Proofs/NativePhysical.lean](Proofs/NativePhysical.lean) proves that both finite
native targets equal every output of the original `KernelRuntime.reference`
physical project/flush endpoint. It reuses the same full-source proof as the
public IO. The retained `KernelNative` and `TensorBridge` theorems keep the
complete output, read-only-input and exterior physical frames, including unused
scratch preservation for direct/empty producers. Real IO likewise reuses
`RealWhole` and `KernelReal.consumerFormula`; all previous Tensor statements
are unchanged.

[Misc/InterfaceChecks.lean](Misc/InterfaceChecks.lean) checks source identities, full-grid
selection, the native channel partition, actual safety, physical target
agreement, retained raw/Tensor declarations, target non-circularity and axiom
footprints. Each public main statement has a strict shared-IO surface check.
This formatting change adds no numerical law or experimental TODO and does
not change Python, source quotations, admission, precision or numerical domain.
The earlier validation evidence above is retained; this pass performs targeted
Lean builds only, with no new GPU experiment or independent replay.

## Selected implementation pairs (2026-10-21)

`FPEquiv.lean` now exposes the five complete existing FP32 kernel-stage pairs
from `Kernels.lean`: inter carry/dot addition, intra scale multiplication, split
scale multiplication, merge carry/load addition, and output carry/dot addition.
Each candidate changes that one operation inside its original full source.
The proof patterns are reused against the actual repaired quotations, retaining
zero padding, casts, reductions, both K branches, masks and source order. The
merge operand domain explicitly uses the selected source's zero padding.

The reached finite-operand `Site` conditions remain part of each program's
public domain. For the diagonal sources they cover every lane of the `where`,
including both evaluated arms. The model is the existing strict FP32 software
theory; all five main pairs have an empty experimental rule set. The task-local
`PairWhere` connection cites the existing shared strict multiplication theorem.
No scalar decoding or rounding proof is duplicated.

These five full kernel-stage equivalences cover the requested complete kernels
under those stated conditions.
They do not supply a newly composed alternative producer/consumer grid or
prove its reached finite domains from arbitrary native inputs. The previous
complete producer and standalone consumer native IO statements, their unrestricted
raw-word inputs, full grids, output cells and frames remain ordinary theorems in
`Misc/NativeCorrect.lean`; their old `FPEquiv` names remain compatibility aliases in `Misc/Compatibility.lean`.
The independent Real producer/consumer formulas are unchanged.

Optional native observations, Tensor/host applications and reader/audit entries
are in `Misc/`. Import these modules explicitly when needed; the primary
`Correct.lean` and `FPEquiv.lean` entries do not re-export optional results.
Legacy native compatibility names are available from [Misc/Compatibility.lean](Misc/Compatibility.lean).

Source-only cleanup materializes all five selected FP32 originals and their
existing complete candidates as ordinary `triton` quotations in `Kernels.lean`.
The public main entries select FP32 throughout; historical mixed-storage
profiles retain their unchanged macros in `SourceCompatibility.lean` and the
same public match interfaces. Complete function identities for all original
source families and candidates hold by `rfl`, including the compatibility
branches. No operation order, padding, casts, dot options, masks, strides, input
domain, binding, output/frame or theorem declaration changes. This organization
adds no numerical, dtype, native, host or hardware requirement and does not
recount the existing proof completion.
