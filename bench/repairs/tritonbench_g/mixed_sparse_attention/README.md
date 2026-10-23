# mixed_sparse_attention: public kernel IO

The canonical repaired wrapper and original Python JIT body are unchanged.
A direct `_triton_mixed_sparse_attn_fused_mask_kernel` candidate is added.
All 73 pre-existing original/repaired files are preserved in the evidence
baseline. Historical public entries remain under `Misc/HostSource.lean`, `HostContract.lean`,
`Misc/HostCorrect.lean`, `Misc/HostFPEquiv.lean` and `Misc/HostREADME.md`.

`Correct.lean` contains one public `specification` using shared
`ArrayProgramIO.ImplementsOn`; `Misc/NativeCorrect.lean` contains one ordinary native theorem using shared
`ArrayProgramIO.ObservesOn`. `IO.lean`, `RealIO.lean` and `NativeIO.lean` display
the eight named typed arrays, source-free finite read footprints, masked output
window, exact unchanged parsed source, numerical interpretation and complete
supplied grid order. The source quotation remains under shared `Kernels.lean`.

Both entries execute the complete original grid in any permutation. Each CTA
receives fresh registers/IDs and the actual predecessor memory. The output
buffer already exists. The judgments prove actual source access safety,
successful execution, every active result and exact exterior cells; safety is
not inferred merely from output placement or a separately asserted footprint.
`Proofs/IOSourceSafety`, `IOLoopSafety` and `IOKernelSafety` establish safety at
actual intermediate states, including both dynamic sparse loops and the outer
early-return guard. `IOAccessBounds` derives its bounds from the public finite
footprints, and `IOGrid` connects the complete execution and frame.

The public raw domain stays exactly `Separated`, `Contiguous`, sequence lengths
bounded by T, positive BM/BN and a permutation of the whole launch grid. In
particular, this raw theorem does **not** add the stronger host shape,
metadata-count or sparse-index bounds below. Input arrays keep arbitrary Nat
domains and may share read-only regions; only the declared finite footprint
needs backing for access safety. The footprint is conservative, including
metadata for inactive CTAs; successful execution, outputs and frame do not
depend on allocation capacities. Empty grids and arbitrary input raw tags keep
the existing decoded-read semantics. `Proofs/IOScope` proves that every old
Real read-view/native raw input supplies the new binding and the same domain.

The full old operational endpoints are ordinary helpers in `Proofs/RawCorrect`
and `Proofs/RawNative`. Tensor/byte transports remain in `Misc/TensorBridge.lean`
and their existing `Proofs/KernelReal`, `KernelNative` and `KernelFrames`
modules. Their complete statements and proof bodies are retained. Both raw
and new public results reuse `KernelGrid.source`; `KernelIO.schedule_run`
identifies the shared schedule with the old complete source runner. Supporting
proof modules import helpers rather than public reader entries.

## Mathematical and native targets

For an active query row r, construct an occurrence list by concatenating all
count-selected blocks' 64 positions, retaining j < seqlen and j <= r, followed
by all count-selected explicit columns. Explicit columns are noncausal, and
repeated/overlapping selections retain their multiplicity. The independent
Real target is the weighted ratio with

```
weight(r,j) = 2 ^ ((scale * 1.44269504) * sum_d Q[r,d] K[j,d])
output(r,d) = sum_occurrence_j weight(r,j) V[j,d] / sum_occurrence_j weight(r,j)
```

Empty support returns zero. The exact source decimal coefficient is retained.
`KernelReal` uses the explicit ideal Real algebra and the independently proved
occurrence-list formula. It does not assert a numerical error bound for the
native low-precision computation.

`NativeIO` exposes the independent array form of the existing
`KernelNativeReference` initial-word online-softmax recurrence, without calling `Source.finalValues`, `Source.update`, a statement
executor or returned memory. It preserves FP32 scale/query multiplication,
the existing FP16/BF16 storage selector for dot operands/probabilities, seeded
matrix accumulation, max/exp2 and reduction order, guarded normalization, and
final selected-format narrowing then FP32 widening. The checked wrapper still
selects FP16; no new dtype variant is added. The IO channel tag `.real` denotes
the Structural payload carrier, while source AST tags retain actual FP32
loads/computation/store and explicit narrower conversions. `KernelNative` proves that the complete decoder-selected quote reaches
that reference for every written output cell.

Rows at or beyond seqlen are not written by the kernel. Their entry values and
all four bytes remain unchanged. The wrapper's zeros_like accounts for the zero
tails of its separate host result; it is not part of these kernel endpoints.

## Tensor input and storage contract

The selected schema admits dense materialized CUDA FP32 Q/K/V/output, signed
int32 sparse metadata, one device, positive H, D in {16,32,64,128}, and 64-by-64
tiles. Q/K/V/output use the canonical dense strides. Counts are bounded by
metadata capacities; selected offsets/columns are within sequence length;
sequence lengths are between zero and T. Existing signed32 formed-index, u64
pointer/storage and grid-axis conditions are explicit. Read-only inputs may
alias, including mixed FP32/int32 views; output is disjoint from input backing.
Empty B or T executes no programs.

`KernelContract.Admission` is the declarative complete entry schema inherited
from the checked source. `checkSparse` is an executable finite signed-metadata
checker with an equivalence proof. The Real-scale/FP32-word relation is explicit
in the schema; Python numeric-to-FP32 conversion remains a local ABI port.

`KernelTensor` binds actual shared Tensor objects and storage addresses to the
source's initial cells. Floating roles obtain shared `Torch.FP32.TensorLaws`;
int32 metadata uses coherent signed-word storage observations instead of FP32
dtype laws. Output placement derives finite source-store coverage, bounds and
separation. Both endpoints commit exactly those source stores through
`Memory.API`, read the actual output Tensor addresses back, and preserve every
other backing byte. Input-byte and inactive-output-byte frame theorems are
explicit public endpoints.

## Compiler boundary and validation

The selected native object interpretation is SM80, four warps and two stages,
with scalar/cast/exp2 operations, fixed m16n8k16 FP16 matrix instruction ports,
and complete local-register/XOR2/XOR1 reductions. Variable-width dot chains and
reductions are constructed from these local operations. There is no whole-dot,
whole-reduction, whole-output or whole-lowering equality premise. The object
byte decoder and selected local routing scheme remain abstract ports. No
concrete verified decoder or formal mapping for saved compiler profiles is
claimed. Other GPU code-generation schemes are not certified by this SM80
inventory. ISA functions are model data without primitive-equals-software laws.

Evidence under `.veritile-cache/kernel-evidence/mixed_sparse_attention/` includes:

- Identical normalized 67-statement canonical Python/readable DSL ASTs.
- The existing CPU contract suite's 65 passing cases.
- Two direct-kernel checks for nonzero inactive tails, exact backing-byte/input
  frames, empty support, duplicate occurrences and noncausal explicit columns.
- Nine exact-AST profiles through the actual installed JIT binder and `_pack_args`,
  covering all feature widths, scalar-one specialization, tails, empty sparse
  capacities, unaligned-to-16 pointers and negative-zero scale; all 45 selected
  TTIR/TTGIR/LLIR/PTX/cubin artifacts are preserved and hashed.

CPU interpretation and offline compilation are corroborating evidence, not GPU
numerical measurements. Local primitive/compiler ports are not deferred
numerical-experiment TODOs. Historical numerical-obligation files remain
preserved records. Host allocation/fills, launch/autotuner lifecycle, resource
admission and synchronization are outside this task. The shared grid-product
compatibility issue remains separately deferred as instructed.

The format pass validates the task-local dependency closure against the shared
compiled interfaces and records source/dependency hashes. This targeted build
does not constitute independent replay or new GPU measurement.

The main entries quantify geometry, layout/separation, positive tile sizes and
a complete grid order. Their shared IO judgments bind the named arrays and
state the independent Real/native targets, with the original sequence-length
condition in the public domain. Operational states, execution and cell-level
reasoning stay in `Proofs/`. `Misc/TensorBridge.lean` retains all Tensor results and
input/inactive/empty frames through the same established source-execution
foundation; `Misc/Audit.lean` checks both public and retained endpoints.

`FPEquiv.fused_mask_equiv` compares the complete original grid with an
actual direct candidate in `Kernels.lean`, using the same eight arrays,
footprints, sequence-length domain, geometry and arbitrary complete grid order.
The candidate fuses the sparse-column loop's consecutive `qk = tl.zeros(...)`
and `qk = tl.where(mask, qk, -inf)` assignments into
`qk = tl.where(mask, tl.zeros(..., dtype=tl.float32), -inf)`. This removes one
intermediate score-register assignment per column iteration; no measured speed
claim is made. Both sparse loops, early return, dot/reduction order, FP32 zero
precision, low-format casts and final narrowing/widening are preserved.

`Proofs/FusedMask.lean` proves the local selection and register-overwrite
identity. `FusedContext` and `FusedKernel` lift exact full-state execution and
trace-safety equality through the actual loops and guard; `FusedGrid` lifts
these to every grid invocation and establishes success, all typed output cells
and both exterior frames. The proof uses no FP arithmetic identity and no
experimental rule, and retains all input word patterns. The native runtime
and ordered dot model remain the established execution-model boundary.

Both original and candidate are ordinary direct `triton` definitions in
`Kernels.lean`; the old `Proofs/Whole002/Kernels.lean` path is a compatibility
import. Existing single-source native results remain ordinary optional
theorems in `Misc/NativeCorrect.lean`. Neither main entry imports `Misc/`.

`HostContract.lean` remains a shared contract helper: the main `Proofs/KernelContract.lean` uses its metadata and admission definitions. Optional host applications and readers are under `Misc/`.
