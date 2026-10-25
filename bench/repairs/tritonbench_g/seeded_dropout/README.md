# Seeded dropout: checked FP32 kernel and index variant

The independent pointwise target is
`if randReal(seed, i, bits32, 10) > p then x[i] / (1 - p) else 0`.
`randReal` decodes the actual canonical Philox10 FP32 random payload; it is a
deterministic finite distribution. The surrounding mathematical arithmetic is
Real arithmetic. No statistical independence or native FP32 exactness is assumed.

`Kernels.lean` contains the complete readable FP32 source and its existing
candidate. The sole candidate change is `pid * BLOCK_SIZE` to
`BLOCK_SIZE * pid`. Both retain the original runtime seed and probability,
canonical RNG, strict comparison, eager division, omitted load padding, and
the original store, without an added output cast. Their block size is 1024. Public mathematical
and FP results use these same canonical sources.

Acceptance covers the kernel. Python supplies input assertions; allocator,
CUDA context, grid submission, empty-launch behavior and returned-object
identity are outside the required public proof. Previously proved host results
remain available in `Misc/HostCorrect.lean`, `Misc/HostFPEquiv.lean` and their
existing helpers. The original benchmark Python and the previous repaired
wrapper and JIT body are unchanged. The index candidate is a separate JIT body.

## Input and memory contract

- The input is a strided, contiguous FP32 CUDA Tensor without an unresolved
  negative view bit and with fewer than `2**31` elements. Scalar and empty
  shapes are admitted. Nonempty data pointers must be four-byte aligned;
  empty tensors have no pointer check.
- `p` is an actual Python `float` with `0 <= p < 1`. NaNs and infinities are
  rejected. Both signed zeros and values below one that narrow to FP32 one
  remain admitted. No nonzero-denominator condition is added.
- `seed` is an actual Python `int`, excluding Bool, in `[-2**63, 2**64)`.
  The normal JIT branch selects i32, then i64 outside the i32 range, then u64
  for values at least `2**63`. The source preserves these three integer types;
  it does not add a u32 Python-int branch.
- Every FP32 input payload remains admitted, including zeros, subnormals,
  infinities and NaNs. Masked-off load values remain arbitrary.

`TensorBridge.python_checks_iff_preconditions` proves exact dynamic admission in
both directions, including non-Tensor and scalar-type rejection.
`Proofs/Contract.lean` derives seed ABI selection from the checked range.
`KernelMemory.checked_input_memory` uses the shared Tensor storage laws to
connect checked contiguity, negative-view and alignment guards to the actual
pointer and logical input values. No copy or normalization is introduced.

The kernel contract states capacities for the input and output regions and a
one-dimensional launched program ID below `ceil(n/1024)`. The signed32 index
and mask theorem derives exact native offsets for every lane, including tails,
and active accesses fit those capacities. No padded allocation is required.
The complete-memory frame preserves every cell outside the active output tile.

## Public proofs

`IO.lean` owns the canonical `KernelPublic.io` record. `Correct.dropoutIO`
instantiates its Real interpretation; the FP view reuses the same channels,
active windows and launch ID with its profiled backend.
`seeded_dropout_kernel_correctness` states its independent mathematical formula
through the shared IO judgment. The proof establishes actual source trace safety,
successful execution, all active logical outputs and the complete exterior frame.
The canonical decoder theorem is reused from `Proofs/WholeMath.lean`.

`IO.lean` displays the original IO and an explicit variant containing the
existing index-commuted source. The single `FPEquiv.index_commuted_equiv`
specification compares `original ≡[[]] indexCommuted` through the shared
implementation view of `ProgramIO.Equivalent`: both sources are safe and terminate, all
active typed outputs agree, and both preserve the complete exterior frame.
The seed range, extent, program ID and backend comparison availability retain
the original scope. They are interpreter/ABI conditions, not output or numerical
rewrite assumptions. Arbitrary floating algebras and all operand words remain
covered under that primitive interface.

`Proofs/SourceSafety.lean` checks the original and candidate prefix accesses,
proves the candidate reaches the same actual prefix state, then reuses the
proved canonical RNG suffix. `Proofs/PublicIO.lean` composes those trace proofs
with the existing execution/value/frame foundation. Former raw statements remain
ordinary theorems in `Proofs/KernelReal.lean` and `Proofs/KernelEquivalence.lean`,
including the complete-memory successful pair, unconditional failure-inclusive
state equality, and both source-identity theorems. The retained FP assumption
audit reports **none**. Failure equality alone is not acceptance evidence.

There is no pending numerical transformation experiment. RNG decoding,
probability extraction, comparison, arithmetic, FP32 load/store conversion,
rounding and FTZ are primitive semantic boundaries. The former seven primitive
conformance families are no longer task experiment obligations. No denominator
simplification, RNG replacement, division rewrite or floating transformation was
introduced. The Lean probability parameter records its Real value; that value
alone does not identify Python's signed-zero payload. No literal-`p=0` identity,
native GPU accuracy or compiler-correctness claim is made.

## Validation

The current source migration replays the primary and auxiliary Lean modules,
with standard-base axiom and FP-assumption audits. Current replay evidence is
recorded separately from the historical CPU results below.

The earlier recorded run of
`python bench/repairs/tritonbench_g/seeded_dropout/Proofs/check_contract.py`
checked unchanged original kernel/host-body ASTs, the candidate's sole product
swap, 216 admitted wrapper cases, 21 rejections and nine installed JIT seed ABI
endpoints. CPU execution then checks 252 complete paired kernel cases and 648
program executions. Cases include empty/scalar/tail/multiblock extents, all
three seed ABIs, signed zeros, subnormals, nonfinite payloads, arbitrary inactive
NaN loads, exact original/candidate memory equality and exterior bytes. The
published Philox10 zero-key vector is checked independently.

CUDA allocation and launches are mocked. The CPU replay validates indexing,
RNG controls and storage under its declared primitive model; no GPU execution
or new compiler/numerical experiment was performed.

`Misc/TensorBridge.lean` contains the optional Python-check correspondence. The
primary `Correct.lean` and `FPEquiv.lean` entries retain the kernel results and
do not import the bridge.

The 2026-10-21 pair-entry revision writes the existing complete implementations
as `original ≡[[]] candidate` in `FPEquiv.lean`. `IO.lean` exposes both actual
source schedules and the shared `ProgramIO.Implementation` view. The existing
`EquivalentOn`/`Equivalent` proof is reused in `Proofs/ProgramPair.lean`, so both
actual traces remain safe, both runs succeed, all active output cells agree and
both exterior frames remain conclusions. The original comparison availability, extent, launched program ID and all three
seed ABI branches remain explicit. The index commutation uses no floating
numerical rule or input-value guard.
Both Python bodies and the existing numerical evidence are unchanged. The
current Lean migration retains every source statement and FP32 call argument;
no new GPU experiment is performed.

Optional application proofs and readers are under `Misc/`. The earlier generic
self-cast host result remains explicitly named `legacy_checked_fp32` and uses
`legacyRunChecked`. The current host reader uses `ActualTile.scalar`, which
contains the actual FP32 probability threshold and no extra store self-cast.

## Statement correspondence and proof connection

`jit-correspondence.json` checks all six Lean quotations against the two unchanged Python bodies,
covering the original/candidate pair for each i32/i64/u64 call ABI. Each contains the same
nine statements. Pointer types, scalar types, constexpr block size and call
arguments occur outside `triton { ... }`. No `seed_arg`/`BLOCK_SIZE_arg`
assignments, `tl.source_integer` directives or extra `.to(tl.float32)` are added.

`ParametricExecutionModel.actual` and `actual_candidate` identify each complete
pasted program with the corresponding block-size-parametric proof model at
1024. These equations retain every statement. `ActualTile` proves canonical
random execution, successful runs, every active output and every exterior cell.
`SourceSafety` checks the actual source trace through those complete equations;
`KernelExecution.execution_eq` independently proves full original/candidate
state equality, including failure, under arbitrary primitive backends.

The historical quotes in `LegacyPastedSurface`/`LegacySurface` include two
extra register assignments and an output self-cast. Their old theorems remain
explicitly historical. `SourcePreservation.legacyBackend` is a defined adapter
that applies FP32 call probability and makes only that historical self-cast an
identity; it is not an identity-cast assumption on arbitrary M. On the unchanged
seed/size/grid contract, `SourcePreservation.kernel` and `indexCommutedKernel`
prove successful execution and equality of the complete memory observation
between the current source and the historical model under this adapter. The
optional host interpretation uses this explicit connection; allocator/setup
failure outcomes, all admitted payloads, scalar/empty/tail shapes and the exact
memory frame remain covered. The primary actual FP pair requires no adapter or
new numerical premise.
