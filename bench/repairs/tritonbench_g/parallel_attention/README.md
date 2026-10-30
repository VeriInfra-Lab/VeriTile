# Checked FP32 parallel quadratic attention

Current selected public entries: [Contract.lean](KernelOnly/Contract.lean), [Correct.lean](KernelOnly/Correct.lean), [FPEquiv.lean](KernelOnly/FPEquiv.lean). The [contract review](../../../validation/contracts.json) records their current scope and assumptions.

## Historical implementation and audit notes

The notes below describe earlier revisions and their recorded boundaries. Their old file lists, receipts and completion counts are historical; they do not establish the current selected contract.

Reference responsibilities (2026-10-23): `KernelOnly/Reference.lean` retains the exact Real scalar model and independent shard/gradient formulas; source-to-formula bridge proofs are in `KernelOnly/Proofs/Reference.lean`. The separate main-FP-required ordered model in `KernelOnly/NativeReference.lean` remains on the main proof path.

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `KernelOnly/Misc/TensorBridge.lean`, `KernelOnly/Misc/NativeCorrect.lean`, `KernelOnly/Misc/Public.lean`, `KernelOnly/Misc/PublicV2.lean`, `KernelOnly/Misc/FPEquivV2.lean`.

The selected review entry is `KernelOnly/Misc/PublicV2.lean`. It exposes complete
raw forward/backward Real grids and the native `FPEquiv.forward_grid` and
`FPEquiv.backward_grid` results through `KernelOnly/Misc/FPEquivV2.lean`.
The simultaneous Tensor results `TensorBridge.forward_joint` and
`TensorBridge.backward_joint`, together with the older per-Tensor applications,
live in [KernelOnly/Misc/TensorBridge.lean](https://github.com/VeriInfra-Lab/VeriTile/blob/d67b1d0444c2e0fb8066414c2041129b53bf2931/bench/repairs/tritonbench_g/parallel_attention/KernelOnly/Misc/TensorBridge.lean).
The auxiliary `KernelOnly/Misc/Public.lean` audit imports that bridge explicitly.

`parallel_attention_repaired.py` implements the complete `parallel_rebased`
forward path and its custom backward. The original Python is unchanged. This
continues the earlier repair while closing actual storage, device, byte-memory,
returned-handle and checked-domain boundaries. It does not replace the direct
launches with segmented launches or normalize/copy Q, K or V.

## Repair and admitted calls

The retained kernel repair explicitly zero-pads all 20 loads, gives the
normalizer one writer (`i_v == 0`), masks ragged future/diagonal query lanes in
DK/DV, and chooses key/value tiles of at least 16. All 18 dot sites keep
`allow_tf32=False`. Forward tiles are 128×32; backward tiles are 64×32. BK/BV
are the existing power-of-two choices capped at 128. All source loops,
coefficient operand orders, five stores, two-stage/four-warp options and direct
grids are retained. Only dO/dZ are materialized with `.contiguous()` in backward.

The checked public entry accepts ordinary, resolved, strided FP32 CUDA Tensor
objects on one device, with Q/K shape `[B,H,T,K]`, V shape `[B,H,T,V]`,
`1 ≤ K ≤ 128`, and `V ≥ 1`. Inputs must be contiguous. Actual nonzero storage
offsets, arbitrary irrelevant singleton strides, and physical input aliases
are allowed. An explicit redundant active-stride check makes the canonical
source-address binding formal. There are no Q/K/V copies.

The custom backward independently rechecks saved Q/K/V and admits ordinary
resolved, strided FP32 CUDA dO/dZ tensors on their device, with shapes
`[B,H,T,V]` and `[B,H,T]`. Incoming nonnegative strides may be expanded or
noncontiguous; actual touched storage spans must be aligned and bounded.
Lazy-negative/conjugate flags are rejected before the empty return and before
the retained `.contiguous()` calls. No resolution copies were added. After
those calls, the actual prepared tensors are checked again for dtype, device,
shape, resolved flags, contiguous active strides, storage span/alignment, and
formed-pointer room before allocation or launch. Empty backward skips copying
and preparation but still applies the incoming checks.

Flags are Python bools. Epsilon is a Python int or float, excluding bool,
convertible to Python float; NaN and infinity are admitted. The public scale
is `K**-0.5` or integer one. `return_both` takes priority over normalization.
The lower-level `ParallelBasedFunction.apply` alias retains its explicit-scale
interface; the checked public scale claims concern `parallel_rebased`.

Nonempty calls check actual storage spans, FP32 alignment, signed-address
storage bounds and room for formed pointers. They retain the conservative
signed32 guard

```
NV = ceil(V / BV)
formed_limit = (NV * B * H * T + 256) * max(K, V) + 128 < 2**31
B * H <= 65535
ceil(T / 64) <= 65535
```

The actual forward grid is `(NV, ceil(T/128), B*H)` and backward grid is
`(NV, ceil(T/64), B*H)`. Empty B/H/T return empty allocated results and gradients
without Triton launches. Backward empty gradients retain `.empty_like()` stride
behavior, including noncanonical empty strides. Forward, backward and eager normalization enter Q's
owning CUDA-device context and restore the caller's previous device.

`Contract.lean` and `Proofs/CallContract.lean` formalize both successful
acceptance and rejection. The domain contains no output or successful-run
premise. Ordinary successful Torch allocation, saved-Tensor/version preservation
and runtime query observations are explicit API contracts.
`Proofs/BackwardContract.lean` separately proves incoming/prepared acceptance
reflection, incoming byte-storage bounds, prepared canonical addresses, and
formed-pointer bounds. Its `BackwardBinding`/`PreparedBinding` tie the public
backward premises to saved actual inputs, checked incoming descriptors, and
prepared kernel handles. The rank/stride-length correspondence is the ordinary
Torch metadata query ABI. Single-output endpoints retain the logical dY view
of Torch's outer autograd operations; the custom boundary checks their produced
dO/dZ arguments. Historical helper theorem scopes are unchanged.

## Independent mathematical and floating results

For each head, the independent Real specification is causal quadratic
attention:

```
S[t,j] = scale * sum_e Q[t,e] * K[j,e]
W[t,j] = S[t,j]^2 for j <= t, otherwise 0
Z[t]   = sum_j W[t,j]
O[t,p] = sum_j W[t,j] * V[j,p]
```

`Correct.correct_real` follows the actual returned allocation, metadata and
caller-device restoration. `Correct.both_vjp`, `normalized_vjp`, and `raw_vjp`
prove directional derivatives of independent loss functions using saved Q/K/V,
saved scale, later cotangent views, and actual returned gradient handles.
Normalization's Real derivative requires nonzero denominators. No theorem
differentiates floating rounding. `Correct.real_frame` states the complement
frame.

`FPEquiv.returned_reference` compares actual returned handles with the independent
ordered scalar graph in `Spec.lean`. `FPEquiv.kernel_bytes` identifies every
stored byte of O/Z/DQ/DK/DV after the selected host reduction. `fp_equivalent`
includes the entire byte heap, return identity, saved input identities, launch
and restored devices. Pair-return and single-output backward byte endpoints
cover saved scale, raw propagation, and the actual denominator retained by
DivBackward. Forward, backward and single-output byte frames are explicit;
`caller_storage_frame` preserves every byte of all three caller storage
allocations, including gaps and input aliases, under ordinary fresh-allocation
separation. `Checked.gradient_metadata` binds actual gradient allocations to the
semantic handles used by the VJP results.

`Proofs/Physical.lean` executes each event against the evolving byte heap and
performs four concrete little-endian byte stores. Its observation theorems
commute that execution with the word semantics; no completed tensor is passed
to the executor. Packing is proved for every FP32 bit pattern. `Materialize`
proves strided/expanded cotangent copying and the unchanged-storage contiguous
branch. `Pointer` proves actual singleton-aware stride addresses, separately
sign-extended pointer-add stages, active storage bounds, and 64-bit no-wrap.
Existing `Machine*` and `GridEnumeration` proofs cover signed32 intermediates,
loop endpoints/post-advances, the direct grid, masks, unique stores and coverage.

The remaining numerical assumptions are `ScalarAgreement`'s ten fixed-arity
scalar families: constants, add, multiply, FMA, division, negation, same-dtype
cast and the finite public scale conversion. No tensor, completed dot,
reduction, host-call, memory/frame or output numerical law is assumed. The
closed software profile uses IEEE-value RN-even scalar operations with explicit
NaN/subnormal behavior and the independent 128-entry scale-word table. This
profile does not assert native GPU conformance for those scalar rules.

## Source and compiler binding

Repair `Kernels.lean` is a source-only entry for the complete source graph;
source/helper execution, bounds, independent specifications and proofs remain
separate. `Proofs/Source.lean` embeds the exact current Python AST and SHA.
Original unmodified source declarations were moved into original
`bench/tritonbench_g/parallel_attention/Kernels.lean`; their names, bodies,
unspecified-load behavior and historical theorem scopes were preserved.

The native reference is the selected source-specific SM90 FP32 SIMT circuit.
The checker resolves dot/carry routing, scalar coefficient order, normalizer
trees, specialization constants and seeds from current compiler artifacts.
It checks all 128 CTA threads and conservation of scalar FMA counts. Host
reductions have a separate finite selector based on pinned Torch
ReduceConfig/TensorIterator code, including coalesced singleton reductions and
ordered subiterators. The normalization autograd DAG is traced separately.

This is a bounded compiler binding with ordinary typed compiler/ABI contracts,
not a universal compiler correctness theorem. The source/domain and Real proofs
are shape-parametric. Native evidence currently covers eight specialization
pairs; an unseen specialization needs its own structural admission. It does
not acquire a checked descriptor merely because the Python guard accepts its
shape. See `checks/COMPILER-CONTRACT.md` and
`checks/HOST-COMPILER-CONTRACT.md`. Parser correctness is not claimed as a Lean
result. No GPU execution was performed.

## Current evidence

- `validation.json`: 11 actual Python/Triton CPU-interpreter forward/backward
  cases, 10 empty/no-launch cases, 12 rejected-domain cases, four actual-autograd
  lazy-negative rejections (dO/dZ, active/empty), input-storage bytes
  preserved and fresh output/gradient allocations. CUDA residency and owning
  device contexts alone are observation shims; Torch storage, views, reductions,
  autograd and kernel bodies execute.
- `backward-contract.json`: two original-host lazy-negative pointer witnesses,
  actual autograd lazy-flag forwarding, and 21 rejected backward/storage controls.
  Expanded/copied and zero-stride views pass. Device/residency/conjugate controls
  use explicit observation shims; impossible storage metadata uses synthetic
  query records. A contiguous lazy-negative tensor retains its identity and
  positive raw bytes while exposing negative logical values. The original host
  passes that pointer to the kernel; its scalar pointer semantics disagree with
  the mathematical derivative. This is not reported as GPU execution.
- `cuda-lowering.json`: 16 actual-JIT-binder compilations, covering tiny, ragged,
  ordinary/interchunk and value-sharded shapes, each with/without scaling and
  forward/backward. Artifacts are under `/tmp/parallel-attention-lowering`.
- `lowering-binding.json`: eight current structural specialization admissions.
  `checks/mutation-guards.json`: six rejected corrupt coefficient, seed,
  shuffle and carry programs.
- `host-schedule.json`: nine finite selector controls and five rejected mutants;
  this is synthetic metadata coverage, not capture of a CUDA runtime.
  `host-autograd-trace.json`: current CPU ATen normalization/backward DAG.
- `source-binding.json`: exact source/AST/Lean-file hashes and eighteen rejected
  source mutations. `Proofs/Reader.lean` audits the public statements and axioms.

The historical defect witness `pa_unspecified_padding_correct` executes the
original source with an unspecified tail value `u` and obtains both O and Z
as `u^4` despite zero active Q/K. It proves a permitted source-semantic defect;
these values are not reported as observed CUDA outputs. Duplicate normalizer
writers, missing ragged masks and unsupported tiny dot tiles are separate
structural defects covered by the repair checks.

The independent Real IO entry is `KernelOnly/Correct.lean`; the ordinary native
IO theorems are in `KernelOnly/Misc/NativeCorrect.lean`. These show the actual
full source schedules, all five
input/output channel shapes, exact masked windows, the source FP32
interpretation and an empty private-scratch list. Forward exposes O/Z;
backward exposes DQ/DK value-shard partials and DV. Z retains the original
shard-zero guard. Host shard reductions are outside these kernel statements.

`Correct.attentionIO` implements the independent squared causal attention
and derivative formulas under the original `ShapeContract`, with the actual
runtime Real scale kept explicit. The ordinary `FPEquiv.attentionIO` theorem in
`KernelOnly/Misc/NativeCorrect.lean` uses the shared
`ProgramIO.Observes` relation: exact symbolic source execution observed through
the existing native scalar/compiler model. It preserves every symbolic input,
every raw word/tag entry image and the original unrestricted native descriptor
shape/flags. In particular, it does not import Real's `ShapeContract`, exclude
mismatched raw tags, or narrow K to BK. `PrimitiveLaws` remains the same seven
fixed-arity instruction families; no transformation law was added.
`KernelOnly/FPEquiv.lean` records `missing_pair_target` and has no main
equivalence specification.

The shared IO judgments prove finite input binding, actual source access
safety, successful complete-grid execution, all typed active outputs and the
complete exterior cell frame. `Proofs/IOSafety*` follow the real prefix, loops,
branches and stores; `IOBounds` derives their bounds from the declared IO
windows. `IORealInputs` connects only finite enabled logical inputs to the
independent formulas. `IONativeAdapter.forward_adapter` and
`backward_adapter` prove that the new symbolic schedules match the original
entry/source/finish adapters for every raw state, including finishing after
each original phase. Their observed typed outputs and every exterior raw cell
agree. No result about residual local registers is added.

All former stronger raw grid theorems remain in the existing `KernelOnly`
proof modules. `KernelOnly/Misc/TensorBridge.lean` retains `forward_tensor`,
`backward_tensor`, `forward_joint` and `backward_joint`, with simultaneous
physical outputs and a common backing heap, derived from the same original
NativeStage grid foundation. `PublicV2` and `FPEquivV2` remain compatibility
readers. This IO-format revision changes no Python source, kernel quotation,
input contract or numerical assumption and performs no new GPU/numerical
experiment.
