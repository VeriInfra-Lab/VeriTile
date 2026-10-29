# RWKV6 scalar ABI checked candidate

This separate candidate adds one assertion to the repaired forward wrapper:
on the actual launch branch `T or output_final_state`, a raw Python `int` scale
must satisfy `-(2**63) <= scale < 2**64`. The kernel body and every other Python
AST node are unchanged. No scalar is converted to Python `float`. Floats,
including Inf, NaN, overflowing-to-FP32 finite values and signed zero, retain
their original admission. With `T=0` and no final state, arbitrary integers
remain admitted and saved unchanged because no JIT binding occurs.

The original source and its independently accepted ABI-defect diagnosis remain
in `../fused_rwkv6_kernel/`. This directory is a repair candidate, not evidence
that the original unrestricted integer contract was correct.

## Scope and independent target

For traversal step `n`, let `t=n` in forward mode and `t=T-1-n` in reverse mode.
The source keeps a `[BV,BK]` register tile and writes key-shard partials:

```
O[ik,z,t,f] = sum over this key shard e:
  (H[n,z,e,f] + K[z,t,e]*V[z,t,f]*U[z%H,e]) * (Q[z,t,e]*scale)
H[n+1,z,e,f] = H[n,z,e,f]*exp(W[z,t,e]) + K[z,t,e]*V[z,t,f]
```

The optional seed is the actual source `0 + load(H0).to(float32)`, and the
optional terminal store writes `H[T]`. Native semantics retain every padded
reduction lane, source cast, operand order and local compiler contraction.
The Real target is an independent exact mathematical recurrence; it does not
assert native arithmetic equals real arithmetic.

Only this launched forward kernel is being proved. Torch `o.sum(0)`, allocation,
autograd internals and runtime implementation are outside the new proof scope.
The existing host results remain available in the original repair directory.

## Source organization and proof status

- `fused_rwkv6_kernel_scalar_abi_checked.py`: sole new branch-local assertion.
- `Contract.lean`: exact candidate admission and the check/contract equivalence;
  includes unchanged float and no-launch domains.
- `Kernels.lean`: ordinary typed `triton` sources with explicit `tl.param scale`,
  including the full register loop, reverse traversal, per-step partial stores
  and final state store. `kernel` is definitionally equal to the prior complete
  source; `earlyUKernel` moves the unchanged U pointer, key mask and U load
  immediately after the head ID.
- `IO.lean`: explicit logical Q/K/W `[Z,T,K]`, V `[Z,T,V]`, U
  `[H,K]` and optional H0 `[Z,K,V]` input channels; original affine head
  pitches; O `[NK,Z,T,V]` and optional Ht `[Z,K,V]` output channels. The
  actual `scale` register is bound at each launch, and the full original grid
  is selected exactly when `T != 0` or final state is requested.
- `IO.lean` also assembles the independent Real outputs and the generic ordered
  scalar/reduction formula. Their existing recurrence definitions and proof
  connections remain in `Proofs`; `Misc/NativeReference.lean` retains the
  optional native output observer. Both traversal directions, padded lanes,
  source casts and raw scalar interpretation are retained.
- `Correct.lean`: one `ProgramIO.Implements` specification for the complete
  grid and independent mathematical formula.
- `Misc/NativeCorrect.lean`: the ordinary theorem `FPEquiv.rwkv_native` retains the
  original `ProgramIO.Observes` statement, proof and primitive/compiler/reduction
  parameters.
- `fused_rwkv6_kernel_u_early.py`: complete optimization candidate; the only
  Python AST change from the selected scalar-ABI source moves `p_u`, `mask_bk`
  and `b_u` earlier in the kernel prologue. Host code is unchanged.
- `FPEquiv.lean`: `u_early_equiv` compares `Pair.original` with `Pair.earlyU`
  using `≡[[]]`. Both use the original complete grid, logical input domain,
  raw scalar binding, all native payloads and optional initial/final branches.
  `Proofs/PairPrefix.lean` proves complete prefix state equality and safety;
  `Proofs/Pair.lean` lifts it through the unchanged time loop and full grid,
  including successful runs, typed outputs and exact exterior frames.
- `Proofs/`: existing entry/body/range/exit execution plus finite logical input
  binding, actual reached load/store safety, exact raw-cell frame and typed
  output coverage. `IOExecution` reuses the existing source-memory and scalar
  model theorems for the actual full `runSchedule`; `IOResults` specializes
  this complete implementation to Real/native meanings.
- `Misc/RawCorrect.lean` and `Misc/RawNative.lean`: exact old raw endpoints,
  including finite four-byte native stores, arbitrary initial bytes, all input
  bytes, exterior bytes and the no-launch identity.
- `Misc/PhysicalIOBinding.lean`: the original physical reader/byte-decoded words
  supply the same logical inputs and address-labelled native leaves. Read
  aliases are preserved. Concrete IO bounds are proved, including arbitrary
  head pitches admitted by the old `Sizes`-only kernel theorem.
- `Misc/TensorBridge.lean`: unchanged full Tensor theorem statements, with only
  imports and proof references redirected to the retained Raw helpers.
- `SOURCE_MAP.md`: statement-by-statement mapping and semantic boundaries.
- `check_scalar_abi.py`: exact AST delta, unchanged raw identity, actual wrapper
  branch capture, real Triton binder/packer and targeted SM80 compilation.

The pointer channel in the existing structural interpreter uses natural
offsets. The source may form negative *dead* offsets at reverse/T=0 or the last
backedge. The proof establishes active signed-address agreement,
zero-trip/final range termination, and independence of the exit code from those
dead pointer registers. No shared-core extension or source pointer-flow repair
is introduced. The original signed setup and physical-address-room proofs are
retained and reused.

## Validation and model boundaries

The scalar checker passed 18 wrapper profiles: four newly rejected out-of-range
integers, two unchanged no-launch huge integers, and twelve legal bindings,
including three SM80 compilations producing twelve compiler artifacts. The
candidate kernel AST is identical to the retained repair's kernel AST; removing
the one assertion restores the complete old Python AST.

Evidence: `.veritile-cache/kernel-evidence/fused_rwkv6_kernel_scalar_abi_checked/scalar-abi/report.json`.
Two additional installed-specializer controls confirm raw integer `1` becomes
`constexpr`, while float `1.0` remains FP32. The native target preserves this
difference without treating multiplication by one as an IEEE identity.
Evidence: `.veritile-cache/kernel-evidence/fused_rwkv6_kernel_scalar_abi_checked/scalar-specialization/report.json`.
No GPU execution or independent numerical replay has been performed.

There are no registered numerical experiment assumptions. Raw scalar ABI
conversion, primitive instruction semantics and structural compiler/reduction
choices are explicit model boundaries. They must not be described as pending
numerical experiments. The quoted program and observation proofs are discharged; their public
statements contain no whole-execution or computed-output premise. Runtime
grid-product compatibility remains the shared deferred follow-up and is not
expanded here.

The frozen packet is `.veritile-cache/task-deliveries/fused_rwkv6_kernel_scalar_abi_checked/delivery.json`; it pins candidate source snapshots, all 17 build receipts, reused ABI/compiler evidence, and the unchanged original diagnosis.

The public entries now expose genuine shared IO judgments. Source identity,
channels, affine windows, masks, scalar arguments and complete schedule are
visible in `SourceIO.lean`. Successful execution, access safety, complete typed
outputs and exact raw exterior frames are proved conclusions. The original
`Sizes` domain remains explicit; no new pitch/contiguity restriction is added.
The narrower physical admission remains in the original Tensor contract.

The public source model and retained physical/byte endpoints reuse the same
`Source.source_memory`, `program_model` and complete-grid proof foundation.
The byte transport still executes the finite original output trace; the public
IO presentation does not replace that endpoint with only regional values.
Only `O` and enabled `Ht` cells are writable. With `T=0` and final output disabled,
the source schedule is empty and the complete raw memory frame is unchanged.

The campaign check covers the independent Real entry, the actual early-U pair,
all retained native/raw endpoints, physical binding and Tensor consumer. The
original Python source, contract and existing theorem statements are unchanged.
The new pair changes only instruction scheduling: issuing U earlier can overlap
its read latency with pointer construction and the optional initial-state read.
No performance improvement is claimed without measurement. No numeric rewrite,
experiment rule, GPU run or independent numerical replay was added.

The campaign packet is
`.veritile-cache/finish-pending-105-20261021/fused_rwkv6_kernel/delivery.json`.
It records full source preservation, the exact candidate AST delta, targeted
builds and a fresh joint main/optional check. The previous delivery above
remains historical evidence for the scalar-ABI repair.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Declaration namespaces, theorem statements, proof bodies and the
retained optional theorem scope are preserved. The required entries do not re-export
optional native conclusions. This file organization adds no mathematical or
numerical completion claim.
The retained raw compatibility applications are also under `Misc/`; the main
readers do not import them merely to re-export their optional conclusions.
