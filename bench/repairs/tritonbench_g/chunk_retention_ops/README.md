# Checked chunk retention repair

This task restores the approved `Whole001` repair and preserves its declarations in
`VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001`. The repaired source is
`chunk_retention_ops_repaired.py`; the original benchmark folder is unchanged.
The current selected entries are `KernelOnly/Contract.lean`,
`KernelOnly/Correct.lean` and `KernelOnly/FPEquiv.lean`; see
[the kernel interface scope](KernelOnly/README.md). Correct exposes four
independent Real kernel specifications. FPEquiv records the pending connection
to the existing candidates. Native observations and Tensor applications require
explicit imports from `KernelOnly/Misc/`. The historical top-level declarations,
loss-polynomial/VJP proofs and typed kernel bodies remain.

The archive already repairs ragged-chunk reverse decay, history/dhistory block
bounds, initial/final-state gradients, minimum legal dot tiles, zero padding,
wide head offsets, and launch-device contexts. This revision retains those four
JIT bodies exactly. It tightens host checks for backing extent/alignment and
resolved storage, validates allocations, rejects aliases of a supplied `d_h0`,
requires an exact Python integer `BT=64`, and checks backward cotangents before
the existing normalization copies. First-order backward rejects a final-state
cotangent when no final state was returned.

## Admitted calls

- CUDA FP32 resolved strided tensors, positive `B,H,T,K,V`, canonical dense
  `q/k/v/initial_state`, same device, and the archived index/grid/capacity limits.
  Read-only primal arguments may alias. Nonzero storage offsets are supported.
- `BT=64`, `BK/BV=max(16,min(64,next_power_of_2(dimension)))`, and the original
  autotune choices of 1, 2, or 4 warps. No tile or dtype policy was replaced.
- Exact finite Python-float scale within FP32 magnitude range, including the
  original `K ** -0.5` default; checkpoint level is exact integer 0 or 1.
- Optional seed and final state; optional terminal cotangent; an initial-state
  gradient only when a seed exists and `ctx.needs_input_grad[3]` is true.
- Raw autograd cotangents may be expanded or strided. The archived
  `contiguous()` copies remain, and absent output cotangents use `zeros_like(v)`.
  Standalone helpers still require their documented contiguous kernel inputs.
- `d_h0` supplied to `chunk_bwd_dh_fn` must have a full backing disjoint from
  `q/k/v/do/d_ht`, even when two active views would not overlap.

`Contract.lean` and `HelperContract.lean` contain metadata predicates and Boolean
guards. `Proofs/Admission.lean` and `Proofs/HelperAdmission.lean` prove both
directions of admission for the full API, later backward call, and all four
standalone helpers. Launch bounds are derived from admitted dimensions.

## Retained top-level proof results

The checked entries are:

- `Checked.chunk_retention_ops_repaired_real_returns`
- `Checked.chunk_retention_ops_repaired_real_backward`
- `Checked.chunk_retention_ops_repaired_fp_returns`
- `Checked.chunk_retention_ops_repaired_fp_backward`

The Real target is the independent dense retention recurrence and its adjoints.
The native target is an input-only operation reference with the source's scalar
FMA/reduction order and the observed `Tensor.sum` reduction tree. It is a
conditional bit-level result, not an assertion that floating arithmetic equals
exact Real arithmetic. Every returned byte is related to the actual returned
Tensor descriptor and data pointer. Optional-state results and the seven-slot
autograd return structure are retained.

`PreparedInputs` records only individual input-load, allocation-preservation,
copy, and zero-fill facts. Its input snapshot uses each original Tensor's
observed strides; it contains no history or output equation. `Versions` relates
ordinary saved-Tensor reads to the actual completed forward call. At checkpoint
level 1 the source executes the H recomputation with final-state output disabled;
no equality to a recomputed history result is assumed. At checkpoint level 0,
the saved history is covered by the ordinary Tensor version obligation.

`CheckedMemory` derives writable address injectivity from tensor metadata and
full-backing freshness. Missing/future allocations are absent; they do not
require invented runtime buffers. `MemoryABI` specifies single-word stores and
little-endian byte observations. The public native results preserve all bytes
of original caller backings, including offset prefixes/tails and copied raw
cotangents, and all bytes outside the exact source store footprint.

The compiler/runtime boundary remains explicit: production AST and SHA256,
typed source bodies including checkpoint recomputation, value-independent
scalar/load/store graph identity, allocation/device/metadata facts, legal
completion, and the actual reduction-tree structure. Offline compilation is
evidence about those profiles; it is not a verified Triton-to-machine-code
compiler proof. The only numerical premises are eight fixed-arity scalar laws:
int64-to-FP32 conversion, add, subtract, multiply, FMA, exp2, log2, and default
host scale. There is no whole-dot, whole-kernel, history, or result oracle.
Raw words retain their NaN/sign/payload bits; the proof adds no normalization.
`ScaleABI` additionally binds the checked Python scalar's exact rational value
and negative-zero observation to its FP32 launch encoding. Explicit/default
scale selection must equal that encoding; `SavedScale` binds the later
`ctx.scale` restoration to the same Python scalar and launch word.

## Validation

Run from the repository root:

```sh
python bench/repairs/tritonbench_g/chunk_retention_ops/check_source.py
python bench/repairs/tritonbench_g/chunk_retention_ops/check_lowering.py --output-dir /tmp/retention-ops-lowering
```

The CPU replay executes the actual wrappers and all four actual JIT bodies in
Triton's interpreter. It passes 7 complete forward/backward cases with repeated
backward, 26 rejection controls, and 52 source launches. Independent dense
double-precision recurrence/autograd comparisons cover output, final state,
`dQ/dK/dV/dH0`, checkpointing, tails, aliased primals, expanded/strided cotangents,
and disabled initial gradients. Full input backing bytes and allocation
canaries are unchanged. Focused helper controls cover supplied-gradient aliases
and exact `BT` type checks.

Offline Triton 3.7.1 lowering passes 36 SM80 profiles: four kernels, three
`(K,V)` combinations `(1,1)/(17,65)/(65,17)`, and all three warp choices. TTIR,
TTGIR, LLVM IR, and PTX are saved by the script. PTX uses scalar FP32 FMA with
TF32/MMA absent. Optional state paths are enabled in those profiles. Validation
used PyTorch 2.13.0+cu130. No GPU numerical execution was performed.

Production SHA256:
`cd9a80f852cfa6502a4be8ab4be723d6f6d49caa34088dff0f5544f8aa9c7d90`.


## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.

The selected main FP entry now proves an actual four-stage program packet pair
`PairIO.original ≡[[]] PairIO.scaleFirst`. It reuses the existing forward-output
scale-first candidate at exactly the repaired FP32 source precision; the three
other original stages remain unchanged. The forward-output quotation is truly
shared with `chunk_retention/Whole001/Kernels.lean`: `Kernels.lean` exposes the
shared full candidate source and `candidate_source_identity` proves the exact
selected binding. Only `(carry+dot)*scale` becomes `scale*(carry+dot)`; the live
dot seed, ordered FMA chains, casts, masks, loop order and stores are retained.
This provides the scalar scale as first multiplicand; no measured speedup is
claimed. `ScaleFirstSafety` and `ScaleFirstPair` preserve every arbitrary PID,
all eight output families, address-labelled input binding, both source traces'
safety and success, all observed typed output cells, and both exact raw exterior
frames. There is no new finite restriction or grid/host/Tensor obligation. The
strict all-word `ScalarOpsLaws.mul_comm` certificate is reused through the
original explicit `Numeric.ScalarLaws` and literal boundary. No experimental
transformation rules or deferred numerical experiments are used.
