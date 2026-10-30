# kldiv_ops separate repair candidate

Candidate status: frozen for independent whole-task review after all 39 Lean
modules, public signatures and axiom dependencies passed private checking.
The unchanged original is not certified by this candidate. No registration,
shared proof, progress record, Git state or frozen repair has been modified.

## Source and selected production profile

Original SHA256: `36512c93f35738488b343c4b2db4a2bb5b11ae6967976673099794efa9d8c2b1`.
Candidate Python SHA256: `90fa8cf3819ccbbdb72f10c20e050fa4faed6d01bad2ead31a46b3d4208fc446`.
The evidence directory is
`/root/.codex/visualizations/2026/10/16/01a14539-f72c-7c83-aa71-3a0014d6e60a/repair-kldiv_ops-001`.

The selected benchmark production profile is matching FP32 prediction, target,
cotangent and destination, FP32 output, runtime FP32 epsilon, both log_target
branches and all four reductions. This is not a certification of other storage
or mixed-dtype profiles. Python's original harness and floating expression order
are retained. The block is min(16384,next_power_of_2(max(1,V))); the original
4/8/16/32 warp selection expression remains (the 32 branch is unreachable under
the cap). Offline evidence is Triton 3.7.1, CUDA SM80, warp size32, stages3,
default fusion and Torch default floating dtype FP32. No GPU was available or used.

## Reviewable repair and exact domain

Forward resolves lazy flags and packs both logical input views. Backward packs
target, clones the normalized entry cotangent before writes, writes a fresh dense
work tensor and copies it into the original logical destination. This preserves
identity on the scalar-one branch and supports physical aliases between inputs,
gradient and destination. Strided destination gaps are preserved. Tensor-library
copy failure on overlapping destinations remains an ordinary actual API failure.

Both kernels widen dimensions, strides, program IDs and aranges to i64. Reduced
forward explicitly zeroes inactive loss lanes before tl.sum, preventing inactive
NaNs from contaminating the reduction. Empty width uses a positive block.
Packed row strides are V, and reduced output stride is1.

The source adds explicit logical capacity checks after forward packing and after
backward work allocation. An undersized target raises ValueError("target has
fewer elements than prediction"); an undersized destination raises
ValueError("new_grads has fewer elements than target"). This is a deliberate
repair of the original logical-role overrun: even when spare backing allowed
the old launch to finish, it read/wrote beyond that tensor's logical elements.
The guard retains the real intermediate state and occurs before launch.

Forward prediction must be rank2 [BT,V], as required by Python unpacking; target
can have any rank/shape with numel≥BT*V. Backward target must be rank2, destination
can have any rank/shape with enough elements. No equal-shape gate is imposed.
Both operations use the first BT*V flattened logical elements. A larger backward
destination receives the derivative in that prefix and the actual untouched
empty_like payload in the remaining logical cells. Its subsequent multiplication
uses ordinary Torch broadcasting with the entry cotangent, including rank0,
vector/row/column/full tensors and higher-rank returns. A scalar numeric-one
cotangent returns the identical original destination object. No backward mean
or batchmean factor is invented; normalization is carried by the caller's entry
cotangent. Empty axes, tails, nonzero offsets, legal nonnegative strides,
read-only aliases, lazy negative views, NaNs/infinities/signed zeros/subnormals,
and arbitrary epsilon bit patterns remain represented.

## Mathematical and executable contracts

The independent loss is t*(log(max(t,eps))-y), or exp(t)*(t-y). Element output,
sum, sum/(BT*V), and sum/BT are defined over logical coordinates without source
loops or temporary storage. Backward is the prediction derivative, -t or
-exp(t), multiplied by the entry cotangent. `Real.loss_increment` proves this
derivative algebraically. Real division by0 is totalized; this is not an
identification with FP32 NaN/infinity outcomes.

`Correct.lean` and `FPEquiv.lean` are independent readers of the same Kernels.
`Forward`/`Backward` prove complete typed row/grid execution. `Coverage`,
`BackwardCoverage`, `FPGraph`, `ForwardValues`, `BackwardValues` connect all
active cells and actual returned tensors to an independent ordered FP graph.
`RealGrid`/`RealHost`/`RealBackward` connect that graph's ideal Real interpretation to the
independent loss and all four reductions. `Broadcast.broadcast_indices` proves
actual projected operand bounds, including empty and arbitrary-rank shapes.

`Storage`, `Library`, `Preparation`, `HostForward`, `HostBackwardFinish`, and
`HostBackward` bind those executions to one physical byte heap, actual returned
objects, normalized entry values, clone freshness, copyback, reductions and
ordinary library outcomes. `SourceBytes` identifies final committed memory with
exact active source-store prefixes. `ByteRealization.fp32_lossless` proves all
four bytes roundtrip, including every NaN payload, so an untouched work tail
cannot be silently canonicalized. `Native`, `NativeBinding`, `NativeFull` and
`NativeReport` derive signed i64 intermediates and active 64-bit byte addresses
from actual successful allocation metadata and the real source guard, without
an extra caller-size or shape assumption.

`AllFrames` covers successful and failed whole calls. Forward preserves all
entry bytes; backward preserves all entry bytes outside the original logical
destination, including allocation padding and gaps. Its PrimitiveFrames fields
are ordinary local library/allocator/dispatch frame contracts. A copy failure
may leave partial destination writes; later failures do not roll them back.
`FailureTrace` distinguishes capacity failure and each post-write failure stage;
`Launch` proves a complete valid typed kernel cannot fail after setup succeeds.
The explicit consistency construction `LibraryWitness` checks only local
primitive contracts and computes a full nonempty forward typed grid, never a desired
kernel result. It is an inhabitance model, not runtime validation in Python.

## Exact remaining local obligations

`LocalNumerics.default_error` and `log_error` express the actual scalar
discrepancy as the local max/log/exp/sub/mul residuals (including the nonlinear
max-to-log term and the exp/sub cross term). `UnaryBound`/`BinaryBound` state
precise primitive-site obligations over their actual operand/result finite
domains; no bound is asserted without numerical validation.

The concrete software profile fixes RN-even separate add/sub/mul, identity
same-FP32 load/cast, constants, sign negation, numeric equality and Torch
pointwise multiplication. The software specialization is not asserted bitwise equal to native
exceptional-value behavior. The generic full graph theorem retains arbitrary
actual primitive words; native local correspondence and error bounds must
respect NaN payload/sign and subnormal behavior rather than assuming a global
canonical-NaN match.
Max, tl.log, tl.exp and each block's reduceSum remain explicit local ports.
Torch sum(dim=0), global sum(), and division by the Python integer scalar remain
separate local ports; the division contract includes scalar conversion and
zero denominator. No whole-output, address, memory, successful-execution or
broadcast-index law is deferred.

For the pinned native profile, tl.log contains internal polynomial FMAs with
FTZ behavior; tl.exp scales by log2(e) and uses approximate ex2. Those internal
intrinsics are part of their local ports. The source subtraction/multiplication
remain separate, and each block reduction plus sequential carry add retains its
actual profile. A finite error bound requires finite actual operands/results at
each evaluated site (including inactive pre-remask lanes), and a positive log
argument wherever a real log error is requested. None of those finite-only
conditions narrows the total execution or byte/frame theorem.

Native runtime admission of the one-dimensional (BT,) grid, zero-grid behavior
and device grid limits remain a dispatch correspondence obligation. Large legal
empty shapes are not excluded by an invented numerical bound. Typed source
frames and all active integer/address bounds are proved independently of that
local runtime admission question.

The two printed empty FP-atom lists in FPEquiv concern only the reused complete
backward bodies' pointer-product commutation. Complete returned-value equalities
have their own full source execution proofs; the lists are not a hardware or
repaired-vs-original-wrapper equivalence claim.

## Evidence

Current source-matched controls: `cpu-controls-1792274201030945041.json` (47 groups,
four original defects), `shape-controls-1792274187688725483.json` (12 groups,
including two additional original spare-backing defects),
`clone-controls-1792275129215665904.json` (preserved-stride clone, aliased and
separate destination), `source-binding-1792274205297326628.json` (strict complete AST repair erasure,
45 originals preserved), and `offline/1792274243215140348/receipt.json`
(16 source-extracted SM80 compile configurations, every loop i64).
CPU controls execute actual original/candidate wrappers and JIT bodies through
a raw-pointer CPU TL adapter and real Torch metadata. They are neither GPU runs
nor native numerical conformance. Private targeted Lean receipts record full
object families (all five companion suffixes accounted for as present or absent), exact sources/imports, failed attempts, preservation
checks and independent no-sorry axiom audits. The final manifest identifies the
successful immutable receipt for each owned module and all evidence hashes.
