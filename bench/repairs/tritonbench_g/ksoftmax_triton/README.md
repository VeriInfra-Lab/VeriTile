# ksoftmax_triton: contiguous FP32 repair

The separate candidate preserves the complete forward and backward arithmetic,
all log/causal/mask branches, and the six original autotune warp choices. It
repairs mask row addressing and causal future gradients, widens integer address
arithmetic, and checks a direct contiguous contract. The original benchmark is
unchanged. There are no snapshots, allocations, transfers or copyback operations.

## Supported API

`softmax(Y, X, M=None, log=False, mask_type=None, causal=False)` writes the
supplied Y and returns None. X/Y have the same rank-three shape `[B,Q,K]`.
`mask_type` is None, `qk`, or `bk`. A used mask has shape `[Q,K]` or `[B,K]`,
respectively. With no mask mode, M is ignored, including its metadata.

`softmax_backward(GradIn, GradOut, Out, log=False, causal=False)` writes the
supplied GradIn and returns None. Its three tensors share one rank-three shape.
The saved output has the interpretation selected by `log` and `causal`.

Every active tensor role must be strided FP32 CUDA, contiguous in the default
format, without a lazy negative view bit, and on one common device. Both flags
must be Python bools. Empty calls return after the common checks and perform no
pointer query or launch. On nonempty calls, `B,Q < 2**31`, `K <= 2**30` and
`B*Q*K < 2**61`; all pointers are 4-byte aligned. The destination's complete
live byte interval must be disjoint from every read tensor's live interval.
Read tensors may alias each other. Contiguous offset views and disjoint views
within one allocation are supported. Canonical strides `Q*K,K` correctly handle
singleton dimensions with arbitrary harmless metadata strides.

All FP32 bit patterns are admitted. Finiteness requirements apply only to the
separate mathematical interpretation and optional local numerical comparisons.
The candidate selects the input's device, retains default FP32 computation and
explicitly enables the original default fusion behavior. Both kernels widen
program IDs, arange lanes, stride arguments and K before address operations.
The shared Triton launcher's large grid product issue remains separately deferred
under `bench/TODO.md`; no native correctness claim covers that unresolved runtime.

## Defects and repair

The original wrapper uses `M.stride(-1)` for a mask row; contiguous masks need
row stride K. Query and batch mask roles also need their respective shape checks.
The causal log backward loads future saved values as zero, then computes
`0-exp(0)*sum(active_grad)`, writing a nonzero gradient to a coordinate absent
from the forward map. The candidate adds the final causal zero selection and
passes K as the mask row stride. `Counterexamples` proves the old -2 future
value, the repaired zero and the differing mask addresses.

Focused CPU checks also reproduce discarded last-stride reads, destination-hole
corruption, negative-view mismatch, and a shifted input/destination race across
rows. These structural bugs are addressed by the source changes and guards.
Their resolution is not deferred as a numerical assumption.

## Proof entry points

`Kernels.lean` holds source definitions only. `Correct.lean` and `FPEquiv.lean`
are independent public roots; auxiliary proofs are in `Proofs/`.

The current revision proves the complete forward and backward kernel source,
without wrapper or allocator implementation obligations. `KernelExecution.row`
calls the actual typed-source interpreter, and `KernelExecution.exec` runs every
row or takes the exact empty identity branch. Successful execution is a proved
conclusion. The historical wrapper endpoints are retained byte-for-byte in
`Proofs/HostCorrect.lean` and `Proofs/HostFPEquiv.lean`.

- `Correct.forward_kernel_real_words` starts with actual FP32 Tensor words,
  decoded by the existing exact rational decoder. It proves the independent
  unshifted softmax/log-softmax partition for finite or negative-infinity active
  input/mask values, provided each nonempty row has at least one finite effective
  score. Negative-infinity masks yield zero probability or negative-infinity log
  output. No restriction is imposed on unread causal/tail values or an ignored M.
- `Correct.backward_kernel_real_words` proves the finite saved-value JVP.
  `Correct.log_backward_kernel_real_words` also covers negative-infinity saved
  log values, using the independent algebraic formula with exp(-infinity)=0.
  This extended formula does not assert differentiation at infinite coordinates.
  Both endpoints include actual source success, every output and the full source
  memory frame, with zero future causal gradients.
- `FPEquiv.forward_kernel_native` and `backward_kernel_native` cover every FP32
  pattern, including NaNs, infinities and signed zero. They connect actual Tensor
  storage bases, offsets and strides to source loads, selected 64-bit ABI pointers
  and scalar arguments, and all four bytes of every output. All initial Tensor
  roles are bound to one physical word map and its corresponding initial bytes;
  read aliases share those words. The exterior byte and source memory frames are
  proved. Compiler metadata contains only local instruction choices, valid
  ordered reductions with each lane exactly once, and source/address metadata.
- `FPEquiv.empty_kernel` requires no compiled object, pointer binding or numeric
  domain. `KernelSchedule.reversed_schedule_equivalent` executes complete source
  rows in reverse order and proves equality of the entire source memory for every
  algebra, including all exceptional native words.

The following historical results supply reusable source/value/frame lemmas:

- Dynamic argument admission is proved equivalent to `ForwardContract` and
  `BackwardContract`, including ignored masks, strict bool classification and
  empty pointer bypass. Guard order affects the error category, not admission.
- `Correct.forward_real` constructs the complete wrapper result and proves the
  independent unshifted softmax/log-softmax formula for arbitrary Real input and
  mask values at active coordinates. Causal future log outputs retain negative
  infinity via `WithBot ℝ`. `Correct.backward_real` proves the independent JVP,
  with zero future causal gradients, and the full source memory frame.
- `FPEquiv.forward_words` and `backward_words` construct the complete native-graph
  wrapper outputs and frame for all bit patterns. Finite reduction instruction
  trees and per-lane FMA operand/sign choices are local compiler metadata, not
  assumed row results. Explicit load nodes prevent contraction across previous
  stores. `forward_bytes` and `backward_bytes` expose every caller-visible output
  byte through the original destination pointer plus the full exterior byte
  frame. Initial backing agreement and a local four-byte store lemma establish
  the concrete observation binding.
- The reused `Source`, `FP`, `RowValue`, `Grid`, `GridValue`, `Congruence`, `Math`
  and `ValueMath` closure retains all loads, branches, reductions, casts and
  stores. Every row's actual source execution is total in the stated model.
  Unique row ownership, arbitrary row schedules and complete output coverage
  are derived; there is no assumed successful grid or output equality.
- `Pointers` proves guarded i32 primitive widening, signed-i64 parameter and
  offset arithmetic, qk/bk mask addressing, active input bounds, 64-bit byte
  pointers, alignment and destination/read-byte separation. Padded pointer
  formation is retained but inactive lanes need no allocation padding.

The ordinary boundary consists of truthful Tensor metadata, valid backing spans,
initial FP32 word/four-byte agreement, the documented next-power-of-two library
contract, selected local compiler lowering, and successful device/runtime
calls without external memory races. Generic Torch/compiler correctness and GPU
hardware execution are not claimed. The old generic memory/snapshot/allocator/
copyback framework is unnecessary and is not imported by this repair.

The mathematical domains are separate from the unchanged Python guards. Native
semantics still covers all-negative-infinity, positive-infinity and NaN rows;
Torch-matching NaN results on such rows are not classified as source defects.

## Validation and numerical status

The current kernel revision has no numerical transformations and no deferred
numerical experiments. Its candidate only reverses the complete row schedule;
the native and schedule endpoints do not depend on `Words.LocalExact`,
`Words.NumericDomain`, or `body_exact_if_local`. Primitive instruction, rounding,
FTZ, exp/log and API semantics are ordinary model-conformance boundaries, not
numerical experiments required by this revision. The eight ports described below
belong only to the historical optional software-reference bridge.

Run `python bench/repairs/tritonbench_g/ksoftmax_triton/check_contract.py`.
The current checker passes 74 admission/branch cases, 38 rejections and six
original counterexamples. It compares exact kernel ASTs after removing only the
explicit integer widening and repaired final causal selection, interprets the
actual Python source over CPU storage, checks all masks/flags and tail sizes,
empty calls, singleton strides, offsets, read aliases, return values and exterior
storage. CUDA metadata/launch are mocked; these checks do not validate GPU
rounding or hardware behavior.

`check_kernel_boundaries.py` adds 58 actual-source cases for partial/all
negative-infinity masks, positive infinity, NaN, unread causal exceptional values,
and backward saved/gradient exceptional words across every branch. Torch
comparisons use equal-NaN classification. The previous 74/38/6 checks and offline
144-profile compilation were retained, rather than repeated for this revision.

All task Lean modules pass targeted compilation and standard-base-only axiom
checks; public independent references pass noncircularity checks. The existing
build toolchain and driver are reused. Original source and historical acceptance
records are unchanged.

This revision refreshes only the task's dependency modules into the workspace
overlay because a historical cached Algebra layout was stale. The refreshed Real
and graph fp32Load fields are checked directly in the final combined import audit;
no shared/base cache or historical source is changed.

Current offline Triton 3.7.1 SM80 compilation passes 144 profiles: K=1,3,33;
all forward mask/log/causal and backward log/causal combinations; warp choices
1,4,32. Other original warp choices remain represented by the generic source and
compiler profile. Artifacts are in `/tmp/veritile-ksoftmax-lowering/`.
Observed code retains the signed-i64 address products, rounded FP32 pointwise
operations, encoded-log2e multiplication followed by `ex2.approx.f32`, full
FP32 division and the complete libdevice log implementation. The log primitive
includes its own FTZ polynomial FMAs. K=1 plain backward and log backward retain
their observed FMA contraction/operand choices. This is offline compilation,
not GPU execution; CUDA is unavailable.

The historical `Words.LocalExact` bridge has eight optional primitive-conformance
ports: add, sub, mul, FMA, full
division, max, approximate exp2 and full libdevice log. Basic arithmetic uses
nearest-even/no-FTZ software FP32 references. Add admits finite pairs and the
negative-infinity padding pair; subtraction admits finite or negative-infinity
left operands with finite right operands; multiplication admits finite pairs and
negative infinity times encoded log2e; FMA requires three finite operands. The
four intrinsic ports cover all bit patterns and retain the exact selected
intrinsic behavior. The log port compares the complete implementation rather
than treating its internal FTZ FMAs as ordinary no-FTZ FMA calls.

`Words.NumericDomain` recursively checks actual intermediate operands at every
rounding node, including reductions and exponent scaling. Finite leaves alone
are insufficient. `body_exact_if_local` composes these ports with the whole
source result. No native numerical conformance or Real error bound is claimed.
The original delivery preserves their former shared-batch deferment as historical
evidence; these ports are not pending numerical experiments for the current
kernel revision. The classification supplement records this distinction without
changing the frozen original packet or any historical proof.
