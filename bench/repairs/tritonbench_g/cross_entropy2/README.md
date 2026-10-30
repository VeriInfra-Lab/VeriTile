Optional Tensor/native applications, host results and audit readers are in `Misc/`; canonical `Correct.lean` and `FPEquiv.lean` do not re-export these optional results. Declaration names and proof scopes are preserved.

Current FPEquiv result (2026-10-21): the complete forward and backward
original programs satisfy `≡[[]]` with their existing integer column-index
commutation candidates. `Kernels.lean` directly quotes all four full programs
and binds caller types, scalar values and pointer offsets outside the quotation.
`jit-correspondence.json` pins all four complete raw Python function bodies,
including all 15 forward and 17 backward statements, original casts, masks,
conditions, signatures and decorators. The only candidate change is the existing
integer column/block multiplication commutation.

The source's int32 label pointer and explicit int64 widening are preserved.
The non-owning-label branch, ignored-label branch, zero-padded logit load,
post-scale negative-infinity mask and unconditional z-loss write are retained.
In split mode z-loss is still written as zero. Signed class-count binding keeps
its integer-to-floating conversion. No new source statement is added.

Nested arithmetic loads lower to branch-local compiler temporaries, retaining
compute-load precision and explicit casts. The complete lowered bodies have
16/18 top-level statements. The old expansion omitted these load markers and
the forward loss-store conversion; the actual generic execution and independent
ordered floating reference now retain them. No equality with the old floating
graph, arbitrary cast identity, or compiler-schedule invariance across different
graphs is assumed. `Proofs/SourcePreservation.lean` retains the historical
expansion as a diagnostic snapshot and checks its difference from current source.
Independent Real formulas and complete original/candidate FP equivalence are
revalidated against current execution with the original input scope.

The shared parser accepts repeated Boolean conjunctions with the original
comparison precedence and left grouping. A plain conditional assignment such as
`loss = lse if not SPLIT else 0.0` keeps its Python logical-line boundary. The
source quotation itself is unchanged; AST regression checks compare it with
explicitly parenthesized source and verify the following statement's scope.

`PairIO.lean` preserves the selected signed64 labels/ownership protection,
explicit pointer bases, runtime scalar ABI, masks and all output windows.
`Proofs/SourcePair.lean` proves exact complete-state execution equality and
source-safety equivalence using `Int.mul_comm`; `PairEquivalence.lean` composes
the original successful execution, typed output and exterior-frame results.
The pair retains unrestricted FP payloads, all signed labels, gradient/input
aliases and the established single-invocation row/block scope. There is no
floating rewrite, experimental assumption or new numerical validation obligation.
Independent Real formulas and optional native/Tensor results retain their scope.

# cross_entropy2 kernel proofs

## Kernel-only proof boundary

The root `Contract.lean` and `Correct.lean` expose the selected repaired FP32
forward and backward kernels; `FPEquiv.lean` proves the actual complete program pairs.
The optional native theorems are in `Misc/NativeCorrect.lean`. Both stages use the readable typed
`forwardSyntax` and `backwardSyntax` declarations in `Kernels.lean`. Production
Python, guards, arithmetic order, precision and casts are unchanged.

The mathematical entry and optional native theorems use the shared `ProgramIO` judgments. `IO.lean`
shows the actual source, row/block program IDs, four runtime ABI bindings,
logical channels, strides/bases, masks, output locations and empty scratch.
`Correct.forward/backward` implement the independent mathematical formulas;
The ordinary `FPEquiv.forward/backward` theorems in `Misc/NativeCorrect.lean`
use `ProgramIO.Observes` with the original native
instruction/compiler observer. Native inputs remain arbitrary symbolic terms,
so the previous input domain is preserved. The public statements contain no
raw state, run/exec or existential final-state machinery.

`Proofs/KernelSafety.lean` and `KernelTailSafety.lean` prove actual source
access safety through every branch: labels, masked logits, conditional target
gather, ignored-row upstream loads and all stores. `Proofs/PublicIO.lean`
and `PublicBackward.lean` combine those proofs with the existing successful
execution and typed output/frame results. The independent scalar expressions
in `Proofs/OrderedReference.lean` preserve every runtime scalar, precision and operation
order. `Proofs/RealIO.lean` applies the existing mathematical results to the
same final output cells.

The exact old raw entry propositions are ordinary theorems in
`Proofs/RawCorrect.lean` and `Proofs/RawNative.lean`. Optional Tensor helpers
import those foundations directly, keeping both reader entries independent of
`TensorBridge`. Regional backward IO retains its existing permission to alias
gradient and logits; the stronger Tensor application below retains the
wrapper's fresh-output condition. The direct-source readability refactor preserves the complete original program.
Admission, numerical model and experiment status remain unchanged.

`TensorBridge.forward_real` constructs the ideal initial state from actual bound Tensor
words and sign-extended int32 labels. For every valid row/block program it
proves successful execution, all three output values (loss, LSE and z-loss),
and the complete untouched-memory frame. The independent formulas retain
logit scaling, optional smoothing, partition ownership and the LSE-square
term. Split programs store their local loss contribution and literal zero
z-loss. The ignored loss and z-loss are zero while LSE is still computed.

`TensorBridge.backward_real` proves successful execution and the independent gradient
formula at every active lane, together with the full frame. Its supplied LSE
is an independent input; it need not equal a forward result. Upstream strides,
masked tails, non-owning targets and ignored literal-zero gradient stores are
included. Gradient storage is fresh, matching this source's wrapper.

`TensorBridge.forward_native` and `TensorBridge.backward_native` compare the actual source outputs with
independently ordered native instruction graphs. The source retains FP32 casts,
explicit reduction trees, exp/log, primitive operation order and compiler-selected
FMA contraction and multiplication operand order. The shared native profile
uses runtime `.leaf` payloads whose values are erased by compiler `shape`.
It does not equate fused arithmetic with separately rounded multiplication and
addition. Primitive and compiler profiles are ordinary local model boundaries;
there are no whole-output, successful-execution or numerical-transformation
premises and no deferred hardware-conformance experiments.

Smoothing, logit scale and LSE square scale are runtime FP32 words. Runtime
negative zero is retained. The int64 total-class count is converted to FP32 at
its arithmetic uses. Three input-only scalar projection equations link the
mathematical parameters to these same words. The ideal-real interpretation of
input words is an input abstraction, not an IEEE decoder or a new finite-logit
admission guard. `HAS_SMOOTHING` remains independent of the smoothing value.
The post-scaling mask retains negative infinity even for negative or zero
logit scales.

The input contract binds Tensor shapes, strides, storage offsets, physical
pointers, allocation capacities and coherent backing bytes. Labels use their
actual four-byte int32 storage and explicit signed int64 widening. Input typed
views may share coherent physical backing; output allocations are separate.
Singleton contiguous row stride zero remains admitted. Access lemmas prove
all active addresses are backed, the widened native masks are exact, target
selection precedes subtraction safely, and forward byte addresses do not wrap.
One-block scalar outputs have rank one; multi-block scalar outputs are block-major rank two.
Host allocation, aggregation, copying, autograd and whole-grid scheduling are
outside these kernel claims.

The implementation is organised as follows:

- `Proofs/KernelNative.lean` and `KernelBackward.lean`: successful actual typed
  execution, independent ordered graphs, exact stores, readbacks and frames.
- `Proofs/KernelMath.lean`, `KernelReal.lean`, `KernelCorrect.lean`, and
  `KernelTensor.lean`: independent formulas, ideal arithmetic, execution
  composition and actual Tensor input binding.
- `Proofs/KernelContract.lean` and `KernelAccess.lean`: Tensor/backing contracts,
  active access bounds, int32 widening and native index/pointer reasoning.
- `Proofs/KernelProfile.lean`: FMA-aware native Tensor endpoints using the shared
  reduction, instruction and operand-order model.

The Lean declarations use only the standard axiom base. Input contracts,
scalar projection and mathematical/native output specifications are audited
for non-circularity. `Proofs/check_kernel.py` executes only the two source JITs
in Triton's CPU interpreter. Sixteen controls cover negative/zero/positive
scales, split and unsplit outputs, tails, int32 ignored labels with NaN inputs,
non-owning partition targets, enabled and disabled smoothing, supplied LSE,
singleton row stride zero, negative-zero runtime scalars and large int64 class
metadata. The checks verify full output backing frames and reproduce the
original negative-scale tail and missing split z-loss store defects.
`Proofs/source_binding.py` pins the selected sources and checks exact JIT and
host-body AST preservation. These controls do not claim GPU primitive conformance.

The former root files are preserved byte for byte under
`Proofs/HostContract.lean`, `Misc/HostCorrect.lean`, and
`Misc/HostFPEquiv.lean`. Historical imports use these preserved files; the public entries do not import TensorBridge or numerical-admission
modules. No new host proof work is included in this revision.

## Historical checked-wrapper documentation

The public filenames and theorem names below refer to the preserved historical
wrapper revision under `Proofs/Host*`.

# cross_entropy2 checked repair

The original Python is unchanged. The separate repair completes row outputs and
split reductions, writes z-loss on every repaired path, preserves scaling and
smoothing order, fixes padded lanes under zero/negative scaling, and safely
handles targets outside a supplied class partition. The accepted repair is
preserved byte for byte in `cross_entropy2_conditional.py`.

`cross_entropy2_repaired.py` adds explicit direct-input admission to the accepted
complete stages. The two JIT ASTs and host computations are unchanged after
removing the new checks. FP32 source arithmetic, signed int32 labels, signed
int64 class/address intermediates, default FMA fusion, exp/log, reduction order,
and casts remain intact.

## Admission

Both stages accept actual strided, contiguous CUDA Tensors with no lazy negative
bit and aligned nonempty pointers. Logits are FP32 `[R,C]`, labels are int32
`[R]`, and `C>0`. Labels and logits share a device, which must be the current CUDA
launch device. Backward additionally requires direct contiguous FP32 upstream
and supplied LSE vectors `[R]` on that device. Nonzero storage offsets, shared
backing allocations and partial trailing bytes are allowed; inputs are never
mutated. The existing `resolve_neg().contiguous()` calls are no-ops on this
checked domain and do not admit more layouts or dtypes.

Smoothing, logit scale and LSE-square scale must be Python floats and finite.
Smoothing is in `[0,1]`, square scale is nonnegative, and logit scale may be
positive, zero or negative. Class metadata and block size have exact Python int
type, smoothing/split flags have exact bool type, ignored index fits signed
int64, `0 < total_classes < 2^63`, and
`0 <= class_start_idx <= class_start_idx + C <= total_classes`.
Block size is any power of two from 1 through 65,536. For nonempty batches,
`R <= 2^31-1` and `ceil(C/BLOCK_SIZE) <= 65535`. Empty batches skip launch and
retain their empty output behavior.

Each label must be the ignored index or a valid global class index. A valid
label outside the supplied partition is allowed. The comparison widens int32
labels to int64 only in the guard so large class metadata/ignored indices cannot
truncate during validation; the kernels retain int32 input labels.

## Public results

`Contract.lean` states the exact common, forward and backward guards.
`Proofs/Contract.lean` proves successful checked calls iff those guards hold and
the unchanged source call succeeds, preserving its exact returned state. It
also derives finite native launch bounds, including the empty-batch branch.
`Proofs/CheckedInvocation.lean` composes the same checks with invocation-local
compiler and Torch graph selection.

`Correct.checked_forward_correct` returns complete FP32 loss, LSE and z-loss
vectors and proves all caller cells preserved. `checked_backward_correct`
proves the gradient using the caller-supplied LSE, which is an ordinary backward
input. `checked_forward_backward_correct` derives backward's actual LSE from
the preceding forward. No expected saved LSE or expected output is assumed.

For supplied logits `x`, scale `a`, effective smoothing `h` (zero when the
smoothing flag is false), `T=total_classes`, `b=class_start_idx`, and square
scale `z`, the independent Real target uses `u_j=a*x_j` and
`L=log(sum_j exp(u_j))` over **all supplied columns**. Its nonignored loss is
`L - (1-h)*u_target_if_owned - (h/T)*sum_j u_j + z*L^2`.
The standalone gradient is
`(g*a)*((1+2*z*L_supplied)*exp(u_j-L_supplied) - [b+j=target]*(1-h) - h/T)`.
Ignored loss, z-loss and gradient are zero; LSE is still computed.
A proper partition does not implicitly supply remote logits or perform a
distributed collective.

`FPEquiv.checked_complete_profiled_bytes` retains the accepted full source
result with actual invocation-specific device/Torch reduction graphs and
backward FMA/output order. It proves actual loss/LSE/z-loss/gradient bytes,
forward-to-backward source composition, complete caller byte frames, native
launch bounds, and selected backend metadata. Ordinary tensor allocation,
pointer/backing and local compiler/runtime contracts remain explicit
non-numerical environment boundaries. The unchanged conditional public entries
remain available separately.

## Local numerical deferrals

`Floating.Obligations` relates native/reference scalar literal/natural
conversion, negative infinity, add/subtract/multiply/divide/negate/FMA/exp/log/
maximum/isInf, and Torch scalar add/multiply/subtract/exp/log/maximum/isInf.
There is no list, reduction, tensor, state or whole-result numerical port.
Actual reduction graphs and their input order are structural selections.

`Peepholes` lists scalar add/multiply/FMA operand exchange, negative-FMA operand
exchange, multiplication-by-two versus self-add, and subtract-one versus
add-negative-one. `ConstantOne` and `ExpImplementation` separately expose the
one conversion and exp2 lowering. These remain explicit all-bit-pattern local
obligations, including NaNs and signed zero. No installed-GPU numerical
conformance or correctly-rounded transcendental result is claimed.

## Checks

All 72 task Lean modules compile with standard-base axiom audits, including the
three checked-contract modules and public adapters. Task-local CE1 tensor/native
helpers replace obsolete vector-addition dependencies without shared edits.

`python bench/repairs/tritonbench_g/cross_entropy2/check_contract.py` verifies
exact JIT/host AST preservation, 44 rejection cases and 26 actual Triton CPU
interpreter/reference/frame cases. It covers explicit block widths through
65,536, split/unsplit and smoothing flags, zero/negative scales, proper
partitions, int64 ignored metadata, supplied-LSE backward, aliases, offsets,
empty batches and partial-byte backing tails. CPU execution adapts CUDA
metadata/current-device observations; it is not GPU conformance evidence.

`Correct.forward` and `Correct.backward` expose the independent mathematical
formulas through `ProgramIO`, including scalar ABI bindings, logical rows,
integer labels and ordinary memory separation. The ordinary `FPEquiv.forward`
and `FPEquiv.backward` theorems in `Misc/NativeCorrect.lean` expose native
observations and frames through the same IO with the existing compiler/instruction
profiles. `Misc/TensorBridge.lean` retains the prior
forward/backward Tensor applications as ordinary Real/native theorems,
including label bytes, all three forward outputs and every active gradient.
The supporting Tensor correspondences derive the raw helper premises; source
execution is supplied by the kernel results.

## Current source migration evidence

Current source correspondence, proof replay, trust audit and environment limits
are recorded in `bench/review-checkpoints/2026-10-28/native-cross-entropy2-function-bodies/`.
Original Python, public specifications, input domains and independent Real
formulas are unchanged. No numerical law, domain restriction or GPU experiment
is introduced. Repository-wide migration and human review remain pending.
