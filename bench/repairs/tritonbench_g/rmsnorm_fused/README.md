# `rmsnorm_fused`: kernel-only selected repair

Preserved raw endpoint readers are also in Misc: `Misc/RawCorrect.lean`, `Misc/RawNative.lean`. Their full declarations are unchanged.

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/PublicChecks.lean`, `Misc/HostCorrect.lean`, `Misc/HostFPEquiv.lean`.

Current source-pair review (2026-10-21): The current `FPEquiv.lean` headline compares the actual repaired runtime-epsilon source with the original specified output-row address-commutation candidate. The candidate is applied to the selected body, retaining its signed int64 PID, signed multiplication, Int-to-Nat pointer conversion, runtime eps register, both passes and undefined inactive W. `Proofs/SourcePair.lean` proves the exact integer pointer equality and source context; no numerical rule/domain is used. The previous full-grid `kernel_native` keeps its exact full name, statement and proof as an ordinary theorem in `Misc/NativeCorrect.lean`. The main pair is uniform in PID, and no new native whole-grid pair, compiler or GPU claim is made.

The selected source is `rmsnorm_fused_repaired.py`. It computes, for every row
`r` and feature `k`, `x[r,k] / sqrt(sum_j x[r,j]^2 / N + eps) * weight[k]`.
Input, weight, accumulation, reciprocal RMS and output use FP32. There is no
extra narrowing before multiplication by the weight. Both source loops execute
once in the admitted domain `0 < N <= B <= 16384`. Padded input lanes are zero;
inactive weight lanes are undefined and never stored.

The Python source and its ordered guards are unchanged by this kernel-only
revision. The admitted tensors are strided FP32 CUDA tensors on the same device,
contiguous with storage offset zero and no lazy negative/conjugate flag. Weight
shape is `[N]`; input rank is positive and final width is positive. Epsilon is a
finite positive Python float. Grid/byte extent and active pointer alignment
checks remain intact. Empty leading dimensions are admitted without a launch
or unused pointer/alignment requirement. Input and weight may overlap or share
storage. Fresh output storage is an ordinary `empty_like` API boundary.

Original-source defects remain separate from the selected repair: a gapped
`2x2` input with row stride 4 can cause writes past its packed output, and
`row=2^17, N=2^14` overflows the original signed int32 row product. The repaired
source promotes PID to int64 and its guards admit only the direct layouts it
actually accesses. These are address/layout defects, not a different formula.

`Kernels.source` is the public readable typed DSL. It retains the int64 cast,
two loops, extra variance mask, input casts and undefined inactive weight load.
The first `eps=0.0` declaration only types an external FP32 register and is
removed before execution; `Runtime.bind` supplies the actual scalar argument.
`check_kernel_source.py` compares every JIT statement to this quotation after
only the declared `stride=N` and block-name substitutions. `Kernels.kernel` and
the older proof family are retained solely for historical compatibility.

The complete Tensor applications are:

- `TensorBridge.kernel_correct`: successful actual DSL execution, mathematical
  RMSNorm on the actual Tensor's logical storage indices, every output value,
  exact complete physical Real-memory update, and the full region/physical frame.
- `TensorBridge.kernel_native_equivalence`: successful source graph execution,
  actual Tensor pointers and bytes, concrete four-byte store execution, every
  output word, complete physical byte equality with an independent native
  recurrence, and preservation of both read-only inputs and all outside bytes.

`Binding.Tensor` contains the shared `VeriTile.Torch.FP32.Tensor` object and
its ordinary API laws. Metadata admission, physical indices, storage capacity
and `dataPtr` therefore refer to the same object. `KernelIndex` proves the
widened signed arithmetic and all active pointer bytes. Inactive masked pointer
expressions do not impose a padded-allocation requirement. `KernelBytes` proves
little-endian codec round trips and individual source/native store commutation;
`KernelPhysical` composes those stores over the complete grid. The public
results do not assume execution success, output values or a whole-memory result.

The native target is `Ordered.reference`, an independent ordered recurrence.
Reduction trees contain each padded lane exactly once; tree roots may depend
on the output lane. Local policy data records zero-accumulator handling,
split/FMA choices, operand order, FP32 cast elimination and division-by-one
elimination. `singleton_fma` explicitly covers the `N=B=1` branch that exposes
`x*x + runtime_epsilon` to FMA. No FP commutativity, associativity, or equivalence
between FMA and split arithmetic is assumed. Runtime epsilon is a separate word
bound by `Instructions.scalarABI`, so equal-valued 0/1/N literals cannot be
mistaken for the runtime argument. Positive finite host epsilon may narrow to
zero or infinity. All input words, including NaNs and infinities, remain admitted.

Primitive IEEE/target instructions (including FTZ, exceptional words and NaN
payload behavior), binary64-to-FP32 scalar ABI conversion, and the compiler's
local lowering policy/tree are explicit model boundaries. They are not numerical
experiment obligations or deferred whole-kernel equivalence assumptions. The
Real theorem is an ideal arithmetic theorem and makes no unconditional claim
that native FP results equal Real arithmetic. Python/Torch/CUDA implementation,
allocator correctness, and compiler conformance to a chosen local profile are
not themselves verified here. No GPU run or independent cold build is claimed.

Historical host endpoints are preserved as ordinary optional theorems in
`Misc/HostCorrect.lean` and `Misc/HostFPEquiv.lean`. Their earlier scope and
validation notes are archived in `README.host-history.md`; they are not public
kernel-only specifications and were not expanded in this revision.

Validation uses only `/tmp/veritile_workspace_lean.py` with task modules in
import order. The final delivery packet records current source hashes, workspace
build receipts, actual imported artifact hashes, logs and the joint axiom audit.
CPU validation: `check_kernel_source.py` passes the complete source comparison
and original address counterexamples; `check_contract.py` passes 32 rejections,
7 admitted cases and 12 exact CPU Triton-interpreter controls. These controls
check source/guards/indexing only and do not justify numerical model primitives.

`Correct.kernel_real` and the ordinary `FPEquiv.kernel_native` theorem in
`Misc/NativeCorrect.lean` use the shared `ProgramIO`
judgment. `IO.lean` declares the complete X/W input spans, Y output span,
empty scratch list, every row invocation, launch IDs and runtime epsilon
register. The formulas receive independent logical arrays. Every permutation
of the full row grid remains covered, including an empty leading dimension.

The public judgment now connects those declarations to actual Structural
source access safety, successful complete execution, all typed output cells
and the exact exterior frame. `Proofs/KernelSafety.lean` checks the two actual
source loops and their evaluated X/W/X/Y accesses. It needs bounds only for
active lanes; it introduces no padding-allocation or numerical premise.
`Proofs/IOInterface.lean` identifies the shared schedule with the existing
source runner. `Proofs/IOReal.lean` and `Proofs/IONative.lean` compose the
existing execution/value proofs with the logical-array and observation maps.

The previous raw-grid statements remain ordinary theorems in
`Misc/RawCorrect.lean` and `Misc/RawNative.lean`. The complete actual-Tensor,
pointer and byte observations remain in `Misc/TensorBridge.lean` with the original
contract and numerical boundaries. Reader integration audits are in
`Misc/PublicChecks.lean`, outside reusable `Proofs/` modules.

The public IO format pass uses the scoped
`.veritile-cache/public-io-format-20261020/workspace_lean.py` overlay and records
its exact targeted build receipts separately. It does not rerun GPU numerical
experiments or claim an independent cold proof replay.
