# Checked FP32 SwiGLU forward kernel

The original benchmark explicitly uses FP32. `Kernels.lean` lists complete
ordinary quotations for that original profile and the existing integer-index
candidate. Both FP32 input casts, sigmoid, left-associated multiplications,
FP32 output, masks and kernel headers are retained. The historical generic
storage-family alias remains available for compatibility.

The independent mathematical formula is `x*y/(1+exp(-x))`. The two operands
are the original tensor's last-axis halves. `Correct.lean`’s `swiglu_kernel_correct`
exposes `swigluIO : ProgramIO` with both input channels, their strided
windows, the actual source invocation and output mask. Its shared judgment
proves actual source access safety, successful execution, every active
logical output and the full exterior regional memory frame. Both program axes,
independent row strides, all six block configurations and tail masks remain
covered. `TensorBridge.swiglu_tensor_kernel_correct` composes this result with the existing
split/reshape storage binding to express outputs in original logical elements.
X and Y share a physical allocation; no artificial separation is required.
`Proofs/Binding.lean` binds storage cells to one reader keyed by device and
absolute byte address for each entry/prepared snapshot. Different storage
objects can describe the same bytes; they cannot carry unrelated input values.

The wrapper supplies assertions. It checks strided FP32 CUDA payloads without
lazy negative bits, a positive even final dimension, matching devices, signed
int32 launch bounds, output row uniqueness, live pointer alignment and
input/output separation. For a nonempty launch the checked half-open byte spans
are `[x.data_ptr(), x.data_ptr()+4*((M-1)*sx+2*N))` and
`[out.data_ptr(), out.data_ptr()+4*((M-1)*so+N))`. They must not overlap,
regardless of storage identity or storage-base equality. These enclosing spans
include row padding; touching endpoints and disjoint slices of overlapping
storage objects remain allowed. The source's existing contiguous/reshape preparation
is preserved. No extra copy or normalization was added. Empty batch axes remain
supported. `checkPayload_iff`, `checkLaunch_iff` and `checked_indices` connect the
executable guards to their contract, fixed-width indices and actual pointers.
`active_bytes_separated` derives disjointness of all four bytes of each active
output/input element directly from that guard. Both checked public kernel
specializations expose this result. Ordinary Torch view/allocation facts are
API bindings, not output assumptions; distinct storage IDs do not imply
disjoint allocation ranges. The reused `WholeIndex.FiniteAdmission` no longer
carries its unused whole-allocation-disjointness premise. All existing index
proofs are reused without that unsupported restriction.

`FPEquiv.lean` displays the selected complete FP32 source and the existing
integer row-stride commutation candidate as `original ≡[[]] indexCommuted`.
`Proofs/SourceEquivalence.lean` reuses the original prefix proof; the FP32
load/cast, sigmoid, multiplication order and store suffix are identical.

`Misc/NativeCorrect.lean`’s ordinary `swiglu_kernel_native` theorem preserves complete successful source graph execution,
each observed native output and the exterior typed-memory frame for every FP32
word. Its independent reference uses the observed sigmoid lowering:

```
s = div.full(1, add(exp2.approx(mul(sub(+0,x), encoded_log2e)), 1))
out = mul(mul(s,x), y)
```

The reference preserves the native operand order and encoded FP32 constants.
Raw loads become graph leaves, so no contraction can cross an input load.
Instruction realization is the shared primitive/compiler modeling boundary.
There is no primitive-conformance experiment or numerical transformation
assumption. Real mathematics does not assert exact native arithmetic.
`TensorBridge.swiglu_tensor_kernel_native` initializes the graph's input leaves from the
shared physical reader and derives its original logical-input formula using
the storage/view binding, including storage offsets. It has no finite-word or
primitive-conformance premise.

Prior host, autotuning, returned-object and primitive-parameterized results
remain in `Proofs/LegacyCorrect.lean` and `Proofs/LegacyFPEquiv.lean`. They are
optional retained results, not additional proof targets for this kernel.
Their allocation-indexed host memory statements are retained as abstract
results; they are not a claim that separately named overlapping storage objects
have independent physical memory. The primary public endpoints prove kernel access safety,
execution and the regional frame. TensorBridge separately retains physical
input bindings and active-byte separation.

`check_contract.py` compares the actual original/repaired JIT ASTs, preserves
all original host statements, checks guards and runs CPU storage output/frame
cases across the six configurations. It covers supplied strided outputs,
original reshape/copy paths, offsets, tails and empty batches. A regression
replays the exact former overlap guard to reproduce its corruption, rejects
both overlap orders under the current guard before launch, and checks four
touching/disjoint shared-storage cases across all six configurations. CUDA is mocked.
The paired `../swiglu_triton/check_cuda_lowering.py` inspects actual SM90 PTX
register dependencies for all six block configurations without GPU execution.
Neither check is an independent GPU numerical experiment. Runtime grid-product
overflow and segmented launch compatibility remain outside this task.

Pointer/storage interpretation follows the shared Torch
[`data_ptr`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html),
[`storage_offset`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html)
and [`reshape`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html)
contracts. Scalar FP32 alignment follows
[PTX addresses as operands](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands).

Concrete original-source counterexamples are retained in `check_contract.py`: Odd final width 3 gives X width 2 and Y width 1, so the original N=2 mask overreads Y. The original signed32 row product also wraps for (2**30)*2. The checked contract excludes these cases explicitly; the unchecked original is not retroactively certified.

The previous checked repair had a separate alias defect: its overlap check ran
only when storage base pointers were equal. With
`xy=torch.arange(1,17,dtype=torch.float32).reshape(2,8)` and
`out=torch.empty(0,dtype=torch.float32).set_(xy.untyped_storage()[4:],0,(2,4),(4,1))`,
the storage bases differ by 4 aligned bytes but the live spans overlap. The
old repair admitted the input. One legal reverse-row `BLOCK_N=32` replay gave
row 0 `[3.655293,206.081070,400.062988,648.118286]` instead of
`[3.655293,10.569565,20.004055,31.424440]`. The current repair rejects this call.
This is a prior repair contract defect, not a numerical assumption or a change
to the original benchmark. The original Python and kernel arithmetic are
preserved; the correction adds no copies and no experimental TODO.

`Misc/TensorBridge.lean` contains the optional Real/native Tensor applications,
physical input and index bindings, and Python-check equivalences as ordinary
theorems. The primary `swiglu_kernel_correct` and `swiglu_index_commuted_equiv`
specifications are in `Correct.lean` and `FPEquiv.lean`; the native formula
observation remains available through `Misc/NativeCorrect.lean`.

The IO-format pass reuses `Proofs/PublicIO.lean` to assemble existing execution,
readback and frame proofs. `Proofs/SourceSafety.lean` checks both program axes
and every actual typed load/store. The former raw endpoints remain ordinary
theorems in `Proofs/KernelRealResult.lean` and `Proofs/KernelNativeResult.lean`;
Tensor applications reuse those same execution foundations. Both main readers
are independent and contain one public specification. Source precision,
input contracts and native instruction boundaries are unchanged.

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean), [NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.

`Proofs/SourcePreservation.lean` proves equality of both complete selected
source functions to the previous shared typed family by `rfl`, including all
regions, strides, extents and block configurations. This source-only
presentation change preserves the accepted domains, precision, main pair and
all Real/native/Tensor statements; the shared historical source is unchanged.
