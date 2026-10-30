# Paged K/V cache copy

For each source row `s`, let `position = lengths[s] - 1`,
`block = tables[s, position // block_size]`, and `slot = position % block_size`.
For every head `h` and dimension `d`, the independent target copies initial
`K[s,h,d]` into K cache and initial `V[s,h,d]` into V cache.
V always uses coordinate `(block,h,slot,d)`; K uses that coordinate in the legacy
layout or `(block,h,d//width,slot,d%width)` in the split layout. Every other
memory cell is unchanged. Each universally quantified `(row,head)` program
executes the complete static split loop, covering every `d < head_dim`.

## Canonical source and input contract

`Kernels.lean` contains the readable typed Triton source consumed by both public
proofs. Payload loads/stores are FP32, metadata storage and original signed
control arithmetic are INT32. The original Python JIT AST is unchanged.
The actual second JIT changes only the scalar K-source address expression from
`row*stride_kt + head*stride_kh` to `head*stride_kh + row*stride_kt`.
Both wrappers share `_checked_copy` and the same assertions; there is no
performance claim and no new payload normalization.

The quotation keeps the original shifted static range
`arange(q*width,(q+1)*width)` and `static_range(HEAD_DIM // KCACHE_X)`.
The shared frontend lowers that exact span to the existing offset-plus-lanes
IR. `Proofs/SourcePreservation.lean` checks the entire original and candidate
bodies against the retained complete source, for every parameter value.
The exact range identity also remains in `Proofs/KernelContract.lean`.
The full static loop remains in the canonical source and the execution proof;
it is not replaced by one representative split or a handwritten source AST.
Triton 3.7.1's `TritonSemantic.arange` checks positive power-of-two range length
and INT32 bounds. The existing width/dimension contract supplies those bounds.

The contract retains six contiguous strided tensors on one CUDA device, no
unresolved negative views, FP32 K/V/cache storage and INT32 length/table storage.
K/V independently admit normalized `[N,H,D]` or `[N,1,H,D]`; original singleton
`squeeze(1)` views remain. Length/table row counts match N. V cache is
`[B,H,S,D]`; K cache is either the same shape or `[B,H,D/X,S,X]`.
Heads, dimension, block size and power-of-two width are positive. The selector
is Boolean. Nonempty actual data pointers are four-byte aligned and all six
data spans are disjoint. Element extents, dimension, block size and table width
are below `2^31`. Historical launch/grid and tile-resource guards remain.
Lengths are in `[1,block_size*table_width]`, selected block IDs are valid, and
selected `(block,slot)` pairs are unique. Empty N is admitted without a launch.
No new range restriction was introduced for this proof conversion.

`Contract.lean` defines the independent `KernelCopy`, the selected contiguous
strides and the initial-memory interpretation. Shared Tensor metadata is
connected to `dataPtr + 4*index`: `logicalState` interprets payloads at those
actual byte addresses and decodes INT32 metadata there. `logicalState_binding`
and `logicalState_payload` establish the initial readback; they assume no output.
`TensorBridge.python_checks_iff` proves that formal guard success is exactly the declared
input contract. Physical capacity is an ordinary backing-memory premise,
separate from the Python assertions.

## Public proofs and preserved results

`Misc/TensorBridge.lean` holds these optional Tensor application results and Python-check
correspondence; neither primary entry imports it.

`IO.lean` declares the actual source, signed length/table metadata inputs,
complete `[dims/width,width]` K/V input and output tiles, metadata-dependent
address windows, masks and one complete static-loop invocation. The two
metadata loads remain unconditional, including when the split count is zero.
`CopyIO.copyValues` is the independent identity formula for both payload tiles;
`CopyIO.OutputUnique` states the existing destination injectivity condition in
logical metadata and geometry.

`Correct.lean` states the shared `ProgramIO.ImplementsOn` judgment and applies
`Proofs/IOCorrect.lean` in one step. The shared judgment connects input bindings,
actual source access safety, successful complete execution, both output tiles
and the exterior frame. `Proofs/KernelSafety.lean` proves safety for the actual
metadata prelude and every static split; `Proofs/KernelValues.lean` reuses the
existing source execution theorem for snapshot readback. Real interprets FP32
payload loads as identity; integer metadata/address operations retain the source
signed quotient, remainder and final natural-address conversion. The old raw
Algorithm result remains in `Proofs/KernelRealResults.lean`.
`TensorBridge.kv_cache_copy_real` retains the original admitted Tensor formula.
`KernelReal.split_count`, `targetInt_eq`, and `target_unique` establish the
correspondence to the same raw execution/readback/frame lemmas used by the main.
`TensorBridge.tensor_physical_correct` retains the actual caller-pointer result.
`TensorBridge.tensor_input_correct` additionally concludes the admitted Tensor logical values:
`Proofs/InputBinding.lean` supplies one generic, input-only `InputReadLaw`, tying
both K/V logical arrays to the same physical payload heap at those exact Tensor
addresses. This is a representation premise, with no output or numerical law.
`FPEquiv.lean` contains the single existing standard source-equivalence
specification: both actual source versions successfully execute the entire
static loop and have identical complete typed memory. Supporting snapshot copy
and frame theorems live in `Proofs/KernelNativeResults.lean` and retain the actual
FP32 load callback. Payloads have no finite-only
restriction; NaNs, infinities, signed zero and subnormals remain admitted.
There are no deferred numerical transformation assumptions.
`TensorBridge.tensor_kernel_results` starts both versions from `logicalFPState`
of the same admitted metadata and shared physical payload heap, using the same
`InputReadLaw` as Real. It retains full successful snapshot/equality/frame results
and proves the logical K/V source values remain present on both sides.

The prior finite/event source model is retained unchanged in `Proofs/Model.lean`.
`Native`, `Source`, `Execution`, `Memory` and `Direct` retain exact finite-width
index/pointer readback, source preservation, grid/split coverage, disjoint writes
and caller-buffer results. `EventCorrect` preserves the previous whole-grid
Real event theorem. `HostCorrect` preserves the optional historical raw-FP32,
caller-object and exterior-byte theorem. Neither public entry imports that host
proof; no new runtime, allocator or wrapper correctness proof was added.

## Original-source counterexamples

The original kernel is valid on the explicit contract above. The original
unguarded wrapper accepts inputs outside it that do not satisfy the copy target:

- A backing K cache `[1,1,2,4]` sliced with `[...,::2]` has logical shape
  `[1,1,2,2]` and final-axis stride 2. For `K[0,0]=[1,2]`, length 1 and block 0,
  the original JIT ignores `stride_kcd`, writes physical offsets 0 and 1, and
  leaves the logical row `[1,-1]` when initialized to -1. It also overwrites a
  backing-padding cell. The contiguous assertion explicitly rejects this input.
- With row 65536 and row stride 65536, the original INT32 product wraps to 0
  instead of `2^32`. Contiguous `K,V.shape=[65537,1,65536]`, unit block size,
  length 1 and distinct valid block IDs realize the source-address counterexample.
  The existing `<2^31` extent assertion rejects this input.

These diagnoses are separate from the admitted-contract proof and the integer
address-commuted candidate. The original benchmark remains unchanged. Shared
Triton grid-product overflow/segmented-launch compatibility remains deferred;
only the previously present grid guard is retained here.

## Validation and modeling references

`python bench/repairs/tritonbench_g/kv_cache_copy/check_contract.py` replays each
actual JIT AST on CPU backing storage: each variant passes 8 admissions and
85 rejections, both layouts, full static loops, original squeeze views, mixed
input ranks, raw FP32 classes and exterior-byte sentinels. It pins the original
AST against the benchmark, verifies the candidate differs only in the selected
add swap, reproduces both counterexamples above and checks shifted range order.
CUDA context/launch are mocked. GPU execution and independent GPU numerical
experiments are not claimed.

Targeted Lean builds check public entries and retained optional results, with
standard-base axiom audits and an empty FP-assumption report. Contiguous Tensor
backing, same-format load/store and compiler/device execution are model boundaries,
not numerical experiment TODOs. References:
[Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html),
[Triton store](https://triton-lang.org/main/python-api/generated/triton.language.store.html),
[Triton arange semantics](https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/language/semantic.py),
[PyTorch data_ptr](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html),
[PyTorch is_contiguous](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html),
and [PTX alignment](https://docs.nvidia.com/cuda/parallel-thread-execution/#addresses-as-operands).

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.

## Source-only cleanup (2026-10-21)

The original and existing offset-commuted candidate are complete ordinary
`quotedOriginal` and `quotedCommuted` quotations. The static split loop,
INT32 metadata, FP32 K/V payloads, all strides and both caches are unchanged.
`Proofs/SourcePreservation.kernel` checks both complete bodies against the
archived prior macro expansions for every Parameters value. Call ports now
explicitly follow the actual source ABI order, independently of first-load order.
Public proofs, domains, precision, numerical dependencies and accepted scope
are retained; this cleanup does not change proof completion status.

### 逐句函数体（2026-10-25）

两份既有原始/候选 Python 均不修改，直接粘贴完整函数体，保留实际
`tl.static_range(HEAD_DIM // KCACHE_X)` 与带非零起点的 `tl.arange`，
INT32 元数据、除法/取余、全部源/目标地址、两次写回和 final return。
18 个运行参数与两个 constexpr 在体外；未使用 stride_kcd 也保留，
并验证完整程序对该实参独立。共享前端处理 constexpr 商及原始 range
拼法，任务源码不改写成手算的 arange 替代代码。
