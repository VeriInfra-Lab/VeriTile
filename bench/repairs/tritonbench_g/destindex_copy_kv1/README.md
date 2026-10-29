# FP32 destination-row copy: kernel contract and proofs

## Target and admitted inputs

The independent formula is `Out[DestLoc[row], head, dim] = K_initial[row, head, dim]`.
Untouched output coordinates and memory outside the kernel write footprint remain
unchanged. The selected payload precision is FP32: the production source does
not constrain dtype; FP16 fixtures are not a production dtype requirement.
Destination metadata remains signed int32. No floating arithmetic is performed.

The wrapper asserts rank-three strided contiguous FP32 payloads, matching head
and feature sizes, rank-one contiguous int32 destinations, enough source rows,
one CUDA device, no unresolved negative-view flags, extents below `2**31`,
a padded tile of at most `2**20` lanes, and four-byte-aligned nonempty pointers.
Destination rows must be nonnegative, in output bounds, and unique. The output
span must not overlap either read span; the read spans may alias. Offset views,
head/feature tails, empty work, and every FP32 bit pattern remain admitted.
The unchanged JIT runs directly on the original buffers; no normalization or
payload copies are introduced. Destination readback validates the index contract.

`Contract.lean` gives the independent `RowCopy` formula and the exact admission
predicate. `TensorBridge.python_checks_iff` proves the checked predicate equivalent
to it. `Proofs/TensorBinding.checked_read` uses the shared
`VeriTile/Torch/FP32.lean` storage laws to connect checked contiguity, the actual
pointer, storage offsets, and the absent negative bit to logical Tensor values.
`TensorBinding` supplies initial signed metadata words and ordinary capacity;
`SourceConformance` derives active int32 offsets, pointer bounds and separation.
These are input/storage facts, not assumptions about future kernel results.

## Canonical sources and public results

`Kernels.lean` displays the complete selected FP32 original and existing
`indexCommutedKernel` as ordinary `triton` quotations. The candidate changes only
`row * stride` to `stride * row`. Both public proofs reference these exact
sources, definitionally equal to the preserved `Legacy` definitions as complete
typed kernels, including input/output ports. `Layout.source` and the older
generic dtype selections remain available for compatibility.

- `TensorBridge.destindex_copy_kv1_real` is a `Spec.Real` theorem for the complete
  selected kernel grid, independent row-copy formula and exterior frame.
- `FPEquiv.index_commuted_equiv` proves both typed sources successfully execute
  and have equal complete typed memory, with `assumptions = []`.
- `TensorBridge.destindex_copy_kv1_fp32` connects the actual finite pointer copy to
  initial raw FP32 words and the exterior byte frame under the checked contract.

`Proofs/Kernel.lean` and `Proofs/PhysicalCorrect.lean` contain the reused kernel
results. `Misc/Host.lean` and `Misc/NativeHost.lean` retain optional historical
wrapper results; neither public entry imports them. Return values, output object
identity, allocations and host launch orchestration are not acceptance tasks.
Same-format FP32 memory access is a modeling boundary, not a numerical experiment.
The exact integer product rewrite needs no numerical assumption or experiment;
inactive lanes impose no finite-value condition.

## Original-source issue

Int32 backing `[0, 1, 2]` viewed with stride two logically requests `[0, 2]`.
For source rows `[11, 22]` and output `[100, 200, 300]`, the original raw metadata
load reads `[0, 1]`, producing `[11, 22, 300]` instead of `[11, 200, 22]`.
The explicit contiguous-destination contract rejects this unsupported layout.
The original benchmark and its counterexample remain separate from this checked
contract revision. The masked kernel itself is unchanged.

## Evidence and boundaries

`check_contract.py` checks unchanged JIT AST, Python/Lean guard correspondence,
ten admissions, thirty rejections, raw FP32 bits, frames, reversed program order,
and the strided-metadata counterexample on CPU backing storage with mocked CUDA.
The two public Lean entries have targeted builds, standard-base axiom audits,
non-circular formula checks and an empty FP-assumption report.
No GPU execution or independent numerical replay is claimed. Shared grid-product
overflow and segmented launch compatibility remain deferred globally.

Modeling references: [shared physical load/store contracts](../../../../documents/GpuMemoryModel.md#physical-byte-address-contracts)
record Triton masks, PTX byte addressing/alignment and ordinary Torch API laws;
[shared FP32 Tensor storage laws](../../../../VeriTile/Torch/FP32.lean)
provide the actual-pointer and logical-value connection reused here.

`IO.lean` now declares the public typed finite channels and full arbitrary
program schedule. Payload input/output shapes are `[count, heads, dims]`;
signed destination metadata has shape `[count]`. The scalar destination load
is unmasked and remains represented even if a payload dimension is empty.
The output window is the destination-indexed scatter. Its admissible logical
metadata is nonnegative and injective; capacity bounds belong to the shared IO
judgment. There is no additional restriction on FP32 payload words.

`Correct.destindex_copy_kv1_kernel_real` and
`FPEquiv.destindex_copy_kv1_kernel_fp32` (the latter in
[Misc/NativeObservation.lean](Misc/NativeObservation.lean)) are short shared
`ProgramIO.ImplementsOn` judgments with the independent logical row-copy target.
`Proofs/SourceSafety.lean` proves the actual source's metadata and masked payload
access safety. `Proofs/LogicalGrid.lean` connects only the finite active reads
to the existing source execution/readback/frame proofs; it does not require
all addresses in an infinite region to contain a particular cell tag.
`Proofs/PublicIO.lean` composes the actual complete source schedule, safety,
outputs and exact exterior frame. This regional judgment does not itself claim
automatic physical pointer transport.

The earlier raw mathematical and word-grid theorems remain ordinary helpers in
`Proofs/KernelRealResult.lean` and `Proofs/KernelNativeResult.lean`. Complete
Tensor applications remain unchanged in `Misc/TensorBridge.lean`, using the same
accepted source and physical correspondence results. No Python, kernel, input
contract, numerical assumption, original counterexample or GPU status changed
in this 2026-10-20 public-format pass.


可选证明入口统一位于 `Misc/`：`Misc/TensorBridge.lean`, `Misc/NativeObservation.lean`, `Misc/Host.lean`, `Misc/NativeHost.lean`。文件迁移保留声明全名、类型、证明和原有状态；主入口不再为兼容重导出可选 native 结果。所需 IO、公式及连接助手继续保留原位置。

### 逐句函数体来源（2026-10-25）

`Kernels.lean` 直接粘贴所选 Python 函数体，保留无 mask 的 destination
读取、独立 strides、所有原始 mask、浮点 load/store 和末尾 `return`。
调用 dtype/运行整数/constexpr 在函数体外。删除旧 Lean 多出的
`k.to(tl.float32)`，按真实隐式写回规则重新验证完整程序。
`index_commuted.py` 是为已有索引候选新写的 Python 对照，明确标注；
原始 Python 不变。旧精度多态辅助 API 保留为单列的 Legacy 来源。

`Proofs/SourcePreservation.lean` 在原有 raw-copy 后端下核对新旧原始/候选
完整执行（包括成功/失败、寄存器和全部内存），使用已有 FP32 同存储
拷贝解释。一般浮点代数不额外假设 cast 恒等；新程序的完整候选等价
与 Real/word/byte/host/Tensor 结果直接重证。
