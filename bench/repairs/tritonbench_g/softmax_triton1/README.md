# Contiguous FP32 row softmax

The independent mathematical target is
`exp(x[r,c]) / sum(j < cols, exp(x[r,j]))`. The source subtracts the row maximum,
loads padding as negative infinity, then evaluates `tl.exp`, `tl.sum` and ordinary
Triton division in their original order. Loads, arithmetic, reductions and stores
are FP32. This Real formula is a mathematical specification, not an assertion
that native softmax equals exact real arithmetic.

The selected Python wrapper accepts rank-two, strided, contiguous FP32 CUDA
matrices without an unresolved negative-view flag. Nonempty inputs require an
actual data pointer aligned to four bytes, fewer than `2**31` rows, and
`next_power_of_2(cols) <= 2**20`. Empty axes return an output without launching.
There is no total-element cap. Nonzero storage offsets, singleton dimensions
with irrelevant noncanonical strides, NaN, infinities and subnormals remain
admitted. The original Tensor is passed directly; no copy or resolution occurs.
`Contract.lean` and `TensorBridge.python_checks_iff_preconditions` describe the guards.

`Kernels.lean` contains the authoritative readable DSL with explicit FP32
precision. The selected source is `softmax_kernel`; `reciprocal_kernel` retains
the existing candidate from the original benchmark proof: only
`numerator / denominator` becomes `numerator * (1 / denominator)`. There is no
new Python candidate or performance claim. Integer offsets retain the selected
Python repair's int64 widening and explicit column-stride signature; the
canonical wrapper supplies row width and unit column strides.

The public proofs concern kernels:

- `Correct.softmax_kernel_correctness` proves successful mathematical execution,
  active access bounds, every output lane, and the complete exterior flat-memory
  frame; `reciprocal_kernel_correctness` in [Misc/ReciprocalCorrect.lean](Misc/ReciprocalCorrect.lean)
  establishes the same mathematical result for the retained candidate as an
  auxiliary theorem. Both proofs are short applications of
  `Proofs/RealIO.lean`, where source access safety, execution, output and frame
  are assembled from the existing row results.
- `TensorBridge.softmax_checked_kernel_correctness` connects the full row formula to
  the original Tensor's logical coordinates and actual backing storage.
- `FPEquiv.softmax_reciprocal_equiv` proves that both complete FP32 sources
  succeed and have identical typed memory under the explicit `DIV-MUL-RCP`
  scalar rule at the computed operands. Only numerator lanes selected by the
  final store mask (`col < cols`) must be finite; the shared denominator must be
  finite and nonzero. The full padded max/sum prefix is unchanged. Inactive
  final quotient registers may differ, since they cannot reach a store.
  `Proofs/Rewrite.lean` checks this observed division site and proves complete
  memory equality without requiring equal final register states; its `equivalence`
  helper also assembles the two executions for the short public proof. This
  transformation condition is not a Python guard.
- `TensorBridge.softmax_checked_fp32` proves successful original-source execution,
  every logical row output and the full typed-memory frame using the retained
  ordered native graph. It accepts every raw FP32 word and needs no reciprocal
  rule. The graph preserves register folds, simultaneous self-first butterflies,
  shared-memory warp aggregation, subtraction, multiplication by raw FP32
  `0x3fb8aa3b`, `ex2.approx`, and `div.full`.
- `FPEquiv.softmax_kernel_fp32` in [Misc/NativeObservation.lean](Misc/NativeObservation.lean)
  exposes that independent native graph through
  the shared `ProgramIO.Implements` relation. The shared descriptor in
  `softmax_triton2/IO.lean` declares active strided windows, constant-word
  input encoding and each output lane's actual blocked-thread owner. This
  task selects its own exact source in the invocation. The reused source
  safety/graph proof covers all active accesses, successful execution and
  every exterior typed cell. The previous raw evaluator endpoint is retained
  in `Proofs/NativeKernelResult.lean`.

`Proofs/InputBinding.lean` extracts the shared Tensor/storage connection from
historical host proofs. Source regions identify the actual backing allocation;
`InputMemory` uses a single physical-address reader, and `input_read` derives
logical values from `TensorLaws`, checked contiguity and the absent negative flag.
`Proofs/Addresses.lean` retains the actual-pointer, masked signed-index,
allocation-bound and alignment results. See
[`documents/GpuMemoryModel.md`](../../../../documents/GpuMemoryModel.md) for
load/store modeling and the cited Triton/PTX/PyTorch sources.

Primitive max, subtraction, multiply, exp2, addition and division instruction
ports, and compiler selection of the stated native graph, are modeling
boundaries. Their base semantics are not per-task experimental TODOs.
The reciprocal transformation reuses the accepted FP32 `DIV-MUL-RCP` report
at the actual active numerator lanes (finite) and shared denominator (finite
and nonzero), after the unchanged load/max/sub/exp/sum prefix. These local
conditions remain in `divisionSite`; inactive final quotients need not agree.
There is no outstanding experiment for this primary result. Operand correlation
alone does not require repeating an applicable admission. The conclusion stays
conditional on that numerical rule, rather than becoming universal IEEE bit equality.

Historical host and returned-object theorems are retained in optional
`Misc/LegacyCorrect.lean` and `Misc/LegacyFPEquiv.lean`. Public kernel entries
do not import them, `Host`, `Wrapper`, or allocator correctness. Shared row,
Real softmax and native graph proofs are reused rather than re-proved.

Original benchmark files remain unchanged. The original wrapper admits layouts
its unit-column addressing does not implement; transposed output can remain
partly unwritten. Signed-int32 row multiplication also overflows for large valid
indices. These original-source defects are distinct from the selected contracted,
widened repair. Runtime grid-product compatibility is deferred globally.

`check_contract.py` exercises the actual Python wrapper and kernel body with
mocked CUDA launch/allocation: 16 real CPU Tensor admissions, 14 rejections,
empty inputs, storage offsets, singleton strides, all pointer misalignments,
warp branches, large metadata, and original layout/index counterexamples.
It is a targeted CPU/source check, not a GPU numerical experiment. Final Lean
builds and source hashes are recorded in the delivery manifest.

`Misc/TensorBridge.lean` contains the optional mathematical Tensor application and admission correspondence as ordinary theorems. Neither primary file imports it. `FPEquiv.softmax_kernel_fp32` in [Misc/NativeObservation.lean](Misc/NativeObservation.lean) uses the shared typed IO on the original shape/stride, row coverage and output-injectivity domain. `TensorBridge.softmax_checked_fp32` retains the original actual Tensor application; the separate reciprocal-equivalence theorem retains its admitted numerical rule and operand conditions.

### 原生函数体与调用边界

原始 `softmax_kernel` 逐句对照所选修复 Python，保留两个 int64 转换、
全部语句、mask 和 FP32 运算/归约次序。`softmax_triton1` 与
`softmax_triton2` 的原始函数体相同，因此共享前者的可读源码入口；
两题分别对照自己的 Python 检查。五个运行时 int64 参数在函数体外以
Int 值绑定，只有 BLOCK_SIZE 是 constexpr；Tensor 视图基址也在调用边界
绑定，不往源程序插入额外语句。所有旧基址、stride、维度与全量输出/frame
和安全范围保留；整数非溢出及实际物理绑定继续使用已有证明。

`reciprocal_candidate.py` 是既有 Lean reciprocal 变换的 Python 对照，明确
标注为本次编写，不冒充历史来源。原始 Python 未改动。主 FP 结果继续仅在
实际最终 store mask 选中的 numerator 为 finite、共享 denominator 为
finite/nonzero 时使用 DIV-MUL-RCP。`site_values_iff` 从当前完整原始前缀
重新证明这些条件的精确对应；未缩小范围或用空执行域规避证明。
