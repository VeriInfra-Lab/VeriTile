# index_select_cat

`IO.lean` declares `Correct.gatherIO`: the signed-INT64 index channel, selected FP32 input
cells, metadata-dependent source window, strided output window, and both tail
masks. `Correct.kernel_correct` uses the shared `ProgramIO.Implements` judgment:
every selected source value is copied to its output coordinate, the actual
source accesses are backed by the declared windows, execution succeeds, and
all exterior typed cells are preserved. Output-address injectivity stays an
explicit geometry condition. Arbitrary BI/BC, source strides and signed index
values remain in the regional model; checked Tensor applications retain their
existing physical index bounds. The Python and both kernel programs are retained.

`FPEquiv.index_select_index_commuted_equiv` is the sole floating entry. It uses
the standard profiled program equivalence for the retained integer-index
commutation and reports zero numerical assumptions. The complete successful
native execution, gather outputs and memory equality remain in
`Proofs/KernelNativeResults.lean`; they do not add a correctness obligation to
the public equivalence statement.

`Misc/TensorBridge.lean` retains the checked Tensor request results and access/check
helpers. Their existing metadata/index correspondence applies the same
`TypedGather.execution` / `RawKernel.complete` foundation as the public IO.
`Proofs/KernelSafety.lean` proves trace safety of the actual source and its
actual evaluated metadata, load/store masks and signed pointer expressions.
Real denotes the mathematical copy; all FP32 payloads remain admitted and no
numerical experiment was added. Independent replay and GPU execution were not
rerun during this format pass.

`Correct.lean` contains the public IO theorem and a single composition call.
Its window/index/mask data lives in `IO.lean`; the execution and safety
connections remain in `Proofs/LogicalIO.lean` and `Proofs/IOCorrect.lean`. No proof helper
imports either public reader entry.

Optional application proofs and readers are under `Misc/`; their declaration namespaces and theorem statements are unchanged. Required main proof helpers remain under `Proofs/`.

Source readability maintenance (2026-10-22): `Kernels.lean` replaces the
whole-source generator with complete ordinary Triton quotations for `kernel`
and `indexCommutedKernel`. Audit-only complete function equalities by `rfl`
cover arbitrary layouts, including signed-index storage, both masks and all
strides. Targeted source/proof builds and fresh joint main/retained optional
imports are recorded in the source-only cache. Domains, programs and theorem
statements are unchanged.

### 逐句函数体

两份既有 Python 原始/候选保持不变；完整函数体直接粘贴到 Kernels。
保留每个 signed64 cast、INT64 索引读取、两维 mask、原始 stride、
形成的所有源/目标指针和最终写回；调用实参与 dtype/constexpr 在体外。

`Proofs/SourcePreservation.lean` 对完整原始/候选分别证明：任意浮点原语、
参数及初始状态下，新旧执行的成功/失败和全部内存单元一致。
实际新代码直接重验原有 Real、FP、逐次访存安全、输出/frame、word/byte、
host/Tensor 范围；整数索引表示变化只调整证明中的精确 cast 和比较推导。
