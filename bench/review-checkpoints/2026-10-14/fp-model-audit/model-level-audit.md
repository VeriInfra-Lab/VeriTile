# FP 模型层级复核（只读）

629 条公开声明的实际保证层级为：425 条 mathematical_contract、33 条 rounding_model_contract、167 条 backend_equivalence、4 条 bit_interpreter_contract。numeric_model 标签不能替代目标谓词与解释器复核。

31 项 mathematical + abstract_rounding 任务中，8 项已有任意后端的候选/原始执行等价，23 项只有 cast/store 舍入模型。57 项带 admitted_fp 且 proof-used atoms 为空的任务都有后端相对等价；不能把它们归为数学 only，也不能称为 GPU 绝对数值验收。

## 8 项结构等价的 abstract_rounding 特例

- `bench/tritonbench_g/fused_recurrent_retention/FPEquiv.lean::fused_recurrent_retention_forward_equiv`
- `bench/tritonbench_g/fused_recurrent_retention/FPEquiv.lean::fused_recurrent_retention_backward_equiv`
- `bench/tritonbench_g/isfinite_kernel/FPEquiv.lean::isfinite_fp32_equiv`
- `bench/tritonbench_g/isfinite_kernel/FPEquiv.lean::isfinite_fp64_equiv`
- `bench/tritonbench_g/ksoftmax_triton/FPEquiv.lean::ksoftmax_backward_index_equiv`
- `bench/tritonbench_g/ksoftmax_triton/FPEquiv.lean::ksoftmax_forward_index_equiv`
- `bench/tritonbench_g/layer_norm_fwd/FPEquiv.lean::layer_norm_fwd_fp32_equiv`
- `bench/tritonbench_g/reversed_cumsum/FPEquiv.lean::reversed_cumsum_compare_right_equiv`
- `bench/tritonbench_g/reversed_cumsum_scalar/FPEquiv.lean::reversed_cumsum_scalar_equiv`
- `bench/tritonbench_g/triton_softmax/FPEquiv.lean::softmax_index_equiv`
- `bench/tritonbench_g/uniform_sampling/FPEquiv.lean::uniform_sampling_equiv`

另有 `bench/tritonbench_g/bgmv_expand_slice/FPEquiv.lean::bgmv_output_first_equiv`：任意 Algebra 的执行相等，但显式需要 NativeAdditionCommutes 与 finite_operands。

## 429 条 mathematical 中的特殊 wrapper

仅发现 isfinite 的四条公开声明是完整 bit-interpreter 合同；Spec.Real 只是 Prop 包装：

- `bench/tritonbench_g/isfinite_kernel/Correct.lean::index_commuted_fp32_correctness`
- `bench/tritonbench_g/isfinite_kernel/Correct.lean::index_commuted_fp64_correctness`
- `bench/tritonbench_g/isfinite_kernel/Correct.lean::original_fp32_correctness`
- `bench/tritonbench_g/isfinite_kernel/Correct.lean::original_fp64_correctness`

实际目标在 Correct.lean:25，Contract32/64 在 32/35；BitBackend.lean:14/24 安装位保持与指数分类；BitClassification.lean:29/30 给出独立有限性公式。它们证明成功执行、输出及 frame，但 native libdevice/compiler 对应仍未绑定。

llama_ff_triton、rms_matmul_rbe、rms_rbe_matmul 的 ByteIO 虽含精确字节解码，外层执行明确为 eraseDType.toAlgKernel。uniform_sampling 保留 Philox/fp32 decoder，但外层算术与输出转换擦除精度；这些仍是 mathematical_contract。f8_conversion_utils 的数学合同是 real copy，不是 FP8 量化正确性。

## 舍入模型缺口

RoundingModel 的 round 输入输出为实数，要求 real identity 与 dtype-wise idempotence。evalOpR 擦除 precision wrapper，只在 cast/store 应用舍入；中间 add/sub/mul/div 保留数学算术。因此 33 条这类声明、其中 23 个仅有这种 FP 层级的任务仍欠实际逐运算 FP 语义/误差保证，不能因没有额外数值公理而完成验收。

## 验收原则

- Classify the instantiated proposition and interpreter reached by its named contracts. Spec.Real is Prop identity; numeric_model tags and theorem filenames do not establish semantics.
- backend_equivalence means candidate execution equals original execution under the same algebra/model and displayed domains. Empty proof-used atoms remove extra arithmetic-law debt, not native refinement debt.
- An equality of Option execution results can hold when both sides fail. Separate successful-execution witnesses and their exact profiles must not be generalized beyond their theorem domains.
- RoundingModel is a real-valued cast/store model, with round_real and round_idem. It does not implement per-operation IEEE rounding, NaN/infinity/signed zero or concrete native dot/FMA/reduction lowering.
- Exact bit or integer primitives inside a real arithmetic contract do not upgrade the whole result to a bit interpreter. Only the four isfinite contracts have complete source execution against a typed bit-classification backend here.
- Task aggregation keeps all model kinds, scope and numerical-law requirements as independent axes. A task with a backend relative-equivalence theorem is stronger than math-only, but still has no absolute GPU error bound or compiler refinement by that fact.
- Unreviewed/changed source must be unknown or require explicit review, never silently no-extra/accepted. Preserve per-declaration hashes and resolve aliases/ProgramSyntax before any automatic promotion.

## 覆盖边界

Read-only source audit; all 629 source entries classified using archived proof-used-atom audit (155) and statement/reference inventory (474); all 429 mathematical declarations/178 containing files marker-scanned; suspicious imported contract wrappers and 8 abstract structural task families examined. No new Lean elaboration, no GPU/native experiment.

所有 629 条的完整键、真实 statement、kind、review_basis、证据路径/行号/hash 均在 JSON 的 rows；49 条非 admitted 标签特殊声明在 explicit_overrides。429 条数学声明文件扫描与隐藏 wrapper 复核分别保留在 mathematical_source_scan/imported_wrapper_checks。此审计没有重跑 Lake/GPU，也不把源码负向扫描冒充内核验证。
