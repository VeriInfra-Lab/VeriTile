# FP 规则细化后的整理（2026-10-21）

**历史报告，已被 [FP 数值规则重扫](../../fp-rule-rescan/README.md) 取代。** 下文的 43 项强制接口迁移和 2 项额外 Real 待办已撤回，具体依据见新报告；原始 JSON 快照保持不变。当前允许直接复用具体模型下的严格证明，不要求主定理参数化为任意 opaque 运算。

**这次整理的是证据分类、规范和逐题待办；没有修改或宣称完成 Lean 接口迁移。**

183 题中，140 题未发现需要改变主接口；43 题保留接口整理项（42 题抽象主 FP 接口，1 题分离 host ABI）。已有具体模型和严格证明继续保留。

其中 2 题还需分离独立 Real 规格：`int8_matmul_kernel`、`int_scaled_matmul`，已包含在 43 题内。原先将 `fp4_to_bf16` 另列为 Real 规格问题的判断修正：它有独立 E2M1 公式和全表精确证明，应保留这项数学成果。

**本轮新增实验需求为 0。** 当前有 10 题使用已有准入规则；这不是 10 个缺实验的题目。原有 2 项延期记录未被本轮擅自清空，见下文范围说明。

## 统一口径

- Real 数学公式及其性质用 Mathlib。必要的整数、量化格式和精确解码保留。
- kernel 主 FP 证明使用 opaque 运算、显式局部规则和程序结构证明。
- 具体 FP 模型及其严格局部定理保留为数值／模型层证据；仅因绑定具体模型，不判定定理错误，也不要求重新实验。
- 严格证明与实验准入分别记录。同名交换律可以在某具体模型下有证明，同时另一个主定理仍通过已准入 R 使用它；不能据名称自动升级证据。
- 基本原语／模型匹配、有限性、布局、mask 和程序安全不是实验放宽规则。实验接受不会证明全域 IEEE 位相等，也不会自动给出整个 kernel 的误差界。

## 43 项接口整理的分类

| 当前主依赖 | 题数 | 整理方式 |
| --- | ---: | --- |
| 具体模型下的结构证明 | 22 | 保留模型实例和证明，主接口暴露抽象原语。 |
| 具体模型及已证局部事实 | 10 | 保留严格事实，通过显式局部规则连接主证明。 |
| 具体模型一致性前提 | 10 | 将实现／配置绑定与 kernel 主证明分层；不转入实验队列。 |
| host ABI 转换边界 | 1 | 分离 epsilon ABI 绑定，保留已有转换模型和证据。 |

这些分类描述实际主依赖，不表示其它类别没有可复用的辅助定理。

| 题目 | 分类 | Real 侧待办 | 依据 |
| --- | --- | --- | --- |
| `attention_fwd_triton1` | 具体模型一致性前提 | 无新增项 | [FPEquiv.lean:19](../../../../../bench/repairs/tritonbench_g/attention_fwd_triton1_tile_checked/FPEquiv.lean#L19) |
| `batched_vecmat_mult` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:15](../../../../../bench/repairs/tritonbench_g/batched_vecmat_mult/FPEquiv.lean#L15) |
| `chunk_retention` | 具体模型一致性前提 | 无新增项 | [FPEquiv.lean:12](../../../../../bench/repairs/tritonbench_g/chunk_retention/KernelOnly/FPEquiv.lean#L12) |
| `chunk_retention_ops` | 具体模型一致性前提 | 无新增项 | [FPEquiv.lean:14](../../../../../bench/repairs/tritonbench_g/chunk_retention_ops/KernelOnly/FPEquiv.lean#L14) |
| `chunked_cumsum_fwd` | 具体模型一致性前提 | 无新增项 | [FPEquiv.lean:20](../../../../../bench/repairs/tritonbench_g/chunked_cumsum_fwd/KernelOnly/FPEquiv.lean#L20) |
| `context_attn_mistral` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:16](../../../../../bench/repairs/tritonbench_g/context_attn_mistral/KernelOnly/FPEquiv.lean#L16) |
| `dequantize_matmul` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:12](../../../../../bench/repairs/tritonbench_g/dequantize_matmul/FPEquiv.lean#L12) |
| `f8_conversion_utils` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:25](../../../../../bench/repairs/tritonbench_g/f8_conversion_utils/FPEquiv.lean#L25) |
| `fast_rms_layernorm` | host ABI 转换边界 | 无新增项 | [FPEquiv.lean:10](../../../../../bench/repairs/tritonbench_g/fast_rms_layernorm/FPEquiv.lean#L10) |
| `fast_rope_embedding` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:12](../../../../../bench/repairs/tritonbench_g/fast_rope_embedding/FPEquiv.lean#L12) |
| `fifth_order_sph_harmonics` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:10](../../../../../bench/repairs/tritonbench_g/fifth_order_sph_harmonics/FPEquiv.lean#L10) |
| `fp4_to_bf16` | 具体模型及已证局部事实 | 无新增项 | [FPEquiv.lean:13](../../../../../bench/repairs/tritonbench_g/fp4_to_bf16/FPEquiv.lean#L13) |
| `fused_recurrent_hgrn` | 具体模型一致性前提 | 无新增项 | [FPEquiv.lean:14](../../../../../bench/repairs/tritonbench_g/fused_recurrent_hgrn/FPEquiv.lean#L14) |
| `fused_recurrent_retention` | 具体模型一致性前提 | 无新增项 | [FPEquiv.lean:62](../../../../../bench/repairs/tritonbench_g/fused_recurrent_retention/FPEquiv.lean#L62) |
| `int8_dequant_matmul` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:9](../../../../../bench/repairs/tritonbench_g/int8_dequant_matmul_scalar_cast/FPEquiv.lean#L9) |
| `int8_matmul_kernel` | 具体模型一致性前提 | 分离独立数学规格 | [FPEquiv.lean:13](../../../../../bench/repairs/tritonbench_g/int8_matmul_kernel_modular_checked/FPEquiv.lean#L13) |
| `int8_matmul_quantization` | 具体模型及已证局部事实 | 无新增项 | [FPEquiv.lean:12](../../../../../bench/repairs/tritonbench_g/int8_matmul_quantization_i32_safe/FPEquiv.lean#L12) |
| `int_scaled_matmul` | 具体模型及已证局部事实 | 分离独立数学规格 | [FPEquiv.lean:24](../../../../../bench/repairs/tritonbench_g/int_scaled_matmul/FPEquiv.lean#L24) |
| `iv_dependent_matmul` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:12](../../../../../bench/repairs/tritonbench_g/iv_dependent_matmul_dot_shape/FPEquiv.lean#L12) |
| `kv_cache_filling` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:9](../../../../../bench/repairs/tritonbench_g/kv_cache_filling/KernelOnly/FPEquiv.lean#L9) |
| `layer_norm_liger` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:19](../../../../../bench/repairs/tritonbench_g/layer_norm_liger/FPEquiv.lean#L19) |
| `lightning_attention` | 具体模型及已证局部事实 | 无新增项 | [FPEquiv.lean:21](../../../../../bench/repairs/tritonbench_g/lightning_attention/KernelOnly/FPEquiv.lean#L21) |
| `matmul_dequantize_int4` | 具体模型一致性前提 | 无新增项 | [FPEquiv.lean:40](../../../../../bench/repairs/tritonbench_g/matmul_dequantize_int4/FPEquiv.lean#L40) |
| `matmul_kernel` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:30](../../../../../bench/repairs/tritonbench_g/matmul_kernel/FPEquiv.lean#L30) |
| `matmul_leakyrelu_fp8` | 具体模型一致性前提 | 无新增项 | [FPEquiv.lean:10](../../../../../bench/repairs/tritonbench_g/matmul_leakyrelu_fp8_scalar_cast/FPEquiv.lean#L10) |
| `matmul_triton_autotune` | 具体模型及已证局部事实 | 无新增项 | [FPEquiv.lean:12](../../../../../bench/repairs/tritonbench_g/matmul_triton_autotune/FPEquiv.lean#L12) |
| `mul_exponent_compensator` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:13](../../../../../bench/repairs/tritonbench_g/mul_exponent_compensator/FPEquiv.lean#L13) |
| `multinomial_sampling` | 具体模型及已证局部事实 | 无新增项 | [FPEquiv.lean:14](../../../../../bench/repairs/tritonbench_g/multinomial_sampling/FPEquiv.lean#L14) |
| `quant_transpose_kernel` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:21](../../../../../bench/repairs/tritonbench_g/quant_transpose_kernel/FPEquiv.lean#L21) |
| `quantize_copy_kv` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:12](../../../../../bench/repairs/tritonbench_g/quantize_copy_kv/FPEquiv.lean#L12) |
| `quantize_kv_transform` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:11](../../../../../bench/repairs/tritonbench_g/quantize_kv_transform/FPEquiv.lean#L11) |
| `relu_strided_buffer` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:12](../../../../../bench/repairs/tritonbench_g/relu_strided_buffer/FPEquiv.lean#L12) |
| `rope_backward_transform` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:10](../../../../../bench/repairs/tritonbench_g/rope_backward_transform/FPEquiv.lean#L10) |
| `rope_embedding` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:10](../../../../../bench/repairs/tritonbench_g/rope_embedding/FPEquiv.lean#L10) |
| `rope_transform` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:11](../../../../../bench/repairs/tritonbench_g/rope_transform/FPEquiv.lean#L11) |
| `rowwise_quantization_triton` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:15](../../../../../bench/repairs/tritonbench_g/rowwise_quantization_triton/FPEquiv.lean#L15) |
| `seeded_dropout` | 具体模型及已证局部事实 | 无新增项 | [FPEquiv.lean:21](../../../../../bench/repairs/tritonbench_g/seeded_dropout/FPEquiv.lean#L21) |
| `spinning_lock_reduction` | 具体模型及已证局部事实 | 无新增项 | [FPEquiv.lean:12](../../../../../bench/repairs/tritonbench_g/spinning_lock_reduction/FPEquiv.lean#L12) |
| `streamk_matmul` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:14](../../../../../bench/repairs/tritonbench_g/streamk_matmul_tile_checked/FPEquiv.lean#L14) |
| `triton_argmax` | 具体模型及已证局部事实 | 无新增项 | [FPEquiv.lean:13](../../../../../bench/repairs/tritonbench_g/triton_argmax_nan_first/FPEquiv.lean#L13) |
| `triton_conv2d_fwd` | 具体模型一致性前提 | 无新增项 | [FPEquiv.lean:10](../../../../../bench/repairs/tritonbench_g/triton_conv2d_fwd/FPEquiv.lean#L10) |
| `triton_linear_activation` | 具体模型下的结构证明 | 无新增项 | [FPEquiv.lean:20](../../../../../bench/repairs/tritonbench_g/triton_linear_activation/FPEquiv.lean#L20) |
| `uniform_sampling` | 具体模型及已证局部事实 | 无新增项 | [FPEquiv.lean:47](../../../../../bench/repairs/tritonbench_g/uniform_sampling/FPEquiv.lean#L47) |

## 严格证明的复用例子

- `int_scaled_matmul`：保留当前 FP.mul 模型下、任意配置和所有 FP32 位值上的乘法交换定理；不换成一个更窄的有限值实验假设。
- `fp4_to_bf16`：保留全部 16 个 nibble、256 个 packed byte 的转换表证明，以及独立 E2M1 数学值对应；保留负零的原始位值结果。
- `lightning_attention`：保留有限 lane 的精确编码、差值及上下三角 mask 定理。
- `uniform_sampling`、`seeded_dropout`、`multinomial_sampling`：复用共享 RNG 模型的精确二进制分数解码／范围证明；不把确定性随机样本解释当成新实验。
- `spinning_lock_reduction`：保留具体模型下正零加正零的精确事实。

完整局部事实及其“主链使用／辅助可用”说明见 [audit.json](audit.json)。常量表、域连接和反例也是可保留的严格事实，不自动算作数值改写或整题完成。

## 当前使用已有准入规则的 10 题

| 题目 | 规则 | 当前证据用途 |
| --- | --- | --- |
| `add_example` | FP32 ADD-COMMUTE | 已有 R 条件下的实际数值改写；复用原准入。 |
| `add_value` | FP32 ADD-COMMUTE | 已有 R 条件下的实际数值改写；复用原准入。 |
| `apply_penalty` | FP32 DIV-MUL-RCP | 已有 R 条件下的实际数值改写；复用原准入。 |
| `chunk_cumsum_kernel` | ADD-COMMUTE | 已有 R 条件下的实际数值改写；复用原准入。 |
| `chunk_cumsum_vector` | ADD-COMMUTE | 已有 R 条件下的实际数值改写；复用原准入。 |
| `fused_activation` | FP32 ADD-COMMUTE | 已有 R 条件下的实际数值改写；复用原准入。 |
| `softmax_triton1` | FP32 DIV-MUL-RCP | 已有 R 条件下的实际数值改写；复用原准入。 |
| `triton_mul2` | FP32 MUL-COMMUTE | 已有 R 条件下的实际数值改写；复用原准入。 |
| `vector_addition` | FP32 ADD-COMMUTE | 已有 R 条件下的实际数值改写；复用原准入。 |
| `vector_addition_custom` | FP32 ADD-COMMUTE | 已有 R 条件下的实际数值改写；复用原准入。 |

这里按当前主定理实际消费的证据分类。ADD-COMMUTE/MUL-COMMUTE 在某些具体模型下可严格证明，不意味着当前经实验准入的 R 证明已自动改用那个具体定理。

原延期记录另行保留：`apply_penalty` 的补充 native FusionRule 不等于当前主 DIV-MUL-RCP 缺失；`softmax_triton1` 的历史操作数覆盖备注不等于必须因相关操作数重跑实验。本轮没有调整它们的验收状态或创建实验任务。

## 记录与检查

- [唯一逐题清单](../../../../tritonbench_g/task_progress.md) / [JSON](../../../../tritonbench_g/task_progress.json)：每题新增 `kernel_revision.fp_rule_review`，将接口待办、Real 待办和数值规则分开。原验收字段、延期、回放与 GPU 状态全部保留。
- [主定理规范](../../../../MAIN_THEOREM_CONVENTIONS.md) 与根 [AGENTS.md](../../../../../AGENTS.md) 使用相同分层规则。
- 复用初次审计的证据，126 题沿用检查，57 题重点复核；未重新展开全库证明。
- 核对 183 个唯一身份、全部选定主声明、553 个未变公共入口及 907 个未变既有证据文件；新增引用共检查 1364 处，所有路径和行号有效。
- 43 个接口迁移项仍为 pending；本轮没有新 Lean 构建、独立证明回放或 GPU 实验。
- `python bench/task_progress.py --check` 通过；既有台账字段逐项保留，检查了 95 个文档链接和逐题待办的生成数量。结果见 [validation.json](validation.json)。
