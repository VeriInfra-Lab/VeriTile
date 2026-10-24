# Real / opaque FP 证明路线审计（2026-10-21）

**历史报告，已被 [FP 数值规则重扫](../fp-rule-rescan/README.md) 取代。** 按用户最终确认的“严格证明／实验准入”规则，183 题中 173 题使用严格模型或结构证明，10 题使用已有准入规则，未发现新的主数值规则证据缺口。此前 43 项强制接口迁移已撤回；2 项额外 Real 待办经独立规格实质复核也已撤回。下文及原始 JSON 保留旧审计快照，不代表当前待办。

对当前清单选中的 **183 题**逐题审查，结果为：**43 题需要按新规则调整，140 题未发现该类问题，0 题待判定**。

“需要调整”指当前公开主结果仍强制具体 FP 语义或具体原语一致性，包含仅需泛化主入口的情况；不表示这些定理为假。历史版本和仅作可选诊断的具体模型不计入此数。

其中 **42 题涉及 kernel 主 FP 执行、观察或证明前提**；**1 题 `fast_rms_layernorm` 仅涉及前向主定理中的 epsilon ABI 转换边界**。另有 **3 题 `fp4_to_bf16`、`int8_matmul_kernel`、`int_scaled_matmul` 的 Correct 也混入具体 FP 模型**，这三题已包含在 43 题内，不重复计数。

## 范围与依据

- 规则：[AGENTS.md](../../../../AGENTS.md)。Real 数学运算通过 Mathlib；FP 数值运算保持 opaque，局部数值改写使用项目定义并准入的 assumptions。
- 版本来源：[task_progress.json](../../../tritonbench_g/task_progress.json) 的当前 `kernel_revision` 与 `public_io_format.main_declarations`，不是历史接受版本。
- 16 个并行审查批次覆盖全部 183 个身份；追踪公开定理实际使用的 IO、后端、观察器和辅助定理。
- 根审查验证了所有选定主声明覆盖、公开入口文件 SHA256、证据文件及行号。完整依赖证据和快照哈希见 [audit.json](audit.json)。
- 保留 FP 位值载体、整数/量化类型或位打包本身不算违规；实际执行的浮点转换、数值比较若固定具体模型，则单独计入。
- 同一 opaque 表达式的结构等价无需新增 numerical assumption。具体模型仅出现在辅助旁支也不计入问题。

## 需要调整的 43 题

每行只计一个原始题目身份。证据链接指向该题当前主入口或实际依赖位置；完整依赖链见 JSON。

| 题目 | 需调整的依赖 | 当前证据 |
|---|---|---|
| `attention_fwd_triton1` | 主结果要求 mul/FMA 对齐固定 IEEE 模型。 | [FPEquiv.lean:19](../../../../bench/repairs/tritonbench_g/attention_fwd_triton1_tile_checked/FPEquiv.lean#L19) |
| `batched_vecmat_mult` | 主图解释器固定具体 FP.mul/add/fma。 | [FPEquiv.lean:15](../../../../bench/repairs/tritonbench_g/batched_vecmat_mult/FPEquiv.lean#L15) |
| `chunk_retention` | 主 ScalarLaws 强制具体 IEEE 算术。 | [FPEquiv.lean:12](../../../../bench/repairs/tritonbench_g/chunk_retention/KernelOnly/FPEquiv.lean#L12) |
| `chunk_retention_ops` | 主定理强制 ScalarLaws 对齐具体 IEEE 算术。 | [FPEquiv.lean:14](../../../../bench/repairs/tritonbench_g/chunk_retention_ops/KernelOnly/FPEquiv.lean#L14) |
| `chunked_cumsum_fwd` | 主定理要求具体 IEEE ScalarLaws。 | [FPEquiv.lean:20](../../../../bench/repairs/tritonbench_g/chunked_cumsum_fwd/KernelOnly/FPEquiv.lean#L20) |
| `context_attn_mistral` | 实际分支使用具体 FP 数值比较。 | [FPEquiv.lean:16](../../../../bench/repairs/tritonbench_g/context_attn_mistral/KernelOnly/FPEquiv.lean#L16) |
| `dequantize_matmul` | 主整数→FP32 转换固定 FP.round。 | [FPEquiv.lean:12](../../../../bench/repairs/tritonbench_g/dequantize_matmul/FPEquiv.lean#L12) |
| `f8_conversion_utils` | 主转换固定 IEEE 解码和 FP8 舍入。 | [FPEquiv.lean:25](../../../../bench/repairs/tritonbench_g/f8_conversion_utils/FPEquiv.lean#L25) |
| `fast_rms_layernorm` | 仅前向主定理的 epsilon 标量 ABI 强制具体 FP32 舍入。 | [FPEquiv.lean:10](../../../../bench/repairs/tritonbench_g/fast_rms_layernorm/FPEquiv.lean#L10) |
| `fast_rope_embedding` | 主 IO 与目标固定 softwarePrimitives。 | [FPEquiv.lean:12](../../../../bench/repairs/tritonbench_g/fast_rope_embedding/FPEquiv.lean#L12) |
| `fifth_order_sph_harmonics` | 主后端及目标固定 IEEE 标量算术。 | [FPEquiv.lean:10](../../../../bench/repairs/tritonbench_g/fifth_order_sph_harmonics/FPEquiv.lean#L10) |
| `fp4_to_bf16` | 主 BF16 转换固定具体舍入；Real 结果观察同一具体执行。 | [FPEquiv.lean:13](../../../../bench/repairs/tritonbench_g/fp4_to_bf16/FPEquiv.lean#L13) |
| `fused_recurrent_hgrn` | 主目标固定 Native.software，并要求原语一致性。 | [FPEquiv.lean:14](../../../../bench/repairs/tritonbench_g/fused_recurrent_hgrn/FPEquiv.lean#L14) |
| `fused_recurrent_retention` | forward/backward 要求具体 FP 算术一致性。 | [FPEquiv.lean:62](../../../../bench/repairs/tritonbench_g/fused_recurrent_retention/FPEquiv.lean#L62) |
| `int8_dequant_matmul` | 主 FP 转换、乘法、half 窄化和加法具体化。 | [FPEquiv.lean:9](../../../../bench/repairs/tritonbench_g/int8_dequant_matmul_scalar_cast/FPEquiv.lean#L9) |
| `int8_matmul_kernel` | 主 FP 转换强制具体解码/截断；Correct 同样依赖。 | [FPEquiv.lean:13](../../../../bench/repairs/tritonbench_g/int8_matmul_kernel_modular_checked/FPEquiv.lean#L13) |
| `int8_matmul_quantization` | producer/GEMM 主路径固定 FP.mul/round/half 转换。 | [FPEquiv.lean:12](../../../../bench/repairs/tritonbench_g/int8_matmul_quantization_i32_safe/FPEquiv.lean#L12) |
| `int_scaled_matmul` | 展开 FP.mul 证明交换；Correct 目标含具体舍入后缀。 | [FPEquiv.lean:24](../../../../bench/repairs/tritonbench_g/int_scaled_matmul/FPEquiv.lean#L24) |
| `iv_dependent_matmul` | 主后端及 half 窄化/扩展固定具体模型。 | [FPEquiv.lean:12](../../../../bench/repairs/tritonbench_g/iv_dependent_matmul_dot_shape/FPEquiv.lean#L12) |
| `kv_cache_filling` | 主 FP→整数转换固定具体 PTX 软件模型。 | [FPEquiv.lean:9](../../../../bench/repairs/tritonbench_g/kv_cache_filling/KernelOnly/FPEquiv.lean#L9) |
| `layer_norm_liger` | 主观察器固定具体 FP.add/sub/mul/div/fma。 | [FPEquiv.lean:19](../../../../bench/repairs/tritonbench_g/layer_norm_liger/FPEquiv.lean#L19) |
| `lightning_attention` | backward mask 的 FP32 舍入、减法和比较仍具体化。 | [FPEquiv.lean:21](../../../../bench/repairs/tritonbench_g/lightning_attention/KernelOnly/FPEquiv.lean#L21) |
| `matmul_dequantize_int4` | 主前提要求 half 乘法/FMA/转换符合具体软件模型。 | [FPEquiv.lean:40](../../../../bench/repairs/tritonbench_g/matmul_dequantize_int4/FPEquiv.lean#L40) |
| `matmul_kernel` | 主结果固定软件 FP32 后端和 FP16 转换。 | [FPEquiv.lean:30](../../../../bench/repairs/tritonbench_g/matmul_kernel/FPEquiv.lean#L30) |
| `matmul_leakyrelu_fp8` | 主 LocalPorts 对齐具体乘法、half 转换及比较。 | [FPEquiv.lean:10](../../../../bench/repairs/tritonbench_g/matmul_leakyrelu_fp8_scalar_cast/FPEquiv.lean#L10) |
| `matmul_triton_autotune` | 激活标量运算及 FP16 转换固定具体模型。 | [FPEquiv.lean:12](../../../../bench/repairs/tritonbench_g/matmul_triton_autotune/FPEquiv.lean#L12) |
| `mul_exponent_compensator` | 主执行与目标固定具体 FP.mul。 | [FPEquiv.lean:13](../../../../bench/repairs/tritonbench_g/mul_exponent_compensator/FPEquiv.lean#L13) |
| `multinomial_sampling` | 主采样路径的 Random.rand 固定 FP.round/FP.mul。 | [FPEquiv.lean:14](../../../../bench/repairs/tritonbench_g/multinomial_sampling/FPEquiv.lean#L14) |
| `quant_transpose_kernel` | 主 nativeRound 固定具体 FP→整数转换。 | [FPEquiv.lean:21](../../../../bench/repairs/tritonbench_g/quant_transpose_kernel/FPEquiv.lean#L21) |
| `quantize_copy_kv` | 主 FP→整数转换固定解码/截断/饱和。 | [FPEquiv.lean:12](../../../../bench/repairs/tritonbench_g/quantize_copy_kv/FPEquiv.lean#L12) |
| `quantize_kv_transform` | 主 FP→整数转换固定解码、截断与饱和。 | [FPEquiv.lean:11](../../../../bench/repairs/tritonbench_g/quantize_kv_transform/FPEquiv.lean#L11) |
| `relu_strided_buffer` | 结构改写主入口固定具体 FP 后端及比较；可泛化。 | [FPEquiv.lean:12](../../../../bench/repairs/tritonbench_g/relu_strided_buffer/FPEquiv.lean#L12) |
| `rope_backward_transform` | 主入口固定具体 add/sub/mul/FMA。 | [FPEquiv.lean:10](../../../../bench/repairs/tritonbench_g/rope_backward_transform/FPEquiv.lean#L10) |
| `rope_embedding` | 主 IO 与目标固定 softwarePrimitives。 | [FPEquiv.lean:10](../../../../bench/repairs/tritonbench_g/rope_embedding/FPEquiv.lean#L10) |
| `rope_transform` | 主目标与执行固定 FP.mul/FP.fma。 | [FPEquiv.lean:11](../../../../bench/repairs/tritonbench_g/rope_transform/FPEquiv.lean#L11) |
| `rowwise_quantization_triton` | 主 llrint 转换固定解码和 nearest-even 舍入。 | [FPEquiv.lean:15](../../../../bench/repairs/tritonbench_g/rowwise_quantization_triton/FPEquiv.lean#L15) |
| `seeded_dropout` | 共享 tl.rand 执行固定 FP.round/FP.mul。 | [FPEquiv.lean:21](../../../../bench/repairs/tritonbench_g/seeded_dropout/FPEquiv.lean#L21) |
| `spinning_lock_reduction` | 主后端固定 FP.add，依赖具体求值得到的加零定理。 | [FPEquiv.lean:12](../../../../bench/repairs/tritonbench_g/spinning_lock_reduction/FPEquiv.lean#L12) |
| `streamk_matmul` | 主执行固定软件原子加法及 half 转换。 | [FPEquiv.lean:14](../../../../bench/repairs/tritonbench_g/streamk_matmul_tile_checked/FPEquiv.lean#L14) |
| `triton_argmax` | 主执行固定具体 IEEE 数值比较/位序。 | [FPEquiv.lean:13](../../../../bench/repairs/tritonbench_g/triton_argmax_nan_first/FPEquiv.lean#L13) |
| `triton_conv2d_fwd` | 两个主 FP 结果均要求具体 IEEE FMA 一致性。 | [FPEquiv.lean:10](../../../../bench/repairs/tritonbench_g/triton_conv2d_fwd/FPEquiv.lean#L10) |
| `triton_linear_activation` | 主图解释固定 IEEE 算术及 FP16 转换。 | [FPEquiv.lean:20](../../../../bench/repairs/tritonbench_g/triton_linear_activation/FPEquiv.lean#L20) |
| `uniform_sampling` | 共享 RNG 路径固定 FP.round/FP.mul。 | [FPEquiv.lean:47](../../../../bench/repairs/tritonbench_g/uniform_sampling/FPEquiv.lean#L47) |

## 未发现该类问题的 140 题

这些结论仅限本次规则审查，不等于新增的整体正确性验收。包括已通过 opaque 运算或已准入规则完成证明的 `add_example`、`add_value`、`relu_triton_kernel` 等。

- `adam_update_triton`
- `add_example`
- `add_value`
- `apply_penalty`
- `attention_forward_triton`
- `attention_fwd_triton2`
- `attention_fwd_triton3`
- `attention_kernel`
- `attention_kernel_aligned`
- `attention_llama`
- `attention_score`
- `attn_fwd_causal`
- `attn_fwd_triton`
- `bgmv_expand_slice`
- `bgmv_shrink_kernel`
- `block_sparse_attn`
- `bmm_chunk_bwd`
- `bmm_chunk_fwd`
- `bmm_optimized`
- `cache_transform_triton`
- `chunk_bwd_dqkg`
- `chunk_cumsum_kernel`
- `chunk_cumsum_vector`
- `chunk_delta_fwd`
- `chunk_gate_recurrence`
- `chunk_gated_attention`
- `chunk_gla_fwd`
- `chunk_gla_simple`
- `chunk_linear_attn`
- `context_attn_bloom`
- `context_attn_fwd`
- `context_attn_llama`
- `context_attn_nopad`
- `cosine_compute`
- `cross_entropy1`
- `cross_entropy2`
- `cross_entropy_ops`
- `decay_cumsum`
- `dequantize_rowwise`
- `destindex_copy`
- `destindex_copy_kv1`
- `destindex_copy_kv2`
- `diag_ssm_triton`
- `dropout_triton`
- `embedding_triton_kernel`
- `fast_ce_loss`
- `fast_layernorm`
- `flash_attn`
- `flash_decode2_llama`
- `flash_decode2_phi`
- `fused_activation`
- `fused_layernorm_triton`
- `fused_recurrent_delta`
- `fused_rotary_embedding`
- `fused_rwkv6_kernel`
- `geglu_tanh_triton`
- `index_select_bwd`
- `index_select_cat`
- `int4_matmul`
- `int8_quantization`
- `isfinite_kernel`
- `kcache_copy_triton`
- `kldiv_compute`
- `kldiv_ops`
- `kldiv_triton`
- `ksoftmax_triton`
- `kv_cache_copy`
- `l2_norm_bwd`
- `l2_norm_triton1`
- `l2_norm_triton2`
- `layer_norm_fwd`
- `layer_norm_ops`
- `layer_norm_triton`
- `layer_norm_welfold`
- `layernorm_fwd_triton`
- `llama_ff_triton`
- `log_softmax`
- `logsumexp_fwd`
- `lora_expand_gemv`
- `masked_add_cuda`
- `masked_select`
- `matmul_dequant_int4`
- `matmul_dequantize`
- `matmul_leakyrelu`
- `matmul_persistent_triton`
- `matmul_tma`
- `matmul_triton1`
- `matmul_triton2`
- `matrix_reduction`
- `matrix_transpose`
- `matrix_vector_multip`
- `max_reduction`
- `mean_reduction`
- `mixed_sparse_attention`
- `nested_loops_processing`
- `parallel_attention`
- `parallel_retention_attention`
- `pow_scalar_tensor`
- `quantize_global`
- `quantize_kv_copy`
- `rbe_triton_transform`
- `relu_triton_kernel`
- `reversed_cumsum`
- `reversed_cumsum_scalar`
- `rms_matmul_rbe`
- `rms_norm_triton`
- `rms_rbe_matmul`
- `rmsnorm_fused`
- `rmsnorm_fused_llama`
- `rmsnorm_implementation`
- `rmsnorm_triton`
- `rotary_emb`
- `rotary_emb_nopad`
- `rotary_transform`
- `rotary_transform_ops`
- `sgmv_expand_slice`
- `sin_computation`
- `sin_kernel`
- `softmax_flaggems`
- `softmax_optimize`
- `softmax_reducev`
- `softmax_triton1`
- `softmax_triton2`
- `softmax_triton3`
- `square_matrix`
- `swiglu_backward`
- `swiglu_fwd`
- `swiglu_triton`
- `token_attn_llama2`
- `token_attn_mistral`
- `token_attn_reduceV`
- `token_softmax_bloom`
- `token_softmax_llama`
- `triton_attention`
- `triton_matmul`
- `triton_mul2`
- `triton_softmax`
- `var_len_copy`
- `vector_addition`
- `vector_addition_custom`

## 其他状态保持单独记录

- 未改源码、定理、Python 契约、数值准入表或原有逐题验收状态。
- 没有重新构建、独立 Lean 证明回放或 GPU 实验；这是当前源码与证明依赖审计。
- `softmax_triton1`、`apply_penalty` 的当前主 FP 结果符合 opaque 路线；旧数值延期登记及补充 native 变换问题分别保留，不混入本次 43 题计数。
