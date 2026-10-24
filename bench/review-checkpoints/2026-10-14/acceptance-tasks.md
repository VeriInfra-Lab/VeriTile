# 184项：实际证明层级与未完成项

正常输入、配置和阶段范围继续保留。相对等价、后端值契约、软件位解释器和实际硬件对应分别记录；固定例子不扩大参数化证明范围。

[当前收据](acceptance-after-value-repair/receipt.json)、[41项修复范围](../../tritonbench_g/fp_value_repairs.md)、[剩余缺口](../../tritonbench_g/fp_proof_gaps.md)。

| 任务 | 实际证明层级 | 声明内额外数值律 | 声明模型 |
|---|---|---|---|
| `adam_update_triton` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `add_example` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `add_value` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `apply_penalty` | 有依赖已有报告前提的程序相对等价证明 | fp32_div_mul_rcp_real | backend_equivalence, mathematical_contract |
| `attention_forward_triton` | 有依赖已有报告前提的程序相对等价证明 | fp32_div_mul_rcp_real | backend_equivalence, mathematical_contract |
| `attention_fwd_triton1` | 有依赖已有报告前提的程序相对等价证明 | fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `attention_fwd_triton2` | 有依赖已有报告前提的程序相对等价证明 | fp32_div_mul_rcp_real | backend_equivalence, mathematical_contract |
| `attention_fwd_triton3` | 有依赖已有报告前提的程序相对等价证明 | fp32_div_mul_rcp_real | backend_equivalence, mathematical_contract |
| `attention_kernel` | 有依赖已有报告前提的程序相对等价证明 | fp32_div_mul_rcp_real | backend_equivalence, mathematical_contract |
| `attention_kernel_aligned` | 存在数值反例：需修订模型或适用域 | ROUND-FP16-ID | mathematical_contract, rounding_model_contract |
| `attention_llama` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `attention_score` | 有依赖已有报告前提的程序相对等价证明 | fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `attn_fwd_causal` | 有依赖已有报告前提的程序相对等价证明 | fp32_div_mul_rcp_real | backend_equivalence, mathematical_contract |
| `attn_fwd_triton` | 有依赖已有报告前提的程序相对等价证明 | fp32_div_mul_rcp_real | backend_equivalence, mathematical_contract |
| `batched_vecmat_mult` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `bgmv_expand_slice` | 有条件相对等价证明：精确数值绑定待补 | BGMV-NATIVE-ADD-COMMUTE | backend_equivalence, mathematical_contract |
| `bgmv_shrink_kernel` | 有依赖已有报告前提的程序相对等价证明 | fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `block_sparse_attn` | 有依赖已有报告前提的程序相对等价证明 | fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `bmm_chunk_bwd` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `bmm_chunk_fwd` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `bmm_optimized` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `cache_transform_triton` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `chunk_bwd_dqkg` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `chunk_cumsum_kernel` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `chunk_cumsum_vector` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `chunk_delta_fwd` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `chunk_gate_recurrence` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `chunk_gated_attention` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `chunk_gla_fwd` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real, fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `chunk_gla_simple` | 有依赖已有报告前提的程序相对等价证明 | fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `chunk_linear_attn` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real, fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `chunk_retention` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real, fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `chunk_retention_ops` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real, fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `chunked_cumsum_fwd` | 有依赖已有报告前提的程序相对等价证明 | fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `context_attn_bloom` | 有依赖已有报告前提的程序相对等价证明 | fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `context_attn_fwd` | 有条件相对等价证明：精确数值绑定待补 | fp32_mul_commute_real, fp32_mul_commute_typed | backend_equivalence, mathematical_contract |
| `context_attn_llama` | 有条件相对等价证明：精确数值绑定待补 | fp32_mul_commute_real, fp32_mul_commute_typed | backend_equivalence, mathematical_contract |
| `context_attn_mistral` | 有依赖已有报告前提的程序相对等价证明 | fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `context_attn_nopad` | 有依赖已有报告前提的程序相对等价证明 | fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `cosine_compute` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `cross_entropy1` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `cross_entropy2` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `cross_entropy_ops` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `decay_cumsum` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `dequantize_matmul` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `dequantize_rowwise` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `destindex_copy` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `destindex_copy_kv1` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `destindex_copy_kv2` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `diag_ssm_triton` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `dropout_triton` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `embedding_triton_kernel` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `f8_conversion_utils` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `fast_ce_loss` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `fast_layernorm` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `fast_rms_layernorm` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `fast_rope_embedding` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `fifth_order_sph_harmonics` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `flash_attn` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `flash_decode2_llama` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `flash_decode2_phi` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `fp4_to_bf16` | 缺源码 | 无额外律；硬件对应另列 | 缺源 |
| `fp4_to_bf16_conversion` | 缺源码 | 无额外律；硬件对应另列 | 缺源 |
| `fused_activation` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `fused_layernorm_triton` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `fused_recurrent_delta` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `fused_recurrent_hgrn` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `fused_recurrent_retention` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `fused_rotary_embedding` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `fused_rwkv6_kernel` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `geglu_tanh_triton` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `index_select_bwd` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `index_select_cat` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `int4_matmul` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 无额外律；硬件对应另列 | backend_value_contract, mathematical_contract |
| `int8_dequant_matmul` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `int8_matmul_kernel` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `int8_matmul_quantization` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `int8_quantization` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `int_scaled_matmul` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `isfinite_kernel` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, bit_interpreter_contract |
| `iv_dependent_matmul` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `kcache_copy_triton` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 无额外律；硬件对应另列 | backend_value_contract, mathematical_contract |
| `kldiv_compute` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `kldiv_ops` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `kldiv_triton` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract, rounding_model_contract |
| `ksoftmax_triton` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `kv_cache_copy` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 无额外律；硬件对应另列 | backend_value_contract, mathematical_contract |
| `kv_cache_filling` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `l2_norm_bwd` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `l2_norm_triton1` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `l2_norm_triton2` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `layer_norm_fwd` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `layer_norm_liger` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `layer_norm_ops` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `layer_norm_triton` | 缺源码 | 无额外律；硬件对应另列 | 缺源 |
| `layer_norm_welfold` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `layernorm_fwd_triton` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `lightning_attention` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `llama_ff_triton` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `log_softmax` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `logsumexp_fwd` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 无额外律；硬件对应另列 | backend_value_contract, mathematical_contract |
| `lora_expand_gemv` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `masked_add_cuda` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `masked_select` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 无额外律；硬件对应另列 | backend_value_contract, mathematical_contract |
| `matmul_dequant_int4` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `matmul_dequantize` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `matmul_dequantize_int4` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 无额外律；硬件对应另列 | backend_value_contract, mathematical_contract |
| `matmul_kernel` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `matmul_leakyrelu` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `matmul_leakyrelu_fp8` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `matmul_persistent_triton` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `matmul_tma` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `matmul_triton1` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `matmul_triton2` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `matmul_triton_autotune` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `matrix_reduction` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `matrix_transpose` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `matrix_vector_multip` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `max_reduction` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `mean_reduction` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract, rounding_model_contract |
| `mixed_sparse_attention` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 无额外律；硬件对应另列 | backend_value_contract, mathematical_contract |
| `mul_exponent_compensator` | 有依赖已有报告前提的程序相对等价证明 | fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `multinomial_sampling` | 缺源码 | 无额外律；硬件对应另列 | 缺源 |
| `nested_loops_processing` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract, rounding_model_contract |
| `parallel_attention` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `parallel_retention_attention` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 无额外律；硬件对应另列 | backend_value_contract, mathematical_contract |
| `pow_scalar_tensor` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `quant_transpose_kernel` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `quantize_copy_kv` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `quantize_global` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `quantize_kv_copy` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `quantize_kv_transform` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `rbe_triton_transform` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 无额外律；硬件对应另列 | backend_value_contract, mathematical_contract |
| `relu_strided_buffer` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract, rounding_model_contract |
| `relu_triton_kernel` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `reversed_cumsum` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `reversed_cumsum_scalar` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `rms_matmul_rbe` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `rms_norm_triton` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `rms_rbe_matmul` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `rmsnorm_fused` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `rmsnorm_fused_llama` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `rmsnorm_implementation` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `rmsnorm_triton` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract, rounding_model_contract |
| `rope_backward_transform` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `rope_embedding` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `rope_transform` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `rotary_emb` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `rotary_emb_nopad` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `rotary_transform` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `rotary_transform_ops` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `rowwise_quantization_triton` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | backend_value_contract, bit_interpreter_contract, mathematical_contract |
| `seeded_dropout` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `sgmv_expand_slice` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `sin_computation` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `sin_kernel` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `softmax_flaggems` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `softmax_optimize` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `softmax_reducev` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `softmax_triton1` | 有依赖已有报告前提的程序相对等价证明 | fp32_div_mul_rcp_real | backend_equivalence, mathematical_contract |
| `softmax_triton2` | 有依赖已有报告前提的程序相对等价证明 | fp32_div_mul_rcp_real | backend_equivalence, mathematical_contract |
| `softmax_triton3` | 有依赖已有报告前提的程序相对等价证明 | fp32_div_mul_rcp_real | backend_equivalence, mathematical_contract |
| `spinning_lock_reduction` | 缺源码 | 无额外律；硬件对应另列 | 缺源 |
| `square_matrix` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `streamk_matmul` | 缺源码 | 无额外律；硬件对应另列 | 缺源 |
| `swiglu_backward` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `swiglu_fwd` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `swiglu_triton` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `token_attn_llama2` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `token_attn_mistral` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `token_attn_reduceV` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `token_softmax_bloom` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 无额外律；硬件对应另列 | backend_value_contract, mathematical_contract |
| `token_softmax_llama` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 无额外律；硬件对应另列 | backend_value_contract, mathematical_contract |
| `triton_argmax` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 无额外律；硬件对应另列 | bit_interpreter_contract, mathematical_contract |
| `triton_attention` | 存在数值反例：需修订模型或适用域 | ROUND-FP16-ID | mathematical_contract, rounding_model_contract |
| `triton_conv2d_fwd` | 存在数值反例：需修订模型或适用域 | ROUND-FP16-ID | mathematical_contract, rounding_model_contract |
| `triton_linear_activation` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `triton_matmul` | 仅cast/store舍入模型：缺算术浮点证明 | 无额外律；硬件对应另列 | mathematical_contract, rounding_model_contract |
| `triton_mul2` | 有依赖已有报告前提的程序相对等价证明 | fp32_mul_commute_real | backend_equivalence, mathematical_contract |
| `triton_softmax` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `uniform_sampling` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `var_len_copy` | 有后端参数化的程序相对等价证明 | 无额外律；硬件对应另列 | backend_equivalence, mathematical_contract |
| `vector_addition` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
| `vector_addition_custom` | 有依赖已有报告前提的程序相对等价证明 | fp32_add_commute_real | backend_equivalence, mathematical_contract |
