# 题级 IO 文件统一（2026-10-23）

当前选定的 183 题全部检查：116 题合并，67 题原已符合要求。

同一接口的内存描述和逻辑张量视图集中在题级 `IO.lean`，保留原来的内部 namespace、公开名称和定义体。完整判断等价、执行、安全与数学连接证明仍在 `Proofs/`。真实不同阶段或编码的接口按逐题理由保留；共享 Logical IO 类型仍是共享库。

本轮涉及 326 个现存源码文件、删除 147 个已迁空接口文件。改动是定义搬迁、导入调整以及必要的局部 elaboration 修复；原 kernel、数学公式、输入条件、范围、FP 候选和数值规则保持。GLA 的显式展开与 RMS 的 kernel 全名限定均有失败/修复及命题保持证据。

整题证明统计保持：182 题已完成，1 题带数值延期完成。文件整理量不计为新增证明完成数。唯一既有数值延期仍为 `matmul_leakyrelu`；没有新增数值实验或准入规则。

## 验证与证据

修改题按受影响的最小 IO/连接/主入口链构建，记录逐题联合导入；未改动题明确区分新检查与复用的既有证据。使用新构建 overlay，旧报告和旧对象保持冻结。缓存兼容模块仅存在于构建证据中，不作为当前题级源码或 fresh 源码回放。

当前验证分类：`{'targeted_passed': 112, 'unchanged_reused': 67, 'passed_with_recorded_limit': 4}`。独立复核覆盖 183 题，检查原定义/主入口保持、当前源选择、IO 组织、导入关系以及实际日志和构建收据。它是迁移独审，不是新的全库数学审计。

- [发布输入及不可变证据](publication.json)
- [当前 IO 逐题审计](io-audit.json)
- [保留的数学范围与当前连接](main-formulas.json)
- [主入口职责检查](reader-roles.json)
- [集成检查结果](validation.json)

集成检查全部通过。

未运行新的 GPU/数值实验，也没有独立重建所有历史证明依赖。旧的 ABI/深诊断器限制与已确认的 helper 重编限制逐题保留，不计为新的数学缺口。

## 有记录的构建限制

- [chunk_gla_fwd](tasks/chunk_gla_fwd.json)：Only the changed chains, selected public modules and explicitly missing/stale baseline prerequisites were rebuilt. Unchanged dependency proof objects are reused from read-only prior successful overlays; this is not a rebuild of the complete proof closure.；This organization-only campaign changes no numerical rule, precision, arithmetic order, actual FP pair, schedule, admissible domain, mathematical formula or public premise; no new GPU/native/host/numerical experiment acceptance was introduced.；The selected FPEquiv source hash is unchanged. Its five exact public declarations were freshly checked, and the unchanged cached selected module was used by the fresh public joint. The expensive attached full #print_fp_assumptions/module audit was not rerun.
- [int_scaled_matmul](tasks/int_scaled_matmul.json)：The new overlay reuses unaffected successful dependency objects. This is targeted current-source compilation and public joint checking, not a clean rebuild of the complete dependency closure.；Inherited limit retained: current-baseline FPEquiv/combined FP replay is unavailable because CandidateIO is absent and inherited Algebra.mk ABI differs. Both unchanged actual FP pair declarations reuse existing strict-overlay certificates with 39 source/receipt/object/dependency pins freshly checked. The historical IO source pin is checked against the complete before snapshot, and its entire original definition block is byte-preserved in the merged current IO. The fresh joint covers the two Real declarations only.
- [kv_cache_filling](tasks/kv_cache_filling.json)：Inherited, previously documented deep #print_fp_assumptions kernel_native Safety.safeOp.match_7/Algebra ABI mismatch. The public FPEquiv source is byte-identical to the frozen selection. Its exact declarations, proofs and ordinary axiom audit were compiled in Checks.ExactFPEquiv with only the deep diagnostic command replaced by a comment. Public joint imports that exact checked copy. No theorem or proof was removed.
- [llama_ff_triton](tasks/llama_ff_triton.json)：Untouched dependencies and Contract objects are reused from the recorded successful queue09 baseline; this is a targeted IO/import verification, not a fresh rebuild of the full proof closure.；No new numerical experiments, GPU runs, native/host correctness checks, or global proof-completion claims.；Current SourceSafety proof body is unchanged; fresh compilation fails in both the pre-migration source with the read-only original overlay and the migrated source. The existing successful safety object is reused, with an overlay-only SourceIO import shim; no SourceIO.lean remains in the final task source. Current IO, LogicalReader, Correct, FPEquiv and both public joints were freshly checked.

## 逐题清单

| 题目 | 本轮 IO 组织 | 证据 |
| --- | --- | --- |
| `adam_update_triton` | 合并 | [逐题记录](tasks/adam_update_triton.json) |
| `add_example` | 已符合要求，无需改动 | [逐题记录](tasks/add_example.json) |
| `add_value` | 已符合要求，无需改动 | [逐题记录](tasks/add_value.json) |
| `apply_penalty` | 合并 | [逐题记录](tasks/apply_penalty.json) |
| `attention_forward_triton` | 合并 | [逐题记录](tasks/attention_forward_triton.json) |
| `attention_fwd_triton1` | 合并 | [逐题记录](tasks/attention_fwd_triton1.json) |
| `attention_fwd_triton2` | 合并 | [逐题记录](tasks/attention_fwd_triton2.json) |
| `attention_fwd_triton3` | 合并 | [逐题记录](tasks/attention_fwd_triton3.json) |
| `attention_kernel` | 合并 | [逐题记录](tasks/attention_kernel.json) |
| `attention_kernel_aligned` | 合并 | [逐题记录](tasks/attention_kernel_aligned.json) |
| `attention_llama` | 合并 | [逐题记录](tasks/attention_llama.json) |
| `attention_score` | 合并 | [逐题记录](tasks/attention_score.json) |
| `attn_fwd_causal` | 合并 | [逐题记录](tasks/attn_fwd_causal.json) |
| `attn_fwd_triton` | 合并 | [逐题记录](tasks/attn_fwd_triton.json) |
| `batched_vecmat_mult` | 合并 | [逐题记录](tasks/batched_vecmat_mult.json) |
| `bgmv_expand_slice` | 合并 | [逐题记录](tasks/bgmv_expand_slice.json) |
| `bgmv_shrink_kernel` | 合并 | [逐题记录](tasks/bgmv_shrink_kernel.json) |
| `block_sparse_attn` | 合并 | [逐题记录](tasks/block_sparse_attn.json) |
| `bmm_chunk_bwd` | 合并 | [逐题记录](tasks/bmm_chunk_bwd.json) |
| `bmm_chunk_fwd` | 合并 | [逐题记录](tasks/bmm_chunk_fwd.json) |
| `bmm_optimized` | 合并 | [逐题记录](tasks/bmm_optimized.json) |
| `cache_transform_triton` | 合并 | [逐题记录](tasks/cache_transform_triton.json) |
| `chunk_bwd_dqkg` | 合并 | [逐题记录](tasks/chunk_bwd_dqkg.json) |
| `chunk_cumsum_kernel` | 合并 | [逐题记录](tasks/chunk_cumsum_kernel.json) |
| `chunk_cumsum_vector` | 合并 | [逐题记录](tasks/chunk_cumsum_vector.json) |
| `chunk_delta_fwd` | 合并 | [逐题记录](tasks/chunk_delta_fwd.json) |
| `chunk_gate_recurrence` | 合并 | [逐题记录](tasks/chunk_gate_recurrence.json) |
| `chunk_gated_attention` | 合并 | [逐题记录](tasks/chunk_gated_attention.json) |
| `chunk_gla_fwd` | 合并 | [逐题记录](tasks/chunk_gla_fwd.json) |
| `chunk_gla_simple` | 合并 | [逐题记录](tasks/chunk_gla_simple.json) |
| `chunk_linear_attn` | 合并 | [逐题记录](tasks/chunk_linear_attn.json) |
| `chunk_retention` | 合并 | [逐题记录](tasks/chunk_retention.json) |
| `chunk_retention_ops` | 合并 | [逐题记录](tasks/chunk_retention_ops.json) |
| `chunked_cumsum_fwd` | 合并 | [逐题记录](tasks/chunked_cumsum_fwd.json) |
| `context_attn_bloom` | 合并 | [逐题记录](tasks/context_attn_bloom.json) |
| `context_attn_fwd` | 合并 | [逐题记录](tasks/context_attn_fwd.json) |
| `context_attn_llama` | 合并 | [逐题记录](tasks/context_attn_llama.json) |
| `context_attn_mistral` | 合并 | [逐题记录](tasks/context_attn_mistral.json) |
| `context_attn_nopad` | 合并 | [逐题记录](tasks/context_attn_nopad.json) |
| `cosine_compute` | 已符合要求，无需改动 | [逐题记录](tasks/cosine_compute.json) |
| `cross_entropy1` | 合并 | [逐题记录](tasks/cross_entropy1.json) |
| `cross_entropy2` | 合并 | [逐题记录](tasks/cross_entropy2.json) |
| `cross_entropy_ops` | 合并 | [逐题记录](tasks/cross_entropy_ops.json) |
| `decay_cumsum` | 合并 | [逐题记录](tasks/decay_cumsum.json) |
| `dequantize_matmul` | 合并 | [逐题记录](tasks/dequantize_matmul.json) |
| `dequantize_rowwise` | 合并 | [逐题记录](tasks/dequantize_rowwise.json) |
| `destindex_copy` | 合并 | [逐题记录](tasks/destindex_copy.json) |
| `destindex_copy_kv1` | 合并 | [逐题记录](tasks/destindex_copy_kv1.json) |
| `destindex_copy_kv2` | 合并 | [逐题记录](tasks/destindex_copy_kv2.json) |
| `diag_ssm_triton` | 已符合要求，无需改动 | [逐题记录](tasks/diag_ssm_triton.json) |
| `dropout_triton` | 已符合要求，无需改动 | [逐题记录](tasks/dropout_triton.json) |
| `embedding_triton_kernel` | 已符合要求，无需改动 | [逐题记录](tasks/embedding_triton_kernel.json) |
| `f8_conversion_utils` | 已符合要求，无需改动 | [逐题记录](tasks/f8_conversion_utils.json) |
| `fast_ce_loss` | 已符合要求，无需改动 | [逐题记录](tasks/fast_ce_loss.json) |
| `fast_layernorm` | 已符合要求，无需改动 | [逐题记录](tasks/fast_layernorm.json) |
| `fast_rms_layernorm` | 合并 | [逐题记录](tasks/fast_rms_layernorm.json) |
| `fast_rope_embedding` | 已符合要求，无需改动 | [逐题记录](tasks/fast_rope_embedding.json) |
| `fifth_order_sph_harmonics` | 已符合要求，无需改动 | [逐题记录](tasks/fifth_order_sph_harmonics.json) |
| `flash_attn` | 已符合要求，无需改动 | [逐题记录](tasks/flash_attn.json) |
| `flash_decode2_llama` | 已符合要求，无需改动 | [逐题记录](tasks/flash_decode2_llama.json) |
| `flash_decode2_phi` | 已符合要求，无需改动 | [逐题记录](tasks/flash_decode2_phi.json) |
| `fp4_to_bf16` | 合并 | [逐题记录](tasks/fp4_to_bf16.json) |
| `fused_activation` | 合并 | [逐题记录](tasks/fused_activation.json) |
| `fused_layernorm_triton` | 合并 | [逐题记录](tasks/fused_layernorm_triton.json) |
| `fused_recurrent_delta` | 合并 | [逐题记录](tasks/fused_recurrent_delta.json) |
| `fused_recurrent_hgrn` | 合并 | [逐题记录](tasks/fused_recurrent_hgrn.json) |
| `fused_recurrent_retention` | 合并 | [逐题记录](tasks/fused_recurrent_retention.json) |
| `fused_rotary_embedding` | 合并 | [逐题记录](tasks/fused_rotary_embedding.json) |
| `fused_rwkv6_kernel` | 合并 | [逐题记录](tasks/fused_rwkv6_kernel.json) |
| `geglu_tanh_triton` | 合并 | [逐题记录](tasks/geglu_tanh_triton.json) |
| `index_select_bwd` | 合并 | [逐题记录](tasks/index_select_bwd.json) |
| `index_select_cat` | 合并 | [逐题记录](tasks/index_select_cat.json) |
| `int4_matmul` | 合并 | [逐题记录](tasks/int4_matmul.json) |
| `int8_dequant_matmul` | 合并 | [逐题记录](tasks/int8_dequant_matmul.json) |
| `int8_matmul_kernel` | 已符合要求，无需改动 | [逐题记录](tasks/int8_matmul_kernel.json) |
| `int8_matmul_quantization` | 已符合要求，无需改动 | [逐题记录](tasks/int8_matmul_quantization.json) |
| `int8_quantization` | 已符合要求，无需改动 | [逐题记录](tasks/int8_quantization.json) |
| `int_scaled_matmul` | 合并 | [逐题记录](tasks/int_scaled_matmul.json) |
| `isfinite_kernel` | 已符合要求，无需改动 | [逐题记录](tasks/isfinite_kernel.json) |
| `iv_dependent_matmul` | 已符合要求，无需改动 | [逐题记录](tasks/iv_dependent_matmul.json) |
| `kcache_copy_triton` | 已符合要求，无需改动 | [逐题记录](tasks/kcache_copy_triton.json) |
| `kldiv_compute` | 已符合要求，无需改动 | [逐题记录](tasks/kldiv_compute.json) |
| `kldiv_ops` | 已符合要求，无需改动 | [逐题记录](tasks/kldiv_ops.json) |
| `kldiv_triton` | 已符合要求，无需改动 | [逐题记录](tasks/kldiv_triton.json) |
| `ksoftmax_triton` | 已符合要求，无需改动 | [逐题记录](tasks/ksoftmax_triton.json) |
| `kv_cache_copy` | 合并 | [逐题记录](tasks/kv_cache_copy.json) |
| `kv_cache_filling` | 合并 | [逐题记录](tasks/kv_cache_filling.json) |
| `l2_norm_bwd` | 合并 | [逐题记录](tasks/l2_norm_bwd.json) |
| `l2_norm_triton1` | 合并 | [逐题记录](tasks/l2_norm_triton1.json) |
| `l2_norm_triton2` | 合并 | [逐题记录](tasks/l2_norm_triton2.json) |
| `layer_norm_fwd` | 合并 | [逐题记录](tasks/layer_norm_fwd.json) |
| `layer_norm_liger` | 合并 | [逐题记录](tasks/layer_norm_liger.json) |
| `layer_norm_ops` | 合并 | [逐题记录](tasks/layer_norm_ops.json) |
| `layer_norm_triton` | 合并 | [逐题记录](tasks/layer_norm_triton.json) |
| `layer_norm_welfold` | 合并 | [逐题记录](tasks/layer_norm_welfold.json) |
| `layernorm_fwd_triton` | 合并 | [逐题记录](tasks/layernorm_fwd_triton.json) |
| `lightning_attention` | 合并 | [逐题记录](tasks/lightning_attention.json) |
| `llama_ff_triton` | 合并 | [逐题记录](tasks/llama_ff_triton.json) |
| `log_softmax` | 合并 | [逐题记录](tasks/log_softmax.json) |
| `logsumexp_fwd` | 合并 | [逐题记录](tasks/logsumexp_fwd.json) |
| `lora_expand_gemv` | 合并 | [逐题记录](tasks/lora_expand_gemv.json) |
| `masked_add_cuda` | 合并 | [逐题记录](tasks/masked_add_cuda.json) |
| `masked_select` | 合并 | [逐题记录](tasks/masked_select.json) |
| `matmul_dequant_int4` | 合并 | [逐题记录](tasks/matmul_dequant_int4.json) |
| `matmul_dequantize` | 合并 | [逐题记录](tasks/matmul_dequantize.json) |
| `matmul_dequantize_int4` | 合并 | [逐题记录](tasks/matmul_dequantize_int4.json) |
| `matmul_kernel` | 合并 | [逐题记录](tasks/matmul_kernel.json) |
| `matmul_leakyrelu` | 合并 | [逐题记录](tasks/matmul_leakyrelu.json) |
| `matmul_leakyrelu_fp8` | 合并 | [逐题记录](tasks/matmul_leakyrelu_fp8.json) |
| `matmul_persistent_triton` | 已符合要求，无需改动 | [逐题记录](tasks/matmul_persistent_triton.json) |
| `matmul_tma` | 已符合要求，无需改动 | [逐题记录](tasks/matmul_tma.json) |
| `matmul_triton1` | 已符合要求，无需改动 | [逐题记录](tasks/matmul_triton1.json) |
| `matmul_triton2` | 已符合要求，无需改动 | [逐题记录](tasks/matmul_triton2.json) |
| `matmul_triton_autotune` | 已符合要求，无需改动 | [逐题记录](tasks/matmul_triton_autotune.json) |
| `matrix_reduction` | 已符合要求，无需改动 | [逐题记录](tasks/matrix_reduction.json) |
| `matrix_transpose` | 已符合要求，无需改动 | [逐题记录](tasks/matrix_transpose.json) |
| `matrix_vector_multip` | 已符合要求，无需改动 | [逐题记录](tasks/matrix_vector_multip.json) |
| `max_reduction` | 已符合要求，无需改动 | [逐题记录](tasks/max_reduction.json) |
| `mean_reduction` | 已符合要求，无需改动 | [逐题记录](tasks/mean_reduction.json) |
| `mixed_sparse_attention` | 合并 | [逐题记录](tasks/mixed_sparse_attention.json) |
| `mul_exponent_compensator` | 已符合要求，无需改动 | [逐题记录](tasks/mul_exponent_compensator.json) |
| `multinomial_sampling` | 合并 | [逐题记录](tasks/multinomial_sampling.json) |
| `nested_loops_processing` | 合并 | [逐题记录](tasks/nested_loops_processing.json) |
| `parallel_attention` | 合并 | [逐题记录](tasks/parallel_attention.json) |
| `parallel_retention_attention` | 合并 | [逐题记录](tasks/parallel_retention_attention.json) |
| `pow_scalar_tensor` | 合并 | [逐题记录](tasks/pow_scalar_tensor.json) |
| `quant_transpose_kernel` | 合并 | [逐题记录](tasks/quant_transpose_kernel.json) |
| `quantize_copy_kv` | 合并 | [逐题记录](tasks/quantize_copy_kv.json) |
| `quantize_global` | 合并 | [逐题记录](tasks/quantize_global.json) |
| `quantize_kv_copy` | 合并 | [逐题记录](tasks/quantize_kv_copy.json) |
| `quantize_kv_transform` | 合并 | [逐题记录](tasks/quantize_kv_transform.json) |
| `rbe_triton_transform` | 合并 | [逐题记录](tasks/rbe_triton_transform.json) |
| `relu_strided_buffer` | 合并 | [逐题记录](tasks/relu_strided_buffer.json) |
| `relu_triton_kernel` | 已符合要求，无需改动 | [逐题记录](tasks/relu_triton_kernel.json) |
| `reversed_cumsum` | 合并 | [逐题记录](tasks/reversed_cumsum.json) |
| `reversed_cumsum_scalar` | 已符合要求，无需改动 | [逐题记录](tasks/reversed_cumsum_scalar.json) |
| `rms_matmul_rbe` | 合并 | [逐题记录](tasks/rms_matmul_rbe.json) |
| `rms_norm_triton` | 合并 | [逐题记录](tasks/rms_norm_triton.json) |
| `rms_rbe_matmul` | 合并 | [逐题记录](tasks/rms_rbe_matmul.json) |
| `rmsnorm_fused` | 已符合要求，无需改动 | [逐题记录](tasks/rmsnorm_fused.json) |
| `rmsnorm_fused_llama` | 已符合要求，无需改动 | [逐题记录](tasks/rmsnorm_fused_llama.json) |
| `rmsnorm_implementation` | 已符合要求，无需改动 | [逐题记录](tasks/rmsnorm_implementation.json) |
| `rmsnorm_triton` | 已符合要求，无需改动 | [逐题记录](tasks/rmsnorm_triton.json) |
| `rope_backward_transform` | 已符合要求，无需改动 | [逐题记录](tasks/rope_backward_transform.json) |
| `rope_embedding` | 已符合要求，无需改动 | [逐题记录](tasks/rope_embedding.json) |
| `rope_transform` | 已符合要求，无需改动 | [逐题记录](tasks/rope_transform.json) |
| `rotary_emb` | 已符合要求，无需改动 | [逐题记录](tasks/rotary_emb.json) |
| `rotary_emb_nopad` | 已符合要求，无需改动 | [逐题记录](tasks/rotary_emb_nopad.json) |
| `rotary_transform` | 已符合要求，无需改动 | [逐题记录](tasks/rotary_transform.json) |
| `rotary_transform_ops` | 已符合要求，无需改动 | [逐题记录](tasks/rotary_transform_ops.json) |
| `rowwise_quantization_triton` | 已符合要求，无需改动 | [逐题记录](tasks/rowwise_quantization_triton.json) |
| `seeded_dropout` | 已符合要求，无需改动 | [逐题记录](tasks/seeded_dropout.json) |
| `sgmv_expand_slice` | 已符合要求，无需改动 | [逐题记录](tasks/sgmv_expand_slice.json) |
| `sin_computation` | 已符合要求，无需改动 | [逐题记录](tasks/sin_computation.json) |
| `sin_kernel` | 已符合要求，无需改动 | [逐题记录](tasks/sin_kernel.json) |
| `softmax_flaggems` | 已符合要求，无需改动 | [逐题记录](tasks/softmax_flaggems.json) |
| `softmax_optimize` | 已符合要求，无需改动 | [逐题记录](tasks/softmax_optimize.json) |
| `softmax_reducev` | 合并 | [逐题记录](tasks/softmax_reducev.json) |
| `softmax_triton1` | 合并 | [逐题记录](tasks/softmax_triton1.json) |
| `softmax_triton2` | 合并 | [逐题记录](tasks/softmax_triton2.json) |
| `softmax_triton3` | 合并 | [逐题记录](tasks/softmax_triton3.json) |
| `spinning_lock_reduction` | 合并 | [逐题记录](tasks/spinning_lock_reduction.json) |
| `square_matrix` | 已符合要求，无需改动 | [逐题记录](tasks/square_matrix.json) |
| `streamk_matmul` | 合并 | [逐题记录](tasks/streamk_matmul.json) |
| `swiglu_backward` | 合并 | [逐题记录](tasks/swiglu_backward.json) |
| `swiglu_fwd` | 合并 | [逐题记录](tasks/swiglu_fwd.json) |
| `swiglu_triton` | 合并 | [逐题记录](tasks/swiglu_triton.json) |
| `token_attn_llama2` | 合并 | [逐题记录](tasks/token_attn_llama2.json) |
| `token_attn_mistral` | 合并 | [逐题记录](tasks/token_attn_mistral.json) |
| `token_attn_reduceV` | 合并 | [逐题记录](tasks/token_attn_reduceV.json) |
| `token_softmax_bloom` | 合并 | [逐题记录](tasks/token_softmax_bloom.json) |
| `token_softmax_llama` | 已符合要求，无需改动 | [逐题记录](tasks/token_softmax_llama.json) |
| `triton_argmax` | 合并 | [逐题记录](tasks/triton_argmax.json) |
| `triton_attention` | 已符合要求，无需改动 | [逐题记录](tasks/triton_attention.json) |
| `triton_conv2d_fwd` | 已符合要求，无需改动 | [逐题记录](tasks/triton_conv2d_fwd.json) |
| `triton_linear_activation` | 已符合要求，无需改动 | [逐题记录](tasks/triton_linear_activation.json) |
| `triton_matmul` | 已符合要求，无需改动 | [逐题记录](tasks/triton_matmul.json) |
| `triton_mul2` | 已符合要求，无需改动 | [逐题记录](tasks/triton_mul2.json) |
| `triton_softmax` | 已符合要求，无需改动 | [逐题记录](tasks/triton_softmax.json) |
| `uniform_sampling` | 已符合要求，无需改动 | [逐题记录](tasks/uniform_sampling.json) |
| `var_len_copy` | 已符合要求，无需改动 | [逐题记录](tasks/var_len_copy.json) |
| `vector_addition` | 已符合要求，无需改动 | [逐题记录](tasks/vector_addition.json) |
| `vector_addition_custom` | 已符合要求，无需改动 | [逐题记录](tasks/vector_addition_custom.json) |
