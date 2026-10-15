# 浮点证明层级、缺口与数值实验

正常输入条件、明确的配置与阶段范围已获接受；把浮点算术换成实数算术没有获接受。先检查证明实际使用的模型，再列数值实验。

**撤回把129项解释成“无需额外数值实验支持”的口径。** 这个旧计数混入只有数学/抽象舍入结果的任务；空数值假设列表不能补齐浮点证明。

| 任务内已有的证明种类（不代表全任务硬件验收） | 任务数 |
|---|---:|
| 仅cast/store舍入模型：缺算术浮点证明 | 23 |
| 存在数值反例：需修订模型或适用域 | 3 |
| 有后端参数化的程序相对等价证明 | 65 |
| 有依赖已有报告前提的程序相对等价证明 | 43 |
| 有条件相对等价证明：精确数值绑定待补 | 3 |
| 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 30 |
| 有逐算子后端值契约：原语及硬件对应仍待完成 | 11 |
| 缺源码 | 6 |

“有相对等价证明”表示任务内有明确列出的candidate-vs-original声明；不表示该任务全部值契约已获得硬件数值正确性证明。相同运算可以在两侧产生相同误差。位解释器结论和真实GPU指令的对应也单独保留。

本轮逐任务、逐声明的修复范围见 [值语义修复表](fp_value_repairs.md)，剩余模型缺口见 [浮点证明缺口清单](fp_proof_gaps.md)。固定输入的位值例子单列，不提升参数化结论。已有五个待补绑定只覆盖条件等价证明的局部规则。

## 从属统计：声明内额外数值律（不表示完成）

| 数值证据状态 | 任务数 | 公开声明数 |
|---|---:|---:|
| 未使用额外数值律；不代表无需实验 | 129 | 680 |
| 使用已有实验报告中的数值律 | 43 | 61 |
| 需补数值实验或精确绑定证据 | 3 | 3 |
| 含不可直接推广到真实浮点的理想化假设 | 3 | 3 |
| 缺源码 | 6 | 0 |

共 184 项、747 条公开记录。这里仅按声明中最强的额外数值前提计数；即使没有前提，也可能根本没有浮点证明。各规则的受影响任务集合可以重叠。

共享舍入模型要求另涉及 31 项、33 条声明，单列而不冒充新的算法数值律。

## 交给实验的具体配置

以下只列已建立条件等价证明的规则配置；完整字段见 [机器可读实验清单](numeric_experiment_tasks.json)。数学/舍入模型任务需先补具体浮点契约与运算图，不能凭任务名生成交换律或用自反等价填数。

| 实验项 | 状态 | 关系 / dtype / 精度 |
|---|---|---|
| `fp32_add_commute_real` | `existing_report_optional_replay` | Q32(a+b) = Q32(b+a)；{'lean_carrier_channel': 'real', 'lean_compute_precision': 'some .fp32', 'lean_output_channel': 'real', 'fragment_default_precision': 'none (each arithmetic operation explicitly carries .fp32)'} |
| `fp32_mul_commute_real` | `existing_report_optional_replay` | Q32(a*b) = Q32(b*a)；{'lean_carrier_channel': 'real', 'lean_compute_precision': 'some .fp32', 'lean_output_channel': 'real', 'fragment_default_precision': 'none (each arithmetic operation explicitly carries .fp32)'} |
| `fp32_div_mul_rcp_real` | `existing_report_optional_replay` | Q32(a/b) = Q32(a*Q32(1/b))；{'lean_carrier_channel': 'real', 'lean_compute_precision': 'some .fp32', 'lean_output_channel': 'real', 'fragment_default_precision': 'none (each arithmetic operation explicitly carries .fp32)'} |
| `fp32_mul_commute_typed` | `pending_exact_binding` | Q32(a*b) = Q32(b*a)；{'lean_carrier_channel': 'fp32', 'lean_compute_precision': 'some .fp32', 'lean_output_channel': 'fp32', 'fragment_default_precision': 'some .fp32'} |
| `ROUND-FP16-ID` | `not_an_experiment` | R.round .fp16 = id; equivalently ∀ x : ℝ, R.round .fp16 x = x；{'dtypes': ['fp16'], 'precision': 'all real inputs, not merely representable fp16 grid points'} |
| `ROUNDING-MODEL-STRUCTURE` | `shared_model_check` | R.round_real : R.round .real = id; R.round_idem : ∀ dtype x, R.round dtype (R.round dtype x) = R.round dtype x；{'dtypes': ['real', 'fp32', 'fp16', 'bf16', 'f8e4', 'f8e5'], 'precision': 'per FloatDType cast/store function'} |
| `BGMV-ADD-fp64-real` | `pending_exact_binding` | ∀ a b, D .finite a → D .finite b → E.binary (some .fp64) .real .add a b = E.binary (some .fp64) .real .add b a；{'precision': 'fp64', 'dtype': 'real', 'source_configurations': 12} |
| `BGMV-ADD-fp32-real` | `existing_report_optional_replay` | ∀ a b, D .finite a → D .finite b → E.binary (some .fp32) .real .add a b = E.binary (some .fp32) .real .add b a；{'precision': 'fp32', 'dtype': 'real', 'source_configurations': 20} |
| `BGMV-ADD-fp32-fp16` | `pending_exact_binding` | ∀ a b, D .finite a → D .finite b → E.binary (some .fp32) .fp16 .add a b = E.binary (some .fp32) .fp16 .add b a；{'precision': 'fp32', 'dtype': 'fp16', 'source_configurations': 28} |
| `BGMV-ADD-fp32-bf16` | `pending_exact_binding` | ∀ a b, D .finite a → D .finite b → E.binary (some .fp32) .bf16 .add a b = E.binary (some .fp32) .bf16 .add b a；{'precision': 'fp32', 'dtype': 'bf16', 'source_configurations': 4} |
| `BGMV-ADD-fp32-fp32` | `pending_exact_binding` | ∀ a b, D .finite a → D .finite b → E.binary (some .fp32) .fp32 .add a b = E.binary (some .fp32) .fp32 .add b a；{'precision': 'fp32', 'dtype': 'fp32', 'source_configurations': 8} |

## 新值契约所需的原语与硬件对应

这些与代数重写律是两类实验。软件位模型和逐算子后端契约已明确比较对象，但没有在本轮运行 GPU。

[逐任务绑定清单](fp_value_native_tasks.json)保留具体合同、配置和待检查关系；[实验运行说明](../../experiments/floating_point/value_validation/README.md)提供 836 个标量位比较案例。归约树、dot、转换和超越函数仍需各自的原核配置。

| 任务 | 原语配置 | 硬件对应状态 |
|---|---|---|
| `kcache_copy_triton` | `raw_copy` | `not_established` |
| `kv_cache_copy` | `raw_copy` | `not_established` |
| `masked_select` | `raw_copy` | `not_established` |
| `kv_cache_filling` | `fp32_uint8_conversion` | `not_established` |
| `fused_recurrent_hgrn` | `unfused_fp32` | `not_established` |
| `fused_recurrent_delta` | `reduction_fp32` | `not_established` |
| `rope_transform` | `unfused_fp32` | `not_established` |
| `rope_backward_transform` | `unfused_fp32` | `not_established` |
| `triton_argmax` | `ordered_max` | `not_established` |
| `matrix_reduction` | `ordered_max` | `not_established` |
| `rotary_transform` | `unfused_fp32` | `not_established` |
| `rotary_transform_ops` | `unfused_fp32` | `not_established` |
| `fused_rotary_embedding` | `unfused_fp32_and_raw_copy` | `not_established` |
| `token_softmax_bloom` | `softmax_fp32` | `not_established` |
| `token_softmax_llama` | `softmax_fp32` | `not_established` |
| `quant_transpose_kernel` | `unfused_fp32` | `not_established` |
| `fused_rwkv6_kernel` | `reduction_fp32` | `not_established` |
| `rotary_emb_nopad` | `unfused_fp32` | `not_established` |
| `rbe_triton_transform` | `mixed_transcendental` | `not_established` |
| `rope_embedding` | `unfused_fp32` | `not_established` |
| `matrix_vector_multip` | `reduction_fp32` | `not_established` |
| `quantize_global` | `unfused_fp32` | `not_established` |
| `int8_quantization` | `unfused_fp32` | `not_established` |
| `logsumexp_fwd` | `mixed_transcendental` | `not_established` |
| `layer_norm_welfold` | `reduction_fp32` | `not_established` |
| `fused_layernorm_triton` | `reduction_fp32` | `not_established` |
| `quantize_kv_copy` | `fp32_int8_conversion` | `not_established` |
| `matmul_dequant_int4` | `packed_int4_binary16` | `not_established` |
| `matmul_dequantize` | `packed_int4_binary16_typed_dot` | `not_established` |
| `rowwise_quantization_triton` | `fp32_binary16_max` | `not_established` |
| `quantize_kv_transform` | `fp32_int8_conversion` | `not_established` |
| `fifth_order_sph_harmonics` | `unfused_fp32` | `not_established` |
| `int4_matmul` | `packed_int4_binary16_typed_dot` | `not_established` |
| `matmul_dequantize_int4` | `packed_int4_binary16_typed_dot` | `not_established` |
| `parallel_attention` | `fp32_typed_dot` | `not_established` |
| `parallel_retention_attention` | `mixed_transcendental` | `not_established` |
| `int8_matmul_kernel` | `signed_int8_dot_int32_wrap` | `not_established` |
| `int8_dequant_matmul` | `int8_dot_fp32_scale_fp16_store` | `not_established` |
| `int8_matmul_quantization` | `int8_quantizer_and_scaled_dot` | `not_established` |
| `int_scaled_matmul` | `int8_dot_fp32_scale_int32_store` | `not_established` |
| `mixed_sparse_attention` | `sparse_attention_fp16_fp32` | `not_established` |

## 待核对的数值假设

### fp32_add_commute_real

**fp32_add_commute_real**

- 类型：`reported_atom`；证据状态：`reported_evidence_available`。
- 精确关系：Q32(a+b) = Q32(b+a)
- 适用域：finite a and finite b
- dtype / 精度 / 配置：{'lean_carrier_channel': 'real', 'lean_compute_precision': 'some .fp32', 'lean_output_channel': 'real', 'fragment_default_precision': 'none (each arithmetic operation explicitly carries .fp32)'}
- 影响：22 项、34 条公开声明。

实验或修复要求：

No absent aggregate row. If applying to a new target compiler/device or claiming native whole-kernel behavior, replay the exact probe/configuration and validate the real-carrier/fp32 interpretation and actual kernel site lowering; retain the model premise explicitly.

现有证据：

- [VeriTile/Triton/Float/ScalarArithmetic.lean](../../VeriTile/Triton/Float/ScalarArithmetic.lean)
- [VeriTile/Triton/Float/ScalarArithmetic.lean](../../VeriTile/Triton/Float/ScalarArithmetic.lean)
- [VeriTile/Triton/Float/ScalarArithmetic.lean](../../VeriTile/Triton/Float/ScalarArithmetic.lean)
- [VeriTile/Triton/Float/ProfiledRewrite.lean](../../VeriTile/Triton/Float/ProfiledRewrite.lean)
- [VeriTile/Triton/Float/ProfiledRewrite.lean](../../VeriTile/Triton/Float/ProfiledRewrite.lean)
- [VeriTile/Spec.lean](../../VeriTile/Spec.lean)
- [Published scoped aggregate report; no new GPU replay](../../experiments/floating_point/report/summary.json)
- [Proof-used atom audit, exact keys and report validation](../../bench/review-checkpoints/2026-10-14/numerical-assumption-audit/fp/audit.json)

受影响声明：

- [adam_update_triton::update_commuted_equiv](../../bench/tritonbench_g/adam_update_triton/FPEquiv.lean)
- [add_example::add_commuted_equiv](../../bench/tritonbench_g/add_example/FPEquiv.lean)
- [add_value::add_value_commuted_equiv](../../bench/tritonbench_g/add_value/FPEquiv.lean)
- [attention_llama::attention_carry_first_equiv](../../bench/tritonbench_g/attention_llama/FPEquiv.lean)
- [attention_llama::attention_fp8_carry_first_equiv](../../bench/tritonbench_g/attention_llama/FPEquiv.lean)
- [batched_vecmat_mult::vecmat_block_first_equiv](../../bench/tritonbench_g/batched_vecmat_mult/FPEquiv.lean)
- [bmm_chunk_bwd::bmm_residual_equiv](../../bench/tritonbench_g/bmm_chunk_bwd/FPEquiv.lean)
- [bmm_chunk_fwd::bmm_dot_first_equiv](../../bench/tritonbench_g/bmm_chunk_fwd/FPEquiv.lean)
- [bmm_optimized::bmm_dot_first_equiv](../../bench/tritonbench_g/bmm_optimized/FPEquiv.lean)
- [chunk_bwd_dqkg::chunk_dot_left_equiv](../../bench/tritonbench_g/chunk_bwd_dqkg/FPEquiv.lean)
- [chunk_cumsum_kernel::chunk_sum_left_equiv](../../bench/tritonbench_g/chunk_cumsum_kernel/FPEquiv.lean)
- [chunk_cumsum_vector::chunk_sum_left_equiv](../../bench/tritonbench_g/chunk_cumsum_vector/FPEquiv.lean)
- [chunk_delta_fwd::chunk_sum_left_equiv](../../bench/tritonbench_g/chunk_delta_fwd/FPEquiv.lean)
- [chunk_gate_recurrence::forward_equiv](../../bench/tritonbench_g/chunk_gate_recurrence/FPEquiv.lean)
- [chunk_gate_recurrence::backward_equiv](../../bench/tritonbench_g/chunk_gate_recurrence/FPEquiv.lean)
- [chunk_gated_attention::recurrent_equiv](../../bench/tritonbench_g/chunk_gated_attention/FPEquiv.lean)
- [chunk_gla_fwd::output_equiv](../../bench/tritonbench_g/chunk_gla_fwd/FPEquiv.lean)
- [chunk_gla_fwd::inter_equiv](../../bench/tritonbench_g/chunk_gla_fwd/FPEquiv.lean)
- [chunk_gla_fwd::merge_equiv](../../bench/tritonbench_g/chunk_gla_fwd/FPEquiv.lean)
- [chunk_linear_attn::forwardState_equiv](../../bench/tritonbench_g/chunk_linear_attn/FPEquiv.lean)
- [chunk_linear_attn::backwardState_equiv](../../bench/tritonbench_g/chunk_linear_attn/FPEquiv.lean)
- [chunk_linear_attn::backwardGradients_equiv](../../bench/tritonbench_g/chunk_linear_attn/FPEquiv.lean)
- [chunk_retention::forwardState_equiv](../../bench/tritonbench_g/chunk_retention/FPEquiv.lean)
- [chunk_retention::backwardState_equiv](../../bench/tritonbench_g/chunk_retention/FPEquiv.lean)
- [chunk_retention::backwardGradients_equiv](../../bench/tritonbench_g/chunk_retention/FPEquiv.lean)
- [chunk_retention_ops::forwardState_equiv](../../bench/tritonbench_g/chunk_retention_ops/FPEquiv.lean)
- [chunk_retention_ops::backwardState_equiv](../../bench/tritonbench_g/chunk_retention_ops/FPEquiv.lean)
- [chunk_retention_ops::backwardGradients_equiv](../../bench/tritonbench_g/chunk_retention_ops/FPEquiv.lean)
- [fused_activation::fused_activation_equiv](../../bench/tritonbench_g/fused_activation/FPEquiv.lean)
- [masked_add_cuda::masked_add_bool_commuted_equiv](../../bench/tritonbench_g/masked_add_cuda/FPEquiv.lean)
- [masked_add_cuda::masked_add_commuted_equiv](../../bench/tritonbench_g/masked_add_cuda/FPEquiv.lean)
- [masked_add_cuda::masked_add_typed_commuted_equiv](../../bench/tritonbench_g/masked_add_cuda/FPEquiv.lean)
- [vector_addition::add_commuted_equiv](../../bench/tritonbench_g/vector_addition/FPEquiv.lean)
- [vector_addition_custom::add_commuted_equiv](../../bench/tritonbench_g/vector_addition_custom/FPEquiv.lean)

### fp32_mul_commute_real

**fp32_mul_commute_real**

- 类型：`reported_atom`；证据状态：`reported_evidence_available`。
- 精确关系：Q32(a*b) = Q32(b*a)
- 适用域：finite a and finite b
- dtype / 精度 / 配置：{'lean_carrier_channel': 'real', 'lean_compute_precision': 'some .fp32', 'lean_output_channel': 'real', 'fragment_default_precision': 'none (each arithmetic operation explicitly carries .fp32)'}
- 影响：17 项、19 条公开声明。

实验或修复要求：

No absent aggregate row. If applying to a new target compiler/device or claiming native whole-kernel behavior, replay the exact probe/configuration and validate the real-carrier/fp32 interpretation and actual kernel site lowering; retain the model premise explicitly.

现有证据：

- [VeriTile/Triton/Float/ScalarArithmetic.lean](../../VeriTile/Triton/Float/ScalarArithmetic.lean)
- [VeriTile/Triton/Float/ScalarArithmetic.lean](../../VeriTile/Triton/Float/ScalarArithmetic.lean)
- [VeriTile/Triton/Float/ScalarArithmetic.lean](../../VeriTile/Triton/Float/ScalarArithmetic.lean)
- [VeriTile/Triton/Float/ProfiledRewrite.lean](../../VeriTile/Triton/Float/ProfiledRewrite.lean)
- [VeriTile/Triton/Float/ProfiledRewrite.lean](../../VeriTile/Triton/Float/ProfiledRewrite.lean)
- [VeriTile/Spec.lean](../../VeriTile/Spec.lean)
- [Published scoped aggregate report; no new GPU replay](../../experiments/floating_point/report/summary.json)
- [Proof-used atom audit, exact keys and report validation](../../bench/review-checkpoints/2026-10-14/numerical-assumption-audit/fp/audit.json)

受影响声明：

- [attention_fwd_triton1::linear_attention_equiv](../../bench/tritonbench_g/attention_fwd_triton1/FPEquiv.lean)
- [attention_score::score_scaling_equiv](../../bench/tritonbench_g/attention_score/FPEquiv.lean)
- [bgmv_shrink_kernel::bgmv_scaling_equiv](../../bench/tritonbench_g/bgmv_shrink_kernel/FPEquiv.lean)
- [block_sparse_attn::block_sparse_scaling_equiv](../../bench/tritonbench_g/block_sparse_attn/FPEquiv.lean)
- [chunk_gla_fwd::intra_equiv](../../bench/tritonbench_g/chunk_gla_fwd/FPEquiv.lean)
- [chunk_gla_fwd::split_equiv](../../bench/tritonbench_g/chunk_gla_fwd/FPEquiv.lean)
- [chunk_gla_simple::scale_equiv](../../bench/tritonbench_g/chunk_gla_simple/FPEquiv.lean)
- [chunk_linear_attn::forwardOutput_equiv](../../bench/tritonbench_g/chunk_linear_attn/FPEquiv.lean)
- [chunk_retention::forwardOutput_equiv](../../bench/tritonbench_g/chunk_retention/FPEquiv.lean)
- [chunk_retention_ops::forwardOutput_equiv](../../bench/tritonbench_g/chunk_retention_ops/FPEquiv.lean)
- [chunked_cumsum_fwd::scaleFirst_equiv](../../bench/tritonbench_g/chunked_cumsum_fwd/FPEquiv.lean)
- [context_attn_bloom::scaleFirst_equiv](../../bench/tritonbench_g/context_attn_bloom/FPEquiv.lean)
- [context_attn_fwd::scaleFirst_equiv](../../bench/tritonbench_g/context_attn_fwd/FPEquiv.lean)
- [context_attn_llama::scaleFirst_equiv](../../bench/tritonbench_g/context_attn_llama/FPEquiv.lean)
- [context_attn_mistral::scaleFirst_equiv](../../bench/tritonbench_g/context_attn_mistral/FPEquiv.lean)
- [context_attn_nopad::scaleFirst_equiv](../../bench/tritonbench_g/context_attn_nopad/FPEquiv.lean)
- [mul_exponent_compensator::mul_compensator_commuted_equiv](../../bench/tritonbench_g/mul_exponent_compensator/FPEquiv.lean)
- [triton_mul2::mul2_commuted_equiv](../../bench/tritonbench_g/triton_mul2/FPEquiv.lean)
- [triton_mul2::mul2_inplace_commuted_equiv](../../bench/tritonbench_g/triton_mul2/FPEquiv.lean)

### fp32_div_mul_rcp_real

**fp32_div_mul_rcp_real**

- 类型：`reported_atom`；证据状态：`reported_evidence_available`。
- 精确关系：Q32(a/b) = Q32(a*Q32(1/b))
- 适用域：finite a and finite b and b != 0; no implicit requirement that reciprocal/output are finite
- dtype / 精度 / 配置：{'lean_carrier_channel': 'real', 'lean_compute_precision': 'some .fp32', 'lean_output_channel': 'real', 'fragment_default_precision': 'none (each arithmetic operation explicitly carries .fp32)'}
- 影响：10 项、10 条公开声明。

实验或修复要求：

No absent aggregate row. If applying to a new target compiler/device or claiming native whole-kernel behavior, replay the exact probe/configuration and validate the real-carrier/fp32 interpretation and actual kernel site lowering; retain the model premise explicitly.

现有证据：

- [VeriTile/Triton/Float/ScalarArithmetic.lean](../../VeriTile/Triton/Float/ScalarArithmetic.lean)
- [VeriTile/Triton/Float/ScalarArithmetic.lean](../../VeriTile/Triton/Float/ScalarArithmetic.lean)
- [VeriTile/Triton/Float/ScalarArithmetic.lean](../../VeriTile/Triton/Float/ScalarArithmetic.lean)
- [VeriTile/Triton/Float/ProfiledRewrite.lean](../../VeriTile/Triton/Float/ProfiledRewrite.lean)
- [VeriTile/Triton/Float/ProfiledRewrite.lean](../../VeriTile/Triton/Float/ProfiledRewrite.lean)
- [VeriTile/Spec.lean](../../VeriTile/Spec.lean)
- [VeriTile/Triton/Float/Reciprocal.lean](../../VeriTile/Triton/Float/Reciprocal.lean)
- [Published scoped aggregate report; no new GPU replay](../../experiments/floating_point/supplement/report/summary.json)
- [Proof-used atom audit, exact keys and report validation](../../bench/review-checkpoints/2026-10-14/numerical-assumption-audit/fp/audit.json)

受影响声明：

- [apply_penalty::apply_penalty_reciprocal_equiv](../../bench/tritonbench_g/apply_penalty/FPEquiv.lean)
- [attention_forward_triton::attention_reciprocal_equiv](../../bench/tritonbench_g/attention_forward_triton/FPEquiv.lean)
- [attention_fwd_triton2::attention_reciprocal_equiv](../../bench/tritonbench_g/attention_fwd_triton2/FPEquiv.lean)
- [attention_fwd_triton3::attention_reciprocal_equiv](../../bench/tritonbench_g/attention_fwd_triton3/FPEquiv.lean)
- [attention_kernel::attention_reciprocal_equiv](../../bench/tritonbench_g/attention_kernel/FPEquiv.lean)
- [attn_fwd_causal::attention_reciprocal_equiv](../../bench/tritonbench_g/attn_fwd_causal/FPEquiv.lean)
- [attn_fwd_triton::attention_reciprocal_equiv](../../bench/tritonbench_g/attn_fwd_triton/FPEquiv.lean)
- [softmax_triton1::softmax_reciprocal_equiv](../../bench/tritonbench_g/softmax_triton1/FPEquiv.lean)
- [softmax_triton2::softmax_reciprocal_equiv](../../bench/tritonbench_g/softmax_triton2/FPEquiv.lean)
- [softmax_triton3::softmax_reciprocal_equiv](../../bench/tritonbench_g/softmax_triton3/FPEquiv.lean)

### fp32_mul_commute_typed

**fp32_mul_commute_typed**

- 类型：`reported_atom`；证据状态：`binding_validation_pending`。
- 精确关系：Q32(a*b) = Q32(b*a)
- 适用域：finite a and finite b
- dtype / 精度 / 配置：{'lean_carrier_channel': 'fp32', 'lean_compute_precision': 'some .fp32', 'lean_output_channel': 'fp32', 'fragment_default_precision': 'some .fp32'}
- 影响：2 项、2 条公开声明。

实验或修复要求：

Validate and record the exact :typed-fp32 binding using actual fp32 operands/output; old real-channel validation is insufficient. Existing offline compilation is type/source evidence only, on Triton 3.8.0 rather than the 3.7.1 numerical report.

现有证据：

- [VeriTile/Triton/Float/TypedScalarArithmetic.lean](../../VeriTile/Triton/Float/TypedScalarArithmetic.lean)
- [VeriTile/Triton/Float/TypedScalarArithmetic.lean](../../VeriTile/Triton/Float/TypedScalarArithmetic.lean)
- [bench/tests/TypedFP32Mul.lean](../../bench/tests/TypedFP32Mul.lean)
- [VeriTile/Triton/Float/ProfiledRewrite.lean](../../VeriTile/Triton/Float/ProfiledRewrite.lean)
- [VeriTile/Triton/Float/ProfiledRewrite.lean](../../VeriTile/Triton/Float/ProfiledRewrite.lean)
- [VeriTile/Spec.lean](../../VeriTile/Spec.lean)
- [Published scoped aggregate report; no new GPU replay](../../experiments/floating_point/report/summary.json)
- [Proof-used atom audit, exact keys and report validation](../../bench/review-checkpoints/2026-10-14/numerical-assumption-audit/fp/audit.json)

受影响声明：

- [context_attn_fwd::scaleFirst_equiv](../../bench/tritonbench_g/context_attn_fwd/FPEquiv.lean)
- [context_attn_llama::scaleFirst_equiv](../../bench/tritonbench_g/context_attn_llama/FPEquiv.lean)

### ROUND-FP16-ID

**ROUND-FP16-ID**

- 类型：`explicit_numeric_law`；证据状态：`counterexample_to_unrestricted_form`。
- 精确关系：R.round .fp16 = id; equivalently ∀ x : ℝ, R.round .fp16 x = x
- 适用域：∀ x : ℝ
- dtype / 精度 / 配置：{'dtypes': ['fp16'], 'precision': 'all real inputs, not merely representable fp16 grid points'}
- 影响：3 项、3 条公开声明。

实验或修复要求：

Do not try to confirm the global identity by sampling. Measure actual intermediate cast operands and show representability only on a precisely stated domain, or change the theorem to retain real rounding. Existing global identity cannot be discharged by a successful finite sample.

现有证据：

- [RoundingModel structure fields](../../VeriTile/Triton/Float/RoundingModel.lean)
- [Only casts are rounded in this expression evaluator; primitive arithmetic retains the mathematical interpretation.](../../VeriTile/Triton/Float/EvalOpR.lean)
- [Host binary16 reference witness: 2049/2048 rounds to 1, a finite fp32-exact input. Not a GPU result.](../../bench/review-checkpoints/2026-10-14/numerical-assumption-audit/nonfp/fp16-identity-counterexample.json)
- [All non-admitted-FP public statements and inherited predicate audit](../../bench/review-checkpoints/2026-10-14/numerical-assumption-audit/nonfp/audit.json)

受影响声明：

- [attention_kernel_aligned::attention_kernel_aligned_io_correctness](../../bench/tritonbench_g/attention_kernel_aligned/AttentionKernelAligned.lean)
- [triton_attention::triton_attention_forward_io_correctness](../../bench/tritonbench_g/triton_attention/TritonAttention.lean)
- [triton_conv2d_fwd::triton_conv2d_fwd_io_correctness](../../bench/tritonbench_g/triton_conv2d_fwd/TritonConv2dFwd.lean)

### BGMV-NATIVE-ADD-COMMUTE

**BGMV-NATIVE-ADD-COMMUTE**

- 类型：`explicit_numeric_law`；证据状态：`binding_validation_pending`。
- 精确关系：∀ site ∈ NativeFP.requirementsFor kernel, ∀ a b, D .finite a → D .finite b → (FP.ProfiledRewrite.engine M).binary site.profile.1 site.profile.2 .add a b = (FP.ProfiledRewrite.engine M).binary site.profile.1 site.profile.2 .add b a
- 适用域：Finite actual reduction result and loaded old output at each reached ADD_INPUTS branch, through the original dynamic output loop; evaluated masked lanes are included. No condition on unreachable branches.
- dtype / 精度 / 配置：[{'precision': 'fp64', 'dtype': 'real', 'configurations': 12}, {'precision': 'fp32', 'dtype': 'real', 'configurations': 20}, {'precision': 'fp32', 'dtype': 'fp16', 'configurations': 28}, {'precision': 'fp32', 'dtype': 'bf16', 'configurations': 4}, {'precision': 'fp32', 'dtype': 'fp32', 'configurations': 8}]
- 影响：1 项、1 条公开声明。

实验或修复要求：

For each selected dtype/precision profile, run actual compiled native add(a,b) and add(b,a), capture the actual reached reduction/output operands and check finiteness, compare the observation used by the backend model (including signed-zero policy), and retain compiler/architecture/options and binary hashes. bf16 widened-fp32 emulation does not alone establish native bf16 correspondence; fp16 needs its own evidence.

现有证据：

- [ScalarAdditionCommutes exact formula](../../bench/tritonbench_g/bgmv_expand_slice/Proofs/NativeFP.lean)
- [Site.Holds follows reached branches/ranges and evaluated operands](../../bench/tritonbench_g/bgmv_expand_slice/Proofs/NativeFP.lean)
- [NativeAdditionCommutes requests only source-selected addition profiles](../../bench/tritonbench_g/bgmv_expand_slice/Proofs/NativeFP.lean)
- [fp32_profiles and fp32_native_of_rules](../../bench/tritonbench_g/bgmv_expand_slice/Proofs/NativeFP.lean)
- [bf16 experiment uses widened fp32 and explicit rounding, not native bf16; fp16/fp64 unsupported](../../bench/tritonbench_g/bgmv_expand_slice/Proofs/AdditionProfiles.lean)
- [Logical countermodel shows fp32-only rules cannot establish native fp16 addition](../../bench/tritonbench_g/bgmv_expand_slice/Proofs/GeneralCountermodel.lean)
- [Current source enumeration: all72 configurations; fp64/real12, fp32/real20, fp32/fp16 28, fp32/bf16 4, fp32/fp32 8. Copy into the new final checkpoint; do not alter the historical profile snapshot.](../../bench/review-checkpoints/2026-10-14/fp-value-repair/bgmv-scope-compat/profiles.json)
- [All non-admitted-FP public statements and inherited predicate audit](../../bench/review-checkpoints/2026-10-14/numerical-assumption-audit/nonfp/audit.json)
- [Lean checks engine default vs explicit fp32 scalar-law definitional equality; nested scopes preserve innermost precision and original operands for allFloatDType; full execution axiom-clean and72 sourceprofile reductions.](../../bench/review-checkpoints/2026-10-14/fp-value-repair/bgmv-scope-compat/ProfileRegression.lean)

受影响声明：

- [bgmv_expand_slice::bgmv_output_first_equiv](../../bench/tritonbench_g/bgmv_expand_slice/FPEquiv.lean)

### ROUNDING-MODEL-STRUCTURE

**ROUNDING-MODEL-STRUCTURE**

- 类型：`shared_model_requirement`；证据状态：`shared_model_contract`。
- 精确关系：R.round_real : R.round .real = id; R.round_idem : ∀ dtype x, R.round dtype (R.round dtype x) = R.round dtype x
- 适用域：RoundingModel round : FloatDType → ℝ → ℝ, all x : ℝ; NaN/Inf are not a separate part of this codomain
- dtype / 精度 / 配置：{'dtypes': ['real', 'fp32', 'fp16', 'bf16', 'f8e4', 'f8e5'], 'precision': 'per FloatDType cast/store function'}
- 影响：31 项、33 条公开声明。

实验或修复要求：

A concrete rounding model must provide the idempotence proof/correspondence for each selected format and its handling of overflow/subnormal modes. An empirical check can recast all representable finite values, with boundary/underflow/overflow probes; finite sampling is not a proof of the quantified real-valued field.

现有证据：

- [RoundingModel structure fields](../../VeriTile/Triton/Float/RoundingModel.lean)
- [Only casts are rounded in this expression evaluator; primitive arithmetic retains the mathematical interpretation.](../../VeriTile/Triton/Float/EvalOpR.lean)
- [All non-admitted-FP public statements and inherited predicate audit](../../bench/review-checkpoints/2026-10-14/numerical-assumption-audit/nonfp/audit.json)

受影响声明：

- [attention_kernel_aligned::attention_kernel_aligned_io_correctness](../../bench/tritonbench_g/attention_kernel_aligned/AttentionKernelAligned.lean)
- [flash_decode2_llama::flash_decode2_llama_final_store_io_correctnessR](../../bench/tritonbench_g/flash_decode2_llama/FlashDecode2Llama.lean)
- [flash_decode2_phi::flash_decode2_phi_final_store_io_correctnessR](../../bench/tritonbench_g/flash_decode2_phi/FlashDecode2Phi.lean)
- [iv_dependent_matmul::iv_dependent_matmul_io_correctness](../../bench/tritonbench_g/iv_dependent_matmul/IvDependentMatmul.lean)
- [kldiv_triton::kldiv_forward_default_none_io_correctness](../../bench/tritonbench_g/kldiv_triton/Correct.lean)
- [layer_norm_liger::layer_norm_liger_scalar_stores_io_correctnessR](../../bench/tritonbench_g/layer_norm_liger/LayerNormLiger.lean)
- [layer_norm_ops::layer_norm_ops_scalar_stores_io_correctnessR](../../bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean)
- [layernorm_fwd_triton::layernorm_fwd_triton_io_correctness](../../bench/tritonbench_g/layernorm_fwd_triton/LayernormFwdTriton.lean)
- [lightning_attention::lightning_attention_bwd_grad_store_io_correctnessR](../../bench/tritonbench_g/lightning_attention/LightningAttention.lean)
- [lora_expand_gemv::gemv_full_io_correctness](../../bench/tritonbench_g/lora_expand_gemv/LoraExpandGemv.lean)
- [matmul_kernel::matmul_kernel_io_correctness](../../bench/tritonbench_g/matmul_kernel/MatmulKernel.lean)
- [matmul_leakyrelu::matmul_leakyrelu_io_correctness](../../bench/tritonbench_g/matmul_leakyrelu/MatmulLeakyrelu.lean)
- [matmul_leakyrelu_fp8::matmul_leakyrelu_fp8_io_correctness](../../bench/tritonbench_g/matmul_leakyrelu_fp8/MatmulLeakyreluFp8.lean)
- [matmul_tma::matmul_tma_f16_io_correctnessR](../../bench/tritonbench_g/matmul_tma/MatmulTma.lean)
- [matmul_triton1::matmul_triton1_io_correctness](../../bench/tritonbench_g/matmul_triton1/MatmulTriton1.lean)
- [matmul_triton2::matmul_triton2_io_correctness](../../bench/tritonbench_g/matmul_triton2/MatmulTriton2.lean)
- [matmul_triton_autotune::matmul_autotune_io_correctness](../../bench/tritonbench_g/matmul_triton_autotune/MatmulTritonAutotune.lean)
- [mean_reduction::mean_dim_kernel_io_correctness](../../bench/tritonbench_g/mean_reduction/Correct.lean)
- [nested_loops_processing::nested3_first_a1_store_io_correctnessR](../../bench/tritonbench_g/nested_loops_processing/Correct.lean)
- [quantize_copy_kv::quantize_copy_kv_io_correctness](../../bench/tritonbench_g/quantize_copy_kv/QuantizeCopyKv.lean)
- [relu_strided_buffer::original_grid_stride_correctness](../../bench/tritonbench_g/relu_strided_buffer/Correct.lean)
- [relu_strided_buffer::candidate_grid_stride_correctness](../../bench/tritonbench_g/relu_strided_buffer/Correct.lean)
- [rmsnorm_triton::rmsnorm_full_io_correctness](../../bench/tritonbench_g/rmsnorm_triton/Correct.lean)
- [sgmv_expand_slice::sgmv_expand_slice_io_correctness](../../bench/tritonbench_g/sgmv_expand_slice/SgmvExpandSlice.lean)
- [softmax_reducev::softmax_reducev_io_correctness](../../bench/tritonbench_g/softmax_reducev/SoftmaxReducev.lean)
- [token_attn_llama2::token_attn_llama2_io_correctness](../../bench/tritonbench_g/token_attn_llama2/TokenAttnLlama2.lean)
- [token_attn_mistral::token_attn_mistral_io_correctness](../../bench/tritonbench_g/token_attn_mistral/TokenAttnMistral.lean)
- [token_attn_reduceV::token_attn_reducev_io_correctness](../../bench/tritonbench_g/token_attn_reduceV/TokenAttnReduceV.lean)
- [triton_attention::triton_attention_forward_io_correctness](../../bench/tritonbench_g/triton_attention/TritonAttention.lean)
- [triton_conv2d_fwd::triton_conv2d_fwd_io_correctness](../../bench/tritonbench_g/triton_conv2d_fwd/TritonConv2dFwd.lean)
- [triton_linear_activation::triton_linear_activation_io_correctness](../../bench/tritonbench_g/triton_linear_activation/TritonLinearActivation.lean)
- [triton_matmul::triton_matmul_f16_io_correctness](../../bench/tritonbench_g/triton_matmul/TritonMatmul.lean)
- [triton_matmul::triton_matmul_f8_io_correctness](../../bench/tritonbench_g/triton_matmul/TritonMatmul.lean)

## 真实后端对应关系

**REAL-ARITHMETIC-BOUNDARY**

NumericDType add/sub/mul/div on all floating tags use Option.map₂ of exact ℝ operations; FloatDType.cast preserves WithBot ℝ through dtype tags.

The 429 mathematical statements are formal real-model results, not exact GPU floating results. Do not turn every such result into a new missing experiment; retain a separate semantics/refinement axis.

- [VeriTile/Triton/Semantics/Scalar.lean](../../VeriTile/Triton/Semantics/Scalar.lean)
- [VeriTile/Triton/Semantics/Scalar.lean](../../VeriTile/Triton/Semantics/Scalar.lean)
- [VeriTile/Triton/Semantics/Scalar.lean](../../VeriTile/Triton/Semantics/Scalar.lean)

**TRANSCENDENTAL-BOUNDARY**

['realExp (some x) = some (Real.exp x)', 'realLog (some x) = some (Real.log x)', 'realSqrt (some x) = some (Real.sqrt x)', 'realRsqrt (some x) = some (1 / Real.sqrt x)', 'realPow (some a) (some b) = some (Real.rpow a b)']

These are definitions of the mathematical model, not unproved exp/log/sqrt laws in the reviewed signatures. Approximation errors, exceptional IEEE inputs, fast-math and instruction correspondence need a separate hardware refinement study.

- [VeriTile/Triton/Semantics/TileOps.lean](../../VeriTile/Triton/Semantics/TileOps.lean)
- [VeriTile/Triton/Semantics/TileOps.lean](../../VeriTile/Triton/Semantics/TileOps.lean)

**INT8-QUANTIZER-BOUNDARY**

realToInt8 (some x) = if 0 ≤ x then Int.floor x else -Int.floor (-x); tritonInt8CastInRange x := -128 ≤ trunc(x) ∧ trunc(x) ≤ 127

The value model uses unbounded Int after truncation. Width overflow/saturation/wraparound must be scoped separately; do not call a pre-rounding/scaled-store theorem a completed int8 conversion proof.

- [VeriTile/Triton/Semantics/Scalar.lean](../../VeriTile/Triton/Semantics/Scalar.lean)

**PRECOMPUTED-STAGE-CONTRACTS**



hPrev/hNext and prepared KTrans/VTile refer to a declared local stage input. User accepts stage scope. In particular fused_recurrent_delta hNext is a supplied post-update carry and not discharged merely by an adjacent state-update clause; do not promote the stage theorem to end-to-end.

- [bench/tritonbench_g/fused_recurrent_delta/FusedRecurrentDelta.lean](../../bench/tritonbench_g/fused_recurrent_delta/FusedRecurrentDelta.lean)
- [bench/tritonbench_g/fused_recurrent_hgrn/FusedRecurrentHgrn.lean](../../bench/tritonbench_g/fused_recurrent_hgrn/FusedRecurrentHgrn.lean)
- [bench/tritonbench_g/lightning_attention/LightningAttention.lean](../../bench/tritonbench_g/lightning_attention/LightningAttention.lean)

**UNSPECIFIED-PADDING-STATE**

s.undef = (fun _ _ => 0), or ∀ rg off, s.undef rg off = 0

50 public statement texts mention an undef condition. Some are the finite-flat bridge convention, some select a zero oracle for masked/boundary loads. This is not an addition/rounding experiment. For hardware use preserve the zero-oracle profile or separately show undefined lanes are unobserved; no blanket arbitrary-undef claim is made. This textual count excludes conditions hidden only inside named IO definitions.


- Existing scalar reports are trusted scoped empirical evidence. They are not new GPU replay or universal IEEE equalities. A separate backend refinement is required for a hardware-wide claim.

- RoundingModel requires round_real and round_idem in the actual structure. Its introductory comment is stale about optional idempotence; the structure fields are authoritative.

- Normal input and documented configuration/stage scopes are accepted by the user. This does not change the formulas, omit eager-lane guards, or silently prove omitted source stages.


## 逐项重新统计

| 任务 | 浮点证据层级 | 声明内数值前提状态 | 额外数值假设 | 共享模型要求 | 完整源码转换阻塞 |
|---|---|---|---|---|---|
| `adam_update_triton` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `add_example` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `add_value` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `apply_penalty` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_div_mul_rcp_real | 无 | 无此标记 |
| `attention_forward_triton` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_div_mul_rcp_real | 无 | 无此标记 |
| `attention_fwd_triton1` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_mul_commute_real | 无 | 无此标记 |
| `attention_fwd_triton2` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_div_mul_rcp_real | 无 | 无此标记 |
| `attention_fwd_triton3` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_div_mul_rcp_real | 无 | 无此标记 |
| `attention_kernel` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_div_mul_rcp_real | 无 | 无此标记 |
| `attention_kernel_aligned` | 存在数值反例：需修订模型或适用域 | 含不可直接推广到真实浮点的理想化假设 | ROUND-FP16-ID | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `attention_llama` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `attention_score` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_mul_commute_real | 无 | 无此标记 |
| `attn_fwd_causal` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_div_mul_rcp_real | 无 | 无此标记 |
| `attn_fwd_triton` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_div_mul_rcp_real | 无 | 无此标记 |
| `batched_vecmat_mult` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `bgmv_expand_slice` | 有条件相对等价证明：精确数值绑定待补 | 需补数值实验或精确绑定证据 | BGMV-NATIVE-ADD-COMMUTE | 无 | 无此标记 |
| `bgmv_shrink_kernel` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_mul_commute_real | 无 | 无此标记 |
| `block_sparse_attn` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_mul_commute_real | 无 | 无此标记 |
| `bmm_chunk_bwd` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `bmm_chunk_fwd` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `bmm_optimized` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `cache_transform_triton` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `chunk_bwd_dqkg` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `chunk_cumsum_kernel` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `chunk_cumsum_vector` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `chunk_delta_fwd` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `chunk_gate_recurrence` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `chunk_gated_attention` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `chunk_gla_fwd` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real, fp32_mul_commute_real | 无 | 无此标记 |
| `chunk_gla_simple` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_mul_commute_real | 无 | 无此标记 |
| `chunk_linear_attn` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real, fp32_mul_commute_real | 无 | 无此标记 |
| `chunk_retention` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real, fp32_mul_commute_real | 无 | 无此标记 |
| `chunk_retention_ops` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real, fp32_mul_commute_real | 无 | 无此标记 |
| `chunked_cumsum_fwd` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_mul_commute_real | 无 | 无此标记 |
| `context_attn_bloom` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_mul_commute_real | 无 | 无此标记 |
| `context_attn_fwd` | 有条件相对等价证明：精确数值绑定待补 | 需补数值实验或精确绑定证据 | fp32_mul_commute_real, fp32_mul_commute_typed | 无 | 无此标记 |
| `context_attn_llama` | 有条件相对等价证明：精确数值绑定待补 | 需补数值实验或精确绑定证据 | fp32_mul_commute_real, fp32_mul_commute_typed | 无 | 无此标记 |
| `context_attn_mistral` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_mul_commute_real | 无 | 无此标记 |
| `context_attn_nopad` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_mul_commute_real | 无 | 无此标记 |
| `cosine_compute` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `cross_entropy1` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `cross_entropy2` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `cross_entropy_ops` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `decay_cumsum` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `dequantize_matmul` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `dequantize_rowwise` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `destindex_copy` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `destindex_copy_kv1` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `destindex_copy_kv2` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `diag_ssm_triton` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `dropout_triton` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `embedding_triton_kernel` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `f8_conversion_utils` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `fast_ce_loss` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `fast_layernorm` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `fast_rms_layernorm` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `fast_rope_embedding` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `fifth_order_sph_harmonics` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `flash_attn` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `flash_decode2_llama` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `flash_decode2_phi` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `fp4_to_bf16` | 缺源码 | 缺源码 | 无 | 无 | 无此标记 |
| `fp4_to_bf16_conversion` | 缺源码 | 缺源码 | 无 | 无 | 无此标记 |
| `fused_activation` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `fused_layernorm_triton` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `fused_recurrent_delta` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `fused_recurrent_hgrn` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `fused_recurrent_retention` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `fused_rotary_embedding` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `fused_rwkv6_kernel` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `geglu_tanh_triton` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `index_select_bwd` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `index_select_cat` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `int4_matmul` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `int8_dequant_matmul` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `int8_matmul_kernel` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `int8_matmul_quantization` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `int8_quantization` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `int_scaled_matmul` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `isfinite_kernel` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `iv_dependent_matmul` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `kcache_copy_triton` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `kldiv_compute` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `kldiv_ops` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `kldiv_triton` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `ksoftmax_triton` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `kv_cache_copy` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `kv_cache_filling` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `l2_norm_bwd` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `l2_norm_triton1` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `l2_norm_triton2` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `layer_norm_fwd` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `layer_norm_liger` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `layer_norm_ops` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `layer_norm_triton` | 缺源码 | 缺源码 | 无 | 无 | 无此标记 |
| `layer_norm_welfold` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `layernorm_fwd_triton` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `lightning_attention` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `llama_ff_triton` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `log_softmax` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `logsumexp_fwd` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `lora_expand_gemv` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `masked_add_cuda` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `masked_select` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `matmul_dequant_int4` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `matmul_dequantize` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `matmul_dequantize_int4` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `matmul_kernel` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `matmul_leakyrelu` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `matmul_leakyrelu_fp8` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `matmul_persistent_triton` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `matmul_tma` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `matmul_triton1` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `matmul_triton2` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `matmul_triton_autotune` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `matrix_reduction` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `matrix_transpose` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `matrix_vector_multip` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `max_reduction` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `mean_reduction` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `mixed_sparse_attention` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `mul_exponent_compensator` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_mul_commute_real | 无 | 无此标记 |
| `multinomial_sampling` | 缺源码 | 缺源码 | 无 | 无 | 无此标记 |
| `nested_loops_processing` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `parallel_attention` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `parallel_retention_attention` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `pow_scalar_tensor` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `quant_transpose_kernel` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 是；已证阶段仍保留 |
| `quantize_copy_kv` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `quantize_global` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `quantize_kv_copy` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `quantize_kv_transform` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rbe_triton_transform` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `relu_strided_buffer` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `relu_triton_kernel` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `reversed_cumsum` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `reversed_cumsum_scalar` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rms_matmul_rbe` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rms_norm_triton` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rms_rbe_matmul` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rmsnorm_fused` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rmsnorm_fused_llama` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rmsnorm_implementation` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rmsnorm_triton` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `rope_backward_transform` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rope_embedding` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rope_transform` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rotary_emb` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rotary_emb_nopad` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rotary_transform` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rotary_transform_ops` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `rowwise_quantization_triton` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `seeded_dropout` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `sgmv_expand_slice` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `sin_computation` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `sin_kernel` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `softmax_flaggems` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `softmax_optimize` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `softmax_reducev` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `softmax_triton1` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_div_mul_rcp_real | 无 | 无此标记 |
| `softmax_triton2` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_div_mul_rcp_real | 无 | 无此标记 |
| `softmax_triton3` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_div_mul_rcp_real | 无 | 无此标记 |
| `spinning_lock_reduction` | 缺源码 | 缺源码 | 无 | 无 | 无此标记 |
| `square_matrix` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `streamk_matmul` | 缺源码 | 缺源码 | 无 | 无 | 无此标记 |
| `swiglu_backward` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `swiglu_fwd` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `swiglu_triton` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `token_attn_llama2` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `token_attn_mistral` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `token_attn_reduceV` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `token_softmax_bloom` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `token_softmax_llama` | 有逐算子后端值契约：原语及硬件对应仍待完成 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `triton_argmax` | 含参数化位解释器值契约：逐阶段范围及硬件对应另列 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `triton_attention` | 存在数值反例：需修订模型或适用域 | 含不可直接推广到真实浮点的理想化假设 | ROUND-FP16-ID | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `triton_conv2d_fwd` | 存在数值反例：需修订模型或适用域 | 含不可直接推广到真实浮点的理想化假设 | ROUND-FP16-ID | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `triton_linear_activation` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `triton_matmul` | 仅cast/store舍入模型：缺算术浮点证明 | 未使用额外数值律；不代表无需实验 | 无 | ROUNDING-MODEL-STRUCTURE | 无此标记 |
| `triton_mul2` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_mul_commute_real | 无 | 无此标记 |
| `triton_softmax` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `uniform_sampling` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `var_len_copy` | 有后端参数化的程序相对等价证明 | 未使用额外数值律；不代表无需实验 | 无 | 无 | 无此标记 |
| `vector_addition` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |
| `vector_addition_custom` | 有依赖已有报告前提的程序相对等价证明 | 使用已有实验报告中的数值律 | fp32_add_commute_real | 无 | 无此标记 |

## 复核

```sh
python3 bench/numerical_obligations.py --check
```

检查包括全部公开记录的成员对应、实际声明文本、源文件和证据哈希。新增、删除或更改声明必须重新审阅，不能自动归入“无额外数值律”。实验状态不会因为配置/阶段范围已获接受而自动提升。
