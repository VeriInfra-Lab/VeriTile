# 浮点与固定宽语义证明缺口

**原41项的数学模型缺口已补充值语义合同；仍有23项只有cast/store舍入模型，3项含已有反例的数值假设，另6项缺源码。**

41项的旧数学定理保留原标签。新增合同逐项保留已接受的输入、配置、阶段范围，并区分后端回调与软件位解释器。硬件对应仍未建立，不能把这次修复称为41项GPU验收通过。

[41项修复范围及公开声明](fp_value_repairs.md)、[逐项硬件实验清单](fp_value_native_tasks.json)、[184项当前统计](numerical_obligations.md)。原始41项缺口保存在[历史基线](../review-checkpoints/2026-10-14/fp-value-repair/before/bench/tritonbench_g/fp_proof_gaps.md)。

## 仍未补齐的23项cast/store模型

这些契约允许cast/store舍入，但普通算术仍按实数执行。需要为已接受阶段补齐逐算子的精度、顺序和独立结果证明；仅提供round_idem不能关闭缺口。

| 任务 | 已有cast/store声明 |
|---|---|
| `flash_decode2_llama` | [flash_decode2_llama_final_store_io_correctnessR](flash_decode2_llama/FlashDecode2Llama.lean) |
| `flash_decode2_phi` | [flash_decode2_phi_final_store_io_correctnessR](flash_decode2_phi/FlashDecode2Phi.lean) |
| `iv_dependent_matmul` | [iv_dependent_matmul_io_correctness](iv_dependent_matmul/IvDependentMatmul.lean) |
| `layer_norm_liger` | [layer_norm_liger_scalar_stores_io_correctnessR](layer_norm_liger/LayerNormLiger.lean) |
| `layer_norm_ops` | [layer_norm_ops_scalar_stores_io_correctnessR](layer_norm_ops/LayerNormOps.lean) |
| `layernorm_fwd_triton` | [layernorm_fwd_triton_io_correctness](layernorm_fwd_triton/LayernormFwdTriton.lean) |
| `lightning_attention` | [lightning_attention_bwd_grad_store_io_correctnessR](lightning_attention/LightningAttention.lean) |
| `lora_expand_gemv` | [gemv_full_io_correctness](lora_expand_gemv/LoraExpandGemv.lean) |
| `matmul_kernel` | [matmul_kernel_io_correctness](matmul_kernel/MatmulKernel.lean) |
| `matmul_leakyrelu` | [matmul_leakyrelu_io_correctness](matmul_leakyrelu/MatmulLeakyrelu.lean) |
| `matmul_leakyrelu_fp8` | [matmul_leakyrelu_fp8_io_correctness](matmul_leakyrelu_fp8/MatmulLeakyreluFp8.lean) |
| `matmul_tma` | [matmul_tma_f16_io_correctnessR](matmul_tma/MatmulTma.lean) |
| `matmul_triton1` | [matmul_triton1_io_correctness](matmul_triton1/MatmulTriton1.lean) |
| `matmul_triton2` | [matmul_triton2_io_correctness](matmul_triton2/MatmulTriton2.lean) |
| `matmul_triton_autotune` | [matmul_autotune_io_correctness](matmul_triton_autotune/MatmulTritonAutotune.lean) |
| `quantize_copy_kv` | [quantize_copy_kv_io_correctness](quantize_copy_kv/QuantizeCopyKv.lean) |
| `sgmv_expand_slice` | [sgmv_expand_slice_io_correctness](sgmv_expand_slice/SgmvExpandSlice.lean) |
| `softmax_reducev` | [softmax_reducev_io_correctness](softmax_reducev/SoftmaxReducev.lean) |
| `token_attn_llama2` | [token_attn_llama2_io_correctness](token_attn_llama2/TokenAttnLlama2.lean) |
| `token_attn_mistral` | [token_attn_mistral_io_correctness](token_attn_mistral/TokenAttnMistral.lean) |
| `token_attn_reduceV` | [token_attn_reducev_io_correctness](token_attn_reduceV/TokenAttnReduceV.lean) |
| `triton_linear_activation` | [triton_linear_activation_io_correctness](triton_linear_activation/TritonLinearActivation.lean) |
| `triton_matmul` | [triton_matmul_f16_io_correctness](triton_matmul/TritonMatmul.lean)<br>[triton_matmul_f8_io_correctness](triton_matmul/TritonMatmul.lean) |

## 仍需修订的数值模型

`attention_kernel_aligned`、`triton_attention`、`triton_conv2d_fwd` 的全域 `round_fp16 = id` 假设已有反例。需修改参考模型，或证明实际中间值满足明确的精确适用域；增加抽样不能证明这个已被反驳的全域命题。

## 缺少源码

`fp4_to_bf16`、`fp4_to_bf16_conversion`、`layer_norm_triton`、`multinomial_sampling`、`spinning_lock_reduction`、`streamk_matmul`。

## 实验边界

[原有五个局部数值实验配置](numeric_experiment_handoff.md)继续保留；新增41项的原语/编译配置验证见上方逐项清单。有限样本可以反驳绑定或支持已记录的配置，不能代替普遍性证明。

[机器可读缺口表](fp_proof_gaps.json)由当前声明审阅和保留的历史基线生成；`python3 bench/fp_value_repairs.py --check` 检查一致性。
