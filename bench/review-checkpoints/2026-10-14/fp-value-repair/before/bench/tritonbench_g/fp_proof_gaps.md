# 浮点与固定宽语义证明缺口

**41项只有数学契约，23项只有cast/store舍入契约，另3项有全域舍入恒等反例。它们均未补齐有限精度目标。** 接受输入、配置、阶段范围不等于接受实数算术替代浮点算术。

本轮逐项核查并修正统计/验收门禁，没有新增这些任务的浮点证明。不得使用source=source、自反等价、原样候选、把Real运算改名为fp32，或直接假定最终输出正确来关闭缺口。

41项中的1项为纯整数核心，实际要补固定宽dot/溢出语义；不能为了统计FP而编造浮点实验。其余按实际操作分为浮点计算21、混合整数/FP6、量化7、typed搬运/比较6。

## 41项数学模型任务的具体工作

| 任务 | 当前接受的范围 | 该范围内还缺什么 |
|---|---|---|
| [fifth_order_sph_harmonics](../../bench/tritonbench_g/fifth_order_sph_harmonics/FifthOrderSphHarmonics.lean) | 11 个五阶球谐前向多项式输出。 | 为实际常量、乘法链和加减表达式建立逐节点 dtype/舍入及融合选项语义，证明独立有限精度多项式结果或真实不同候选。 |
| [fused_layernorm_triton](../../bench/tritonbench_g/fused_layernorm_triton/FusedLayernormTriton.lean) | Welford/方差 reduction 与消费给定 mean/rstd 的 normalize 分阶段结论。 | 把已选 reduction 和 normalize 中的 fp32/bf16 转换、Welford 顺序、libdevice.rsqrt 和仿射乘加保留到 FP 执行/参考；实数方差恒等式不能直接迁入 FP。 |
| [fused_recurrent_delta](../../bench/tritonbench_g/fused_recurrent_delta/FusedRecurrentDelta.lean) | 给定 HPrev/HNext 的前后向局部步骤与若干输出事实。 | 在已选步骤中建立 typed load/cast、逐步 sum/product/add 的 FP 执行和独立 rounded carry/output 规格；实际 kernel 与目标的差异尚需确定。 |
| [fused_recurrent_hgrn](../../bench/tritonbench_g/fused_recurrent_hgrn/FusedRecurrentHgrn.lean) | 给定 carry 的前向一步、后向 dx/dg 一步与 dx 两步；dx 有实数 flat IO。 | 为 carry*gate+input、dx=carry+DO、dg 乘法及输出 cast 给出实际 dtype 的 FP 步骤规格/证明；现有 signed-loop typing 修复不提供这种数值证明。 |
| [fused_rotary_embedding](../../bench/tritonbench_g/fused_rotary_embedding/FusedRotaryEmbedding.lean) | Q 旋转与 V-cache chain 两个选定子核。 | Q 旋转需 typed 乘法/加减与存储转换结果；V-cache 子核需独立的 typed payload/address preservation。两者均未有任务级 FP 比较定理。 |
| [fused_rwkv6_kernel](../../bench/tritonbench_g/fused_rwkv6_kernel/FusedRwkv6Kernel.lean) | 给定 BHPrev 的 state update、output 与条件末状态事实。 | 为 fp32 carry 更新、tl.exp、sum、输入/输出 cast 保留顺序并建立 FP 规格；不能用实数 exp 和递推闭式替代 native 执行。 |
| [int4_matmul](../../bench/tritonbench_g/int4_matmul/Int4Matmul.lean) | 选定分组/块布局的 packed-int4 解码与 matmul 读回。 | 需要固定宽位提取/符号解释、解量化乘法及 fp16 转换、dot 的输入/累加精度和顺序、输出转换的实际语义与独立结果；无现成 FP 候选。 |
| [int8_dequant_matmul](../../bench/tritonbench_g/int8_dequant_matmul/Int8DequantMatmul.lean) | 逐行反量化 matmul、可选 bias、fp16 标记输出。 | 证明 int8/int32 dot 的固定宽行为或已声明不溢出域，再保留 scale/bias 的 FP 运算和 fp16 store；当前 typed tag 的 cast 在旧 eval 下仍是数学转换。 |
| [int8_matmul_kernel](../../bench/tritonbench_g/int8_matmul_kernel/Int8MatmulKernel.lean) | 打包整数提取、整数 dot 与输出。 | 优先补机器整数语义：输入转换、位移/掩码、有符号解释、int32 dot 累加溢出及输出宽度；与机器整数参考或真实不同候选比较。不能凭任务在 FP 清单里就编造浮点交换律。 |
| [int8_matmul_quantization](../../bench/tritonbench_g/int8_matmul_quantization/Int8MatmulQuantization.lean) | 分开的 per-row quantize 与 matmul 阶段。 | quantize 需保留 abs/max、scale 除法、浮点转 int8 的取整/宽度语义；matmul 需 int32 dot、转 fp32、a_scale*b_scale 和输出转换。分别给出阶段 FP/机器整数规格。 |
| [int8_quantization](../../bench/tritonbench_g/int8_quantization/Int8Quantization.lean) | 消费 ScalePre 的预取整 Q/K 缩放/存储片段。 | 当前片段也需真实 dtype 的 preScale 乘法/scale 除法与 store 的 FP 规格；它不是 int8 量化结果证明。若声称真实 quantized 输出，才需补 sign bias 和 int8 cast。 |
| [int_scaled_matmul](../../bench/tritonbench_g/int_scaled_matmul/IntScaledMatmul.lean) | 第一源整数 GEMM 含实际 undef-tail 贡献；第二源按行 scale 输出。 | 第一阶段需固定宽整数载入/dot/输出语义，保留原始尾部 oracle；第二阶段需真实 int32→浮点转换、scale 乘法和输出类型。现有 floor/unbounded Int 与实数乘法不能给出这些。 |
| [kcache_copy_triton](../../bench/tritonbench_g/kcache_copy_triton/KcacheCopyTriton.lean) | sequence-length-one 的 cache-copy 片段。 | 补独立 typed/bit payload 的源地址→目标地址读回与 frame/failure 证明；若目标是优化等价，还需真实不同的地址/调度候选与相同数值运算。 |
| [kv_cache_copy](../../bench/tritonbench_g/kv_cache_copy/KvCacheCopy.lean) | sequence-length-one 的 K/V x-block copy。 | 补选定 dtype 的 typed payload 复制、region separation 和地址映射的 FP/bit 语义结果；候选未定义，不能以原样源作候选。 |
| [kv_cache_filling](../../bench/tritonbench_g/kv_cache_filling/KvCacheFilling.lean) | 未量化 K/V tile copy 与消费 MetaPre 的量化 metadata store。 | 为这两个已选片段证明 typed payload、真实 storage conversion、地址和 mask 的结果；现有实数 copy 投影不足以覆盖任意 FP bit value。 |
| [layer_norm_welfold](../../bench/tritonbench_g/layer_norm_welfold/LayerNormWelfold.lean) | 分开的 reduction 与消费给定 mean/rstd 的 normalize。 | 保留实际 variance/reduction 顺序、fp32/bf16 casts、libdevice.rsqrt 和 normalize 的 FP 运算图，建立独立 rounded reference/真实候选比较。 |
| [logsumexp_fwd](../../bench/tritonbench_g/logsumexp_fwd/LogsumexpFwd.lean) | 单 program 活跃块、可选 scale 的 log-sum-exp。 | 建立 max/sub/exp/sum/log 及 scale 的真实 FP 执行顺序和类型；native tl.exp/tl.log 与实数函数的对应待绑定。 |
| [masked_select](../../bench/tritonbench_g/masked_select/MaskedSelect.lean) | 消费 select mask 与 prefix-sum metadata 的 selection-copy。 | 补输入 payload 的 typed/bit 保持、mask→bool 转换与 prefix 地址的 FP/离散语义输出证明；优化候选需有真实源级差异。 |
| [matmul_dequant_int4](../../bench/tritonbench_g/matmul_dequant_int4/MatmulDequantInt4.lean) | 公开 headline 实际只覆盖 standalone dequantize 输出阶段。 | 补 packed nibble 提取的固定宽语义、scale/zero-point 的实际 FP 类型与乘法、结果 storage conversion；证明独立 dequantization 规格。 |
| [matmul_dequantize](../../bench/tritonbench_g/matmul_dequantize/MatmulDequantize.lean) | standalone dequantize、matmul4 及 SPLIT_K=1 matmul 的三个源码阶段。 | 分别补固定宽 nibble 解码、fp16 操作数转换、fp16×fp16→fp32 dot/累加、fp32→real 通道桥接和实际 store 的有限精度语义；当前保留 typed AST 但旧 evaluator 擦除宽度。 |
| [matmul_dequantize_int4](../../bench/tritonbench_g/matmul_dequantize_int4/MatmulDequantizeInt4.lean) | 完整 matmul4 的选定 NO_GROUPS 两支与实际尾部读地址。 | 补 packed/int-width 解码和实际 source b.to(a.dtype)、dot 累加、输出转换的 FP 规格；当前 real-to-real cast 的保留只是源形状证据。 |
| [matrix_reduction](../../bench/tritonbench_g/matrix_reduction/MatrixReduction.lean) | 单矩阵 tile 的逐行 max 与输出 IO。 | 补真实浮点比较/max reduction 的规格、dtype 存储与 payload 行为；有限输入域下可用顺序/选择证明，NaN、signed-zero/tie 行为需明确域或语义。 |
| [matrix_vector_multip](../../bench/tritonbench_g/matrix_vector_multip/MatrixVectorMultip.lean) | 一个 BLOCK_M 范围的矩阵向量乘积。 | 补源输入转 fp32、逐项乘法、tl.sum 的实际树/累加精度及 output store 的 FP 规格；未给新归约候选前不列特定重排律。 |
| [mixed_sparse_attention](../../bench/tritonbench_g/mixed_sparse_attention/MixedSparseAttention.lean) | 指定 sparse metadata 的前向值事实与 AccPre/LPre epilogue。 | 为选定 forward 和 epilogue 保留 q scale 转换、dot、exp2、max/sum、division、p/output casts 的 FP 语义与独立参考；current real closed form 不能证明浮点结果。 |
| [parallel_attention](../../bench/tritonbench_g/parallel_attention/ParallelAttention.lean) | 前向 O/Z 与后向 DK/DV，已修正 unspecified padding 参考。 | 保留原 load oracle、allow_tf32=false、scale、dot、平方/乘法、sum 和输出 casts，建立选定阶段的 FP typed execution/reference；当前 real identity cast 是明确缺口。 |
| [parallel_retention_attention](../../bench/tritonbench_g/parallel_retention_attention/ParallelRetentionAttention.lean) | 前向 O 与后向 DK/DV 的 causal/retention 配置。 | 保留原 oracle、allow_tf32=false、exp2/log2 衰减、scale、dot、casts 与累加顺序，建立 FP 阶段规格/证明；实数 retention 公式不提供 native 精度。 |
| [quant_transpose_kernel](../../bench/tritonbench_g/quant_transpose_kernel/QuantTransposeKernel.lean) | faithful llrint/int8 路径显式投影失败；已有转置 scaled-real store 片段。 | 先确定验收目标：对已接受 scaled-store 片段补实际 dtype 乘法/cast FP 结果；若要 quantized 输出，补 llrint rounding-mode、int8 narrowing 语义及 faithful lowering。 |
| [quantize_global](../../bench/tritonbench_g/quantize_global/QuantizeGlobal.lean) | 给定 AbsmaxInv 的 scaled-real store；faithful llrint/int8 投影显式失败。 | 已选片段需 FP scale/multiply/store 规格；实际量化输出则需 llrint 的精确取整和 int8 转换，先解决 lowering 而不是新增无关交换律。 |
| [quantize_kv_copy](../../bench/tritonbench_g/quantize_kv_copy/QuantizeKvCopy.lean) | groupwise KV 量化值和 scale 的数学转换结果。 | 建立 abs/max/127 与 scale cast、src/scale、float→int8 的实际 FP/固定宽语义，尤其 scale=0、范围边界和结果转换；source duplicate kernel 定义必须按声明行定位。 |
| [quantize_kv_transform](../../bench/tritonbench_g/quantize_kv_transform/QuantizeKvTransform.lean) | 指定 head-dimension/destination 布局的 KV 量化值和 scale。 | 补 max(abs)/127 的浮点舍入、scale dtype、division、int8 conversion 和输出存储的精确规格/对应；当前数学 cast 不等于硬件量化。 |
| [rbe_triton_transform](../../bench/tritonbench_g/rbe_triton_transform/RbeTritonTransform.lean) | 指定 token-frequency 模型的 rotary pair 输出。 | 补 native libdevice.pow、freq cast、tl.sin/tl.cos 和 pair 乘加的真实 dtype 语义与参考；实际 transcendental 实现需绑定而非直接 Real.sin/cos。 |
| [rope_backward_transform](../../bench/tritonbench_g/rope_backward_transform/RopeBackwardTransform.lean) | backward Q/K full-surface 值事实及独立 head-store IO。 | 保留源 load→fp32 转换、cos/sin 参数、反向旋转乘加和store转换，证明选定 full-surface/half-stage 的 FP 输出；没有独立 FP candidate。 |
| [rope_embedding](../../bench/tritonbench_g/rope_embedding/RopeEmbedding.lean) | 四个前/后向 half-store 契约及 full-surface lowering。 | 为每个已选 half-store 的输入 cast、乘法/加减、输出类型给独立 FP 结果/真实候选；不能凭数学旋转公式宣称浮点等价。 |
| [rope_transform](../../bench/tritonbench_g/rope_transform/RopeTransform.lean) | forward Q/K full-surface 值与独立 head-store IO。 | 补实际 fp32 load casts、正向旋转乘法/加减和store conversion 的 FP 执行/参考，并验证原始源对应。 |
| [rotary_emb_nopad](../../bench/tritonbench_g/rotary_emb_nopad/RotaryEmbNopad.lean) | 选定 Q 旋转与 cache 子表面、明确 half/head/metadata 条件。 | Q 子核需真实浮点乘加/cast 结果，cache 子核需 typed payload/address preservation；不能把两个实数子结论命名 FP 作为完成。 |
| [rotary_transform](../../bench/tritonbench_g/rotary_transform/RotaryTransform.lean) | 按 interleaved/varlen/offset 条件选定的 rotary row/body 与 metadata IO。 | 补源 fp32 cast、乘加、conjugate/sign selection、output store 的 FP 规格，保留实际选择路径和每个被计算 lane。 |
| [rotary_transform_ops](../../bench/tritonbench_g/rotary_transform_ops/RotaryTransformOps.lean) | 带 flags/paired-half/varlen metadata 的 rotary row/body IO。 | 补实际 load cast、multiply/add/sub/where 与store类型的 FP 执行/独立参考；现有 frontend/meta IO 是实数层结果。 |
| [rowwise_quantization_triton](../../bench/tritonbench_g/rowwise_quantization_triton/RowwiseQuantizationTriton.lean) | 包含行 absmax producer 与 scaled-real 输出，未覆盖 faithful llrint/final-int8。 | 当前阶段需真实 max/abs、scale division、multiply 和store 的 FP 规格；如果目标为 int8 结果，另补 llrint/narrowing 及零scale域。 |
| [token_softmax_bloom](../../bench/tritonbench_g/token_softmax_bloom/TokenSoftmaxBloom.lean) | 给定 logits/start/length 的逐序列 softmax。 | 补实际 fp32 load cast、max/sub/exp/sum/div 和store的 FP 顺序/reference；尤其 eager masked lane 和有限输入的非有限中间结果。 |
| [token_softmax_llama](../../bench/tritonbench_g/token_softmax_llama/TokenSoftmaxLlama.lean) | 给定 logits/start/length 的逐序列 softmax 与注入输出地址。 | 补 fp32 cast、max/sub/exp/sum/div 和store的 FP 执行及独立参考，保留实际 mask 与reduction顺序。 |
| [triton_argmax](../../bench/tritonbench_g/triton_argmax/TritonArgmax.lean) | 独立 first-stage value/index、second-stage 与 single-block dim argmax。 | 补浮点 max/argmax 的比较、tie-index、NaN/signed-zero 域约定与输出index宽度，建立已选阶段的 typed结果/真实非平凡候选。 |

这些任务尚无选定的独立FP规格/有实质变化的候选，具体局部实验关系为unknown。先建立准确的运算图和比较对象，再提取可复用或待实验的数值规律；不会按kernel名称猜测交换律、结合律或倒数律。

## 23项cast/store舍入模型任务

这些结论允许模型在cast/store处舍入，但普通算术仍使用实数运算，withPrecision不会建立完整的逐算子精度语义。必须补算术精度/舍入和相应结果证明；仅提供round_idem不能关闭这一层缺口。

| 任务 | 当前舍入模型声明 |
|---|---|
| `flash_decode2_llama` | [flash_decode2_llama_final_store_io_correctnessR](../../bench/tritonbench_g/flash_decode2_llama/FlashDecode2Llama.lean) |
| `flash_decode2_phi` | [flash_decode2_phi_final_store_io_correctnessR](../../bench/tritonbench_g/flash_decode2_phi/FlashDecode2Phi.lean) |
| `iv_dependent_matmul` | [iv_dependent_matmul_io_correctness](../../bench/tritonbench_g/iv_dependent_matmul/IvDependentMatmul.lean) |
| `layer_norm_liger` | [layer_norm_liger_scalar_stores_io_correctnessR](../../bench/tritonbench_g/layer_norm_liger/LayerNormLiger.lean) |
| `layer_norm_ops` | [layer_norm_ops_scalar_stores_io_correctnessR](../../bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean) |
| `layernorm_fwd_triton` | [layernorm_fwd_triton_io_correctness](../../bench/tritonbench_g/layernorm_fwd_triton/LayernormFwdTriton.lean) |
| `lightning_attention` | [lightning_attention_bwd_grad_store_io_correctnessR](../../bench/tritonbench_g/lightning_attention/LightningAttention.lean) |
| `lora_expand_gemv` | [gemv_full_io_correctness](../../bench/tritonbench_g/lora_expand_gemv/LoraExpandGemv.lean) |
| `matmul_kernel` | [matmul_kernel_io_correctness](../../bench/tritonbench_g/matmul_kernel/MatmulKernel.lean) |
| `matmul_leakyrelu` | [matmul_leakyrelu_io_correctness](../../bench/tritonbench_g/matmul_leakyrelu/MatmulLeakyrelu.lean) |
| `matmul_leakyrelu_fp8` | [matmul_leakyrelu_fp8_io_correctness](../../bench/tritonbench_g/matmul_leakyrelu_fp8/MatmulLeakyreluFp8.lean) |
| `matmul_tma` | [matmul_tma_f16_io_correctnessR](../../bench/tritonbench_g/matmul_tma/MatmulTma.lean) |
| `matmul_triton1` | [matmul_triton1_io_correctness](../../bench/tritonbench_g/matmul_triton1/MatmulTriton1.lean) |
| `matmul_triton2` | [matmul_triton2_io_correctness](../../bench/tritonbench_g/matmul_triton2/MatmulTriton2.lean) |
| `matmul_triton_autotune` | [matmul_autotune_io_correctness](../../bench/tritonbench_g/matmul_triton_autotune/MatmulTritonAutotune.lean) |
| `quantize_copy_kv` | [quantize_copy_kv_io_correctness](../../bench/tritonbench_g/quantize_copy_kv/QuantizeCopyKv.lean) |
| `sgmv_expand_slice` | [sgmv_expand_slice_io_correctness](../../bench/tritonbench_g/sgmv_expand_slice/SgmvExpandSlice.lean) |
| `softmax_reducev` | [softmax_reducev_io_correctness](../../bench/tritonbench_g/softmax_reducev/SoftmaxReducev.lean) |
| `token_attn_llama2` | [token_attn_llama2_io_correctness](../../bench/tritonbench_g/token_attn_llama2/TokenAttnLlama2.lean) |
| `token_attn_mistral` | [token_attn_mistral_io_correctness](../../bench/tritonbench_g/token_attn_mistral/TokenAttnMistral.lean) |
| `token_attn_reduceV` | [token_attn_reducev_io_correctness](../../bench/tritonbench_g/token_attn_reduceV/TokenAttnReduceV.lean) |
| `triton_linear_activation` | [triton_linear_activation_io_correctness](../../bench/tritonbench_g/triton_linear_activation/TritonLinearActivation.lean) |
| `triton_matmul` | [triton_matmul_f16_io_correctness](../../bench/tritonbench_g/triton_matmul/TritonMatmul.lean)<br>[triton_matmul_f8_io_correctness](../../bench/tritonbench_g/triton_matmul/TritonMatmul.lean) |

## 已有反例的3项

`attention_kernel_aligned`、`triton_attention`、`triton_conv2d_fwd` 的全域 `round_fp16 = id` 不能由普通fp16舍入满足。需要修改参考模型或为真实中间值证明精确适用域，不能靠增加抽样将全域假设测通。

## 实验与验收的关系

[五个已明确的局部实验配置](numeric_experiment_handoff.md)仍保留，但不能替代上述证明缺口。[主统计](numerical_obligations.md)按实际定理模型展示已有相对等价、条件等价、位解释器和未补齐的数学/舍入层结果。已有相对等价本身也不证明GPU对独立数值规格的绝对正确性。

所有逐项声明、接受范围、Python操作行号、源码SHA和审阅依据见 [机器可读缺口表](fp_proof_gaps.json)；原始数学组审阅见 [冻结审计](../review-checkpoints/2026-10-14/fp-model-audit/math-only-audit.json)。
