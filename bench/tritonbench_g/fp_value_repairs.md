# 原41项数学模型任务的值语义修复

原41项均已为先前接受的配置/阶段补充逐算子后端或位解释器合同，共新增 118 条公开记录。旧数学结论保留原标签。

**这不是41项GPU验收通过。** 参数化后端合同仍需绑定原语；软件位合同仍需对应真实编译器和设备。固定输入的例子单列，只验证该实例。混合任务含位合同，也不代表所有阶段都已摆脱后端回调。

每条合同检查实际执行成功及独立初始输入的结果/写回（辅助helper与固定例子按其实际声明），不把最终输出等式当前提。正常输入、配置、已接受的阶段范围继续保留。

[机器可读绑定清单](fp_value_native_tasks.json)、[实验运行说明](../../experiments/floating_point/value_validation/README.md)、[全部184项状态](numerical_obligations.md)。本轮没有GPU执行记录。

| 任务 | 参数化后端合同 | 参数化位合同 | 固定例子 | 已覆盖范围与保留边界 |
|---|---:|---:|---:|---|
| [fifth_order_sph_harmonics](fifth_order_sph_harmonics/FPCorrect.lean) | 1 | 1 | 0 | Full eleven-channel source forward, original operation order and constants, independent initial-memory values and complete frame; FP32 inputs/output, stride > 10 and active-input cover. Backward was outside the old accepted stage. Native separate rounding, coefficient lowering, FTZ/NaN and load/store binding remain pending. |
| [fused_layernorm_triton](fused_layernorm_triton/FPCorrect.lean) | 4 | 1 | 3 | All previously accepted normalize stages and general Welford helper/update, complete one-block reduction through an explicit ordered tree, actual FP32 equality guards, mean/rstd stores and full frame. Native helper version, concrete compiler reduction tree and rsqrt still need binding. Software arithmetic and helper bit examples do not discharge these bindings. |
| [fused_recurrent_delta](fused_recurrent_delta/FPCorrect.lean) | 12 | 12 | 1 | All twelve previously accepted fp32 value stages: output/vnew/state, dk headwise/scalar/correction, dv headwise/scalar, dbeta headwise/scalar and dq headwise/scalar. Full recurrent/reverse-loop and host composition remain outside the original stage scope. Concrete software reduction is sequential; native tree correspondence is unestablished. Software output scale=1; abstract per-operation output is parameterized by scale. |
| [fused_recurrent_hgrn](fused_recurrent_hgrn/FPCorrect.lean) | 4 | 4 | 1 | All four previously accepted fp32 stages: forward carry, backward dx, dx-two-step and dg predecessor branches. Complete recurrent/reverse loops and host composition were outside the accepted stages and remain outside. |
| [fused_rotary_embedding](fused_rotary_embedding/FPCorrect.lean) | 1 | 1 | 0 | Complete selected Q pair rotation and complete V leader guard/context-length/block-table/payload chain, with readback and frame. Other Python paths remain outside the prior selected value stages; scalar fp32 lowering and raw copy binding remain unestablished. |
| [fused_rwkv6_kernel](fused_rwkv6_kernel/FPCorrect.lean) | 5 | 1 | 1 | All three accepted state/output/conditional-final stages, including the actual conditional final-state store. Concrete output profile has scale=1 and sequential software reduction; state and conditional-final retain an explicit native exp primitive and mixed conversion graph. Full recurrent loops remain outside the previous stage scope. |
| [int4_matmul](int4_matmul/FPCorrect.lean) | 1 | 0 | 0 | Complete accepted SPLIT_K=1 loop with arbitrary trip count under the full-block K profile; signed packed nibbles, direct half dequantization, fp32 accumulator, half output and frame. Parametric GEMM retains explicit typed dot primitive. Software sequential dot witnesses are separate ordinary theorems, not universal native bit equality; compiler contraction and native dot/tree remain unbound. Accepted K = blockK * steps and pid1 = 0 remain required; arbitrary ragged K is not claimed. |
| [int8_dequant_matmul](int8_dequant_matmul/FPCorrect.lean) | 0 | 1 | 0 | Full original dynamic K loop and both bias branches, concrete signed8 dot/int32 wrap, fp32 integer conversion and ordered scale/divisor multiplies, half cast and optional half bias addition, active output/frame. FP32 scales and supplied runtime divisor, fp16 output/bias and ordinary shape/address/input typing premises remain the declared profile. Original SPLIT_K=1, EVEN_K=true and K=blockK*steps and pid1=0 remain the accepted profile. |
| [int8_matmul_kernel](int8_matmul_kernel/FPCorrect.lean) | 0 | 1 | 0 | Full original four packed-field outer iterations and arbitrary complete K-loop, signed input narrowing, unsigned packed-byte extraction, concrete int8 dot and int32 wrap, initial-memory output and frame. Positive block width, K = 4 * blockK * steps, physical int32 A/uint8 B and ordinary output injectivity remain the declared profile. |
| [int8_matmul_quantization](int8_matmul_quantization/FPCorrect.lean) | 0 | 2 | 0 | Complete two-pass dynamic-K quantization producer and complete dynamic-K GEMM consumer, real signed8 conversion and int32 wrap, concrete fp32 scale arithmetic and fp16 storage, active outputs and frame. Producer active float-to-int8 domain, fp32 source/B-scale, fp16 A-scale/output, ordinary address/shape premises remain; host composition and native/compiler binding are not claimed. |
| [int8_quantization](int8_quantization/FPCorrect.lean) | 1 | 1 | 0 | Accepted supplied-scale pre-rounding Q/K stage: fp32 multiply/divide, two stores and complete frame. Concrete bit profile covers K preScale=1; Q retains supplied literal primitive. Scale producer and final biased uint8/int8 result remain outside the previously accepted stage. Q coefficient lowering remains explicit. |
| [int_scaled_matmul](int_scaled_matmul/FPCorrect.lean) | 0 | 2 | 0 | Both original kernels: block-pointer raw K loop with genuine unspecified byte tails and int32 wrap; scaled dynamic K kernel with both EVEN_K branches, fp32 integer conversion/multiplication and actual toward-zero int32 destination, independent initial-state values/readback/frame. Kernel1 positive blockK, physical signed-byte storage and active output injectivity; kernel2 original K coverage/EVEN_K premises and active defined FP-to-int32 conversion. Native/compiler binding remains open; arbitrary inactive conversion policy does not affect active output. |
| [kcache_copy_triton](kcache_copy_triton/FPCorrect.lean) | 1 | 0 | 0 | Sequence-length-one split-x copy, five named machine storage formats plus an abstract .real carrier (no native fp64 binding). Multi-token launches remain outside the previously accepted stage. |
| [kv_cache_copy](kv_cache_copy/FPCorrect.lean) | 1 | 0 | 0 | Sequence-length-one K/V x-block, independent model storage channels: five named machine formats plus abstract .real, with no inferred native fp64 binding. Supplied context/block-table metadata, active-address injectivity and distinct caches remain the input scope. |
| [kv_cache_filling](kv_cache_filling/FPCorrect.lean) | 1 | 2 | 0 | Accepted metadata writeback plus complete unquantized K/V tile copies using actual fp32-to-uint8 toward-zero conversion, active-only definedness and independent value/frame. Quantizer producers and multi-token loop composition remain outside these supplied-metadata/tile stages. |
| [layer_norm_welfold](layer_norm_welfold/FPCorrect.lean) | 3 | 1 | 1 | Actual three-loop source for XBLOCK=1 and 0<rnumel<=RBLOCK, plus accepted arbitrary normalization chunk; original tails and alias-write order retained. Full source retains rsqrt primitive; bit closure is currently for the normalization chunk. Multi-block loops and native reduction tree remain outside this profile. |
| [logsumexp_fwd](logsumexp_fwd/FPCorrect.lean) | 2 | 0 | 1 | Accepted per-program row stage in both HAS_SCALE branches, including -inf padding, max/sub/exp/sum/log/add/store. Native max/exp/log and reduction order remain explicit primitives; the concrete scalar adapter uses scale=1. Host second-stage logsumexp is outside scope. |
| [masked_select](masked_select/FPCorrect.lean) | 1 | 0 | 0 | Selection-copy consuming supplied mask/prefix metadata; five named machine floating formats plus abstract .real and signed int64 payload (native width binding remains explicit). Prefix-sum producer remains outside the accepted stage; signed index widths and bounds need their normal native mapping. |
| [matmul_dequant_int4](matmul_dequant_int4/FPCorrect.lean) | 0 | 1 | 0 | Complete accepted standalone tile: signed32 shifts and eight nibbles, exact integer difference, true binary16 conversion/multiplication, both source bridge and independent initial-memory value/frame. Native integer shift/count/width and direct-half rounding lowering remain unbound; there is no additional model numerical law. |
| [matmul_dequantize](matmul_dequantize/FPCorrect.lean) | 2 | 1 | 0 | All three accepted stages: standalone binary16 dequantization, first full grouped-dequant GEMM with both NO_GROUPS branches, and second full GEMM. Parametric GEMM retains explicit typed dot primitive. Software sequential dot witnesses are separate ordinary theorems, not universal native bit equality; compiler contraction and native dot/tree remain unbound. The second GEMM stage has K = blockK * steps and pid1 = 0; the first stage retains its own ceil/tail behavior. |
| [matmul_dequantize_int4](matmul_dequantize_int4/FPCorrect.lean) | 1 | 0 | 0 | Full arbitrary K loop in both NO_GROUPS branches; original two half products and half subtraction, typed dot, fp32 accumulation, store and frame. Parametric GEMM retains explicit typed dot primitive. Software sequential dot witnesses are separate ordinary theorems, not universal native bit equality; compiler contraction and native dot/tree remain unbound. |
| [matrix_reduction](matrix_reduction/FPCorrect.lean) | 0 | 1 | 0 | Complete original single matrix tile, fp16/fp32, positive reduction width, non-NaN inputs. Actual half-to-single bit conversion, exact numerical maximum, selected input bit readback and full frame. Native max tree/tie/NaN semantics and fp16-to-fp32 lowering still need binding; signed-zero tie selection is explicit in the software model. |
| [matrix_vector_multip](matrix_vector_multip/FPCorrect.lean) | 1 | 1 | 0 | Full original loop under the accepted one-block profile: 0 < BLOCK_M and M <= BLOCK_M, including M=0. Initial +0, load casts, output cast and frame retained. The sequential software sum order is explicit; native reduction order and multi-block launches are not established. |
| [mixed_sparse_attention](mixed_sparse_attention/FPCorrect.lean) | 3 | 0 | 2 | Complete previously accepted fp16 active-program source scope: all eight fixed block iterations, one column iteration under BLOCK_N >= 16, normalization division, fp16 cast and masked output stores, plus the separately materialized fp32 scratch epilogue. Independent initial-memory per-operation recurrence and complete output memory/frame; preserves inactive updates and masked metadata undef. All used M and D callbacks require native binding, including fp16 load/casts, fp32 scalar operations/literals, exp2, maxnum, typed dot with default source options and reductions. The concrete fp32 division epilogue keeps toHalf as an explicit primitive; its rounding witness is not closed fp16 bit interpretation. Two maximum nodes have exact local source-graph restoration bridges tied to Triton core.py/semantic.py hashes. Scalar-cond shape normalization, address/domain bounds and native compilation/launch legality remain binding boundaries. |
| [parallel_attention](parallel_attention/FPCorrect.lean) | 2 | 1 | 1 | Complete previously accepted forward O/Z and backward DK/DV source stages at FP32 storage, both arbitrary loops, independent values and full frames. Native dot/reduction/contraction and operation binding remain pending. Only expressly instantiated software paths count as bit closure. Host composition and stages outside prior accepted scope are not inferred. |
| [parallel_retention_attention](parallel_retention_attention/FPCorrect.lean) | 2 | 0 | 0 | Complete previously accepted forward and backward source stages at FP32 storage, all recurrence loops, independent values and full memory frames. Native dot/reduction/contraction and operation binding remain pending. Only expressly instantiated software paths count as bit closure. Host composition and stages outside prior accepted scope are not inferred. |
| [quant_transpose_kernel](quant_transpose_kernel/FPCorrect.lean) | 0 | 1 | 0 | Previously accepted scaled/pre-rounding stage: concrete fp32 multiplication by scale and by 127, then masked transposed writeback and frame. Actual quantized integer conversion/output is outside the explicitly accepted pre-rounding stage and is not claimed. |
| [quantize_global](quantize_global/FPCorrect.lean) | 0 | 1 | 0 | Accepted pre-llrint scaled-output stage: actual fp32 multiplication by supplied scale and 127, readback and frame. llrint, final int8 cast and scale producer remain outside the previously accepted stage. |
| [quantize_kv_copy](quantize_kv_copy/FPCorrect.lean) | 0 | 1 | 0 | Complete accepted grouped stage: actual fp32 load/abs/max/two divisions, signed int8 toward-zero conversion, fp32 scale store, active readback and frame. Definedness requires only active integer conversions finite and in range; arbitrary inactive policy is proved irrelevant. Native max and division/cast lowering remain unbound. |
| [quantize_kv_transform](quantize_kv_transform/FPCorrect.lean) | 0 | 1 | 0 | Complete accepted fp16/fp32 transform: true abs/max/division, binary16 scale store/reload, toward-zero int8 conversion, active values and full frame. Active integer conversions require finite/in-range quotients; inactive policy independence is proved. Native scalar/cast/max bindings remain pending. |
| [rbe_triton_transform](rbe_triton_transform/FPCorrect.lean) | 1 | 0 | 0 | Entire previously accepted rotary pair, both independently masked stores, fp16 storage and fp32 computation. No closed bit model yet. Every reached M callback remains unbound: literal/fromNat, source and destination casts, pow/div/mul/add/sub and sin/cos. The independent initial-state formula retains these primitives and precision tags explicitly. |
| [rope_backward_transform](rope_backward_transform/FPCorrect.lean) | 1 | 1 | 0 | Full original Q/K source including BACKWARD_PASS=true and all four masked stores, fp32 inputs and coefficients. Same native/configuration and coefficient-producer boundary as rope_transform. |
| [rope_embedding](rope_embedding/FPCorrect.lean) | 0 | 1 | 0 | All four accepted independent forward/backward half-store paths, actual register casts and backward fneg; concrete initial-memory fp32 value and frame. Half-store means a vector half, not fp16 storage. Full in-place loop composition was outside the accepted independent half-store scope; native compiler/runtime binding remains open. |
| [rope_transform](rope_transform/FPCorrect.lean) | 1 | 1 | 0 | Full original Q/K source, both branches and all four masked stores, fp32 inputs and coefficients, arbitrary dimensions/strides. Precomputed coefficient producers and host grid/contiguity were outside the accepted scope; other format combinations are not covered by this fp32 contract. |
| [rotary_emb_nopad](rotary_emb_nopad/FPCorrect.lean) | 0 | 4 | 0 | All four accepted Q, K, V2-Q and V2-cache paths; masks, both lanes, typed metadata and initial-memory value/frame. Original V2 invalid-token cache addressing consumes undefined loaded metadata; this behavior is preserved and explicitly witnessed. Host/global composition remains outside accepted stage scope. |
| [rotary_transform](rotary_transform/FPCorrect.lean) | 0 | 1 | 0 | Selected paired-row full rotation, fixed/varlen metadata branches, fp32 per-operation software rounding. Full 2D/interleaved/conjugate/offset-tensor paths were outside the prior value scope. |
| [rotary_transform_ops](rotary_transform_ops/FPCorrect.lean) | 0 | 1 | 0 | Selected paired-row full rotation, fixed/varlen metadata branches, own source-shape correspondence and fp32 per-operation software rounding. Full 2D/interleaved/conjugate/offset-tensor paths were outside the prior value scope. |
| [rowwise_quantization_triton](rowwise_quantization_triton/FPCorrect.lean) | 1 | 1 | 0 | Complete accepted pre-llrint row stage with true abs, masked maximum, FP32 div/mul, row output and actual binary16 maximum store; independent initial-memory values and full frame. llrint and final int8 output remain outside the accepted stage. Native maximum NaN/tie policy, division, mixed casts, fusion and launch binding remain unverified. |
| [token_softmax_bloom](token_softmax_bloom/FPCorrect.lean) | 1 | 0 | 0 | Complete per-sequence row chain: typed load/padding, fp32 maximum/sub/exp/sum/div, target-format cast and masked output. Parametric storage; original fp16 profile included. Native exp/max/reduction tree/cast semantics and compiler lowering remain unbound. Full token-logit producers and host composition stay external. |
| [token_softmax_llama](token_softmax_llama/FPCorrect.lean) | 1 | 0 | 0 | Complete per-sequence row chain: typed load/padding, fp32 maximum/sub/exp/sum/div, target-format cast and masked output. Parametric storage; original fp32 profile included. Native exp/max/reduction tree/cast semantics and compiler lowering remain unbound. Full token-logit producers and host composition stay external. |
| [triton_argmax](triton_argmax/FPCorrect.lean) | 0 | 3 | 0 | First stage (both index-width branches), independent second stage, and dimension reduction in original single-block 0<N<=BN scope. fp16/fp32, non-NaN inputs, actual numeric order and lowest-lane ties including signed zero. Multi-block dimension-loop and cross-stage host composition were not accepted; native primitive/index-width lowering needs binding. |

## 逐声明及原语绑定

### fifth_order_sph_harmonics

原语配置：`unfused_fp32`；硬件对应：`not_established`。

- [`spherical_forward_typed_correctness`](fifth_order_sph_harmonics/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`spherical_forward_software_fp32`](fifth_order_sph_harmonics/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### fused_layernorm_triton

原语配置：`reduction_fp32`；硬件对应：`not_established`。

- [`normalize_source_fp32`](fused_layernorm_triton/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`normalize_software_fp32`](fused_layernorm_triton/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`normalize_nonempty_rounding`](fused_layernorm_triton/FPCorrect.lean)：`bit_interpreter_contract`，`fixed_nonempty_witness`。
- [`welford_helper_fp32`](fused_layernorm_triton/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`welford_reduction_fp32`](fused_layernorm_triton/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`welford_software_with_rsqrt`](fused_layernorm_triton/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`welford_nonempty_rounding`](fused_layernorm_triton/FPCorrect.lean)：`backend_value_contract`，`fixed_nonempty_witness`。
- [`welford_bit_examples`](fused_layernorm_triton/FPCorrect.lean)：`bit_interpreter_contract`，`fixed_nonempty_witness`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 实际浮点eq/lt/le/gt/ge比较与合同一致；特别核对NaN、signed-zero、Inf和比较结果控制的分支。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- exp/exp2/log/log2/pow/sin/cos/rsqrt的实际指令与精确回调绑定；按各合同实际使用的原语和输入域记录位输出，不能假设实数精确函数。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### fused_recurrent_delta

原语配置：`reduction_fp32`；硬件对应：`not_established`。

- [`output_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`vnew_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`state_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`vnew_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`output_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`state_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`output_nonempty_rounding`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`fixed_nonempty_witness`。
- [`dkHeadwise_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dkScalar_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dkCorrection_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dvHeadwise_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dvScalar_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dbetaHeadwise_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dbetaScalar_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dqHeadwise_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dqScalar_fp32_stage`](fused_recurrent_delta/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dkHeadwise_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`dkScalar_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`dkCorrection_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`dvHeadwise_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`dvScalar_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`dbetaHeadwise_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`dbetaScalar_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`dqHeadwise_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`dqScalar_software_fp32`](fused_recurrent_delta/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### fused_recurrent_hgrn

原语配置：`unfused_fp32`；硬件对应：`not_established`。

- [`forward_fp32_stage`](fused_recurrent_hgrn/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dx_fp32_stage`](fused_recurrent_hgrn/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dx_two_fp32_stage`](fused_recurrent_hgrn/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`dg_fp32_stage`](fused_recurrent_hgrn/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`forward_software_fp32`](fused_recurrent_hgrn/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`dx_software_fp32`](fused_recurrent_hgrn/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`dx_two_software_fp32`](fused_recurrent_hgrn/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`dg_software_fp32`](fused_recurrent_hgrn/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`forward_nonempty_rounding`](fused_recurrent_hgrn/FPCorrect.lean)：`bit_interpreter_contract`，`fixed_nonempty_witness`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### fused_rotary_embedding

原语配置：`unfused_fp32_and_raw_copy`；硬件对应：`not_established`。

- [`fused_rotary_q_fp32_correctness`](fused_rotary_embedding/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`fused_rotary_vcache_typed_correctness`](fused_rotary_embedding/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### fused_rwkv6_kernel

原语配置：`reduction_fp32`；硬件对应：`not_established`。

- [`state_fp32_stage`](fused_rwkv6_kernel/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`output_fp32_stage`](fused_rwkv6_kernel/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`conditional_final_fp32_stage`](fused_rwkv6_kernel/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`state_fp32_with_exp_primitive`](fused_rwkv6_kernel/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`conditional_final_fp32_with_exp_primitive`](fused_rwkv6_kernel/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`output_software_fp32`](fused_rwkv6_kernel/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`output_nonempty_rounding`](fused_rwkv6_kernel/FPCorrect.lean)：`bit_interpreter_contract`，`fixed_nonempty_witness`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- exp/exp2/log/log2/pow/sin/cos/rsqrt的实际指令与精确回调绑定；按各合同实际使用的原语和输入域记录位输出，不能假设实数精确函数。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### int4_matmul

原语配置：`packed_int4_binary16_typed_dot`；硬件对应：`not_established`。

- [`int4_matmul_fp16_correctness`](int4_matmul/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- packed解码、signedness、int8窄化、移位和int32点积累加符合固定宽解释器；核对wrap或饱和模式。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 浮点dot的输入/累加精度、原始操作次序及allow_tf32等配置与显式dot回调对应；普通位例不证明任意dot回调正确。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### int8_dequant_matmul

原语配置：`int8_dot_fp32_scale_fp16_store`；硬件对应：`not_established`。

- [`int8_dequant_matmul_fp_correctness`](int8_dequant_matmul/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- packed解码、signedness、int8窄化、移位和int32点积累加符合固定宽解释器；核对wrap或饱和模式。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### int8_matmul_kernel

原语配置：`signed_int8_dot_int32_wrap`；硬件对应：`not_established`。

- [`int8_matmul_word_correctness`](int8_matmul_kernel/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- packed解码、signedness、int8窄化、移位和int32点积累加符合固定宽解释器；核对wrap或饱和模式。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### int8_matmul_quantization

原语配置：`int8_quantizer_and_scaled_dot`；硬件对应：`not_established`。

- [`int8_matmul_quantization_producer_machine_correctness`](int8_matmul_quantization/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`int8_matmul_quantization_consumer_machine_correctness`](int8_matmul_quantization/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- packed解码、signedness、int8窄化、移位和int32点积累加符合固定宽解释器；核对wrap或饱和模式。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- maximum/argmax的比较、NaN策略、signed-zero与tie/index选择符合合同；不能用无条件compare+where替代maximum。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- 实际源浮点到目标有符号/无符号位宽的转换，在active输出的有限可表示域上符合toward-zero合同；域外lane任意policy的独立性单列。llrint不在此规则内。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### int8_quantization

原语配置：`unfused_fp32`；硬件对应：`not_established`。

- [`per_block_scaled_fp32`](int8_quantization/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`per_block_scaled_k_software`](int8_quantization/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### int_scaled_matmul

原语配置：`int8_dot_fp32_scale_int32_store`；硬件对应：`not_established`。

- [`block_pointer_word_correctness`](int_scaled_matmul/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`scaled_word_correctness`](int_scaled_matmul/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- packed解码、signedness、int8窄化、移位和int32点积累加符合固定宽解释器；核对wrap或饱和模式。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 实际源浮点到目标有符号/无符号位宽的转换，在active输出的有限可表示域上符合toward-zero合同；域外lane任意policy的独立性单列。llrint不在此规则内。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### kcache_copy_triton

原语配置：`raw_copy`；硬件对应：`not_established`。

- [`kcache_copy_typed_correctness`](kcache_copy_triton/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### kv_cache_copy

原语配置：`raw_copy`；硬件对应：`not_established`。

- [`kv_cache_copy_typed_correctness`](kv_cache_copy/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### kv_cache_filling

原语配置：`fp32_uint8_conversion`；硬件对应：`not_established`。

- [`kv_metadata_typed_correctness`](kv_cache_filling/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`kv_k_uint8_typed_correctness`](kv_cache_filling/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`kv_v_uint8_typed_correctness`](kv_cache_filling/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际源浮点到目标有符号/无符号位宽的转换，在active输出的有限可表示域上符合toward-zero合同；域外lane任意policy的独立性单列。llrint不在此规则内。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### layer_norm_welfold

原语配置：`reduction_fp32`；硬件对应：`not_established`。

- [`normalize_chunk_fp32`](layer_norm_welfold/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`normalize_software_fp32`](layer_norm_welfold/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`normalize_nonempty_rounding`](layer_norm_welfold/FPCorrect.lean)：`bit_interpreter_contract`，`fixed_nonempty_witness`。
- [`source_one_block_fp32`](layer_norm_welfold/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`source_one_block_fp32_with_rsqrt`](layer_norm_welfold/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- exp/exp2/log/log2/pow/sin/cos/rsqrt的实际指令与精确回调绑定；按各合同实际使用的原语和输入域记录位输出，不能假设实数精确函数。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### logsumexp_fwd

原语配置：`mixed_transcendental`；硬件对应：`not_established`。

- [`row_block_fp32`](logsumexp_fwd/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`row_block_fp32_with_primitives`](logsumexp_fwd/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`row_block_nonempty`](logsumexp_fwd/FPCorrect.lean)：`backend_value_contract`，`fixed_nonempty_witness`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- maximum/argmax的比较、NaN策略、signed-zero与tie/index选择符合合同；不能用无条件compare+where替代maximum。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- exp/exp2/log/log2/pow/sin/cos/rsqrt的实际指令与精确回调绑定；按各合同实际使用的原语和输入域记录位输出，不能假设实数精确函数。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### masked_select

原语配置：`raw_copy`；硬件对应：`not_established`。

- [`masked_select_typed_correctness`](masked_select/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### matmul_dequant_int4

原语配置：`packed_int4_binary16`；硬件对应：`not_established`。

- [`matmul_dequant_int4_fp16_correctness`](matmul_dequant_int4/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- packed解码、signedness、int8窄化、移位和int32点积累加符合固定宽解释器；核对wrap或饱和模式。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### matmul_dequantize

原语配置：`packed_int4_binary16_typed_dot`；硬件对应：`not_established`。

- [`matmul_dequantize_standalone_fp16_correctness`](matmul_dequantize/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`matmul_dequantize_second_fp16_correctness`](matmul_dequantize/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`matmul_dequantize_first_fp16_correctness`](matmul_dequantize/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- packed解码、signedness、int8窄化、移位和int32点积累加符合固定宽解释器；核对wrap或饱和模式。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 浮点dot的输入/累加精度、原始操作次序及allow_tf32等配置与显式dot回调对应；普通位例不证明任意dot回调正确。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### matmul_dequantize_int4

原语配置：`packed_int4_binary16_typed_dot`；硬件对应：`not_established`。

- [`matmul_dequantize_int4_fp16_correctness`](matmul_dequantize_int4/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- packed解码、signedness、int8窄化、移位和int32点积累加符合固定宽解释器；核对wrap或饱和模式。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 浮点dot的输入/累加精度、原始操作次序及allow_tf32等配置与显式dot回调对应；普通位例不证明任意dot回调正确。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### matrix_reduction

原语配置：`ordered_max`；硬件对应：`not_established`。

- [`matrix_reduction_machine_correctness`](matrix_reduction/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- maximum/argmax的比较、NaN策略、signed-zero与tie/index选择符合合同；不能用无条件compare+where替代maximum。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### matrix_vector_multip

原语配置：`reduction_fp32`；硬件对应：`not_established`。

- [`mv_one_block_typed_correctness`](matrix_vector_multip/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`mv_one_block_software_fp32`](matrix_vector_multip/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### mixed_sparse_attention

原语配置：`sparse_attention_fp16_fp32`；硬件对应：`not_established`。

- [`forward_fp16`](mixed_sparse_attention/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`forward_nonempty`](mixed_sparse_attention/FPCorrect.lean)：`backend_value_contract`，`fixed_nonempty_witness`。
- [`epilogue_fp16`](mixed_sparse_attention/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`epilogue_software_with_cast`](mixed_sparse_attention/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`epilogue_nonempty_rounding`](mixed_sparse_attention/FPCorrect.lean)：`backend_value_contract`，`fixed_nonempty_witness`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- maximum/argmax的比较、NaN策略、signed-zero与tie/index选择符合合同；不能用无条件compare+where替代maximum。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- exp/exp2/log/log2/pow/sin/cos/rsqrt的实际指令与精确回调绑定；按各合同实际使用的原语和输入域记录位输出，不能假设实数精确函数。
- 浮点dot的输入/累加精度、原始操作次序及allow_tf32等配置与显式dot回调对应；普通位例不证明任意dot回调正确。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### parallel_attention

原语配置：`fp32_typed_dot`；硬件对应：`not_established`。

- [`forward_fp32`](parallel_attention/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`backward_fp32`](parallel_attention/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`forward_software_fp32`](parallel_attention/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`forward_nonempty_rounding`](parallel_attention/FPCorrect.lean)：`bit_interpreter_contract`，`fixed_nonempty_witness`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 浮点dot的输入/累加精度、原始操作次序及allow_tf32等配置与显式dot回调对应；普通位例不证明任意dot回调正确。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### parallel_retention_attention

原语配置：`mixed_transcendental`；硬件对应：`not_established`。

- [`forward_fp32`](parallel_retention_attention/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`backward_fp32`](parallel_retention_attention/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- exp/exp2/log/log2/pow/sin/cos/rsqrt的实际指令与精确回调绑定；按各合同实际使用的原语和输入域记录位输出，不能假设实数精确函数。
- 浮点dot的输入/累加精度、原始操作次序及allow_tf32等配置与显式dot回调对应；普通位例不证明任意dot回调正确。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### quant_transpose_kernel

原语配置：`unfused_fp32`；硬件对应：`not_established`。

- [`quant_transpose_scaled_store_fp32_correctness`](quant_transpose_kernel/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### quantize_global

原语配置：`unfused_fp32`；硬件对应：`not_established`。

- [`quantize_global_scaled_fp32`](quantize_global/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### quantize_kv_copy

原语配置：`fp32_int8_conversion`；硬件对应：`not_established`。

- [`quantize_kv_group_fp32_int8_correctness`](quantize_kv_copy/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- maximum/argmax的比较、NaN策略、signed-zero与tie/index选择符合合同；不能用无条件compare+where替代maximum。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- 实际源浮点到目标有符号/无符号位宽的转换，在active输出的有限可表示域上符合toward-zero合同；域外lane任意policy的独立性单列。llrint不在此规则内。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### quantize_kv_transform

原语配置：`fp32_int8_conversion`；硬件对应：`not_established`。

- [`quantize_kv_transform_machine_correctness`](quantize_kv_transform/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- maximum/argmax的比较、NaN策略、signed-zero与tie/index选择符合合同；不能用无条件compare+where替代maximum。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- 实际源浮点到目标有符号/无符号位宽的转换，在active输出的有限可表示域上符合toward-zero合同；域外lane任意policy的独立性单列。llrint不在此规则内。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### rbe_triton_transform

原语配置：`mixed_transcendental`；硬件对应：`not_established`。

- [`rbe_triton_backend_correctness`](rbe_triton_transform/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- exp/exp2/log/log2/pow/sin/cos/rsqrt的实际指令与精确回调绑定；按各合同实际使用的原语和输入域记录位输出，不能假设实数精确函数。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### rope_backward_transform

原语配置：`unfused_fp32`；硬件对应：`not_established`。

- [`rope_backward_fp32_correctness`](rope_backward_transform/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`rope_backward_software_fp32`](rope_backward_transform/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### rope_embedding

原语配置：`unfused_fp32`；硬件对应：`not_established`。

- [`rope_embedding_fp32_correctness`](rope_embedding/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### rope_transform

原语配置：`unfused_fp32`；硬件对应：`not_established`。

- [`rope_fp32_correctness`](rope_transform/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`rope_software_fp32`](rope_transform/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### rotary_emb_nopad

原语配置：`unfused_fp32`；硬件对应：`not_established`。

- [`rotary_nopad_q_fp32_correctness`](rotary_emb_nopad/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`rotary_nopad_k_fp32_correctness`](rotary_emb_nopad/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`rotary_nopad_v2_q_fp32_correctness`](rotary_emb_nopad/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`rotary_nopad_v2_cache_fp32_correctness`](rotary_emb_nopad/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### rotary_transform

原语配置：`unfused_fp32`；硬件对应：`not_established`。

- [`rotary_transform_fp32_correctness`](rotary_transform/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### rotary_transform_ops

原语配置：`unfused_fp32`；硬件对应：`not_established`。

- [`rotary_transform_ops_fp32_correctness`](rotary_transform_ops/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### rowwise_quantization_triton

原语配置：`fp32_binary16_max`；硬件对应：`not_established`。

- [`rowwise_scaled_typed_correctness`](rowwise_quantization_triton/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。
- [`rowwise_scaled_software_correctness`](rowwise_quantization_triton/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- maximum/argmax的比较、NaN策略、signed-zero与tie/index选择符合合同；不能用无条件compare+where替代maximum。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### token_softmax_bloom

原语配置：`softmax_fp32`；硬件对应：`not_established`。

- [`token_softmax_typed_correctness`](token_softmax_bloom/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- maximum/argmax的比较、NaN策略、signed-zero与tie/index选择符合合同；不能用无条件compare+where替代maximum。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- exp/exp2/log/log2/pow/sin/cos/rsqrt的实际指令与精确回调绑定；按各合同实际使用的原语和输入域记录位输出，不能假设实数精确函数。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### token_softmax_llama

原语配置：`softmax_fp32`；硬件对应：`not_established`。

- [`token_softmax_typed_correctness`](token_softmax_llama/FPCorrect.lean)：`backend_value_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际逐步add/sub/mul/div/neg/abs与合同的软件值或指定后端回调对应；保留舍入、FTZ/DAZ、NaN、signed-zero和融合配置。普通除法与div_rn分别绑定。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- maximum/argmax的比较、NaN策略、signed-zero与tie/index选择符合合同；不能用无条件compare+where替代maximum。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- exp/exp2/log/log2/pow/sin/cos/rsqrt的实际指令与精确回调绑定；按各合同实际使用的原语和输入域记录位输出，不能假设实数精确函数。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

### triton_argmax

原语配置：`ordered_max`；硬件对应：`not_established`。

- [`argmax_stage1_bit_correctness`](triton_argmax/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`argmax_stage2_bit_correctness`](triton_argmax/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。
- [`argmax_dim_single_block_bit_correctness`](triton_argmax/FPCorrect.lean)：`bit_interpreter_contract`，`parametric_value_contract`。

待验证的对应关系：

- 原始load/store与mask/undef oracle保持指定存储位；数值cast单列。
- 实际源/目标格式转换与软件cast或指定回调对应；记录tie、overflow、subnormal及NaN payload规则。
- maximum/argmax的比较、NaN策略、signed-zero与tie/index选择符合合同；不能用无条件compare+where替代maximum。
- 原核的每个归约站点使用合同指定的相同树/次序、初值、算术精度与mask；软件顺序fold不自动对应GPU树。
- 仅核对合同实际到达的浮点字面量、padding/初值、±Inf及fromNat/fromInt站点：固定源整数宽度、目标precision、常数折叠与实际位编码；不能从算术探针推定这些转换正确。
- 编译原核和合同对应的具体shape/stride/constexpr配置，逐站点核对上述绑定、地址宽度和写回；组件样本不代替原核对应。

## 复核

```sh
python3 bench/fp_value_repairs.py --check
```

检查精确的原41项成员集合、当前源哈希、公开声明文本与模型/角色审阅一致性；不由文件名自动判断证明强度。
