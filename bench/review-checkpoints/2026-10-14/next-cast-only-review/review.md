# 剩余 23 项 cast/store-only 的下一批补证准备

只读审查；未改源、metadata 或验收分类，未编译、未运行 GPU。基线仍是 **23 cast/store-only、3 refuted、6 missing**。共检查 23 个任务的 63 条公开 specification；精确声明、源片段和 SHA 见 `source-evidence.json`。

建议顺序：**flash_decode2_llama → quantize_copy_kv → token_attn_llama2 → token_attn_reduceV → token_attn_mistral**。前三项分别建立简单 scalar阶段、复用现成量化链、建立无重排假设的乘法/求和profile；后两项复用完整 PV stream。

## 全部 23 项分组

| 组 | 数量 | 任务 | 缺口 |
|---|---:|---|---|
| supplied_scalar_stages | 1 | flash_decode2_llama | 给定中间值的 div / maximum / copy；原公开阶段不包含完整 stage2 exp 递推 |
| scalar_products_and_streams | 4 | token_attn_llama2, token_attn_reduceV, token_attn_mistral, lora_expand_gemv | 逐元素乘法、指定顺序归约与动态循环；没有 tl.dot，但不能消掉 padding 或改写成无序 Real 求和 |
| mixed_quantizer | 1 | quantize_copy_kv | half / single 格式转换、abs/max/div、half scale、signed int8 转换与双输出 |
| normalization | 3 | layer_norm_liger, layer_norm_ops, layernorm_fwd_triton | 均值/方差或梯度归约、sqrt/rsqrt/div 与输出；按各 headline 的 producer / supplied slice 分别覆盖 |
| online_exponential_stages | 2 | flash_decode2_phi, softmax_reducev | 运行 maximum、exp、scale/add、归约与最终 div；Phi 仅已接受 step / recurrence faces |
| floating_dot | 12 | iv_dependent_matmul, matmul_kernel, matmul_leakyrelu, matmul_leakyrelu_fp8, matmul_tma, matmul_triton1, matmul_triton2, matmul_triton_autotune, sgmv_expand_slice, triton_matmul, triton_linear_activation, lightning_attention | 浮点 dot 与累积顺序、混合输入/输出格式；不能以整数 dot 或 opaque callback 当具体浮点实现 |

## 每项实际公开范围与缺口

### flash_decode2_llama

归一化 store(Acc/SumExp)、running-max 单步、Final copy；完整 stage2 kernel 仅有 lowering conjunct。

div32; maximum32 的 NaN / zero / tie 选择; typed load/store。未纳入阶段无须虚构 exp。

公开依据：`bench/tritonbench_g/flash_decode2_llama/FlashDecode2Llama.lean:444::flash_decode2_llama_normalization_output_summary_general`; `bench/tritonbench_g/flash_decode2_llama/FlashDecode2Llama.lean:484::flash_decode2_llama_running_max_output_summary_general`; `bench/tritonbench_g/flash_decode2_llama/FlashDecode2Llama.lean:681::flash_decode2_llama_final_store_io_correctness`; `bench/tritonbench_g/flash_decode2_llama/FlashDecode2Llama.lean:769::flash_decode2_llama_final_store_io_correctnessR`

### token_attn_llama2

完整 Q/K score-stage：block_mask∈{0,1}、B_Loc gather、axis1 product sum、任意 sm_scale 与地址/维度、masked output。

mul32 +显式sum树+scale literal/operand转换+mul32；host scalar不能被默认 literal callback 偷偷留下。

公开依据：`bench/tritonbench_g/token_attn_llama2/TokenAttnLlama2.lean:1262::token_attn_llama2_output_summary_general`; `bench/tritonbench_g/token_attn_llama2/TokenAttnLlama2.lean:2709::token_attn_llama2_io_correctness`

### token_attn_reduceV

完整 suppliedProb/index PV stream；spbs=srts=1，B_N>0，S≤T*B_N 的 IO pre，任意循环/partialtail，完整 unmasked output vector。

mul/reduce/add、initial +0、outputcast；不得缩为单block或把 paddingadd/signzero当自动恒等。

公开依据：`bench/tritonbench_g/token_attn_reduceV/TokenAttnReduceV.lean:1540::token_attn_reducev_output_summary_general`; `bench/tritonbench_g/token_attn_reduceV/TokenAttnReduceV.lean:3238::token_attn_reducev_io_correctness`

### token_attn_mistral

完整 sliding-window PV stream；spbs=srts=1，B_N>0，任意 attSeq/window，attSeq≤T*B_N 的 IO pre，完整 unmasked output vector。

mul/reduce/add/cast；Prob 与 V masks不同，必须保留末块完整 B_N arithmetic，0*Inf/NaN不得删除。

公开依据：`bench/tritonbench_g/token_attn_mistral/TokenAttnMistral.lean:1783::token_attn_mistral_output_summary_general`; `bench/tritonbench_g/token_attn_mistral/TokenAttnMistral.lean:3681::token_attn_mistral_io_correctness`

### lora_expand_gemv

Full.bgmv_loop_surface：完整输出 n-loop，任意 split_n_length，B_N>0，K≤B_K，out≠lora，cn_stride>0；选定 ADD_INPUTS=false/CAST_TYPE=false 的 masked input active body，sentinel 已由阶段排除。

逐元素 mul + axis1 sum + store、全部 output loop/tail；fp16/bf16 输入与 sum 提升需显式绑定。原公开范围不是任意 Python branch。

公开依据：`bench/tritonbench_g/lora_expand_gemv/LoraExpandGemv.lean:1043::gemv_full_output_summary`; `bench/tritonbench_g/lora_expand_gemv/LoraExpandGemv.lean:2293::gemv_full_io_correctness`

### quantize_copy_kv

完整 accepted per-destination quantizer：每个 active head 的所有 BLOCK_DMODEL 列，max/127→half scale、src/scale→int8、双 store；hD>0及原两个 injectivity/region separation。

真实 half/single arithmetic、finite toward-zero signed8 conversion、NaN/overflow undefined policy；inactive heads独立。

公开依据：`bench/tritonbench_g/quantize_copy_kv/QuantizeCopyKv.lean:617::destindex_copy_quantize_kv_output_summary_general`; `bench/tritonbench_g/quantize_copy_kv/QuantizeCopyKv.lean:1383::quantize_copy_kv_io_correctness`

### layer_norm_liger

forward Y/Mean/RSTD 三输出和独立 supplied MeanPre/RSTDPre scalar stores；不包含 Python backward。

sum/mean/sub/square/variance/eps/rsqrt/affine 的各次舍入与三次 store；原 wrapper 的 W/B 行 stride 使用范围以实际目标 kernel 为准。

公开依据：`bench/tritonbench_g/layer_norm_liger/LayerNormLiger.lean:589::layer_norm_liger_forward_output_summary`; `bench/tritonbench_g/layer_norm_liger/LayerNormLiger.lean:803::layer_norm_liger_scalar_stores_io_correctness`; `bench/tritonbench_g/layer_norm_liger/LayerNormLiger.lean:921::layer_norm_liger_scalar_stores_io_correctnessR`

### layer_norm_ops

forward plain/rms/residual headlines 仅 supplied value/statistic/residual stores；backward C1/C2 reduction、DX-from-C、one-row DW/DB、residual add 双 store。

各 backward arithmetic producer 均要 typed 化；给出 forward copy 不能关闭整个任务，也无需把未接受的完整 forward producer加入本轮。

公开依据：`bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean:3903::layer_norm_ops_fwd_plain_bias_all_outputs_compute_correct_general`; `bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean:3936::layer_norm_ops_fwd_rms_bias_all_outputs_compute_correct_general`; `bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean:3965::layer_norm_ops_fwd_residual_bias_all_outputs_compute_correct_general`; `bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean:4011::layer_norm_ops_bwd_rms_core_outputs_compute_correct_general`; `bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean:4053::layer_norm_ops_bwd_plain_bias_core_outputs_compute_correct_general`; `bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean:4125::layer_norm_ops_bwd_residual_add_all_outputs_compute_correct_general`; `bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean:5499::layer_norm_ops_scalar_stores_io_correctness`; `bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean:5607::layer_norm_ops_scalar_stores_io_correctnessR`

### layernorm_fwd_triton

strided full N row 的均值/方差/输出多次循环，B>0、X≠Y、W≠Y；原 IO stream extent 与读写布局保留。

fp32 loop accumulation / reductions / div / sqrt / reciprocal / affine / y.to(output dtype)，所有 N 和 tail。

公开依据：`bench/tritonbench_g/layernorm_fwd_triton/LayernormFwdTriton.lean:2524::layernorm_fwd_triton_output_summary`; `bench/tritonbench_g/layernorm_fwd_triton/LayernormFwdTriton.lean:3899::layernorm_fwd_triton_io_correctness`

### flash_decode2_phi

masked accumulator step、sum-exp step、runningMaxAfter 初态/递推恒等式、masked normalization store、Final copy；full loop 仅 lowering。

sub/exp/mul/add/div/max 的真实 profile；不能把 runningMaxAfter 定义递推本身算实际循环执行。

公开依据：`bench/tritonbench_g/flash_decode2_phi/FlashDecode2Phi.lean:760::flash_decode2_phi_masked_accumulator_output_summary_general`; `bench/tritonbench_g/flash_decode2_phi/FlashDecode2Phi.lean:826::flash_decode2_phi_running_max_output_summary_general`; `bench/tritonbench_g/flash_decode2_phi/FlashDecode2Phi.lean:866::flash_decode2_phi_normalization_output_summary_general`; `bench/tritonbench_g/flash_decode2_phi/FlashDecode2Phi.lean:1051::flash_decode2_phi_final_store_io_correctness`; `bench/tritonbench_g/flash_decode2_phi/FlashDecode2Phi.lean:1143::flash_decode2_phi_final_store_io_correctnessR`

### softmax_reducev

完整 modeled在线softmax-weighted V stream；genuine summary 原有 seq>0、seq%B_N=0、running maximum finite-witness；IO 自己的 pre 必须一起保留。

max/sub/exp/sum/mul/add/div；保留 block carry 顺序与 masked metadata，不用 ideal exp identity 重排。

公开依据：`bench/tritonbench_g/softmax_reducev/SoftmaxReducev.lean:2624::softmax_reducev_genuine_output_compute_correct_general`; `bench/tritonbench_g/softmax_reducev/SoftmaxReducev.lean:4495::softmax_reducev_io_correctness`

### iv_dependent_matmul

pre-load IV-dependent 完整已选 K=BK*numKBlocks；hBK>0，SCN=1，BN≤SCM，保留原输入边缘/输出约束。

fp dot + 每轮 accumulator add + fp16 store；保留 preload 与 next-load 顺序。

公开依据：`bench/tritonbench_g/iv_dependent_matmul/IvDependentMatmul.lean:966::iv_dependent_matmul_closed_form_correct`; `bench/tritonbench_g/iv_dependent_matmul/IvDependentMatmul.lean:1956::iv_dependent_matmul_io_correctness`

### matmul_kernel

固定布局 kernel 的 BM/BN/BK/T 参数，BN≤4096；完整指定 K stream。

三参数 tl.dot(a,b,acc) 的 fused accumulate 语义、fp16 cast；不能把它当 dot 后独立 add。

公开依据：`bench/tritonbench_g/matmul_kernel/MatmulKernel.lean:712::matmul_kernel_closed_form_correct`; `bench/tritonbench_g/matmul_kernel/MatmulKernel.lean:1389::matmul_kernel_io_correctness`

### matmul_leakyrelu

grouped layout 下 K=BK*numKBlocks，BK>0，SCN=1，BN≤SCM 的已选 leaky-ReLU 路径。

dot 后 add、>=0 compare、0.01 的 fp literal 与 mul、where、half store。

公开依据：`bench/tritonbench_g/matmul_leakyrelu/MatmulLeakyrelu.lean:1078::matmul_leakyrelu_closed_form_correct`; `bench/tritonbench_g/matmul_leakyrelu/MatmulLeakyrelu.lean:2118::matmul_leakyrelu_io_correctness`

### matmul_leakyrelu_fp8

与原声明一致的 whole-K grouped matmul + leaky-ReLU，SCN=1，BN≤SCM；文件名不保证任何 fp8编码。

三参数 dot accumulate、compare/literal/mul、half output；输入真实格式必须按 source/profile 审核，不能因目录名用 f8e4。

公开依据：`bench/tritonbench_g/matmul_leakyrelu_fp8/MatmulLeakyreluFp8.lean:1078::matmul_leakyrelu_fp8_closed_form_correct`; `bench/tritonbench_g/matmul_leakyrelu_fp8/MatmulLeakyreluFp8.lean:2117::matmul_leakyrelu_fp8_io_correctness`

### matmul_tma

单个所建模 tile 的两个 output faces(fp32与fp16)，原 output injectivity；descriptor/TMA hardware 不在算法投影保证中。

实际浮点 dot + optionalhalfcast；source/projection bridge 与 descriptor/native memory 同步分开列。

公开依据：`bench/tritonbench_g/matmul_tma/MatmulTma.lean:482::matmul_tma_f32_closed_form_correct`; `bench/tritonbench_g/matmul_tma/MatmulTma.lean:627::matmul_tma_f16_closed_form_correct`; `bench/tritonbench_g/matmul_tma/MatmulTma.lean:883::matmul_tma_f32_io_correctness`; `bench/tritonbench_g/matmul_tma/MatmulTma.lean:1289::matmul_tma_f16_io_correctnessR`

### matmul_triton1

指定 contiguous surface 参数，K=BLOCK_K*numKBlocks，BLOCK_K>0，N≤NS。

dot(..., allow_tf32=False) 后 add，保留 disabled TF32 选项；非所有 GPU dot 都同一个顺序。

公开依据：`bench/tritonbench_g/matmul_triton1/MatmulTriton1.lean:651::matmul_triton1_closed_form_correct`; `bench/tritonbench_g/matmul_triton1/MatmulTriton1.lean:1328::matmul_triton1_io_correctness`

### matmul_triton2

grouped matmul whole-K，BK>0；summary injective cOffset，IO用 SCN=1 / BN≤SCM。

dot 后独立 accumulator add、最终 store dtype，所有 accepted grouped addresses/masks。

公开依据：`bench/tritonbench_g/matmul_triton2/MatmulTriton2.lean:993::matmul_triton2_output_summary_general`; `bench/tritonbench_g/matmul_triton2/MatmulTriton2.lean:1851::matmul_triton2_io_correctness`

### matmul_triton_autotune

K=BK*numKBlocks，SCN=1，BN≤SCM；ACTIVATION Bool 两分支均纳入；不验证 host tuner。

三参数 dot accumulate、optional leaky-ReLU compare/0.01mul、halfcast；两分支均不能遗漏。

公开依据：`bench/tritonbench_g/matmul_triton_autotune/MatmulTritonAutotune.lean:1150::matmul_autotune_closed_form_correct`; `bench/tritonbench_g/matmul_triton_autotune/MatmulTritonAutotune.lean:2175::matmul_autotune_io_correctness`

### sgmv_expand_slice

一个 sequence-row block；K=BK*numKBlocks，BK>0，原 output injectivity；保留 supplied seq/adapter metadata 和 slice offset。

mixed inputs dot、fp32 accumulate、output cast；当前 theorem 目标的 branch配置以 explicit surface 为准，不能用一块K witness替代完整 numKBlocks。

公开依据：`bench/tritonbench_g/sgmv_expand_slice/SgmvExpandSlice.lean:1062::sgmv_expand_slice_one_row_block_output_summary`; `bench/tritonbench_g/sgmv_expand_slice/SgmvExpandSlice.lean:2168::sgmv_expand_slice_io_correctness`

### triton_matmul

两个独立 output faces fp16 与 f8e4；K=BK*numKBlocks，SCN=1，BN≤SCM；whole-K grouped layout。

三参数dot accumulate、half rounding 与真实 e4m3fn/f8e4 边界；MixedBits half不足以关闭 fp8分支。

公开依据：`bench/tritonbench_g/triton_matmul/TritonMatmul.lean:1280::triton_matmul_f16_closed_form_correct`; `bench/tritonbench_g/triton_matmul/TritonMatmul.lean:1328::triton_matmul_f8_closed_form_correct`; `bench/tritonbench_g/triton_matmul/TritonMatmul.lean:2324::triton_matmul_f16_io_correctness`; `bench/tritonbench_g/triton_matmul/TritonMatmul.lean:2383::triton_matmul_f8_io_correctness`

### triton_linear_activation

完整 accepted K stream、HAS_BIAS/SHOULD_SAVE_ACT_INPUTS、ACTIVATION字符串路径，存 C和可选ACT_INPUTS；保留全layout和IO T>0条件。

dot+add+bias、relu/tanh/erf/fast_gelu 常数和运算次序、typed output；所有实际支持 activation 分支，不接受只 identity。

公开依据：`bench/tritonbench_g/triton_linear_activation/TritonLinearActivation.lean:1598::triton_linear_activation_output_summary_general`; `bench/tritonbench_g/triton_linear_activation/TritonLinearActivation.lean:2712::triton_linear_activation_io_correctness`

### lightning_attention

full fwd + two bwd surfaces 只有 lowering；数值结论是 forward KV carry step、o_inter dot、带输入 carry premises 的完整 causal forward body，及 supplied GradPre store。

dot(K,V)、dot(Q,KV)、QK causal mask 后乘 V、add/casts；把 carry premise 改为独立机器状态 invariant，而非假设最终输出。不能宣称 backward producer 已被这些 headlines 覆盖。

公开依据：`bench/tritonbench_g/lightning_attention/LightningAttention.lean:1547::lightning_attention_output_summary_general`; `bench/tritonbench_g/lightning_attention/LightningAttention.lean:1769::lightning_attention_bwd_grad_store_io_correctness`; `bench/tritonbench_g/lightning_attention/LightningAttention.lean:1846::lightning_attention_bwd_grad_store_io_correctnessR`

## 优先 5 项的实现范围

### 1. flash_decode2_llama

无已接受 dot/exp/whole-loop 数值证明需要迁移；div32直接 ScalarBackend，最大选择用现有 FP.lt/eq 建立独立 bit spec，copy复用 typed load/store。

完整接受范围：

- 保留全部 4 个 public headline：normalization_output_summary_general、running_max_output_summary_general、final_store_io_correctness、final_store_io_correctnessR。
- 任意 BLOCK_DMODEL、所有现有 batch/head/stride/block_seq_n 参数；归一化和copy保持输出offset injectivity；两步/拷贝来自各自 supplied 输入，不能把它们称为完整 stage2循环。
- Full flash_decode2_llama_surface 的 toAlgorithm? conjunct保留；Python未用的stride_mid_od/stride_mid_o_es/stride_od签名位置不擅自改变地址。

Dtype：本地4组调用 Mid_O/Mid_O_LogExpSum/O 均 fp32；新的 source/contract保留dtype参数或准确标注此原有 fp32 profile。Acc/SumExp/MaxLogic阶段寄存器用fp32编码；不能把未标记 .real当 fp32 cast=id。

独立规格：

- norm[i] = div_cfg(initial Acc[accOffset i], initial SumExp[sumOffset])，含 0/0、±Inf、NaN的具体软件语义；不额外加分母非零来省事。
- maxStep = 已明确 NaN/tie/±0策略的二元 maximum(initial tlogic, initial max)，选择位模式或独立 selection relation；软件策略与GPU绑定分列。
- copy[i] = initial Final[finalOffset i]，逐bit保留 + out frame。

需要的新模块：

- Proofs/FPTypedSource.lean — 保持原阶段数据流、所有 branch/loop/mask 与声明 dtype，并提供旧源/新typed源的可检查 projection。
- Proofs/FPContract.lean — 独立初态/stream reference，exec成功、全部输出readback及memory frame；不是 reference自身的rfl。
- Proofs/FPSoftware.lean — 明确每个实际使用 callback 的bit实现或标为backend-relative。
- FPCorrect.lean — 参数化公开入口，保留原accepted全范围；固定witness仅作回归。
- FPContract按三目标分条：Normalization / MaximumStep / FinalStore；source shape明确从full source对应fragment抽取，不新增完整exp递推声称。

可复用：`VeriTile/Triton/Float/ScalarBackend.lean`, `VeriTile/Triton/Float/ScalarOps.lean`, `VeriTile/Triton/Float/BitValue.lean`, `bench/tritonbench_g/matrix_reduction/Proofs/MachineMax.lean`

Native 实验/编译绑定原语：

- fp32 div 的rounding、FTZ/DAZ、0/0、Inf/NaN；PTX div还是rcp×mul lowering
- fp32 maximum 的NaN传播、sNaN quieting、signed-zero/tie选择；FP.lt 本身不等于 tl.maximum全部语义
- fp32typed load/store bit保持、metadata/address/mask绑定

额外审查点：

- ScalarBackend未实现 .max；必须显式补，不能继承 rest.binary .max 后标为bit实现。
- 该accepted阶段不使用exp；native清单不要按整个Python文件加入exp。

### 2. quantize_copy_kv

quantize_kv_transform 已有同型完整typed producer、MixedBits half↔fp32、max、signed8 conversion、双输出/frame；主要差异是transform多一个dimension mask。

完整接受范围：

- 保留2个public headline的全部功能：源tile→abs/max/127→真正fp16 scale→quotient→signedint8、目的地gather、值与scale两个maskedstore。
- 任意 BLOCK_HEAD/BLOCK_DMODEL/head_num、所有原strides与destination；原hD:0<BLOCK_DMODEL、Out≠OutScale、value/scaleoffset injectivity 保留。
- 每个activehead的所有 BLOCK_DMODEL 列均参与，原Python只有headmask，没有维度mask；不可新增dimension≤width限制或改变head/列范围。

Dtype：原本地输入 fp16、scale fp16、Out int8；复用 mixed carrier；若保留fp32输入补充profile须分列，不能抹掉fp16路径。原 host没有穷举dtype保证，测试dtype不是全域合同。

独立规格：

- loaded[h,d]按原headmask读取原状态半精度payload；abs清sign；按明确比较/NaN策略作非空max；scale=halfRound(div32(promote(max),127.0f))。
- quotient=源实际promotion下的div，scale再widen；integer=toSigned8TowardZero(quotient)的defined值，否则任意明示u(width,bits)。
- 两次独立scatter输出和frame；可对所有bits给total-policy reference，再给active-defined域下TowardZero+SignedRange且与u无关的加强结论。

需要的新模块：

- Proofs/FPTypedSource.lean — 原head-only mask与dtype固定source；定义到TypedTransform.Parameters的mapping，dimension=width。
- Proofs/FPProjection.lean — 对所有Fin width证明额外维度predicate恒真，证明source执行等价（不能仅声称源码文本一样）。
- Proofs/FPContract.lean — 复用或移植transform typed执行/readback/frame，补active-defined independence；若只给InputDefined结论须明列域，不能称无域全值覆盖。
- FPCorrect.lean — parameterized 两输出bit合同与source bridge；可单独FPWitness回归但不顶替主合同。

可复用：`bench/tritonbench_g/quantize_kv_transform/Proofs/TypedSource.lean`, `bench/tritonbench_g/quantize_kv_transform/FPCorrect.lean`, `bench/tritonbench_g/quantize_kv_transform/Proofs/MixedBits.lean`, `bench/tritonbench_g/matrix_reduction/Proofs/MachineMax.lean`, `VeriTile/Triton/Float/IntegerConversionTotal.lean`, `VeriTile/Triton/Float/ScalarBackend.lean`

Native 实验/编译绑定原语：

- half abs sign清除、max NaN/tie/zero及是否先提升精度
- half→fp32、fp32/127、fp32→half round/overflow/subnormal/NaNpayload、halfscale→fp32
- src/scale div的实际promotion和是否中间half再舍入；需编译trace/实验确认，不能由工具软件profile推断硬件
- float→signedint8 toward-zero及defined range [-128,127]；undefined/NaN/overflow结果只做policy绑定，不暗称saturating
- headmask、int8store、halfscale store、目的地元数据

额外审查点：

- inactivehead会产生0/0，不能要求无padding或把u=0称硬件结果。
- 全0 activehead/scale下溢0也会undefined；以total-policy合同保留执行，再将finite/range精确域单列。
- scalehalf路径必须是真32位中间+真16位编码，旧LegacyRealInt8仅历史数学桥。

### 3. token_attn_llama2

没有tl.dot或transcendental，只有fp32 elementwise product、axis1 reduction、scalar scale multiply和一次guardedstore；可复用scalar与显式sum解释器。

完整接受范围：

- 保留output_summary_general与io_correctness两个参数化合同；任意BD/BN/所有strides/kv_group_num/max_input_len/sm_scale与program ids。
- 完整13-statement prelude、block_mask∈{0,1} guard、带变长起点B_Loc gather、Kmask、sum(q*k,1)*sm_scale及outputmask/frame。
- IO T=1来自原guard不是人为单块假设；保持writeMask中的blockactive和windowactive双条件、原output injectivity。
- 数学summary的clean undef与IO pinning语义须在新typed状态对应，不以新增finite/nonzero输入限制替代。

Dtype：本地Q/K/Att_Out均fp32，scale为Python double计算的host标量；必须写清传参后scale实际fp32编码。保持source symbolic scalar contract，并给明确float32 profile；不把sm_scale固定为某个测试值。

独立规格：

- 对active block，独立从initial Q/B_Loc/K地址读取q,k；P[d]=mul_cfg(q[d],k[i,d])；r=EvalSumTree(plan[BD],P,+0)；score=mul_cfg(r,scaleBits)。
- sum保留全部BD叶与所有seed/padding；inactive block不执行、inactive output不写。
- reference独立于Structural.exec，但可参数化一个明确sum树；不要求不同树浮点相等，不使用add_assoc/mul_distrib。

需要的新模块：

- Proofs/FPTypedSource.lean — 保持原阶段数据流、所有 branch/loop/mask 与声明 dtype，并提供旧源/新typed源的可检查 projection。
- Proofs/FPContract.lean — 独立初态/stream reference，exec成功、全部输出readback及memory frame；不是 reference自身的rfl。
- Proofs/FPSoftware.lean — 明确每个实际使用 callback 的bit实现或标为backend-relative。
- FPCorrect.lean — 参数化公开入口，保留原accepted全范围；固定witness仅作回归。
- Proofs/FPSumProfile.lean — scalar bitops + explicit tree reduction，声明所有callback关闭；ScaleEncoding明确symbolic literal→runtimebits，未绑定前不得bit全覆盖。

可复用：`VeriTile/Triton/Float/ScalarBackend.lean`, `VeriTile/Triton/Float/ScalarOps.lean`, `VeriTile/Triton/Float/BitValue.lean`, `bench/tritonbench_g/matrix_vector_multip/Proofs/TypedSoftware.lean`

Native 实验/编译绑定原语：

- fp32mul与后置scale mul；逐操作rounding/FTZ/DAZ/NaN/±0
- tl.sum axis1按实际BD/BN/warp/layout的reductiontree及padding；是否有FMA fusion
- host sm_scale的runtime dtype/常数转换/取整（不是重新实验数学1/sqrt精度）
- maskedQ/K/gather/store与typed payload保持

额外审查点：

- ScalarBackend只有literal0/1，任意sm_scale仍走rest.literal；需新scale编码binding而非假设最终score正确。
- typed product/reduce compute precision必须用实际展开确认；sequentialSum可作为软件profile，不能自动叫GPU tree。

### 4. token_attn_reduceV

纯fp32 mul/sum/add，full-loop可沿已有forRangeDyn invariant；tokenllama建立sumprofile后复用。

完整接受范围：

- 保留两个 public headline、全部runtime sequence S与任意BD/BN，BN>0；保留spbs=1、srts=1、原outputoffset injectivity。
- 完整 range(0,S,BN)，包括S=0、任意多个block及partialtail；IO保持S≤T*BN，不以T=1或S%BN=0收窄。
- metadata B_req_idx/B_Start_Loc/B_Seqlen、Req_to_tokens gather、kv grouping、每步p/V mask、initialfp32+0、输出cast与完整unmasked vectorstore/frame。

Dtype：本地Prob/V/Out均fp32；source output dtype cast要保留，generic typed合同可参数化其他dtype，而明确软件bitprofile仅声明所实现dtype。

独立规格：

- acc0[d]=+0；对每个真实loopstep t，读取完整BN个mask后的p和v，product=mul_cfg，blocksum=EvalSumTree(plan[BN],product,+0)，acc_next=add_cfg(acc,blocksum)。
- 最后按Outdtype cast并写所有BD；reference来自initialstate/pins和明确递推，不能用旧Real Σ_live做FP目标。

需要的新模块：

- Proofs/FPTypedSource.lean — 保持原阶段数据流、所有 branch/loop/mask 与声明 dtype，并提供旧源/新typed源的可检查 projection。
- Proofs/FPContract.lean — 独立初态/stream reference，exec成功、全部输出readback及memory frame；不是 reference自身的rfl。
- Proofs/FPSoftware.lean — 明确每个实际使用 callback 的bit实现或标为backend-relative。
- FPCorrect.lean — 参数化公开入口，保留原accepted全范围；固定witness仅作回归。
- Proofs/FPComplete.lean — 任意动态tripcount归纳、全部tail、初始+0、最终dtypecast；可和mistral复用PV循环lemma但mask函数作为显式输入。

可复用：`VeriTile/Triton/Float/ScalarBackend.lean`, `VeriTile/Triton/Float/ScalarOps.lean`, `VeriTile/Triton/Float/BitValue.lean`, `bench/tritonbench_g/matrix_vector_multip/Proofs/TypedSoftware.lean`

Native 实验/编译绑定原语：

- fp32mul/add的rounding/FTZ/NaN/±0与是否fusion
- axis0 tl.sum的真实BN/tree/seed/padding，和跨blockaccadd顺序
- outputcast实际dtype、maskedvalue/defaultzero、元数据gather与outputstore

额外审查点：

- 末块所有BN算术必须保留；即使load双mask产生0也不能凭Real零恒等消去fp加法的signedzero效果。
- 独立证明最终output对未使用rest callback不依赖；没有显式sum实现不能标bit contract。

### 5. token_attn_mistral

与reduceV同一mul/sum/add主干，多一个window起点和不同的p/V mask；复用完整PVloop，不是另造短prefix。

完整接受范围：

- 保留两个public headline；spbs=srts=1、BN>0、原outputinjectivity；任意batchSeq、attSeq、sliding_window及strides/kvgroup。
- 完整range(0,attSeq,BN)，包括0和tail，IO只使用原attSeq≤T*BN；不新增attSeq=min(batchSeq,window)等未有关系。
- startIndex=max(batchSeq-sliding_window,0) 的元数据计算与B_Att_Start_Loc/B_req_idx/B_Att_Seqlen保持；Prob按attSeq mask，V/gather按startIndex+n<batchSeq mask；unmaskedfinalstore/frame。

Dtype：本地Prob/V/Out均fp32。整型window/metadata最大值是地址运算，不应把它误列为额外浮点maximum原语。

独立规格：

- acc0=+0；逐真实t对全部BN叶分别按两个不同mask构造p与v，mul_cfg后按axis0显式tree求和，再add_cfg到carry。
- 末块p=0但V可能active含Inf/NaN，0*Inf会NaN；必须保留这个源行为而非旧RealΣ_live或添加finiteV限制。
- attSeq=0时写initial+0的Outcast；全部BD输出与frame。

需要的新模块：

- Proofs/FPTypedSource.lean — 保持原阶段数据流、所有 branch/loop/mask 与声明 dtype，并提供旧源/新typed源的可检查 projection。
- Proofs/FPContract.lean — 独立初态/stream reference，exec成功、全部输出readback及memory frame；不是 reference自身的rfl。
- Proofs/FPSoftware.lean — 明确每个实际使用 callback 的bit实现或标为backend-relative。
- FPCorrect.lean — 参数化公开入口，保留原accepted全范围；固定witness仅作回归。
- Proofs/FPComplete.lean — 从reduceV的通用PVloop实例化独立pMask/vMask，证明window/gather索引与源完全相同。

可复用：`VeriTile/Triton/Float/ScalarBackend.lean`, `VeriTile/Triton/Float/ScalarOps.lean`, `VeriTile/Triton/Float/BitValue.lean`, `bench/tritonbench_g/matrix_vector_multip/Proofs/TypedSoftware.lean`

Native 实验/编译绑定原语：

- 与reduceV相同的fp32 mul/add、axis0sumtree、cast/load/store原语
- 实际pMask≠vMask的tail包含Inf/NaN回归；不以改变domain避开
- metadata subtraction/max/group division按原整数域绑定；这不是floatmax实验

额外审查点：

- 不得增加窗口与attSeq一致性前提来让不同mask变相等。
- 泛型backend递推先参数化覆盖，再具体bitprofile关闭每一used callback；fixedtail witness只做检查。

## 共同验收限制

- 保留原完整accepted stage/shape/loop/branch/dtype profile，不以singleblock、emptytile、固定witness替代参数化合同。
- 新增typedsource和独立value-spec必须实际绑定；旧Real evaluator或cast/store RoundingModel不构成机器算术闭合。
- 具体bit合同必须列出实际used primitive并证明不依赖未实现rest；backend-relative合同另记。
- Software reduction/dot schedule、NaNpayload、±0/tie、FTZ/DAZ以及promotion/constantconversion均需native binding；本审查没有做native实验。
- sum树直接执行即可，不需额外admitted标量重排律；禁止借ScalarReduction的associativity/distribution rules来自动证明不同FPtree相等。
- 固定样例能测非空性或边界，但不改变参数化保证层级。

## 现有摘要的精确读法

- flash_decode2_llama：旧report scope文字的producer/step泛称应读为两个分别有kernel的局部算术目标；fullstage只lowering。后续native不应误加exp。
- lightning_attention：旧report提到forward/backward producer/step容易让人误读；真实public数值只有上述forward slices及suppliedgradientstore，两个backward fullsurfaces只是lowering。
- layer_norm_ops：forward三类headline是suppliedstores；真正尚缺算术主要在backwardC1/C2/DX/DW/DB/residualadd，不应由forwardcopy完成去删除整项gap。

All 23 existing cast-store task rows and all 63 public specification statements were inventoried. Python @triton.jit source graphs and public theorem targets were inspected; the five candidates received detailed range/mask/dtype/profile review. This is a source-level plan, not a new proof or complete native compiler semantic audit.
