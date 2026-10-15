用户接受正常输入要求、配置限制及阶段结论。本轮应修改“验收解释与统计轴”，不能删除原有前提或把数学模型改名为 GPU 事实。

建议的统计轴：

| 轴 | 值 / 用途 |
|---|---|
| source_availability | present / missing；184 个目录，178 有源码，6 缺源码 |
| scope_kind | selected_full_kernel / configuration / stage / compatibility_projection / unsupported_full_source；保存原 coverage_level 和具体 scope 文本 |
| scope_user_acceptance | 对正常输入、配置、阶段、保留兼容投影结论记 accepted_for_this_review；missing_source 记 no_source_to_review；这个字段不表示数值假设已满足 |
| numerical_obligations | 按下述三种独立数值义务列清单，不按 specialization 或 admitted_fp 标签猜测 |
| mathematical_or_native_model | mathematical / abstract_rounding / opaque_backend / native_bound；它描述模型，不是一个待通过实验的布尔错误 |
| proof_check | 用现有全库构建与公开声明检查 receipt，保持和上述统计分离 |
| independent_replay | 原隔离环境阻塞保持单独事实，不应冒充数值实验缺口 |

数值义务的三层：

1. additional_theorem_law：实际用到的数值原子、native addition commutation、显式全域 round=id。每个原子按 rule+format+implementation+guard+Lean channel+binding key 去重；不要按调用点重复实验，也不要仅枚举 Rules 容器所有可选规则。现有公有头词法交叉核对为63条 Rules（45任务），1条 NativeAdditionCommutes（1任务），3条 round.fp16=id（3任务）。最终 atom 用量必须以实际 proof / definitions 追踪为准。
2. shared_rounding_model_condition：RoundingModel 的 round_real 和 round_idem 是33条声明/31任务共享模型结构。round_real 只是数学 .real 通道定义。round_idem 是“已经舍入的值再次舍入不变”，与“所有实数都不舍入”的 hfp16 不同。共性实验/硬件绑定可以复用，不需要每 kernel 重测。模型仅为 ℝ→ℝ，NaN、无穷、signed zero、溢出不自动被覆盖。
3. native_binding_boundary：理想 real、抽象 rounding、opaque typed backend 与具体 Triton/CUDA 实现的对应证据。对同一未改动上下文的结构等价不需要凭空补算术假设；对数学公式不能因“无额外 theorem 假设”而宣称 native IEEE 数值一致。每个桥接状态记录 not_claimed / existing_compile_only / scoped_experimental_binding / formal，不把429个mathematical声明全粗算“待实验”。

现有 scope 分类仍可作来源分类，不能继续呈现为101项“缺数值假设”。原分类是60 full、101 specialization、16 partial、1 blocked、6 missing。所有178有源码任务均有可按用户要求审查的完整/配置/阶段结论；这不是178个完整源程序全部功能已证明。逐阶段内容在 partial_stage_review.json。

16 partial 的精确解释：

- flash_attn：原兼容额外-cast family 的4条 projection 标签导致任务整体落到 partial，但已有 exact-source full_flat_correctness 和 actual_source_equivalence；新 scope 概览应该按声明展示，避免旧兼容声明压低新源码结论。
- flash_decode2_llama、flash_decode2_phi：stage2 producer/step 值结论和最终存储阶段；stage1 supplied。
- fused_layernorm_triton、layer_norm_welfold：归约/normalization 的分阶段结论及传入统计量的 IO 阶段。
- fused_recurrent_delta、fused_recurrent_hgrn、fused_rwkv6_kernel：单步/携带状态结论；hgrn 缺完整 reverse fold 的事实保留，但用户可验收现有单步结论。
- kv_cache_filling：unquantized copy 和 supplied metadata store；不含量化 cache producer。
- layer_norm_liger：forward 值与 supplied statistic stores；backward 不在当前声明内。
- layer_norm_ops：forward precomputed stores 与 backward one-row core/reduction statements；不是多row host composition。
- lightning_attention：producer/step 值结论和 supplied-gradient store。
- mixed_sparse_attention：sparse-layout value 与 supplied Acc/L epilogue。
- int8_quantization：ScalePre supplied 的 Q/K store stage，scale/bias/int8 rounding 不在现有值结论。
- quantize_global、rowwise_quantization_triton：pre-rounding scaled-store；后者 max producer 已包含。

quant_transpose_kernel 的两层必须同时展示：

- scaled real-store stage 可标 scope_user_acceptance=accepted_for_this_review（用户明确接受阶段限制）。
- full_source_blocked=true，原因是 faithful surface 的 llrint/int8 擦除实际返回 Except.error；blocked_summary 证明的正是这个错误事实与 companion slice 值结论的合取。现有阶段可以验收，完整量化源码未覆盖。实验测试 llrint/int8 不能使现有 Lean semantics 自动支持该指令；需要建模/桥接代码才可关闭这一 source-model 缺口。

不要修改原 coverage_level 以迁就新口径：它还有旧 manifest、spec-sheet、负面测试、source annotations 的一致性约束。新增轴并修改报告解释更稳妥；若更改原类别，必须同时修改对应源注释而不是只改 JSON。

已有误导性点：RoundingModel.lean 顶部和 structure doc 的“only real identity”与实际 round_idem 字段矛盾。建议只修正文档明确两项字段，不改语义。

独立核对没有运行 Lean 或 GPU，没有修改 root 文件。分析脚本和原始 public statement/hash 记录保留在本目录。
