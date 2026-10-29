# 183 题标准 Triton 源码入口复查

本轮检查唯一清单中全部 183 个 `kernel_revision.kernel_source`，并追踪转发入口实际选择的共享原程序及候选。保留实际选定的缓存版本；发现清单引用过期时，以已验收的源码、完整保持性证据和构建记录校正，不仅凭文件路径推断实际版本。注释、字符串中的 Python AST、未选用的历史宏及导入中的其他程序不计为当前源码。

确认 **89 题**存在 `ComputeKernel.mk inputs outputs (...).surfaceBody` 一类外层重建：**80 题在选定入口，9 题在实际共享源码**。此外，**1 题**的实际源码仍位于 `Proofs/`；**3 题**在源码文件保留了可归入 `Proofs/` 的程序分解助手。类别重叠后共 **92 题有本轮确认的整理项，91 题未发现本轮所查模式**。

此前“源码整理全部完成”的判断不准确，已经在唯一逐题清单中重新登记待整理；旧修复证据保留。本轮没有修改 kernel、证明、Python 输入契约或数值规则。整题证明计数保持 **182 已完成、1 带数值延期完成**；源码呈现问题本身不撤回已成立的证明。91 题未发现这些模式，也不等于所有源码和文件职责已获全面认证。

## 具体分类

- 外层直接包装，例如 `chunk_cumsum_kernel`、`kldiv_compute`：完整 `triton` 被取出程序体，再手工填端口。
- 模板实例重新包装，例如 `l2_norm_bwd`、`l2_norm_triton1`、`int_scaled_matmul`：调用已完整的 kernel 后，又用 `.mk` 重填端口。
- 共享源码中的包装，例如 `attention_score`、`chunk_retention`、`destindex_copy`：入口转发本身合理，整理实际共享定义即可，不复制源码。
- `matmul_leakyrelu`：原程序和已有候选分别经 `Whole.kernel`、`Whole.contractedKernel` 引用 `Proofs/FPTypedSource`、`Proofs/FPFusedSource` 的实际完整源码，需调整位置。
- `bmm_optimized`、`matmul_kernel`、`triton_matmul`：完整源码已经可读；`prelude/body/epilogue` 等分解及等式属于证明辅助。这里的切片并未用于偷换权威源码，不记作程序错误。

`chunked_cumsum_fwd` 的初筛曾命中旧缓存中位于 `Proofs/KernelSource` 的题级前端。继续核对已验收 delivery 和 combined-minimum generation 后，确认它早已移到题根 `Kernels.lean`，有 `source = previousSource` 的完整相等检查。本轮只纠正清单过期的 `kernel_source` 引用与对应修复摘要哈希，保留旧缓存作为历史证据；不重复报为源码未修复。依据为 `.veritile-cache/finish-pending-105-20261021/chunked_cumsum_fwd/{delivery,root-acceptance}.json` 和 `combined-minimum-welford-20261022/SourcePreservation.lean` 的既有通过记录。

## 后续整理的边界

外层包装目前承载输入/输出端口，不能批量删括号后只看编译结果。先核对共享 Metadata 对指针别名、外部指针重绑定、条件分支和端口次序的处理，再逐题证明新旧**完整 ComputeKernel 相等**，或提供明确语义连接，然后针对性构建主入口。上一轮 `matrix_transpose` 的完整相等证据保留，该题本轮无此包装。

真实多阶段组合（如 `rms_matmul_rbe.qkv`）、透明共享源码和兼容转发宏保留。`spinning_lock_reduction`、`streamk_matmul` 的完整源码仍采用题级 reader/`List Statement`；这与隐藏源码生成宏不同，本轮单列说明，既不算外层包装，也不宣称其接口已统一为标准 `ComputeKernel := triton`。不因本次扫描新增硬件、host、dtype 或数值实验任务。

本轮仅做源码结构检查和登记，没有运行 Lean 构建、全源码独立回放或 GPU 实测。机器可读的位置、源码哈希与逐题结论在 [audit.json](audit.json)；`python bench/review-checkpoints/2026-10-22/direct-triton-scan/scan.py --check` 可复查记录是否漂移。该脚本使用明确审阅过的转发路线，不是自动语义分类器。

## 逐题结果

| Task | 本轮发现 | 具体位置 |
| --- | --- | --- |
| `adam_update_triton` | 外层重建 | [Kernels.lean:92](../../../repairs/tritonbench_g/adam_update_triton/Kernels.lean#L92)；[Kernels.lean:97](../../../repairs/tritonbench_g/adam_update_triton/Kernels.lean#L97) |
| `add_example` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/add_example/Kernels.lean#L1) |
| `add_value` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/add_value/Kernels.lean#L1) |
| `apply_penalty` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/apply_penalty/Kernels.lean#L1) |
| `attention_forward_triton` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/attention_forward_triton/Kernels.lean#L1) |
| `attention_fwd_triton1` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/attention_fwd_triton1_tile_checked/Kernels.lean#L1) |
| `attention_fwd_triton2` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/attention_fwd_triton2/Kernels.lean#L1) |
| `attention_fwd_triton3` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/attention_fwd_triton3_scalar_cast/Kernels.lean#L1) |
| `attention_kernel` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/attention_kernel/Kernels.lean#L1) |
| `attention_kernel_aligned` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/attention_kernel_aligned/Kernels.lean#L1) |
| `attention_llama` | 外层重建 | [Kernels.lean:23](../../../repairs/tritonbench_g/attention_llama_scalar_abi_checked/Kernels.lean#L23)；[Kernels.lean:91](../../../repairs/tritonbench_g/attention_llama_scalar_abi_checked/Kernels.lean#L91)；[Kernels.lean:164](../../../repairs/tritonbench_g/attention_llama_scalar_abi_checked/Kernels.lean#L164)；[Kernels.lean:232](../../../repairs/tritonbench_g/attention_llama_scalar_abi_checked/Kernels.lean#L232) |
| `attention_score` | 外层重建 | [SourceDefinitions.lean:20](../../../repairs/tritonbench_g/attention_score_tile_checked/SourceDefinitions.lean#L20)；[SourceDefinitions.lean:104](../../../repairs/tritonbench_g/attention_score_tile_checked/SourceDefinitions.lean#L104)；[SourceDefinitions.lean:197](../../../repairs/tritonbench_g/attention_score_tile_checked/SourceDefinitions.lean#L197) |
| `attn_fwd_causal` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/attn_fwd_causal/Kernels.lean#L1) |
| `attn_fwd_triton` | 外层重建 | [Kernels.lean:24](../../../repairs/tritonbench_g/attn_fwd_triton_tile_checked/Kernels.lean#L24) |
| `batched_vecmat_mult` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/batched_vecmat_mult/Kernels.lean#L1) |
| `bgmv_expand_slice` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/bgmv_expand_slice_physical_tile_checked/Kernels.lean#L1) |
| `bgmv_shrink_kernel` | 外层重建 | [Kernels.lean:16](../../../repairs/tritonbench_g/bgmv_shrink_kernel_physical/Kernels.lean#L16)；[Kernels.lean:57](../../../repairs/tritonbench_g/bgmv_shrink_kernel_physical/Kernels.lean#L57)；[Kernels.lean:98](../../../repairs/tritonbench_g/bgmv_shrink_kernel_physical/Kernels.lean#L98)；[Kernels.lean:139](../../../repairs/tritonbench_g/bgmv_shrink_kernel_physical/Kernels.lean#L139) |
| `block_sparse_attn` | 外层重建 | [Kernels.lean:52](../../../repairs/tritonbench_g/block_sparse_attn/KernelOnly/Kernels.lean#L52)；[Kernels.lean:148](../../../repairs/tritonbench_g/block_sparse_attn/KernelOnly/Kernels.lean#L148) |
| `bmm_chunk_bwd` | 外层重建 | [Kernels.lean:98](../../../repairs/tritonbench_g/bmm_chunk_bwd/Kernels.lean#L98)；[Kernels.lean:147](../../../repairs/tritonbench_g/bmm_chunk_bwd/Kernels.lean#L147) |
| `bmm_chunk_fwd` | 外层重建 | [Kernels.lean:18](../../../repairs/tritonbench_g/bmm_chunk_fwd/Kernels.lean#L18)；[Kernels.lean:78](../../../repairs/tritonbench_g/bmm_chunk_fwd/Kernels.lean#L78)；[Kernels.lean:160](../../../repairs/tritonbench_g/bmm_chunk_fwd/Kernels.lean#L160) |
| `bmm_optimized` | 外层重建、分解助手归位 | [Kernels.lean:23](../../../repairs/tritonbench_g/bmm_optimized_grid_axes_checked/Kernels.lean#L23)；[Kernels.lean:75](../../../repairs/tritonbench_g/bmm_optimized_grid_axes_checked/Kernels.lean#L75)；[Kernels.lean:126](../../../repairs/tritonbench_g/bmm_optimized_grid_axes_checked/Kernels.lean#L126)；[Kernels.lean:177](../../../repairs/tritonbench_g/bmm_optimized_grid_axes_checked/Kernels.lean#L177)；其余 13 处见 audit.json |
| `cache_transform_triton` | 外层重建 | [Kernels.lean:18](../../../repairs/tritonbench_g/cache_transform_triton/Kernels.lean#L18)；[Kernels.lean:38](../../../repairs/tritonbench_g/cache_transform_triton/Kernels.lean#L38) |
| `chunk_bwd_dqkg` | 外层重建 | [Kernels.lean:33](../../../repairs/tritonbench_g/chunk_bwd_dqkg/Kernels.lean#L33)；[Kernels.lean:113](../../../repairs/tritonbench_g/chunk_bwd_dqkg/Kernels.lean#L113)；[Kernels.lean:193](../../../repairs/tritonbench_g/chunk_bwd_dqkg/Kernels.lean#L193)；[Kernels.lean:273](../../../repairs/tritonbench_g/chunk_bwd_dqkg/Kernels.lean#L273)；其余 15 处见 audit.json |
| `chunk_cumsum_kernel` | 外层重建 | [Kernels.lean:12](../../../repairs/tritonbench_g/chunk_cumsum_kernel/Kernels.lean#L12)；[Kernels.lean:30](../../../repairs/tritonbench_g/chunk_cumsum_kernel/Kernels.lean#L30) |
| `chunk_cumsum_vector` | 外层重建 | [Kernels.lean:54](../../../repairs/tritonbench_g/chunk_cumsum_vector/Kernels.lean#L54)；[Kernels.lean:88](../../../repairs/tritonbench_g/chunk_cumsum_vector/Kernels.lean#L88) |
| `chunk_delta_fwd` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/chunk_delta_fwd_static_dot/Kernels.lean#L1) |
| `chunk_gate_recurrence` | 外层重建 | [Kernels.lean:14](../../../repairs/tritonbench_g/chunk_gate_recurrence/Kernels.lean#L14)；[Kernels.lean:47](../../../repairs/tritonbench_g/chunk_gate_recurrence/Kernels.lean#L47)；[Kernels.lean:82](../../../repairs/tritonbench_g/chunk_gate_recurrence/Kernels.lean#L82)；[Kernels.lean:136](../../../repairs/tritonbench_g/chunk_gate_recurrence/Kernels.lean#L136) |
| `chunk_gated_attention` | 外层重建 | [Kernels.lean:17](../../../repairs/tritonbench_g/chunk_gated_attention/Kernels.lean#L17)；[Kernels.lean:41](../../../repairs/tritonbench_g/chunk_gated_attention/Kernels.lean#L41)；[Kernels.lean:78](../../../repairs/tritonbench_g/chunk_gated_attention/Kernels.lean#L78)；[Kernels.lean:142](../../../repairs/tritonbench_g/chunk_gated_attention/Kernels.lean#L142)；其余 52 处见 audit.json |
| `chunk_gla_fwd` | 外层重建 | [Kernels.lean:40](../../../repairs/tritonbench_g/chunk_gla_fwd/Kernels.lean#L40)；[Kernels.lean:89](../../../repairs/tritonbench_g/chunk_gla_fwd/Kernels.lean#L89)；[Kernels.lean:234](../../../repairs/tritonbench_g/chunk_gla_fwd/Kernels.lean#L234)；[Kernels.lean:269](../../../repairs/tritonbench_g/chunk_gla_fwd/Kernels.lean#L269)；其余 6 处见 audit.json |
| `chunk_gla_simple` | 外层重建 | [Kernels.lean:15](../../../repairs/tritonbench_g/chunk_gla_simple_min_dot_tile/Kernels.lean#L15)；[Kernels.lean:52](../../../repairs/tritonbench_g/chunk_gla_simple_min_dot_tile/Kernels.lean#L52) |
| `chunk_linear_attn` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/chunk_linear_attn/Kernels.lean#L1) |
| `chunk_retention` | 外层重建 | [Kernels.lean:15](../../../repairs/tritonbench_g/chunk_retention/Whole001/Kernels.lean#L15)；[Kernels.lean:51](../../../repairs/tritonbench_g/chunk_retention/Whole001/Kernels.lean#L51)；[Kernels.lean:85](../../../repairs/tritonbench_g/chunk_retention/Whole001/Kernels.lean#L85)；[Kernels.lean:118](../../../repairs/tritonbench_g/chunk_retention/Whole001/Kernels.lean#L118)；其余 1 处见 audit.json |
| `chunk_retention_ops` | 外层重建 | [Kernels.lean:18](../../../repairs/tritonbench_g/chunk_retention_ops/Kernels.lean#L18)；[Kernels.lean:56](../../../repairs/tritonbench_g/chunk_retention_ops/Kernels.lean#L56)；[Kernels.lean:97](../../../repairs/tritonbench_g/chunk_retention_ops/Kernels.lean#L97)；[Kernels.lean:135](../../../repairs/tritonbench_g/chunk_retention_ops/Kernels.lean#L135) |
| `chunked_cumsum_fwd` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/chunked_cumsum_fwd/Kernels.lean#L1) |
| `context_attn_bloom` | 外层重建 | [Kernels.lean:24](../../../repairs/tritonbench_g/context_attn_bloom/Kernels.lean#L24)；[Kernels.lean:123](../../../repairs/tritonbench_g/context_attn_bloom/Kernels.lean#L123)；[Kernels.lean:223](../../../repairs/tritonbench_g/context_attn_bloom/Kernels.lean#L223) |
| `context_attn_fwd` | 外层重建 | [Kernels.lean:20](../../../repairs/tritonbench_g/context_attn_fwd/Kernels.lean#L20)；[Kernels.lean:95](../../../repairs/tritonbench_g/context_attn_fwd/Kernels.lean#L95) |
| `context_attn_llama` | 外层重建 | [Kernels.lean:23](../../../repairs/tritonbench_g/context_attn_llama/KernelOnly/Kernels.lean#L23)；[Kernels.lean:105](../../../repairs/tritonbench_g/context_attn_llama/KernelOnly/Kernels.lean#L105) |
| `context_attn_mistral` | 外层重建 | [Kernels.lean:18](../../../repairs/tritonbench_g/context_attn_mistral/KernelOnly/Kernels.lean#L18)；[Kernels.lean:104](../../../repairs/tritonbench_g/context_attn_mistral/KernelOnly/Kernels.lean#L104) |
| `context_attn_nopad` | 外层重建 | [Kernels.lean:18](../../../repairs/tritonbench_g/context_attn_nopad/KernelOnly/Kernels.lean#L18)；[Kernels.lean:98](../../../repairs/tritonbench_g/context_attn_nopad/KernelOnly/Kernels.lean#L98) |
| `cosine_compute` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/cosine_compute/Kernels.lean#L1) |
| `cross_entropy1` | 外层重建 | [Kernels.lean:49](../../../repairs/tritonbench_g/cross_entropy1/Kernels.lean#L49)；[Kernels.lean:105](../../../repairs/tritonbench_g/cross_entropy1/Kernels.lean#L105)；[Kernels.lean:148](../../../repairs/tritonbench_g/cross_entropy1/Kernels.lean#L148)；[Kernels.lean:204](../../../repairs/tritonbench_g/cross_entropy1/Kernels.lean#L204) |
| `cross_entropy2` | 外层重建 | [Kernels.lean:22](../../../repairs/tritonbench_g/cross_entropy2/Kernels.lean#L22)；[Kernels.lean:89](../../../repairs/tritonbench_g/cross_entropy2/Kernels.lean#L89)；[Kernels.lean:136](../../../repairs/tritonbench_g/cross_entropy2/Kernels.lean#L136)；[Kernels.lean:203](../../../repairs/tritonbench_g/cross_entropy2/Kernels.lean#L203) |
| `cross_entropy_ops` | 外层重建 | [Kernels.lean:22](../../../repairs/tritonbench_g/cross_entropy_ops/Kernels.lean#L22)；[Kernels.lean:87](../../../repairs/tritonbench_g/cross_entropy_ops/Kernels.lean#L87)；[Kernels.lean:134](../../../repairs/tritonbench_g/cross_entropy_ops/Kernels.lean#L134)；[Kernels.lean:199](../../../repairs/tritonbench_g/cross_entropy_ops/Kernels.lean#L199) |
| `decay_cumsum` | 外层重建 | [Kernels.lean:55](../../../repairs/tritonbench_g/decay_cumsum/Kernels.lean#L55)；[Kernels.lean:73](../../../repairs/tritonbench_g/decay_cumsum/Kernels.lean#L73)；[Kernels.lean:91](../../../repairs/tritonbench_g/decay_cumsum/Kernels.lean#L91)；[Kernels.lean:119](../../../repairs/tritonbench_g/decay_cumsum/Kernels.lean#L119)；其余 2 处见 audit.json |
| `dequantize_matmul` | 外层重建 | [Kernels.lean:10](../../../repairs/tritonbench_g/dequantize_matmul/Kernels.lean#L10)；[Kernels.lean:48](../../../repairs/tritonbench_g/dequantize_matmul/Kernels.lean#L48) |
| `dequantize_rowwise` | 外层重建 | [Kernels.lean:14](../../../repairs/tritonbench_g/dequantize_rowwise/Kernels.lean#L14)；[Kernels.lean:29](../../../repairs/tritonbench_g/dequantize_rowwise/Kernels.lean#L29) |
| `destindex_copy` | 外层重建 | [Kernels.lean:36](../../../repairs/tritonbench_g/destindex_copy_kv1/Kernels.lean#L36)；[Kernels.lean:75](../../../repairs/tritonbench_g/destindex_copy_kv1/Kernels.lean#L75) |
| `destindex_copy_kv1` | 外层重建 | [Kernels.lean:36](../../../repairs/tritonbench_g/destindex_copy_kv1/Kernels.lean#L36)；[Kernels.lean:75](../../../repairs/tritonbench_g/destindex_copy_kv1/Kernels.lean#L75) |
| `destindex_copy_kv2` | 外层重建 | [Kernels.lean:29](../../../repairs/tritonbench_g/destindex_copy_kv2/Kernels.lean#L29)；[Kernels.lean:48](../../../repairs/tritonbench_g/destindex_copy_kv2/Kernels.lean#L48) |
| `diag_ssm_triton` | 外层重建 | [Kernels.lean:18](../../../repairs/tritonbench_g/diag_ssm_triton/Kernels.lean#L18)；[Kernels.lean:35](../../../repairs/tritonbench_g/diag_ssm_triton/Kernels.lean#L35)；[Kernels.lean:64](../../../repairs/tritonbench_g/diag_ssm_triton/Kernels.lean#L64)；[Kernels.lean:87](../../../repairs/tritonbench_g/diag_ssm_triton/Kernels.lean#L87)；其余 4 处见 audit.json |
| `dropout_triton` | 外层重建 | [Kernels.lean:13](../../../repairs/tritonbench_g/dropout_triton/Kernels.lean#L13)；[Kernels.lean:26](../../../repairs/tritonbench_g/dropout_triton/Kernels.lean#L26) |
| `embedding_triton_kernel` | 外层重建 | [Kernels.lean:22](../../../repairs/tritonbench_g/embedding_triton_kernel/Kernels.lean#L22)；[Kernels.lean:49](../../../repairs/tritonbench_g/embedding_triton_kernel/Kernels.lean#L49) |
| `f8_conversion_utils` | 外层重建 | [Kernels.lean:20](../../../tritonbench_g/f8_conversion_utils/Kernels.lean#L20)；[Kernels.lean:29](../../../tritonbench_g/f8_conversion_utils/Kernels.lean#L29)；[Kernels.lean:38](../../../tritonbench_g/f8_conversion_utils/Kernels.lean#L38)；[Kernels.lean:47](../../../tritonbench_g/f8_conversion_utils/Kernels.lean#L47)；其余 12 处见 audit.json |
| `fast_ce_loss` | 外层重建 | [Kernels.lean:17](../../../repairs/tritonbench_g/fast_ce_loss/Kernels.lean#L17)；[Kernels.lean:43](../../../repairs/tritonbench_g/fast_ce_loss/Kernels.lean#L43)；[Kernels.lean:71](../../../repairs/tritonbench_g/fast_ce_loss/Kernels.lean#L71)；[Kernels.lean:99](../../../repairs/tritonbench_g/fast_ce_loss/Kernels.lean#L99)；其余 20 处见 audit.json |
| `fast_layernorm` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/fast_layernorm/Kernels.lean#L1) |
| `fast_rms_layernorm` | 外层重建 | [Kernels.lean:33](../../../repairs/tritonbench_g/fast_rms_layernorm/Kernels.lean#L33)；[Kernels.lean:55](../../../repairs/tritonbench_g/fast_rms_layernorm/Kernels.lean#L55)；[Kernels.lean:76](../../../repairs/tritonbench_g/fast_rms_layernorm/Kernels.lean#L76)；[Kernels.lean:98](../../../repairs/tritonbench_g/fast_rms_layernorm/Kernels.lean#L98)；其余 7 处见 audit.json |
| `fast_rope_embedding` | 外层重建 | [Kernels.lean:18](../../../repairs/tritonbench_g/fast_rope_embedding/Kernels.lean#L18)；[Kernels.lean:47](../../../repairs/tritonbench_g/fast_rope_embedding/Kernels.lean#L47) |
| `fifth_order_sph_harmonics` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/fifth_order_sph_harmonics/Kernels.lean#L1) |
| `flash_attn` | 外层重建 | [Kernels.lean:30](../../../repairs/tritonbench_g/flash_attn_scalar_abi_checked/Kernels.lean#L30)；[Kernels.lean:98](../../../repairs/tritonbench_g/flash_attn_scalar_abi_checked/Kernels.lean#L98) |
| `flash_decode2_llama` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/flash_decode2_llama/Kernels.lean#L1) |
| `flash_decode2_phi` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/flash_decode2_phi_block_checked/Kernels.lean#L1) |
| `fp4_to_bf16` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/fp4_to_bf16/Kernels.lean#L1) |
| `fused_activation` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/fused_activation/Kernels.lean#L1) |
| `fused_layernorm_triton` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/fused_layernorm_triton/Kernels.lean#L1) |
| `fused_recurrent_delta` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/fused_recurrent_delta/Kernels.lean#L1) |
| `fused_recurrent_hgrn` | 外层重建 | [Kernels.lean:15](../../../repairs/tritonbench_g/fused_recurrent_hgrn/Kernels.lean#L15)；[Kernels.lean:53](../../../repairs/tritonbench_g/fused_recurrent_hgrn/Kernels.lean#L53)；[Kernels.lean:91](../../../repairs/tritonbench_g/fused_recurrent_hgrn/Kernels.lean#L91) |
| `fused_recurrent_retention` | 外层重建 | [Kernels.lean:18](../../../repairs/tritonbench_g/fused_recurrent_retention/Kernels.lean#L18)；[Kernels.lean:75](../../../repairs/tritonbench_g/fused_recurrent_retention/Kernels.lean#L75)；[Kernels.lean:163](../../../repairs/tritonbench_g/fused_recurrent_retention/Kernels.lean#L163)；[Kernels.lean:220](../../../repairs/tritonbench_g/fused_recurrent_retention/Kernels.lean#L220) |
| `fused_rotary_embedding` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/fused_rotary_embedding_tile_checked/Kernels.lean#L1) |
| `fused_rwkv6_kernel` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/fused_rwkv6_kernel_scalar_abi_checked/Kernels.lean#L1) |
| `geglu_tanh_triton` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/geglu_tanh_triton/Kernels.lean#L1) |
| `index_select_bwd` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/index_select_bwd/Kernels.lean#L1) |
| `index_select_cat` | 外层重建 | [Kernels.lean:19](../../../repairs/tritonbench_g/index_select_cat/Kernels.lean#L19)；[Kernels.lean:38](../../../repairs/tritonbench_g/index_select_cat/Kernels.lean#L38) |
| `int4_matmul` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/int4_matmul/Kernels.lean#L1) |
| `int8_dequant_matmul` | 外层重建 | [Kernels.lean:17](../../../repairs/tritonbench_g/int8_dequant_matmul_scalar_cast/Kernels.lean#L17)；[Kernels.lean:86](../../../repairs/tritonbench_g/int8_dequant_matmul_scalar_cast/Kernels.lean#L86) |
| `int8_matmul_kernel` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/int8_matmul_kernel_modular_checked/Kernels.lean#L1) |
| `int8_matmul_quantization` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/int8_matmul_quantization_i32_safe/Kernels.lean#L1) |
| `int8_quantization` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/int8_quantization/Kernels.lean#L1) |
| `int_scaled_matmul` | 外层重建 | [Kernels.lean:179](../../../repairs/tritonbench_g/int_scaled_matmul/Kernels.lean#L179)；[Kernels.lean:185](../../../repairs/tritonbench_g/int_scaled_matmul/Kernels.lean#L185)；[Kernels.lean:191](../../../repairs/tritonbench_g/int_scaled_matmul/Kernels.lean#L191)；[Kernels.lean:198](../../../repairs/tritonbench_g/int_scaled_matmul/Kernels.lean#L198) |
| `isfinite_kernel` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/isfinite_kernel/Kernels.lean#L1) |
| `iv_dependent_matmul` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/iv_dependent_matmul_dot_shape/Kernels.lean#L1) |
| `kcache_copy_triton` | 外层重建 | [Kernels.lean:22](../../../repairs/tritonbench_g/kcache_copy_triton/Kernels.lean#L22)；[Kernels.lean:49](../../../repairs/tritonbench_g/kcache_copy_triton/Kernels.lean#L49) |
| `kldiv_compute` | 外层重建 | [Kernels.lean:14](../../../repairs/tritonbench_g/kldiv_compute/Kernels.lean#L14)；[Kernels.lean:32](../../../repairs/tritonbench_g/kldiv_compute/Kernels.lean#L32) |
| `kldiv_ops` | 外层重建 | [Kernels.lean:215](../../../repairs/tritonbench_g/kldiv_ops/Kernels.lean#L215)；[Kernels.lean:219](../../../repairs/tritonbench_g/kldiv_ops/Kernels.lean#L219)；[Kernels.lean:223](../../../repairs/tritonbench_g/kldiv_ops/Kernels.lean#L223)；[Kernels.lean:227](../../../repairs/tritonbench_g/kldiv_ops/Kernels.lean#L227) |
| `kldiv_triton` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/kldiv_triton/Kernels.lean#L1) |
| `ksoftmax_triton` | 外层重建 | [Kernels.lean:16](../../../repairs/tritonbench_g/ksoftmax_triton/Kernels.lean#L16)；[Kernels.lean:61](../../../repairs/tritonbench_g/ksoftmax_triton/Kernels.lean#L61) |
| `kv_cache_copy` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/kv_cache_copy/Kernels.lean#L1) |
| `kv_cache_filling` | 未发现本轮模式 | [Kernels.lean:1](../../../../.veritile-cache/finish-pending-105-20261021/source-only/kv_cache_filling/sources/bench/repairs/tritonbench_g/kv_cache_filling/Kernels.lean#L1) |
| `l2_norm_bwd` | 外层重建 | [Kernels.lean:64](../../../repairs/tritonbench_g/l2_norm_bwd/Kernels.lean#L64)；[Kernels.lean:67](../../../repairs/tritonbench_g/l2_norm_bwd/Kernels.lean#L67) |
| `l2_norm_triton1` | 外层重建 | [Kernels.lean:48](../../../repairs/tritonbench_g/l2_norm_triton1/Kernels.lean#L48)；[Kernels.lean:51](../../../repairs/tritonbench_g/l2_norm_triton1/Kernels.lean#L51) |
| `l2_norm_triton2` | 外层重建 | [Kernels.lean:56](../../../repairs/tritonbench_g/l2_norm_triton2/Kernels.lean#L56)；[Kernels.lean:59](../../../repairs/tritonbench_g/l2_norm_triton2/Kernels.lean#L59) |
| `layer_norm_fwd` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/layer_norm_fwd_nonzero_tile/Kernels.lean#L1) |
| `layer_norm_liger` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/layer_norm_liger/Kernels.lean#L1) |
| `layer_norm_ops` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/layer_norm_ops/Kernels.lean#L1) |
| `layer_norm_triton` | 外层重建 | [Kernels.lean:19](../../../repairs/tritonbench_g/layer_norm_triton/Kernels.lean#L19)；[Kernels.lean:57](../../../repairs/tritonbench_g/layer_norm_triton/Kernels.lean#L57)；[Kernels.lean:96](../../../repairs/tritonbench_g/layer_norm_triton/Kernels.lean#L96)；[Kernels.lean:118](../../../repairs/tritonbench_g/layer_norm_triton/Kernels.lean#L118)；其余 2 处见 audit.json |
| `layer_norm_welfold` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/layer_norm_welfold/Kernels.lean#L1) |
| `layernorm_fwd_triton` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/layernorm_fwd_triton/Kernels.lean#L1) |
| `lightning_attention` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/lightning_attention/KernelOnly/Kernels.lean#L1) |
| `llama_ff_triton` | 外层重建 | [Kernels.lean:13](../../../repairs/tritonbench_g/llama_ff_triton_scalar_cast/Kernels.lean#L13)；[Kernels.lean:68](../../../repairs/tritonbench_g/llama_ff_triton_scalar_cast/Kernels.lean#L68)；[Kernels.lean:133](../../../repairs/tritonbench_g/llama_ff_triton_scalar_cast/Kernels.lean#L133)；[Kernels.lean:188](../../../repairs/tritonbench_g/llama_ff_triton_scalar_cast/Kernels.lean#L188) |
| `log_softmax` | 外层重建 | [Kernels.lean:11](../../../repairs/tritonbench_g/log_softmax/Kernels.lean#L11)；[Kernels.lean:32](../../../repairs/tritonbench_g/log_softmax/Kernels.lean#L32)；[Kernels.lean:53](../../../repairs/tritonbench_g/log_softmax/Kernels.lean#L53)；[Kernels.lean:73](../../../repairs/tritonbench_g/log_softmax/Kernels.lean#L73) |
| `logsumexp_fwd` | 外层重建 | [Kernels.lean:15](../../../repairs/tritonbench_g/logsumexp_fwd/Kernels.lean#L15)；[Kernels.lean:37](../../../repairs/tritonbench_g/logsumexp_fwd/Kernels.lean#L37) |
| `lora_expand_gemv` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/lora_expand_gemv/ByteSeparated/Kernels.lean#L1) |
| `masked_add_cuda` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/masked_add_cuda/Kernels.lean#L1) |
| `masked_select` | 外层重建 | [Kernels.lean:13](../../../repairs/tritonbench_g/masked_select/Kernels.lean#L13) |
| `matmul_dequant_int4` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/matmul_dequant_int4_grid_y_checked/Kernels.lean#L1) |
| `matmul_dequantize` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/matmul_dequantize/Kernels.lean#L1) |
| `matmul_dequantize_int4` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/matmul_dequantize_int4/Kernels.lean#L1) |
| `matmul_kernel` | 分解助手归位 | [Kernels.lean:72](../../../repairs/tritonbench_g/matmul_kernel/Kernels.lean#L72) |
| `matmul_leakyrelu` | 源码位于 Proofs | [FPTypedSource.lean:15](../../../tritonbench_g/matmul_leakyrelu/Proofs/FPTypedSource.lean#L15)；[FPFusedSource.lean:16](../../../tritonbench_g/matmul_leakyrelu/Proofs/FPFusedSource.lean#L16) |
| `matmul_leakyrelu_fp8` | 外层重建 | [Kernels.lean:19](../../../repairs/tritonbench_g/matmul_leakyrelu_fp8_scalar_cast/Kernels.lean#L19)；[Kernels.lean:69](../../../repairs/tritonbench_g/matmul_leakyrelu_fp8_scalar_cast/Kernels.lean#L69) |
| `matmul_persistent_triton` | 外层重建 | [Kernels.lean:15](../../../repairs/tritonbench_g/matmul_persistent_triton_scalar_cast/Kernels.lean#L15)；[Kernels.lean:73](../../../repairs/tritonbench_g/matmul_persistent_triton_scalar_cast/Kernels.lean#L73)；[Kernels.lean:131](../../../repairs/tritonbench_g/matmul_persistent_triton_scalar_cast/Kernels.lean#L131)；[Kernels.lean:189](../../../repairs/tritonbench_g/matmul_persistent_triton_scalar_cast/Kernels.lean#L189) |
| `matmul_tma` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/matmul_tma_static_checked_v2/KernelOnly/Kernels.lean#L1) |
| `matmul_triton1` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/matmul_triton1/Kernels.lean#L1) |
| `matmul_triton2` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/matmul_triton2/KernelOnly/Kernels.lean#L1) |
| `matmul_triton_autotune` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/matmul_triton_autotune/Kernels.lean#L1) |
| `matrix_reduction` | 外层重建 | [Kernels.lean:44](../../../tritonbench_g/matrix_reduction/Kernels.lean#L44)；[Kernels.lean:54](../../../tritonbench_g/matrix_reduction/Kernels.lean#L54)；[Kernels.lean:68](../../../tritonbench_g/matrix_reduction/Kernels.lean#L68)；[Kernels.lean:78](../../../tritonbench_g/matrix_reduction/Kernels.lean#L78) |
| `matrix_transpose` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/matrix_transpose/Kernels.lean#L1) |
| `matrix_vector_multip` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/matrix_vector_multip/Kernels.lean#L1) |
| `max_reduction` | 外层重建 | [Kernels.lean:91](../../../tritonbench_g/max_reduction/Kernels.lean#L91)；[Kernels.lean:102](../../../tritonbench_g/max_reduction/Kernels.lean#L102)；[Kernels.lean:113](../../../tritonbench_g/max_reduction/Kernels.lean#L113)；[Kernels.lean:124](../../../tritonbench_g/max_reduction/Kernels.lean#L124)；其余 32 处见 audit.json |
| `mean_reduction` | 外层重建 | [Kernels.lean:11](../../../repairs/tritonbench_g/mean_reduction/Kernels.lean#L11)；[Kernels.lean:32](../../../repairs/tritonbench_g/mean_reduction/Kernels.lean#L32) |
| `mixed_sparse_attention` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/mixed_sparse_attention/Kernels.lean#L1) |
| `mul_exponent_compensator` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/mul_exponent_compensator/Kernels.lean#L1) |
| `multinomial_sampling` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/multinomial_sampling/Kernels.lean#L1) |
| `nested_loops_processing` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/nested_loops_processing/Kernels.lean#L1) |
| `parallel_attention` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/parallel_attention/KernelOnly/Kernels.lean#L1) |
| `parallel_retention_attention` | 外层重建 | [QuerySyntax.lean:22](../../../repairs/tritonbench_g/parallel_retention_attention/QuerySyntax.lean#L22)；[CandidateSyntax.lean:83](../../../repairs/tritonbench_g/parallel_retention_attention/CandidateSyntax.lean#L83) |
| `pow_scalar_tensor` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/pow_scalar_tensor/Kernels.lean#L1) |
| `quant_transpose_kernel` | 外层重建 | [Kernels.lean:22](../../../repairs/tritonbench_g/quant_transpose_kernel/Kernels.lean#L22)；[Kernels.lean:50](../../../repairs/tritonbench_g/quant_transpose_kernel/Kernels.lean#L50) |
| `quantize_copy_kv` | 外层重建 | [Kernels.lean:27](../../../repairs/tritonbench_g/quantize_copy_kv/Kernels.lean#L27)；[Kernels.lean:52](../../../repairs/tritonbench_g/quantize_copy_kv/Kernels.lean#L52) |
| `quantize_global` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/quantize_global/Kernels.lean#L1) |
| `quantize_kv_copy` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/quantize_kv_copy/Kernels.lean#L1) |
| `quantize_kv_transform` | 外层重建 | [Kernels.lean:228](../../../repairs/tritonbench_g/quantize_kv_transform/Kernels.lean#L228)；[Kernels.lean:257](../../../repairs/tritonbench_g/quantize_kv_transform/Kernels.lean#L257) |
| `rbe_triton_transform` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/rbe_triton_transform_checked_launch/Kernels.lean#L1) |
| `relu_strided_buffer` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/relu_strided_buffer/Kernels.lean#L1) |
| `relu_triton_kernel` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/relu_triton_kernel/Kernels.lean#L1) |
| `reversed_cumsum` | 外层重建 | [Kernels.lean:53](../../../repairs/tritonbench_g/reversed_cumsum/Kernels.lean#L53)；[Kernels.lean:88](../../../repairs/tritonbench_g/reversed_cumsum/Kernels.lean#L88) |
| `reversed_cumsum_scalar` | 外层重建 | [Kernels.lean:50](../../../repairs/tritonbench_g/reversed_cumsum_scalar/Kernels.lean#L50)；[Kernels.lean:76](../../../repairs/tritonbench_g/reversed_cumsum_scalar/Kernels.lean#L76) |
| `rms_matmul_rbe` | 外层重建 | [Kernels.lean:24](../../../repairs/tritonbench_g/rms_matmul_rbe_scalar_domain/Kernels.lean#L24)；[Kernels.lean:85](../../../repairs/tritonbench_g/rms_matmul_rbe_scalar_domain/Kernels.lean#L85)；[Kernels.lean:146](../../../repairs/tritonbench_g/rms_matmul_rbe_scalar_domain/Kernels.lean#L146)；[Kernels.lean:207](../../../repairs/tritonbench_g/rms_matmul_rbe_scalar_domain/Kernels.lean#L207)；其余 4 处见 audit.json |
| `rms_norm_triton` | 外层重建 | [Kernels.lean:20](../../../repairs/tritonbench_g/rms_norm_triton/Kernels.lean#L20) |
| `rms_rbe_matmul` | 外层重建 | [Kernels.lean:17](../../../repairs/tritonbench_g/rms_rbe_matmul/Kernels.lean#L17)；[Kernels.lean:69](../../../repairs/tritonbench_g/rms_rbe_matmul/Kernels.lean#L69)；[Kernels.lean:140](../../../repairs/tritonbench_g/rms_rbe_matmul/Kernels.lean#L140)；[Kernels.lean:192](../../../repairs/tritonbench_g/rms_rbe_matmul/Kernels.lean#L192) |
| `rmsnorm_fused` | 外层重建 | [Kernels.lean:17](../../../repairs/tritonbench_g/rmsnorm_fused/Kernels.lean#L17)；[Kernels.lean:50](../../../repairs/tritonbench_g/rmsnorm_fused/Kernels.lean#L50) |
| `rmsnorm_fused_llama` | 外层重建 | [Kernels.lean:12](../../../repairs/tritonbench_g/rmsnorm_fused_llama/Kernels.lean#L12)；[Kernels.lean:37](../../../repairs/tritonbench_g/rmsnorm_fused_llama/Kernels.lean#L37)；[Kernels.lean:63](../../../repairs/tritonbench_g/rmsnorm_fused_llama/Kernels.lean#L63)；[Kernels.lean:88](../../../repairs/tritonbench_g/rmsnorm_fused_llama/Kernels.lean#L88) |
| `rmsnorm_implementation` | 外层重建 | [Kernels.lean:16](../../../repairs/tritonbench_g/rmsnorm_implementation/Kernels.lean#L16)；[Kernels.lean:48](../../../repairs/tritonbench_g/rmsnorm_implementation/Kernels.lean#L48) |
| `rmsnorm_triton` | 外层重建 | [Kernels.lean:15](../../../repairs/tritonbench_g/rmsnorm_triton/Kernels.lean#L15)；[Kernels.lean:45](../../../repairs/tritonbench_g/rmsnorm_triton/Kernels.lean#L45) |
| `rope_backward_transform` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/rope_backward_transform/Kernels.lean#L1) |
| `rope_embedding` | 外层重建 | [Kernels.lean:19](../../../repairs/tritonbench_g/rope_embedding/Kernels.lean#L19)；[Kernels.lean:49](../../../repairs/tritonbench_g/rope_embedding/Kernels.lean#L49) |
| `rope_transform` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/rope_transform/Kernels.lean#L1) |
| `rotary_emb` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/rotary_emb/Kernels.lean#L1) |
| `rotary_emb_nopad` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/rotary_emb_nopad/Kernels.lean#L1) |
| `rotary_transform` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/rotary_transform_tile_checked/Kernels.lean#L1) |
| `rotary_transform_ops` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/rotary_transform_ops/Kernels.lean#L1) |
| `rowwise_quantization_triton` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/rowwise_quantization_triton/Kernels.lean#L1) |
| `seeded_dropout` | 外层重建 | [Kernels.lean:24](../../../repairs/tritonbench_g/seeded_dropout/Kernels.lean#L24)；[Kernels.lean:40](../../../repairs/tritonbench_g/seeded_dropout/Kernels.lean#L40)；[Kernels.lean:56](../../../repairs/tritonbench_g/seeded_dropout/Kernels.lean#L56)；[Kernels.lean:75](../../../repairs/tritonbench_g/seeded_dropout/Kernels.lean#L75)；其余 2 处见 audit.json |
| `sgmv_expand_slice` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/sgmv_expand_slice/ByteSeparated/Kernels.lean#L1) |
| `sin_computation` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/sin_computation/Kernels.lean#L1) |
| `sin_kernel` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/sin_kernel/Kernels.lean#L1) |
| `softmax_flaggems` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/softmax_flaggems_finite_seed_v2/Kernels.lean#L1) |
| `softmax_optimize` | 外层重建 | [Kernels.lean:14](../../../repairs/tritonbench_g/softmax_optimize/Kernels.lean#L14)；[Kernels.lean:76](../../../repairs/tritonbench_g/softmax_optimize/Kernels.lean#L76) |
| `softmax_reducev` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/softmax_reducev_neginf_checked/Kernels.lean#L1) |
| `softmax_triton1` | 外层重建 | [Kernels.lean:12](../../../repairs/tritonbench_g/softmax_triton1/Kernels.lean#L12)；[Kernels.lean:32](../../../repairs/tritonbench_g/softmax_triton1/Kernels.lean#L32) |
| `softmax_triton2` | 外层重建 | [Kernels.lean:12](../../../repairs/tritonbench_g/softmax_triton2/Kernels.lean#L12)；[Kernels.lean:33](../../../repairs/tritonbench_g/softmax_triton2/Kernels.lean#L33) |
| `softmax_triton3` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/softmax_triton3/Kernels.lean#L1) |
| `spinning_lock_reduction` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/spinning_lock_reduction/Kernels.lean#L1) |
| `square_matrix` | 外层重建 | [Kernels.lean:13](../../../repairs/tritonbench_g/square_matrix/Kernels.lean#L13)；[Kernels.lean:29](../../../repairs/tritonbench_g/square_matrix/Kernels.lean#L29) |
| `streamk_matmul` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/streamk_matmul_tile_checked/Kernels.lean#L1) |
| `swiglu_backward` | 外层重建 | [Kernels.lean:13](../../../repairs/tritonbench_g/swiglu_backward/Kernels.lean#L13)；[Kernels.lean:41](../../../repairs/tritonbench_g/swiglu_backward/Kernels.lean#L41) |
| `swiglu_fwd` | 外层重建 | [Kernels.lean:12](../../../repairs/tritonbench_g/swiglu_fwd/Kernels.lean#L12)；[Kernels.lean:32](../../../repairs/tritonbench_g/swiglu_fwd/Kernels.lean#L32) |
| `swiglu_triton` | 外层重建 | [Kernels.lean:19](../../../repairs/tritonbench_g/swiglu_triton/Kernels.lean#L19)；[Kernels.lean:34](../../../repairs/tritonbench_g/swiglu_triton/Kernels.lean#L34)；[Kernels.lean:54](../../../repairs/tritonbench_g/swiglu_triton/Kernels.lean#L54)；[Kernels.lean:74](../../../repairs/tritonbench_g/swiglu_triton/Kernels.lean#L74) |
| `token_attn_llama2` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/token_attn_llama2_grid_axes_checked/Kernels.lean#L1) |
| `token_attn_mistral` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/token_attn_mistral_grid_y_checked/Kernels.lean#L1) |
| `token_attn_reduceV` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/token_attn_reduceV_grid_y_checked/Kernels.lean#L1) |
| `token_softmax_bloom` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/token_softmax_bloom/Kernels.lean#L1) |
| `token_softmax_llama` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/token_softmax_llama/Kernels.lean#L1) |
| `triton_argmax` | 外层重建 | [Kernels.lean:14](../../../repairs/tritonbench_g/triton_argmax_nan_first/Kernels.lean#L14)；[Kernels.lean:39](../../../repairs/tritonbench_g/triton_argmax_nan_first/Kernels.lean#L39)；[Kernels.lean:64](../../../repairs/tritonbench_g/triton_argmax_nan_first/Kernels.lean#L64)；[Kernels.lean:89](../../../repairs/tritonbench_g/triton_argmax_nan_first/Kernels.lean#L89)；其余 7 处见 audit.json |
| `triton_attention` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/triton_attention_scalar_abi_checked/Kernels.lean#L1) |
| `triton_conv2d_fwd` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/triton_conv2d_fwd/Kernels.lean#L1) |
| `triton_linear_activation` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/triton_linear_activation/Kernels.lean#L1) |
| `triton_matmul` | 分解助手归位 | [Kernels.lean:48](../../../repairs/tritonbench_g/triton_matmul/Kernels.lean#L48) |
| `triton_mul2` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/triton_mul2/Kernels.lean#L1) |
| `triton_softmax` | 外层重建 | [Kernels.lean:12](../../../repairs/tritonbench_g/triton_softmax/Kernels.lean#L12)；[Kernels.lean:31](../../../repairs/tritonbench_g/triton_softmax/Kernels.lean#L31) |
| `uniform_sampling` | 外层重建 | [Kernels.lean:14](../../../repairs/tritonbench_g/uniform_sampling/Kernels.lean#L14)；[Kernels.lean:49](../../../repairs/tritonbench_g/uniform_sampling/Kernels.lean#L49) |
| `var_len_copy` | 外层重建 | [Kernels.lean:18](../../../repairs/tritonbench_g/var_len_copy/Kernels.lean#L18)；[Kernels.lean:32](../../../repairs/tritonbench_g/var_len_copy/Kernels.lean#L32) |
| `vector_addition` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/vector_addition/Kernels.lean#L1) |
| `vector_addition_custom` | 未发现本轮模式 | [Kernels.lean:1](../../../repairs/tritonbench_g/vector_addition_custom/Kernels.lean#L1) |
