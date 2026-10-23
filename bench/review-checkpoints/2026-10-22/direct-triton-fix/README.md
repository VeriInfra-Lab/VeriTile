# 标准 Triton 源码入口修复

本轮处理前次复查登记的 **92 题**：移除单个 kernel 外层 `.mk ... (...).surfaceBody` 重建，将 `matmul_leakyrelu` 的实际原程序和候选移出 `Proofs/`，将三题的程序分解助手归入 `Proofs/`。追踪实际调用时补齐 RMS、argmax 候选和 attn_fwd 的共享源码。复查全部 183 个选定入口及已确认的共享路线，**本轮所查的剩余源码组织项为 0**。

92 题的完整选定程序保持性证据和针对性源码构建通过；保真覆盖端口、全部语句、参数、精度、分支和实际原程序／候选。对被外层接口覆盖的内部模板，端口整理与完整选定程序相等分别记录，不把模板 body 相同冒充所有 metadata 相同。复用了适用的既有证明对象，没有逐题重证底层浮点数或 host。

**90 题**完成当前 `Correct`／`FPEquiv` 两个主文件的 fresh 检查。另两题分别记录：

- `chunk_gla_fwd`：`Correct` 原文件检查通过；`FPEquiv` 的五个实际主声明、别名、假设及证明逐字保留的源码夹具检查通过，仅省略末尾审计命令。原文件完整检查在附带审计耗时过长后主动停止；不声称这套审计完成，既有审计证据按原适用范围保留。
- `int_scaled_matmul`：`Correct` fresh 检查、完整源码保持性和源码构建通过；`FPEquiv` fresh 检查受旧整数点积助手的 `Algebra.minimum` ABI 不匹配限制。适用的既有主证明证据及本次检查限制分别列出，不宣称该 fresh `FPEquiv` 检查通过。

92 个选定源码模块的一次联合导入检查通过；它不等于所有主证明模块的联合导入。整题正确性计数保持 **182 已完成、1 带数值延期完成**，延期题仍为 `matmul_leakyrelu`。本轮没有改变 Python 输入契约、数学目标、主定理陈述或数值规则，没有新增／执行数值实验，没有进行全库独立源码回放或 GPU 实测。既有跨题整体导入兼容性状态单独保留。

## 共享源码接口

优先使用普通完整 `triton { ... }`。自动推断无法保留原端口顺序、重复端口或可选／未用端口时，在源码顶部写 `tl.ports inputs=[...], outputs=[...]`。这是共享 DSL 的有序接口元数据，不发出指令，不改变读写、运算精度或输入安全前提。`inputs`／`outputs` 使用非保留语法，声明内容不参加程序体精度推断。

共享端口、旧指针别名推断、运行标量、精度四项回归均在最终构建对象上检查通过。旧缓存解释器的私有特化冲突通过在新隔离目录刷新直接 DSL 调用者解决，未修改旧验收缓存。日志及初次失败记录保留。

## 范围与证据

真实多阶段组合、透明共享入口和合理兼容转发保留。`spinning_lock_reduction`、`streamk_matmul` 的明确题级 reader／`List Statement` 接口仍按前次扫描单列；本报告不声称它们已经统一为 `ComputeKernel := triton`。本轮零剩余仅指已明确检查的源码构造模式，不是所有 183 题、全部历史文件或硬件实现的无条件认证。

- [修复前扫描](../direct-triton-scan/README.md)保持原文，作为历史证据。
- [audit.json](audit.json)记录当前源码哈希、逐题完整程序证据和构建边界。
- `python bench/review-checkpoints/2026-10-22/direct-triton-fix/review.py --check` 检查本轮记录是否漂移；它不替代 Lean 检查或自动推断语义可达性。

## 本轮 92 题

| Task | 完整程序保持／源码构建 | 当前主源码检查 | 逐题证据 |
| --- | --- | --- | --- |
| `adam_update_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue09-result.json) |
| `attention_llama` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue13-result.json) |
| `attention_score` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue06-result.json) |
| `attn_fwd_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue14-result.json) |
| `bgmv_shrink_kernel` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue14-result.json) |
| `block_sparse_attn` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue13-result.json) |
| `bmm_chunk_bwd` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue14-result.json) |
| `bmm_chunk_fwd` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue10-result.json) |
| `bmm_optimized` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue05-result.json) |
| `cache_transform_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue04-result.json) |
| `chunk_bwd_dqkg` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue04-result.json) |
| `chunk_cumsum_kernel` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue06-result.json) |
| `chunk_cumsum_vector` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue07-result.json) |
| `chunk_gate_recurrence` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue10-result.json) |
| `chunk_gated_attention` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue01-result.json) |
| `chunk_gla_fwd` | 通过 | 主声明通过；附带审计未完成 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue09-result.json) |
| `chunk_gla_simple` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue08-result.json) |
| `chunk_retention` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue12-result.json) |
| `chunk_retention_ops` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue11-result.json) |
| `context_attn_bloom` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue11-result.json) |
| `context_attn_fwd` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue10-result.json) |
| `context_attn_llama` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue11-result.json) |
| `context_attn_mistral` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue12-result.json) |
| `context_attn_nopad` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue05-result.json) |
| `cross_entropy1` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue09-result.json) |
| `cross_entropy2` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue13-result.json) |
| `cross_entropy_ops` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue14-result.json) |
| `decay_cumsum` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue12-result.json) |
| `dequantize_matmul` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue09-result.json) |
| `dequantize_rowwise` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue13-result.json) |
| `destindex_copy` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue12-result.json) |
| `destindex_copy_kv1` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue12-result.json) |
| `destindex_copy_kv2` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue14-result.json) |
| `diag_ssm_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue10-result.json) |
| `dropout_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue04-result.json) |
| `embedding_triton_kernel` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue06-result.json) |
| `f8_conversion_utils` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue06-result.json) |
| `fast_ce_loss` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue03-result.json) |
| `fast_rms_layernorm` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue07-result.json) |
| `fast_rope_embedding` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue07-result.json) |
| `flash_attn` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue08-result.json) |
| `fused_recurrent_hgrn` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue05-result.json) |
| `fused_recurrent_retention` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue07-result.json) |
| `index_select_cat` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue10-result.json) |
| `int8_dequant_matmul` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue11-result.json) |
| `int_scaled_matmul` | 通过 | 既有代际兼容限制；复用证据另列 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue08-result.json) |
| `kcache_copy_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue12-result.json) |
| `kldiv_compute` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue05-result.json) |
| `kldiv_ops` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue12-result.json) |
| `ksoftmax_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue09-result.json) |
| `l2_norm_bwd` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue13-result.json) |
| `l2_norm_triton1` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue14-result.json) |
| `l2_norm_triton2` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue04-result.json) |
| `layer_norm_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue13-result.json) |
| `llama_ff_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue10-result.json) |
| `log_softmax` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue11-result.json) |
| `logsumexp_fwd` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue06-result.json) |
| `masked_select` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue04-result.json) |
| `matmul_kernel` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue15-result.json) |
| `matmul_leakyrelu` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue15-result.json) |
| `matmul_leakyrelu_fp8` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue07-result.json) |
| `matmul_persistent_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue09-result.json) |
| `matrix_reduction` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue13-result.json) |
| `max_reduction` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue02-result.json) |
| `mean_reduction` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue08-result.json) |
| `parallel_retention_attention` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue10-result.json) |
| `quant_transpose_kernel` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue11-result.json) |
| `quantize_copy_kv` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue12-result.json) |
| `quantize_kv_transform` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue03-result.json) |
| `reversed_cumsum` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue05-result.json) |
| `reversed_cumsum_scalar` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue09-result.json) |
| `rms_matmul_rbe` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue11-result.json) |
| `rms_norm_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue06-result.json) |
| `rms_rbe_matmul` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue14-result.json) |
| `rmsnorm_fused` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue13-result.json) |
| `rmsnorm_fused_llama` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue07-result.json) |
| `rmsnorm_implementation` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue14-result.json) |
| `rmsnorm_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue04-result.json) |
| `rope_embedding` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue06-result.json) |
| `seeded_dropout` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue14-result.json) |
| `softmax_optimize` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue07-result.json) |
| `softmax_triton1` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue08-result.json) |
| `softmax_triton2` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue10-result.json) |
| `square_matrix` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue11-result.json) |
| `swiglu_backward` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue12-result.json) |
| `swiglu_fwd` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue03-result.json) |
| `swiglu_triton` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue08-result.json) |
| `triton_argmax` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue08-result.json) |
| `triton_matmul` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue15-result.json) |
| `triton_softmax` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue05-result.json) |
| `uniform_sampling` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue09-result.json) |
| `var_len_copy` | 通过 | 通过 | [证据](../../../../.veritile-cache/direct-triton-fix-20261022/queue13-result.json) |
