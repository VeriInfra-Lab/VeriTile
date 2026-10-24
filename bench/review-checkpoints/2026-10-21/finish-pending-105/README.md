# Completion of the remaining 105 tasks

Baseline: 78 accepted / 105 pending. Current: 183 accepted / 0 pending.

Only explicitly accepted whole tasks appear below. Source cleanup, helper counts and compiling a declaration alone do not count. User-authorized new candidates must implement an optimization directly related to the original computation.

| Task | Change and scope | Numerical evidence | Root decision |
|---|---|---|---|
| adam_update_triton | Reuse the existing final momentum-addition candidate with the actual runtime FP32 scalar ABI. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/adam_update_triton/root-acceptance.json) |
| attention_forward_triton | Hoist the invariant head mask outside the complete dynamic loop. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/attention_forward_triton/root-acceptance.json) |
| attention_fwd_triton1 | Four fixed BT=32 temporal block-pointer products become shifts in the actual complete source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/attention_fwd_triton1/root-acceptance.json) |
| attention_fwd_triton2 | Fixed BLOCK_M=128 entry row-base product becomes a shift in the complete mixed-precision attention source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/attention_fwd_triton2/root-acceptance.json) |
| attention_fwd_triton3 | Share each iteration's key-tail mask across both actual load sites. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/attention_fwd_triton3/root-acceptance.json) |
| attention_kernel | Issue query loads and scaling earlier, overlapping independent pointer and online-state initialization. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/attention_kernel/root-acceptance.json) |
| attention_kernel_aligned | Schedule Q load/scale earlier in both complete aligned-attention storage branches. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/attention_kernel_aligned/root-acceptance.json) |
| attention_llama | Reuse the carry-first normalizer candidate on complete runtime-scalar FP16/E5M2 attention programs. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/attention_llama/root-acceptance.json) |
| attn_fwd_causal | Share the head quotient in the remainder expression. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/attn_fwd_causal/root-acceptance.json) |
| attn_fwd_triton | Share the computed head quotient in the complete attention source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/attn_fwd_triton/root-acceptance.json) |
| batched_vecmat_mult | Reuse the existing complete block-first carry candidate. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/batched_vecmat_mult/root-acceptance.json) |
| bgmv_expand_slice | Existing output-first addition candidate on the complete signed-index batch/row grid. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/bgmv_expand_slice/root-acceptance.json) |
| bgmv_shrink_kernel | Reuse the existing final FP32 scale-product operand exchange in complete FP16/BF16 split-K sources. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/bgmv_shrink_kernel/root-acceptance.json) |
| block_sparse_attn | Share the head quotient in remainder indexing; explicitly declare runtime scale. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/block_sparse_attn/root-acceptance.json) |
| bmm_optimized | Expose shared quotient in group-remainder calculation. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/bmm_optimized/root-acceptance.json) |
| cache_transform_triton | Byte-gather address candidate derives remainder from the quotient already used by the row lookup. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/cache_transform_triton/root-acceptance.json) |
| chunk_bwd_dqkg | Reuse the absolute chunk end in the final gate mask across the complete gradient kernel. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/chunk_bwd_dqkg/root-acceptance.json) |
| chunk_delta_fwd | Fold static K*V factors in the complete selected delta state/history addresses, preserving the native dot/FMA arithmetic graph. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/chunk_delta_fwd/root-acceptance.json) |
| chunk_gate_recurrence | Reuse the original sum-left recurrence candidate with repaired four-stride seed loads and identical ordered FMA lowering. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/chunk_gate_recurrence/root-acceptance.json) |
| chunk_gla_simple | Reuse the existing causal-mask lane vector in complete GLA row addresses, removing its duplicate construction. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/chunk_gla_simple/root-acceptance.json) |
| chunk_linear_attn | Hoist the common Q/K head-base product out of forward-output key loops while preserving all four attention stages and the final dV reduction. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/chunk_linear_attn/root-acceptance.json) |
| chunk_retention | Reuse scale-first final multiplication in forward output. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/chunk_retention/root-acceptance.json) |
| chunk_retention_ops | Reuse shared forward-output scale-first source in the complete four-stage packet. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/chunk_retention_ops/root-acceptance.json) |
| chunked_cumsum_fwd | Reuse the existing scale-first rate multiplication candidate on the complete prepared-dt and inclusive-cumsum source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/chunked_cumsum_fwd/root-acceptance.json) |
| context_attn_bloom | Reuse the already-computed row-block origin instead of multiplying it again. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/context_attn_bloom/root-acceptance.json) |
| context_attn_fwd | Factor a common prompt offset out of integer minimum, removing one scalar addition per invocation. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/context_attn_fwd/root-acceptance.json) |
| context_attn_llama | Factor the common prompt offset out of the integer block-end minimum. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/context_attn_llama/root-acceptance.json) |
| context_attn_mistral | Reuse the scalar block origin instead of computing its product again in complete context attention. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/context_attn_mistral/root-acceptance.json) |
| context_attn_nopad | Reuse the computed scalar block origin in row addresses. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/context_attn_nopad/root-acceptance.json) |
| cross_entropy1 | Reuse exact integer column-product commutation in actual complete forward and backward programs. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/cross_entropy1/root-acceptance.json) |
| cross_entropy2 | Reuse exact integer column-product commutation in actual complete forward and backward programs. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/cross_entropy2/root-acceptance.json) |
| cross_entropy_ops | Reuse exact integer column-product commutation in actual complete forward and backward programs. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/cross_entropy_ops/root-acceptance.json) |
| fast_layernorm | Reuse forward/dX address candidate; remove a redundant inactive-column select from the complete affine row loop. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/fast_layernorm/root-acceptance.json) |
| fast_rms_layernorm | Existing row-index commutation on complete regular/Gemma forward and DW-then-DX backward schedules. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/fast_rms_layernorm/root-acceptance.json) |
| fast_rope_embedding | Existing head-group integer product commutation; actual launched full kernel, both directions and arbitrary source strides. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/fast_rope_embedding/root-acceptance.json) |
| fifth_order_sph_harmonics | Reuse identical left-associated power subexpressions in complete forward and backward spherical-harmonic sources. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/fifth_order_sph_harmonics/root-acceptance.json) |
| flash_attn | Reuse head-offset product normalization on repaired independent pitches. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/flash_attn/root-acceptance.json) |
| flash_decode2_llama | Share the computed split-row address base. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/flash_decode2_llama/root-acceptance.json) |
| flash_decode2_phi | Share the split-row base in the masked source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/flash_decode2_phi/root-acceptance.json) |
| fused_layernorm_triton | Hoist invariant row/mean/inverse scalar projections before the normalization loop of the complete Welford LayerNorm. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/fused_layernorm_triton/root-acceptance.json) |
| fused_recurrent_delta | Actual runtime FP32 forward/backward pair; eliminate zero key-grid output-address terms under the original NK=1 invocation. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/fused_recurrent_delta/root-acceptance.json) |
| fused_recurrent_hgrn | Fold constexpr T times D before three forward base-address products. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/fused_recurrent_hgrn/root-acceptance.json) |
| fused_recurrent_retention | Reuse compare-right key masks in complete forward/backward sources. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/fused_recurrent_retention/root-acceptance.json) |
| fused_rotary_embedding | Reuse the first-half Q address and a uniform half-stride displacement to remove a per-lane stride multiplication. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/fused_rotary_embedding/root-acceptance.json) |
| fused_rwkv6_kernel | Issue the complete RWKV U read earlier to overlap unrelated pointer setup and optional initial-state loading. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/fused_rwkv6_kernel/root-acceptance.json) |
| index_select_bwd | Tail-aware actual JIT block specialization reduces inactive lanes; preserve ordered additions and all slices. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/index_select_bwd/root-acceptance.json) |
| int4_matmul | Precombine BK times split stride, removing one dynamic integer multiplication per K offset. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/int4_matmul/root-acceptance.json) |
| int8_dequant_matmul | Reuse original row/column index tiles after the full integer-dot loop. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/int8_dequant_matmul/root-acceptance.json) |
| int8_matmul_kernel | Compare actual complete packed-int8 matmul programs across the fifteen retained legal autotune configurations. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/int8_matmul_kernel/root-acceptance.json) |
| int8_matmul_quantization | Remove redundant producer row modulo and always-true GEMM scale-load masks. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/int8_matmul_quantization/root-acceptance.json) |
| int8_quantization | Replace per-lane multiplication of 0.5 by selected signs with direct selection of positive/negative 0.5. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/int8_quantization/root-acceptance.json) |
| int_scaled_matmul | Remove an accumulator copy and reuse invariant row/column vectors in the two complete integer matmul variants. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/int_scaled_matmul/root-acceptance.json) |
| kldiv_ops | Existing signed output-row product commutation; complete forward and backward grids and reduction modes. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/kldiv_ops/root-acceptance.json) |
| kldiv_triton | Reuse signed output-row product commutation for both complete forward and backward source grids. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/kldiv_triton/root-acceptance.json) |
| layer_norm_liger | Share centered forward vector and hoist backward denominator conversion. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/layer_norm_liger/root-acceptance.json) |
| layer_norm_ops | Reuse column comparison in forward and the row origin in the complete bounded backward loop. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/layer_norm_ops/root-acceptance.json) |
| layer_norm_triton | Expand the existing one-iteration forward loop and shorten grouped/final backward vector lifetimes. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/layer_norm_triton/root-acceptance.json) |
| layer_norm_welfold | Complete three-loop LayerNorm source; reuse the row offset between each normalization X load and BF16 Y store. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/layer_norm_welfold/root-acceptance.json) |
| layernorm_fwd_triton | Move each output-weight load after normalization to shorten its live range. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/layernorm_fwd_triton/root-acceptance.json) |
| lightning_attention | Reuse duplicate base offsets in the complete forward and two-stage backward sources. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/lightning_attention/root-acceptance.json) |
| llama_ff_triton | Reuse output-row product commutation in repaired int64/four-mask source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/llama_ff_triton/root-acceptance.json) |
| logsumexp_fwd | Remove redundant full-vector mask restoration only from the unscaled logsumexp specialization. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/logsumexp_fwd/root-acceptance.json) |
| lora_expand_gemv | Specialize cdiv(N,SPLIT_N) to one under the actual SPLIT_N=N launch, removing uniform integer division. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/lora_expand_gemv/root-acceptance.json) |
| matmul_dequant_int4 | Specialize the already-required unit output column stride. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matmul_dequant_int4/root-acceptance.json) |
| matmul_dequantize | Precompute group extents and affine dequantization output origins in four complete kernels. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matmul_dequantize/root-acceptance.json) |
| matmul_dequantize_int4 | Precompute the invariant quantization-group extent in the complete GPTQ source and native graph. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matmul_dequantize_int4/root-acceptance.json) |
| matmul_kernel | Strength-reduce both K-tail masks using remaining K. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matmul_kernel/root-acceptance.json) |
| matmul_leakyrelu | Reuse the existing seeded-dot candidate in place of separate dot/add, with explicit deferred local numerical contraction. | Explicit deferred rules | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matmul_leakyrelu/root-acceptance.json) |
| matmul_leakyrelu_fp8 | Reuse right-side slope multiplication in the actual complete safe-index/scalar-cast matmul source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matmul_leakyrelu_fp8/root-acceptance.json) |
| matmul_persistent_triton | Reuse scalar row-product commutation in the actual nested persistent matmul source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matmul_persistent_triton/root-acceptance.json) |
| matmul_tma | Remove redundant FP32 output identity conversion. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matmul_tma/root-acceptance.json) |
| matmul_triton1 | Hoist row and column masks out of the K loop and reuse them in loads and the final store. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matmul_triton1/root-acceptance.json) |
| matmul_triton2 | Hoist row and column masks out of the K loop and reuse them in loads and the final store. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matmul_triton2/root-acceptance.json) |
| matrix_reduction | Replace whole-tile block-pointer loading by direct affine addresses. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matrix_reduction/root-acceptance.json) |
| matrix_vector_multip | Express the original already-contracted accumulator explicitly with tl.fma across every column loop. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/matrix_vector_multip/root-acceptance.json) |
| mixed_sparse_attention | Fuse zero-tile creation with sparse-column score masking. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/mixed_sparse_attention/root-acceptance.json) |
| mul_exponent_compensator | Complete masked FP32 scaling grid: reuse existing scalar-left constant multiply candidate, all words and all complete program orders. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/mul_exponent_compensator/root-acceptance.json) |
| multinomial_sampling | Statically unroll the existing constexpr token-block loop. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/multinomial_sampling/root-acceptance.json) |
| parallel_attention | Fuse consecutive score assignments in actual forward/DQ source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/parallel_attention/root-acceptance.json) |
| parallel_retention_attention | Fuse adjacent ordered accumulator assignments in complete forward and DQ sources. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/parallel_retention_attention/root-acceptance.json) |
| quantize_kv_copy | Schedule two independent pure output-address assignments before the FP32 quantization pipeline. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/quantize_kv_copy/root-acceptance.json) |
| quantize_kv_transform | Move pure output-address formation before the unchanged complete quantization computation. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/quantize_kv_transform/root-acceptance.json) |
| rbe_triton_transform | Reuse existing real/imaginary load masks for corresponding stores in the complete rotary source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/rbe_triton_transform/root-acceptance.json) |
| rms_matmul_rbe | Complete Q/K/V typed source schedules; existing output-row integer product commutation in all four storage branches. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/rms_matmul_rbe/root-acceptance.json) |
| rms_rbe_matmul | Complete rectangular RMS/RBE matmul grid; preserve existing output-row integer commutation. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/rms_rbe_matmul/root-acceptance.json) |
| rope_backward_transform | Share existing coefficient indices and half-width masks with the complete Q/K backward rotation source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/rope_backward_transform/root-acceptance.json) |
| rope_transform | Reuse coefficient indices and masks in the complete forward rotation source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/rope_transform/root-acceptance.json) |
| rotary_emb_nopad | Reuse the first Q-half address for the high half in complete cached and uncached rotary kernels. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/rotary_emb_nopad/root-acceptance.json) |
| rotary_transform | Share repeated interleaved-lane index arithmetic in all four complete metadata source profiles. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/rotary_transform/root-acceptance.json) |
| rotary_transform_ops | Compute paired rotary lane indices with exact XOR indexing. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/rotary_transform_ops/root-acceptance.json) |
| rowwise_quantization_triton | Move FP16 row-maximum store before INT8 quantization. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/rowwise_quantization_triton/root-acceptance.json) |
| sgmv_expand_slice | Reuse input row/column index vectors when computing the segmented matvec output coordinates. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/sgmv_expand_slice/root-acceptance.json) |
| softmax_reducev | Compute output addresses before normalization in the complete online reduction kernel. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/softmax_reducev/root-acceptance.json) |
| softmax_triton2 | Runtime-guarded reciprocal strength reduction with original-division fallback. | Existing admitted experimental rule(s): DIV-MUL-RCP | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/softmax_triton2/root-acceptance.json) |
| softmax_triton3 | Runtime-guarded reciprocal strength reduction with original-division fallback. | Existing admitted experimental rule(s): DIV-MUL-RCP | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/softmax_triton3/root-acceptance.json) |
| spinning_lock_reduction | Fold constexpr tile area in producer and reducer pointers. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/spinning_lock_reduction/root-acceptance.json) |
| streamk_matmul | Fold constexpr tile area in actual first/reduce/full source addresses. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/streamk_matmul/root-acceptance.json) |
| token_attn_llama2 | Reuse the already computed scalar token-block origin in the complete Q/K score source. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/token_attn_llama2/root-acceptance.json) |
| token_attn_mistral | Actual Python/Lean affine-address optimization computes a uniform scalar origin and removes one vector addition. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/token_attn_mistral/root-acceptance.json) |
| token_attn_reduceV | Hoist a uniform scalar probability origin and remove a vector addition from actual gathered-PV addressing. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/token_attn_reduceV/root-acceptance.json) |
| triton_argmax | Use guarded integer shift strength reduction in complete global first-stage and dimension kernels. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/triton_argmax/root-acceptance.json) |
| triton_attention | Replace fixed128 preprocess row multiplication by shift7. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/triton_attention/root-acceptance.json) |
| triton_conv2d_fwd | Replace fixed-128 convolution row-base multiplication by a shift in both complete operand-precision paths. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/triton_conv2d_fwd/root-acceptance.json) |
| triton_linear_activation | Strength-reduce GROUP_M=8 group width with a shift while retaining generic fallback. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/triton_linear_activation/root-acceptance.json) |
| triton_matmul | Use a guarded group-width shift with the original formula for every other group size. | No experimental numerical rule | [Evidence](../../../../.veritile-cache/finish-pending-105-20261021/triton_matmul/root-acceptance.json) |

Fresh joint entry checks reuse applicable internal proof objects. Independent full-source replay and GPU execution were not run. Previously valid proofs and original-source defect diagnoses retain their recorded scopes.

<!-- final-validation-scope -->

## Final validation scope

The campaign accepted exactly the 105 tasks that were pending in its frozen
baseline. The single task ledger now records 182 completed tasks and one
completed with deferred numerical evidence, with no pending task or active repair
blocker. Actual main-pair numerical use is 178 strict-model/structural results,
four users of the existing FP32 DIV-MUL-RCP admission, and the single deferred
matmul_leakyrelu dot/add-to-dotAcc rule. Its two gates remain NOT_RUN; this is not
an unconditional numerical-equivalence claim.

All 47 source-only deliveries are separately accepted without recounting proof
completion. The source review records 107 fixed and 76 with no identified issue.
The final source/delivery/root-log pin check found no drift across 2,050 unique
accepted source paths; see [the check](final-pin-check.json). Fresh targeted
root joint checks reused applicable internal objects. Independent full-source
replay and GPU execution were not run.

A cross-task combined import is not passing. The non-KV 182-task union in the
isolated minimumFloat/current-Welford generation encounters a duplicate generated
`VeriTile.Triton.FP.Structural.evalExpr.eq_6` declaration between
`attention_fwd_triton1_tile_checked.Proofs.Step` and `Float.Reciprocal`. The
183-name attempt additionally lacks the separately selected KV reduceMin object
generation; it does not test or invalidate the successful current KV task check.
Required helper refreshes for this new generation are not all complete. Actual
failed attempts and their log hashes are recorded in
`task_progress.json/current_cross_task_import_check`; follow-up work is in
[the shared compatibility TODO](../../../TODO.md).

The current KV source-only selection retains its explicit isolated driver at
`.veritile-cache/finish-pending-105-20261021/source-only/kv_cache_filling/build.py`.
Its 47 necessary modules, seven complete-function preservation equalities and
fresh public/optional joint check passed; the original historical packet is
unchanged. No reduceMin model was silently overlaid into the current shared
Core. This generation boundary is separate from the single numerical deferral.

The frozen [combined-generation report](../../../../.veritile-cache/finish-pending-105-20261021/combined-minimum-welford-20261022/README.md) records the representative checks, exact object-family reuse, inherited-receipt limits, incomplete helper refresh and actual failed union attempts. Root verified 138 owned module receipts and their source/object/direct-dependency hashes with zero mismatches; that integrity check does not certify the aggregate import.
