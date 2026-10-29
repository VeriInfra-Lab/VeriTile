# Cleanup of superseded public proof entries

This change retires obsolete public specifications across 37 ports while retaining the current source contracts. The 86 retained public correctness statements have identical hypotheses and conclusions. Floating equivalence statements are unchanged. This cleanup adds no mathematical coverage, native admission, or independent task acceptance.

The review ledger removes 68 entry records (697 to 629); namespace-aware Lean inspection checks 71 retired declarations because three names also occur in the relu candidate namespace. Eleven unreferenced output-summary declarations and their three audit references are deleted. The unused ksoftmax plain-forward proof chain is deleted; its still-required backward source is extracted unchanged. The triton_softmax mathematical IO proof is retained as an ordinary helper in Proofs/RealIO.lean. The checked Lean environment retains 59 helper declarations and 140 current public declarations across the affected correctness/equivalence modules.

Genuine independent cases remain public, including the persistent-matmul K=0 memory-preservation theorem. Public correctness and floating equivalence are kept separately. No remaining specialization is relabeled complete.

The expanded aggregate also found an outdated pow_scalar_tensor precision assertion. Its corrected AST equation explicitly retains the source input cast to fp32; the kernel and public contracts are unchanged. The initial missing audit import and the intermediate precision assertion failure are recorded in separate failed snapshots.

## Validation

All final checks passed on unchanged captured inputs:

- Aggregate: 5,197 Lake jobs, 255 requested targets, 1,841 captured local inputs, exit code 0 and no changed inputs. The captured inputs and terminal build log are included here.
- Lean environment audit: 140 current public declarations remain tagged; 71 retired declarations are untagged; 59 reusable helpers remain available; 14 removed names are absent. The fixture and passing log are in `review/PublicSurface.lean` and `review/public-surface-driver.log`.
- [Source-only corpus audit](../legacy-public-cleanup-source/README.md): all 178 source views passed; 3,563 captured inputs remained unchanged. This run does not include new Python unit-test, native-compilation or independent-acceptance results.
- Coverage validation: 629 review records; all remaining review classifications are preserved. The proof-gap manifest and completion inventory freshness checks passed.
- Registry inspection: 130 affected registry entries match current namespaces. Current specification sheets and their index were refreshed.
- Statement-preservation review: hypotheses and conclusions of all 86 retained public correctness statements across 37 ports are unchanged. All 27 retained public proof bodies in the files with summary deletions are unchanged; the extracted ksoftmax backward source is unchanged.

The exact supplemental artifacts are listed with SHA-256 hashes in `review/SHA256.json`. The public-attribute audit can be replayed from the repository root with `lake env lean bench/review-checkpoints/2026-10-14/legacy-public-cleanup-root/review/PublicSurface.lean` after building the targets in `review/targets-validated.json`. These are local checks, not independent benchmark acceptance.

## Retired entries and current correctness surface

| Port | Retired public entry records | Current correctness entries |
|---|---|---|
| `flash_attn` | `flash_attn_genuine_l_compute_correct_general`, `flash_attn_genuine_output_compute_correct_general`, `flash_attn_io_correctness`, `flash_attn_python_case1_genuine_compute_correct_general`, `flash_attn_python_case2_genuine_compute_correct_general` | `original_correctness`, `candidate_correctness`, `flat_correctness`, `full_flat_correctness` |
| `fused_activation` | `fused_add_mul_activation_kernel_correctness` | `original_real_correct`, `commuted_real_correct` |
| `geglu_tanh_triton` | `backward_formula_correct`, `forward_formula_correct`, `geglu_tanh_backward_kernel_correctness`, `geglu_tanh_forward_kernel_correctness` | `forward_typed_real_correct`, `backward_typed_real_correct` |
| `index_select_bwd` | `index_select_cat_bwd_kernel_correctness` | `source_real_correct` |
| `index_select_cat` | `index_select_cat_fwd_kernel_correctness` | `source_real_correct` |
| `kldiv_compute` | `kldivergence_kernel_correctness` | `typed_real_correct` |
| `ksoftmax_triton` | `ksoftmax_forward_plain_real_correct` | `ksoftmax_backward_original_real_correct`, `ksoftmax_backward_candidate_real_correct`, `ksoftmax_forward_fp32_original_real_correct`, `ksoftmax_forward_fp32_candidate_real_correct` |
| `l2_norm_bwd` | `l2_norm_bwd_kernel_correctness` | `typed_real_correct`, `original_real_correct`, `candidate_real_correct` |
| `l2_norm_triton1` | `l2_norm_fwd_1pass_kernel_correctness` | `typed_real_correct`, `original_real_correct`, `candidate_real_correct` |
| `l2_norm_triton2` | `l2_norm_bwd_kernel_correctness`, `l2_norm_fwd_1pass_kernel_correctness` | `forward_typed_real_correct`, `backward_typed_real_correct` |
| `llama_ff_triton` | `ff_llama_closed_form_correct` | `fp16_flat_correct`, `fp8_flat_correct` |
| `log_softmax` | `log_softmax_backward_kernel_correctness`, `log_softmax_kernel_correctness` | `forward_typed_correct`, `backward_typed_correct`, `forward_region_correct`, `backward_region_correct` |
| `masked_add_cuda` | `masked_add_bool_commuted_real_correct`, `masked_add_kernel_correctness` | `masked_add_integer_real_correct`, `masked_add_fp64_integer_real_correct`, `masked_add_typed_integer_real_correct` |
| `matrix_transpose` | `kernel_correctness` | `transpose_typed_real_correct` |
| `max_reduction` | `max_kernel_1_io_correctness`, `max_kernel_output_summary` | `stage2_math_correct`, `stage1_real_correct`, `stage2_real_correct`, `dimension_real_correct` |
| `mean_reduction` | `mean_dim_kernel_output_summary` | `mean_dim_kernel_io_correctness`, `mean_typed_real_correct` |
| `pow_scalar_tensor` | `pow_scalar_tensor_one_tile_io_correctness`, `pow_scalar_tensor_output_summary_general` | `floating_one_tile_correct`, `floating_grid_correct`, `integer_one_tile_correct`, `integer_grid_correct` |
| `relu_strided_buffer` | `relu_strided_buffer_grid_stride_io_correctnessR`, `relu_strided_buffer_one_tile_io_correctness`, `relu_strided_buffer_output_summary_general` | `original_one_tile_correctness`, `original_grid_stride_correctness`, `candidate_one_tile_correctness`, `candidate_grid_stride_correctness` |
| `rms_matmul_rbe` | `rms_matmul_rbe_closed_form_correct`, `rms_matmul_rbe_qkv_closed_form_correct` | `half_formal_real_correct`, `qkv_half_formal_real_correct`, `fp8_formal_real_correct`, `qkv_fp8_formal_real_correct` |
| `rms_norm_triton` | `rms_norm_kernel_correctness` | `typed_real_correct`, `original_real_correct`, `candidate_real_correct` |
| `rms_rbe_matmul` | `rms_matmul_rbe_closed_form_correct` | `half_formal_real_correct`, `fp8_formal_real_correct` |
| `rmsnorm_fused` | `rms_norm_fwd_fused_io_correctness`, `rms_norm_fwd_fused_output_summary` | `typed_real_correct` |
| `rmsnorm_fused_llama` | `rms_norm_fwd_fused_llama_io_correctness`, `rms_norm_fwd_fused_llama_output_summary` | `original_real_correct`, `candidate_real_correct` |
| `rmsnorm_implementation` | `rmsnorm_implementation_io_correctness`, `rmsnorm_implementation_output_summary` | `typed_real_correct`, `full_real_correct` |
| `rotary_emb` | `rotary_emb_kernel_correctness` | `rotary_emb_correctness` |
| `sin_computation` | `sin_kernel_correctness` | `typed_real_correct` |
| `sin_kernel` | `kernel_function_correctness` | `typed_real_correct` |
| `softmax_flaggems` | `softmax_backward_kernel_inner_one_tile_compute_correct`, `softmax_backward_kernel_non_inner_one_tile_compute_correct`, `softmax_kernel_inner_one_tile_compute_correct`, `softmax_kernel_inner_one_tile_correctness`, `softmax_kernel_non_inner_one_tile_compute_correct` | `forward_non_inner_correctness`, `forward_inner_correctness`, `backward_non_inner_correctness`, `backward_inner_correctness` |
| `softmax_optimize` | `softmax_kernel_online_v2_one_tile_io_correctness`, `softmax_kernel_online_v2_one_tile_output_summary`, `softmax_kernel_online_v2_output_summary` | `full_source_original_correct`, `full_source_candidate_correct` |
| `softmax_triton2` | `softmax_kernel_correctness` | `softmax_original_real_correct`, `softmax_reciprocal_correctness` |
| `softmax_triton3` | `softmax_kernel_correctness`, `softmax_kernel_masked_correctness` | `softmax_typed_real_correct`, `softmax_typed_masked_real_correct` |
| `square_matrix` | `square_kernel_correctness` | `square_typed_real_correct` |
| `swiglu_backward` | `formula_correct`, `swiglu_bwd_kernel_correctness` | `typed_real_correct` |
| `swiglu_fwd` | `formula_correct`, `swiglu_fwd_kernel_correctness` | `typed_real_correct` |
| `swiglu_triton` | `backward_formula_correct`, `forward_formula_correct`, `swiglu_backward_kernel_correctness`, `swiglu_forward_kernel_correctness` | `forward_typed_real_correct`, `backward_typed_real_correct` |
| `triton_softmax` | `softmax_kernel_correctness` | `softmax_typed_real_correct` |
| `var_len_copy` | `var_len_copy_kernel_triton_small_length_output_summary`, `var_len_copy_one_chunk_io_correctness` | `typed_real_correct`, `original_real_correct`, `candidate_real_correct`, `discrete_typed_real_correct`, `discrete_original_real_correct`, `discrete_candidate_real_correct` |
