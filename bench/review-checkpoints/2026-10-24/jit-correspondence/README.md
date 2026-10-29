# Python JIT / Lean source correspondence review

The selected `triton_argmax` revision (`triton_argmax_nan_first`) now has five
complete AST correspondence checks: three original JIT bodies and two changed
candidate bodies. Current public Real and FP proofs execute the new source
through explicit runtime argument bindings. See the example README and
`review.json` for exact proof scope, evidence and model boundaries.

Final verification: both Real and both FP public specifications, all five AST
pairs, five Python tests, nine Lean frontend regression files and the four
formula/reader/IO/ledger guards pass. The isolated build completes 3,559 jobs;
the development/shared-consumer build completes 8,568 jobs. These counts include
replayed external dependencies and are not counts of newly proved theorems.
The clean checkout began without project artifacts; after the first successful
build, the final additive axis guard was checked out from Git and rebuilt in the
same isolated directory. Both build logs and exact source pins are retained.

## Other selected tasks

All 183 selected source pairs were inventoried. **45 other tasks** contain
explicit Python int64 conversions but no corresponding explicit int64 cast in
their selected direct Lean quotation file. **137 tasks remain uncertified by
this screen**; absence of this particular symptom is not a correspondence pass.
The repaired argmax is the only full AST correspondence pass in this inventory.

These are source correspondence findings, not 45 established mathematical or
runtime counterexamples. Some prior proofs explicitly erase widening under an
address-range model. Those proof results retain their original scope, but do
not satisfy the current requirement that the readable source preserve every
Python statement and cast. This review does not repair the other tasks or
recertify their entire semantics.

Manual confirmations:

- `apply_penalty`: Python `cur_batch = tl.program_id(0).to(tl.int64)` and
  widening of loaded sequence endpoints/token IDs; Lean assigns the uncast
  program ID and natural-valued loads. Python runtime stride casts also become
  template arithmetic in the quotation.
- `dequantize_rowwise`: Python widens both program ID and arange; the Lean
  `source` omits both casts. Runtime `inv_127` is embedded as a Lean constant.
- `fast_layernorm`: Python casts `n_cols`, program ID, arange and runtime
  strides. The Lean forward kernel explicitly documents erasing widening,
  uses N/eps splices and specializes stride to N. Its affine kernel also
  normalizes the source while loop to a range loop. These require source repair
  and proof bridges before claiming statement correspondence.

`inventory.json` pins both selected files and every detected Python cast with
its function and line. The table gives the first cast per flagged task. It is a
reproducible lower-bound screen, not a full catalog of all possible differences.

| Task | Python source and first cast line |
| --- | --- |
| `apply_penalty` | `bench/repairs/tritonbench_g/apply_penalty/apply_penalty_repaired.py:15` |
| `bmm_chunk_bwd` | `bench/repairs/tritonbench_g/bmm_chunk_bwd/bmm_chunk_bwd_repaired.py:32` |
| `bmm_chunk_fwd` | `bench/repairs/tritonbench_g/bmm_chunk_fwd/bmm_chunk_fwd_repaired.py:34` |
| `chunk_cumsum_vector` | `bench/repairs/tritonbench_g/chunk_cumsum_vector/chunk_cumsum_vector_repaired.py:39` |
| `chunk_gla_simple` | `bench/repairs/tritonbench_g/chunk_gla_simple_min_dot_tile/chunk_gla_simple_min_dot_tile.py:45` |
| `context_attn_bloom` | `bench/repairs/tritonbench_g/context_attn_bloom/context_attn_bloom_repaired.py:7` |
| `context_attn_fwd` | `bench/repairs/tritonbench_g/context_attn_fwd/context_attn_fwd_repaired.py:40` |
| `context_attn_llama` | `bench/repairs/tritonbench_g/context_attn_llama/context_attn_llama_repaired.py:7` |
| `context_attn_mistral` | `bench/repairs/tritonbench_g/context_attn_mistral/context_attn_mistral_repaired.py:34` |
| `context_attn_nopad` | `bench/repairs/tritonbench_g/context_attn_nopad/context_attn_nopad_repaired.py:17` |
| `decay_cumsum` | `bench/repairs/tritonbench_g/decay_cumsum/decay_cumsum_repaired.py:24` |
| `dequantize_rowwise` | `bench/repairs/tritonbench_g/dequantize_rowwise/dequantize_rowwise_repaired.py:18` |
| `diag_ssm_triton` | `bench/repairs/tritonbench_g/diag_ssm_triton/diag_ssm_triton_repaired.py:18` |
| `embedding_triton_kernel` | `bench/repairs/tritonbench_g/embedding_triton_kernel/embedding_triton_kernel_repaired.py:16` |
| `fast_layernorm` | `bench/repairs/tritonbench_g/fast_layernorm/fast_layernorm_repaired.py:35` |
| `fast_rms_layernorm` | `bench/repairs/tritonbench_g/fast_rms_layernorm/fast_rms_layernorm_repaired.py:37` |
| `fifth_order_sph_harmonics` | `bench/repairs/tritonbench_g/fifth_order_sph_harmonics/fifth_order_sph_harmonics_repaired.py:24` |
| `flash_decode2_llama` | `bench/repairs/tritonbench_g/flash_decode2_llama/flash_decode2_llama_repaired.py:8` |
| `flash_decode2_phi` | `bench/repairs/tritonbench_g/flash_decode2_phi_block_checked/flash_decode2_phi_block_checked.py:8` |
| `fused_layernorm_triton` | `bench/repairs/tritonbench_g/fused_layernorm_triton/fused_layernorm_triton_repaired.py:46` |
| `index_select_cat` | `bench/repairs/tritonbench_g/index_select_cat/index_select_cat_repaired.py:11` |
| `int8_quantization` | `bench/repairs/tritonbench_g/int8_quantization/int8_quantization_repaired.py:17` |
| `iv_dependent_matmul` | `bench/repairs/tritonbench_g/iv_dependent_matmul_dot_shape/iv_dependent_matmul_dot_shape.py:17` |
| `layer_norm_fwd` | `bench/repairs/tritonbench_g/layer_norm_fwd_nonzero_tile/layer_norm_fwd_nonzero_tile.py:30` |
| `layer_norm_liger` | `bench/repairs/tritonbench_g/layer_norm_liger/layer_norm_liger_repaired.py:63` |
| `layer_norm_triton` | `bench/repairs/tritonbench_g/layer_norm_triton/layer_norm_triton_repaired.py:11` |
| `layernorm_fwd_triton` | `bench/repairs/tritonbench_g/layernorm_fwd_triton/layernorm_fwd_triton_repaired.py:15` |
| `log_softmax` | `bench/repairs/tritonbench_g/log_softmax/log_softmax_repaired.py:41` |
| `matmul_triton_autotune` | `bench/repairs/tritonbench_g/matmul_triton_autotune/matmul_triton_autotune_view_safe.py:53` |
| `mul_exponent_compensator` | `bench/repairs/tritonbench_g/mul_exponent_compensator/mul_exponent_compensator_repaired.py:11` |
| `nested_loops_processing` | `bench/repairs/tritonbench_g/nested_loops_processing/nested_loops_processing_repaired.py:8` |
| `quantize_global` | `bench/repairs/tritonbench_g/quantize_global/quantize_global_repaired.py:31` |
| `rbe_triton_transform` | `bench/repairs/tritonbench_g/rbe_triton_transform_checked_launch/rbe_triton_transform_checked_launch.py:15` |
| `reversed_cumsum` | `bench/repairs/tritonbench_g/reversed_cumsum/reversed_cumsum_repaired.py:33` |
| `reversed_cumsum_scalar` | `bench/repairs/tritonbench_g/reversed_cumsum_scalar/reversed_cumsum_scalar_repaired.py:30` |
| `rms_matmul_rbe` | `bench/repairs/tritonbench_g/rms_matmul_rbe_scalar_domain/rms_matmul_rbe_repaired.py:28` |
| `rms_rbe_matmul` | `bench/repairs/tritonbench_g/rms_rbe_matmul_launch_guard/rms_rbe_matmul_launch_guard.py:13` |
| `rowwise_quantization_triton` | `bench/repairs/tritonbench_g/rowwise_quantization_triton/rowwise_quantization_triton_repaired.py:35` |
| `softmax_flaggems` | `bench/repairs/tritonbench_g/softmax_flaggems_finite_seed_v2/softmax_flaggems_repaired.py:64` |
| `softmax_triton1` | `bench/repairs/tritonbench_g/softmax_triton1/softmax_triton1_repaired.py:9` |
| `softmax_triton2` | `bench/repairs/tritonbench_g/softmax_triton2/softmax_triton2_repaired.py:10` |
| `softmax_triton3` | `bench/repairs/tritonbench_g/softmax_triton3/softmax_triton3_repaired.py:16` |
| `token_softmax_bloom` | `bench/repairs/tritonbench_g/token_softmax_bloom/token_softmax_bloom_repaired.py:14` |
| `token_softmax_llama` | `bench/repairs/tritonbench_g/token_softmax_llama/token_softmax_llama_repaired.py:14` |
| `triton_softmax` | `bench/repairs/tritonbench_g/triton_softmax/triton_softmax_repaired.py:8` |

## Reproduce

```bash
python3 -m bench.audit_jit_correspondence --output /tmp/jit-correspondence.json
python3 bench/repairs/tritonbench_g/triton_argmax_nan_first/check_pair_source.py
python3 -m unittest bench.tests.test_jit_correspondence
lake build bench.repairs.tritonbench_g.triton_argmax_nan_first.Correct \
  bench.repairs.tritonbench_g.triton_argmax_nan_first.FPEquiv bench.tests.SourceDType
```

The AST guard is included in the existing `check_pair_source.py` acceptance
command and in unittest discovery. The Lean frontend regression is registered
in the existing `FPFrontendRegressions` Lake library. Independent official-comparator
replay remains unavailable in this environment. No GPU run is claimed.
