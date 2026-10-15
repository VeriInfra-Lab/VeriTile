# attention_kernel_aligned

The explicitly named fp16 surface and its complete output/IO correctness proofs
build successfully. The original Python remains unchanged. The bf16 host branch
is still unsupported by the recorded native compiler and is not counted as a
verified source profile.

`attention_kernel_aligned_fp16_surface` fixes the original `OUT_DTYPE` constexpr
to `tl.float16`. It replaces the old generic
`attention_kernel_aligned_fwd_kernel_aligned_surface`; the corresponding generic
lowering theorem is also retired in favor of
`attention_kernel_aligned_fp16_surface_toAlgorithm_supported`. The previous
arbitrary-`FloatDType` lowering claim did not elaborate with the typed frontend.
The existing public output summary and IO specification already selected fp16;
their mathematical contracts are retained and now connected to the actual typed
source body.

The score dot and accumulation remain fp16. The value dot retains its actual
fp32 result and fp32-to-real erasure cast. Bias remains unscaled. Unchecked
block-pointer loads retain the source's unspecified padding; since the checked
axis list is empty, an explicit lemma proves that the padding value is never
observed. No zero-undef input condition is added.

The real output summary models real arithmetic. The rounded IO specification
still assumes `R.round .fp16 = id`, with its existing positive dimension and
nonempty key-block premises. No new fp32 rounding premise is added: fp32-to-real
cast targets the real channel and is definitionally independent of fp32
rounding. These are algorithm-layer proofs, not bit-accurate native refinement.
Host launch composition remains outside the per-program proof.

Evidence (2026-10-14):

- `../../review-checkpoints/2026-10-14/aligned-fp16-build/` records the complete
  Lean target, exit 0, 143 unchanged captured inputs, and archived exact inputs.
- `../../review-checkpoints/2026-10-14/aligned-native/` records offline Triton
  3.8.0 compilation on CUDA sm90 for the unchanged kernel AST. fp16 compiles;
  bf16 fails at `tl.dot(q, k, out_dtype=OUT_DTYPE)` with
  `out_dtype=bfloat16 is unsupported`. The tested native dimensions are
  `BLOCK_M=BLOCK_N=HEAD=64`, `BIAS_LAST_SIZE=64`, `B0_NUMEL=128`, four warps and
  two stages. No GPU kernel or host wrapper was executed.
- `../../review-checkpoints/2026-10-14/aligned-source-profile/` contains the
  complete source audit and adversarial profile regression results. The new
  `dtype-constexpr` source comparison binds the whole original Python hash,
  successful fp16 and rejected bf16 native records, every native artifact hash,
  and the exact Lean source selector. It substitutes only the five direct dtype
  operands; it neither rewrites the Python file nor certifies other dtypes.

The bf16 source portability gap is a concrete remaining limitation. Supporting
it would require a separately reviewed source change and fresh proof/native
receipts; no such repair is silently applied here.
