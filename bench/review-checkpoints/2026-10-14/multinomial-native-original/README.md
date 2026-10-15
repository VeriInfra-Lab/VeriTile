# Multinomial unchanged-source native observation

The pinned `thunlp/TritonBench@603e28a` Python source has SHA256 `3bd5a7c11f42ade4e2ae69b58019888a419990bfbf3fe1d54d877e50e36de336` and Git blob `f4e073d925beb986b4a1dd3a47e2ae00bb7931f6`. Extraction copied the complete JIT and imports, checking every extracted AST. The CUDA test harness and host wrapper were not invoked.

All 8 requested profiles compiled with installed Triton 3.8.0 for sm90: fp16/bf16/fp32/fp64 Scores, each with i32 or i64 runtime strides/dimensions. Seeds/Offsets/Indices/Outputs remain i64, and original wrapper constants remain BLOCK=8, BLOCK_N=128, num_warps=8. All 40 TTIR/TTGIR/LLVM/PTX/cubin artifacts were copied with verified hashes. Recorded inputs were unchanged; compiler libraries are fingerprinted, not copied into this checkpoint.

TTIR for fp32/i32 contains actual i64-to-i32 `arith.trunci` on offsets and Bool-to-i32 `arith.extui` on the argmax input. This is offline original-source compilation only, not GPU execution, Lean source-AST acceptance, a whole-kernel correctness proof, native refinement, or independent comparator acceptance. The corpus port remains a scaffold pending its complete source/proof surface.
