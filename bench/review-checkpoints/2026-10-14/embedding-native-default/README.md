# Embedding implicit default conversion — native observation

The unchanged original JIT compiled offline in all six fp16/bf16/fp32 × i32/i64 vocabulary-upper-bound profiles for CUDA sm90. Runtime arguments remain runtime arguments; only original block/hidden-size constexprs are specialized. No wrapper or GPU kernel executed.

The i64 profiles retain an explicit `arith.trunci ... : i64 to i32` for the default token, followed by signed extension for comparison. The installed runtime maps the Python value 2^32 to i64. This supports the need to model the masked-load conversion; it is not a native refinement proof or a claim of GPU memory safety.

The original Python, extracted verbatim JIT, runner, complete compiler artifacts, hashes and all profile results are retained. [Triton load documentation](https://triton-lang.org/main/python-api/generated/triton.language.load.html) specifies implicit other conversion.
