# Both canonical precision routes fail ordinary scalar specialization

Canonical source SHA256:
`ad6eb70f7493aff7f323558870765887e09c208c3a547e2c6271fe519efddd7d`.

The actual checked kernel fails Triton 3.7.1 offline compilation on small
admitted positive workloads in both retained dtype routes. The native JIT
specializes integer 1 as a constexpr Python integer, but the source calls
tensor `.to(tl.int64)` on that integer.

| Source dtype/configuration | Admitted shape | First failing expression |
| --- | --- | --- |
| FP16, BK64, stages3 | M=N=1, K=64 | line 33: `M.to(tl.int64)` |
| FP16, BK64, stages3 | M=N=2, K=64 | line 34: `stride_ak.to(tl.int64)` |
| E4M3FN, BK128, stages4 | M=N=1, K=128 | line 33: `M.to(tl.int64)` |
| E4M3FN, BK128, stages4 | M=N=2, K=128 | line 34: `stride_ak.to(tl.int64)` |

All four use the actual BM128/BN256/GROUP8/warps8 configuration and the
ordinary device-property value NUM_SMS=132, producing grid `(1,)`. They pass
the exact source wrapper using real Torch CUDA FakeTensor metadata. Matching
rank-two contiguous tensors are a subset of the admitted arbitrary strided
domain; no additional input restriction is imposed. All-one operands have
the independent mathematical product 64 for the FP16 witnesses (`0x5400`)
and 128 for E4M3 (`0x70`), both exactly representable in the respective output
dtype. Both routes keep their source-selected precision and FP32 live carry.

`check_kernel_counterexample.py` imports the unchanged JIT function and
launch-metadata decorator. It captures exact wrapper dimensions, strides,
configurations and NUM_SMS, then calls `native_specialize_impl` and
`JITFunction._pack_args` in `kernel.arg_names` order. Every constexpr is
asserted equal to its captured value, and all three pointer dtype signatures
are checked. Matching aligned CPU pointer handles supply only the offline
pointer ABI and are never passed to a GPU launcher. Each canonical compile
raises `CompilationError` with `'int' object has no attribute 'to'` before any
matrix result exists.

For each dtype, the same unchanged source compiles with a deliberately forced
all-dynamic scalar ABI. Those two controls validate the compiler path and
show why that artificial signature can hide the actual specialization
failure. They are different ABIs, not two source variants or successful
canonical wrapper executions. No GPU execution is claimed.

This is a current canonical source defect, not a numerical deferral. The
diagnostic exits zero after confirming the four expected failures and two
explicit controls. No kernel-success theorem is asserted for the failing
calls. All 134 preexisting original and repair files remain byte-identical;
their existing abstract mathematical, source, instruction, Tensor and frame
results retain their declared scopes. The prior source diagnoses and both
FP16/E4M3 routes remain preserved. No core, ledger or Lake file was changed,
and no primitive numerical experiment was added.

The frozen evidence is
`.veritile-cache/kernel-evidence/matmul_persistent_triton/delivery.json`.
