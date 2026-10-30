# Byte-separated candidate domain

`lora_expand_gemv_byte_separated.py` is a separate proposed repair of frozen
`lora_expand_gemv_repaired.py` (SHA256
`8f36abb103e6dbe0b646c65a8d763a7d57afdd901a9888ea398987f219e70d81`).
The canonical selection and its confirmed alias defect remain unchanged until
independent acceptance. Candidate proofs live under `ByteSeparated/`.

The sole behavioral change replaces the output/read `_cdata` identity test
with a test on actual complete untyped-storage byte intervals. For each read
operand X, W and indices, query `(p,n)=(storage.data_ptr(),storage.nbytes())`.
For Y query `(q,m)`. The pair is accepted exactly when

```
m == 0 or n == 0 or q + m <= p or p + n <= q.
```

Endpoint arithmetic uses Python integers. Half-open intervals allow adjacent
backings; zero-byte storage is empty. Checks remain after device/layout/shape
validation and before pointer alignment, metadata inspection and dispatch.

Previously different StorageImpl identities were accepted even when DLPack
imports shared physical bytes. Those overlapping imports now reject. Ordinary
disjoint backings remain accepted, irrespective of how they were created.
Read/read aliases remain accepted. Nonempty views sharing one larger backing
remain rejected even if their view windows are disjoint: existing frame claims
protect the **entire** original read backing, including unused bytes. Shared
zero-byte backings now pass the separation check because they contain no byte.
An empty view with a nonempty backing still uses its full backing interval.

All other conditions are identical: contiguous resolved FP32 X/Y, FP16 W,
signed INT32/INT64 indices, common CUDA device, both add modes, singleton 4D
weights, aligned offset views, extra rows/columns, sentinel -1, all prior
dimension/content/alignment bounds, and K in **1..4096**. No finite-value guard
is added. The kernel, launch geometry, strides, reduction dtype/tree,
contraction behavior and numerical interfaces are unchanged. No copy,
allocation, normalization or host-scheduling implementation is added.

The mathematical target is the finite K-term X/W contraction plus original Y
when `add_inputs=True`; sentinel rows and all other Y bytes stay unchanged.
Native correspondence retains the original FP16 products/reduction, conversion
to FP32, and optional FP32 add. Scalar instruction conformance remains the
existing explicit numerical boundary; floating results are not equated with
Real arithmetic.
