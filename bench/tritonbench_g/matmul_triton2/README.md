# matmul_triton2

Current repaired Python implementation: [matmul_triton2.py](matmul_triton2.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

fun (a,b) row col => ∑ k : Fin innerSize, a row k * b k col

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.

`Kernels.lean` contains the repaired Python body and its mask-caching candidate.
It retains the nine runtime `uint64` casts, program/lane casts, the cast of
`group_size_m`, grouped ordering, default `tl.dot(a, b)` options, and the original
post-loop pointer/mask/store sequence. There is no second complete Lean source
or independent whole-kernel executor.

The public `SourceRuntime.SourceBounds` records validity of the explicit scalar
and lane casts and the launch extent. It keeps strides independent and does not
restrict the proof to one block or one autotune configuration.
`SourceBounds.of_contiguous_ready` derives this boundary from the existing
contiguous finite invocation domain. Geometry, positive K-block width and
output separation remain explicit. Both full source schedules have checked
success, access safety, complete outputs and exact raw frames.

The common matrix memory layout and pure targets are shared with
`matmul_triton1`; this task retains its own grouped schedule and default-dot
compiler profile. The local dot profile preserves the source operands, zero
seed, carry and precision options, and does not claim a split-rounding/fusion
arithmetic identity. The Real headline is the independent full-K matrix product.

The original kernel body is pasted with its Python indentation, colons and
comments. `jit-correspondence.json` checks its complete Python AST and the
external call signature; no casts or runtime statements are normalized away.
