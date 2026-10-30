# Token attention scores: original Python bodies

`Kernels.lean` contains the complete original `_fwd_kernel_token_att1` body
from `token_attn_llama2_grid_axes_checked.py` and the existing block-origin
reuse candidate from `token_attn_llama2_reuse.py`. Both bodies preserve all 15
Python statements, including the final bare return, the runtime 0/1 loop,
masked metadata/key loads, reduction, scale and output store. Caller pointer
types, scalar values and constexpr dimensions are supplied outside the body.
`jit-correspondence.json` checks both complete bodies, signatures and bindings.
The candidate only moves `block_stard_index = start_n * BLOCK_N` before
`offs_n` and reuses that integer product. No hardware speedup is claimed.

The stage writes the following gathered scores to the caller-owned output:

```
Out[h, Start[b] + n] =
  sum(d < D, Q[b,h,d] * K[Loc[b,M-Length[b]+n], h/(Hq/Hkv), d]) / sqrt(D)
```

There is no softmax or value reduction. `Correct.lean` states this independent
Real formula on named query/key arrays. `IO.lean` supplies the complete grid
schedule, arbitrary admitted strides, metadata channels, active output mask
and exterior memory frame. `Whole.Valid` remains the public domain: dimensions
16/32/64/128, positive divisible heads, valid lengths/starts and active page
IDs, output injectivity, bounded accesses and the original signed32 envelope.
Repeated page IDs, permitted read aliases, arbitrary inactive location
prefixes and unrestricted FP32 payload words remain admitted. The existing
signed-to-Nat metadata adapter keeps its storage-width and nonnegative
refinement proofs; the source integer select and loop registers are signed.

The historical Lean source inserted FP32 input casts absent from Python.
The current quotation uses the actual typed loads and product. Its typed FP32
sum is bound in `Proofs/SourceBackend.lean` to the selected ordered FP32
reduction primitive with the same shape, axis, keep flag and complete input
layout. The scale uses the established FP32 computation profile; internal
carrier conversions and the implicit destination storage conversion remain
explicit in the execution/reference model. Triton 3.5.1
[`_store_legacy`](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/semantic.py)
converts the stored value to the destination element type. They do not add `.to(...)`
statements to the source, and no identity of arbitrary backend casts is
assumed. `SourceProgram`, `SourceExecution` and `SourceContract` connect the
actual complete source to its memory effects. `SourcePreservation` records
that its AST differs from the historical one.

`FPEquiv.block_origin_reuse_equiv` compares the actual original/candidate
whole grids. `Proofs/ReuseEquivalence.lean` proves full-state execution equality
and trace-safety equivalence for the reordered integer assignments, including
missing-register failures, then lifts them through every admitted grid
permutation. No numerical rewrite or floating input restriction is introduced.
Source safety, successful execution, all output values and exterior frames
are conclusions of the public IO proof.

Optional native and Tensor applications remain under `Misc/`. The native
reference uses address-labelled input words and the source's actual product,
reduction and scale graph. Each selected reduction plan covers all dimensions
exactly once, with its admitted zero padding. Tree and local contraction
choices inspect expression shapes rather than numerical words. Rounded add,
multiply and FMA remain the existing scalar instruction parameters; distinct
floating evaluation orders are not asserted equal. Source-derived FP32
mul/add adjacency is retained when describing contraction choices. The host
scale words remain `3e800000`, `3e3504f3`, `3e000000`, and `3db504f3`.

The original checked Python wrapper and `Contract.lean` are unchanged. The
existing nonempty-launch CUDA bounds are `Hq <= 65535` and
`ceilDiv(M,32) <= 65535`; empty calls retain their original domain. Physical
Tensor applications retain actual pointer addresses, all output cells, raw
and projected native frames, and input-view preservation. Disjoint views may
share an allocation, so the frame does not assert preservation of an entire
allocation containing output writes. The previously deferred launcher grid
product compatibility remains outside this contract.

`check_candidate.py` checks the existing wrapper/candidate transformation and
CPU guard/launch capture when Torch is available. Body-correspondence and
mutation tests require no Torch or GPU. Proof replay and validation limits are
recorded separately in the review checkpoint; no new GPU or numerical
experiment is implied by this source migration.
