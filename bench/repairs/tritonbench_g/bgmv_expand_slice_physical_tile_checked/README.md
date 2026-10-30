# BGMV expand slice

`Kernels.lean` contains the complete original Python function body and the
existing output-first candidate. Each has an INT32 and INT64 metadata call
profile, with pointer dtypes, runtime strides and constexpr bindings outside
the pasted body. The body keeps the docstring, all casts, early return,
`tl.cdiv`, branches, loop, loads, reduction and masked store. The candidate
exchanges only the operands of the optional final FP32 addition.

The selected Python implementation and wrapper are unchanged. They accept
FP32 X/Y, FP16 W, contiguous rank-3 or rank-4 weights, either signed index
width, both addition modes and aligned output views. Metadata -1 skips the
row. The wrapper checks complete physical output/read backing separation,
`256 * next_power_of_2(K) <= 2**20` on nonempty calls, and its existing
shape/device/layout/offset conditions. K=0 remains rejected by Python; empty
batch or output-row calls return without launching.

For every active batch b, adapter a=indices[b], and row n<N, the independent
Real specification in `Correct.lean` is

```
Y_after[b, offset+n] = sum(k<K, X_before[b,k] * W_before[a,n,k])
                     + (Y_before[b, offset+n] if add_inputs else 0)
```

`IO.lean` supplies the complete grid, logical tensors, gathered weight reads,
active write masks and exact exterior frame. The kernel theorem retains its
broader mathematical Call domain, including backed K=0 inputs and nonnegative
metadata with backed reads. It does not acquire extra Python restrictions.
`Proofs/IOExecution.lean`, `AccessSafety.lean`, `SourceSafety.lean` and
`IOGrid.lean` connect execution and every actual access to that interface.

`Proofs/SourceView.lean` provides an internal proof decomposition of the known
CAST_TYPE=True branch. It proves equality of full execution results, including
failure, and equivalence of the actual safety traces for every local algebra
and state. Both metadata profiles are connected to the same complete kernel.
The authoritative source remains the ordinary quotation in `Kernels.lean`.

`Proofs/SourceValues.lean` records the actual ordered floating operations:
FP32 X load and conversion to FP16, FP16 product and reduction, genuine
FP16-to-FP32 widening, then optional FP32 addition. No FP32-to-FP32 cast is
invented after that addition. `Proofs/Direct/` retains the separate legacy
explicit-cast program and its formula; it is not substituted for the new body.

`FPEquiv.lean` proves complete source/output-first equivalence under the shared
strict software FP32 addition certificate, for all words without a finite-value
restriction or experimental numerical rule. `Proofs/PairConnection.lean`
preserves complete states, failures and access safety; `Proofs/PairIO.lean`
connects that result to the unchanged full-grid IO. This is a software-model
statement, with no additional hardware-conformance claim.

The optional native and Tensor endpoints remain in `Misc/`. Their original
input domains, source reduction plans, primitive conformance conditions,
physical backing frames and wrapper guards are retained.

`check_source.py` checks the four raw bodies and the actual Python launch
using `jit-correspondence.json`, without rewriting casts or arithmetic.
It writes its report to stdout and does not require a local evidence cache.
`check_revision.py` retains the historical 14 Torch-backed alias/tile controls;
those require Torch and Triton and are separate from source correspondence.

```
python bench/repairs/tritonbench_g/bgmv_expand_slice_physical_tile_checked/check_source.py
lake build bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Correct \
           bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.FPEquiv
```
