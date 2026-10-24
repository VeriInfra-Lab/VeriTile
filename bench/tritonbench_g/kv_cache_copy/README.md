# kv_cache_copy

Current entry: [mixed five-format raw-bit proof](../../review-checkpoints/2026-10-15/kv-cache-raw-bit-closure/README.md), public `RawBitCorrect.lean::mixed_five_format_raw_bit_correctness`. It proves exact word storage for the complete old accepted masked x-block, with 25 independent K/V format pairs. Both loads and metadata use initial memory; complete encoded memory, success, dual readback and frame are proved. Raw NaN payloads, negative zero, cross-aliasing, masked tails and zero width are retained.

Whole memory needs no cache separation or injectivity; dual readback keeps the old distinct-cache and active V injectivity conditions. Arbitrary legacy carriers remain ordinary theorems. These are software word formats: no native64, byte ABI, full Python static loop or GPU claim is made. Three modules and 20 named checks passed independent root compilation; one new public declaration is counted. See [current integration](../../review-checkpoints/2026-10-15/kv-cache-software-metadata/README.md).

## Historical typed-carrier proof


Source: `kv_cache_copy.py` (TritonBench-G v1).

`Proofs/Execution.lean` retains the accepted masked, single x-block decode source.
K and V independently select six source annotations (36 combinations), with
matching source/cache storage. Five annotations name physical formats: fp16,
bf16, fp32, f8e4 and f8e5. The float64 annotation lowers to abstract `.real`
and has no native 64-bit storage binding; only 25 combinations of the five
named physical formats count as native-format profiles. Both loads execute
before either store. `source_projection` proves exact recovery of
`copy_to_kvcache_seqlen1_xblock` after storage erasure, including all metadata,
address formulas and masks.

`Proofs/TypedCopy.lean::TypedCopy.correctness` proves successful typed execution,
independent K/V readback from the original input snapshot, and preservation
outside both active write sets. K/V cache regions must be distinct for K
readback. V readback requires injectivity only among active lane addresses;
positive component stride is sufficient but is not required by the contract.
Input/output aliases are permitted and modeled by the actual load/store order.
No output values are assumed. `k_typed_readback` and `v_typed_readback` preserve
complete well-typed payloads, including NaN payloads and signed zeros.

The opaque backend needs no arithmetic or rounding law. It uses the raw fp32
load backend from `kcache_copy_triton/Proofs/TypedCopy.lean`. Masked-off load literals
and casts can be arbitrary because no active store observes them. The frontend
`.real` channel carries opaque values, not mathematical real numbers. Its
precision-tagged fp32 use is modeled; unqualified `.real` does not prove fp64.

`Proofs/BitWitness.lean::TypedCopy.bit_success_witness` executes two active lanes in
a partial x-block with fp32 K and fp16 V, checks special-value bit payloads at
strided destinations, and checks untouched inactive destinations. This is a
Lean witness, not a GPU experiment.

The contract covers homogeneous copies within each K/V pair. It does not prove
conversion between different input/cache dtypes, the full static x-block loop,
machine-address overflow or CUDA refinement. The original accepted x-block
scope, Nat metadata abstraction and finite-region/launch obligations remain.

The explicit public entry is `FPCorrect.lean`.

Checks (run in repository root, after dependencies are built):

```sh
lake env lean bench/tritonbench_g/kv_cache_copy/Proofs/Execution.lean
lake env lean bench/tritonbench_g/kv_cache_copy/Proofs/TypedCopy.lean
lake env lean bench/tritonbench_g/kv_cache_copy/Proofs/BitWitness.lean
```
