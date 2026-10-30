# Direct FP32 chunk gradient repair

`chunk_bwd_dqkg_fn(do, q, k, v, g, h, dh, scale)` returns fresh contiguous
`dq`, `dk` with shape `[B,H,T,K]` and `dg` with shape `[NK,B,H,T]`. DG remains
the per-key-tile contribution produced by this kernel. Summing key tiles and
reverse cumulative sum belong to downstream code.

The admitted inputs are same-device CUDA FP32 strided tensors with exact
canonical strides, no conjugate/negative lazy flags, naturally aligned pointers
and pairwise disjoint data spans:

| Inputs | Shape |
| --- | --- |
| q, k | `[B,H,T,K]` |
| v, do | `[B,H,T,V]` |
| g | `[B,H,T]` |
| h, dh | `[B,H,ceil(T/64)*K,V]` |

K and V are positive. Empty B, H or T allocates the original outputs and skips
launch. Python float/int scales, excluding bool, are explicitly rounded to
finite FP32. Both signed zeros are retained. The nonempty guard bounds all
formed source indices by a conservative signed-32 envelope, bounds the CUDA
grid axes, and checks input/output pointer additions against 64-bit wrap.
Output pointer checks run after the original allocations, before launch.

The arithmetic kernel retains the accepted repair: explicit zero padding,
correct history-chunk base and K mask, BT=64, BK/BV in {16,32,64}, and both
four/eight-warp choices. The default FP32/TF32 dots, exp, reductions, operation
order, and `dg.fill_(-1e9)` are unchanged. There are no input normalization
copies or snapshots. The last-row correction uses gated dK before its final
attention dot.

`Contract.lean` proves exact success of the executable admission model.
`Proofs/DirectLayout.lean` derives disjoint physical placement, literal queried
stride equality and native pointer safety. Allocation parameters describe only
the ordinary fresh objects, aligned spans and capacity; they contain no source
execution or safety result.

`Direct.chunk_bwd_dqkg_real` in `Correct.lean` executes the complete flattened
erased source across every tuner trial and selected launch, proves all returned
values equal the independent finite contractions in `Spec.lean`, returns the
same object/base/shape/device descriptors, and gives the exact cell frame.
Its input arrays range over arbitrary Reals independently of byte decoding.

`DirectNative.chunk_bwd_dqkg_native` and `chunk_bwd_dqkg_if_local` in
`FPEquiv.lean` start from the original physical input bytes, execute complete
typed source grids on predecessor states, and return all materialized output
words plus physical byte serialization and frames. They require the explicit
selected compiler/profile/layout mapping described in `SOURCE_BINDING.md`.
The numerical transport premises are scalar instructions and fixed K8 MMA
coordinates only. There is no native-to-Real error bound.

`check_contract.py` runs the actual retained arithmetic AST with CPU arithmetic
and CUDA launch/allocation stand-ins. It checks 12 admitted shape cases under
both warp choices and with/without tuner trials, 93 rejection cases, complete
outputs, argument/return identities, the original fill and backing-byte frames.
It includes time/key/value tails, multiple value and time blocks, several
heads/batches, empty dimensions, negative-zero scale and disjoint views sharing
one storage. This harness makes no GPU numerical claim.

Run the harness with:

```sh
python bench/tritonbench_g/chunk_bwd_dqkg/check_contract.py
```
