# Canonical kernel review: launch-domain defect

This review fixes the source to `ledger.contract_revision.source`,
`token_attn_llama2_repaired.py`, SHA256
`e2c0688906af29ede2ac51f3d15eed7c6d7828796817bc7132578ef5282c9d1c`.
The current guard admits inputs whose actual three-axis launch exceeds CUDA's
documented y or z limit. Complete successful execution over the entire current
admitted domain therefore cannot be certified. The source, guards, public
entries, and existing conditional kernel proofs are preserved unchanged.

This is the gathered QK score stage, with the independent mathematical target

```
Out[h, Start[b]+t] =
  sum_d Q[b,h,d] * K[Loc[b,M-Length[b]+t], h/(Hq/Hkv), d] / sqrt(D)
```

for every `t < Length[b]`. It mutates the supplied output and returns `None`;
there is no softmax or value reduction. The selected precision is FP32 for Q,
K, and output, with signed-int32 Loc/Start/Length. The full floating kernel AST
equals the preserved original AST. It retains 32-token tiles, the 0-or-1 loop,
unmasked D-element Q loads, masked gathered K loads, dimension products and
sum, and the final scale multiply. D is one of 16, 32, 64, 128; the launch uses
two warps and one stage. The native specification retains actual marshalled
host scale bits, complete reduction trees, and permitted FP32 FMA choices.
Scalar instruction meanings, FTZ behavior, and compiler choices are model
parameters, not unresolved per-task numerical experiments. Ideal-Real sqrt
scaling is distinct from a claim that floating execution equals exact Reals.

The unchanged guard requires Q `[B,Hq,D]`, K `[cache,Hkv,D]`, output `[Hq,T]`,
Loc `[B,M]`, and metadata vectors `[B]`, on one CUDA device. Heads are positive
and Hq is divisible by Hkv; the maximum is a nonnegative Python `int`.
All buffers are contiguous and have no negative-view flag. Nonempty launches
require the exact canonical dense strides. The complete shape/stride envelope
must be below `2**31`. The metadata guard checks `0 <= length <= M`,
`0 <= start <= T-length`, valid cache pages only in each active suffix, and
pairwise disjoint nonempty output segments. Inactive prefixes may contain
arbitrary signed-int32 entries; active page IDs may repeat. Same-dtype read
aliases are allowed, while overlapping FP32/int32 read views are rejected.
Output/input view intervals are disjoint. Nonempty view pointers are aligned
to four bytes, and all view extents fit uint64.

Empty B or M still validates header, envelope, and integer metadata before
returning without launch. Such calls do not require canonical inactive strides,
alignment, or view-disjointness checks. For nonempty B and M, even all-zero
lengths do not suppress the wrapper launch. The witnesses below use positive
lengths, so they do not depend on this latter edge case.

The replay invokes the **unchanged actual guard and wrapper** over ordinary
contiguous CPU Torch tensors with an explicit CUDA admission facade. Only
`is_cuda`, the CUDA context, and the launch are replaced. No kernel interpreter,
GPU execution, or numerical experiment runs. The actual guard's local
`envelope` and the exact wrapper launch tuple are captured.

| Witness | Q shape | K shape | Output shape | Loc shape | M | Captured grid | Guard envelope |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| y overflow | `[1,65536,16]` | `[1,1,16]` | `[65536,1]` | `[1,1]` | 1 | `(1,65536,1)` | 3,342,537 |
| z overflow | `[1,1,16]` | `[1,1,16]` | `[1,1]` | `[1,2097121]` | 2,097,121 | `(1,1,65536)` | 8,388,732 |

Both have `Start=[0]`, `Length=[1]`, all locations zero, all Q/K values zero,
and output initially -7. All six buffers are separate, contiguous, and aligned.
Both pass every canonical guard and have a grid product of only 65,536, which
fits signed int32. The independent mathematical score at every output is zero.
The wrapper instead submits a grid prohibited by the CUDA architectural limit.
The evidence infers invalid launch from the specification; it does not claim
an observed CUDA exception or an executed floating result.

The architectural basis is the already frozen CUDA C++ Programming Guide
12.6.3, Table 21, row "Maximum y- or z-dimension of a grid of thread blocks":
65,535 on its listed compute capabilities, including SM80 and SM90. The frozen
Triton 3.7.1 source review and current matching source hashes show that the
wrapper coordinates are passed directly to `config.gridDimY` and
`config.gridDimZ`; no flattening or segmentation changes these coordinates.
No network retrieval or compiler replay was needed. A control at y=65,535,
a control at `ceil(M/32)=65,535`, and an ordinary small control also pass the
original guard and capture the expected tuples.

This defect concerns the wrapper's input/physical-launch contract. It is
separate from deferred signed grid-product overflow in shared launcher code.
The old internal logical-grid, Real, native-expression, and physical-frame
proofs remain useful under their explicit execution models; they do not
establish CUDA launch validity for the two admitted witnesses. All 44 existing
source/document files across the original and repaired task are hash-preserved.
No new Lean build or universal source-success theorem is claimed in this
diagnostic packet.

Evidence is frozen under `.veritile-cache/kernel-evidence/token_attn_llama2/`.
Any later correction must be a separately reviewed, explicit input-contract
candidate with the necessary per-axis limits. No guard restriction, segmented
implementation, dtype change, host/allocator layer, or shared-core change is
introduced by this review.
