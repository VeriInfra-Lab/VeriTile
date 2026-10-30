# Kernel boundary review

The canonical selection is `contract_revision.source` for `token_attn_reduceV`
in `bench/tritonbench_g/task_progress.json`:
`token_attn_reduceV_repaired.py`, SHA256
`1182d7c84e373ae328250b0d8c63886d12f8bde5c36279bbcdc131db1de0e020`.
The source, its guards and all existing Lean proofs remain unchanged.

The independent target is the gathered weighted sum
`sum(n < length[b], P[h,start[b]+n] * V[table[request[b],n],h/(Hq/Hkv),d])`.
P, V, accumulation and output are FP32; request/table/start/length payloads are
signed int32. The kernel keeps every 128-lane masked product, reduction, loop
addition and final output cast. It does not compute softmax. A zero-length row
writes zero to every output coordinate; an empty batch returns without launch.
The current domain retains D in {16,32,64,128}, positive divisible head counts,
contiguous nonnegative views, valid metadata and output/input byte separation.
There is no finite-payload or nonzero-length restriction.

## Admitted launch exceeds the CUDA grid y limit

The existing guard admits these small concrete inputs:

| Argument | Shape | Dtype / contents |
| --- | --- | --- |
| P | `[65536,0]` | FP32, empty |
| V | `[0,1,16]` | FP32, empty cache |
| Out | `[1,65536,16]` | FP32, initially 7 |
| request table | `[1,0]` | int32, empty |
| request / start / length | `[1]` each | int32, each contains 0 |

All metadata/content, extent, alignment and separation checks succeed. Out has
1,048,576 elements and occupies 4 MiB. Batch is one, so this is a launching call
despite its zero sequence length. The actual wrapper submits grid `(1,65536)`
with `BLOCK_DMODEL=16`, `BLOCK_N=128`, four warps, one stage and
`kv_group_num=65536`. Its canonical strides also fit signed int32.

NVIDIA's [CUDA C++ Programming Guide 12.6.3, Table 21](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability)
lists the “Maximum y- or z-dimension of a grid of thread blocks” as **65535**
for the listed compute capabilities 5.0 through 9.0, including SM80 and SM90.
Thus this admitted launch exceeds a single-axis architectural limit. Its grid
product is only 65536: this finding is distinct from the separately deferred
Triton launcher signed grid-product overflow issue.

The source-specific probe runs the unchanged `_checked_arguments` and captures
the unchanged wrapper's actual launch arguments for Hq=1, 65535 and 65536.
It uses genuine CPU Torch tensors with an explicit CUDA admission facade and
a mocked device context; the large grids are captured, not executed. The
official document establishes the architectural limit. No GPU launch failure
or `cuLaunchKernel` error was observed or claimed.

Small actual Triton-interpreter controls cover GQA with Hq=6/Hkv=2, D=16 and
sequence lengths 0, 1 and 129. Every output matches an independent weighted sum,
and all input tensors remain unchanged. The original checker also still passes
all 48 cases. These are CPU checks, not universal native numerical conformance.

## Preserved proofs and subsequent cache work

The existing single DSL quotation and conditional source/grid/Real/native
proofs are retained. A logical grid execution model does not establish that
every Python-admitted grid can be submitted on the documented CUDA devices.
No new head guard, segmented launch, layout conversion or other compatibility
implementation is introduced by this review.

A separate Lean cache probe found that the inherited compiled `withSum` and
`Native.algebra` do not reduce according to their current source definitions:
`(withSum M R).cast = M.cast` and the declared `fp32Load` identity fail by `rfl`.
The ordinary Real multiplication probe passes. This records the existing
Algebra-layout cache mismatch as a necessary dependency refresh before a later
proof update. It is separate from the canonical Python launch defect. The probe
used `--check`; no task module artifact was rebuilt and no core source changed.

Evidence, the reproducible launch probe, official document/excerpt, logs and
the frozen delivery packet are under
`.veritile-cache/token-attn-reducev-kernel-evidence`. The packet pins all original
task source hashes. Whole-domain native launch success is not asserted.
