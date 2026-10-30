# Dual-plane destination copy: checked repair candidate

## Original defect and selected repair

The original dual-plane kernel ignores head counts/strides and accesses rounded
feature widths without masks. Concrete counterexamples remain recorded:

| Original input | Failure |
| --- | --- |
| One row, twelve nope heads, width four | Only head zero is copied; eleven heads retain the old output. |
| Width three with four rounded lanes | The exterior fourth backing word is overwritten; exact-size storage is accessed out of bounds. |
| Destination backing `[0,1,2]` viewed with stride two | Raw loads use `[0,1]` instead of logical `[0,2]`. |

The first two are source defects; the selected repair uses the complete masked
multihead kernel separately for each plane. Original and repaired programs remain
separate. Proof of the repair does not certify the defective original source.

## Target, precision and inputs

For each plane independently, the mathematical target is
`Out_plane[DestLoc[row], head, dim] = Source_plane_initial[row, head, dim]`,
with untouched coordinates and exterior memory preserved. Payload precision
defaults to FP32 because production dtype is unspecified; int32 destination
metadata is retained. Plane head counts and feature widths are independent.

The wrapper asserts [kv1's direct-buffer contract](../destindex_copy_kv1/README.md)
for each plane and shared destination vector: contiguous strided tensors, FP32
payloads/int32 destinations, matching per-plane shapes and enough source rows,
one CUDA device, no unresolved negative bit, signed-int32 extents, bounded tiles,
natural pointer alignment and bounded unique destination rows. Both output spans
are disjoint from all reads and from each other. Read/read aliases, offset views,
tails, independent empty planes and every FP32 bit pattern remain admitted.
No payload allocation, normalization or copyback is introduced.

`Contract.lean` and `TensorBridge.python_checks_iff` expose the exact checked admission.
Shared `TensorBinding.checked_read` uses core Torch storage semantics to connect
contiguity, actual pointers, storage offsets and negative-view flags to logical
values. Capacity and initial signed metadata readback remain explicit storage
bindings; no kernel-output callback is assumed.

## Source and public kernel proofs

`Kernels.lean` aliases only kv1's canonical typed Triton DSL and existing
index-product commutation variant. The repaired Python `_copy_heads` AST agrees
with kv1's original masked source except for its function name.

- `Correct.destindex_copy_real` proves `Spec.Real` for an arbitrary plane's
  complete kernel grid, its independent copy formula and memory frame.
- `FPEquiv.index_commuted_equiv` proves both actual typed variants successfully
  execute and agree on all typed memory, with `assumptions = []`.
- `FPEquiv.destindex_copy_fp32` in [Misc/NativeObservation.lean](Misc/NativeObservation.lean)
  proves exact raw-word copying and the exterior
  word and byte frames for each plane, with finite-index/pointer safety derived
  from the checked contract.

The public task is the common repaired kernel applied to either plane. Historical
simultaneous two-launch composition, output identity and return-value results are
retained in optional `Misc/Host.lean`; host orchestration is not a required
proof. `Composition` retains the independent plane/stage formulas and reusable
frame lemmas without importing host proofs. No numerical law or finite-lane
premise is required for this copy/index rewrite.

## Evidence and remaining boundary

`check_contract.py` checks actual source ASTs, guard correspondence, nine
admissions, forty-seven rejections, all three original counterexamples, raw
special FP32 words, both frames and read buffers using CPU backing storage,
reversed program order and mocked CUDA context/launch. Both public Lean entries
have targeted builds, standard-base axiom audits, non-circular formula checks
and an empty FP-assumption report. No GPU run or independent numerical replay is
claimed. Shared grid-product overflow and segmented launch compatibility remain
deferred globally.

Modeling references: [shared physical load/store contracts](../../../../documents/GpuMemoryModel.md#physical-byte-address-contracts)
record Triton masks, PTX byte addressing/alignment and ordinary Torch API laws;
[shared FP32 Tensor storage laws](../../../../VeriTile/Torch/FP32.lean)
provide the actual-pointer and logical-value connection reused here.

`Misc/TensorBridge.lean` holds the optional Tensor/check correspondence helpers;
`Correct.lean` and `FPEquiv.lean` retain the mathematical and source-equivalence entries;
[Misc/NativeObservation.lean](Misc/NativeObservation.lean) retains the auxiliary native kernel result.


The main Real/FP32 interfaces now use the shared source-safe
`ProgramIO.ImplementsOn` judgment over finite logical channels. `IO.lean` binds
this task's exact kernel alias to the shared descriptor in
`destindex_copy_kv1/IO.lean`: FP32 payload `[count, heads, dims]`, signed destination
metadata `[count]`, the destination-dependent output scatter, no scratch, and
the complete arbitrary-order source schedule. The logical metadata condition is
nonnegative and injective, preserving the earlier Nat destination-map domain.
The independent formula simply copies every payload element to its declared
destination. All FP32 words remain admitted.

The implementation reuses `destindex_copy_kv1.Proofs.PublicIO.implements`, including
actual source trace safety at every predecessor state, successful full schedule,
all typed outputs and the exact exterior-cell frame. The pure integer-index
commutation equivalence stays unchanged. Previous unrestricted raw-state results
are retained in `Proofs/RawResults.lean`. Checked two-plane Tensor applications
and physical byte frames remain unchanged in `Misc/TensorBridge.lean`; the new
regional IO does not claim an additional physical transport proof.

Targeted builds cover the new descriptor/raw-result module, both public entries
and the Tensor consumer. The mathematical/native formula and public surface
audits use shared IO relations and logical metadata only. This format pass does
not rerun the historical CPU checks, independent replay or GPU experiments.

### 共享原始函数体（2026-10-25）

所选 `_copy_heads` 和 `destindex_copy_kv1` 的所选原始函数拥有完全相同
的参数角色及函数体（包括全部 mask 和 `return`）；显式对应清单分别
检查两个原始 Python 文件与 kv1 中同一份可读 `triton { ... }`。
两路 KV 调用的独立 head/feature 维度、实际 IO 和所有证明范围均保留。
已有候选也明确复用同一已标注为新写的索引候选 Python 对照。
