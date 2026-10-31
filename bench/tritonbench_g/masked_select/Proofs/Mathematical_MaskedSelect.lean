import bench.tritonbench_g.masked_select.Sources.Mathematical_Kernels

/-!
# `masked_select` — strict per-kernel correctness

`masked_select_kernel` is a compaction scatter: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of the input, the boolean select mask, and
a precomputed inclusive-prefix-sum of the mask; each selected lane scatters its
input value to `out_ptr[prefix_sum − 1]`, i.e. its compacted output slot.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`masked_select_kernel[grid](...)`, the grid size
`cdiv(n_elements, BLOCK_SIZE)`, the host-side `prefix_sum`/output allocation, and
how the runtime composes per-program writes) is the *trusted boundary*, not a
proof obligation here. Because `pid` is universally quantified, the per-program
statement covers every program of the grid.

The headline is stated on the kernel's compaction-scatter IO signature
`maskedSelectIO` (`BoolScatterMasked2DKernelIO₁`, the scatter genre's bool-gated
skin): which buffer is which argument — data `inp_ptr` (`inp`), select gate
`select_mask_ptr` (`mbuf`, `.bool` channel), prefix-sum `prefix_sum_ptr`
(`idxbuf`, `.nat` index channel), and the scatter target `out_ptr` (`out`) —
the shared read window `pid₀ * BLOCK_SIZE + j`, the **static** bounds mask
`offset < n_elements` (the trace-safety/input superset), the data-dependent
`writeMask` `offset < n_elements ∧ bs j = Bool.true` (the kernel's
`select_mask and mask` store gate), and the data-dependent scatter destination
`write … ids j = ids j − 1` (the loaded `prefix_sum` value minus one). `⊨`
(`BoolScatterMasked2DKernelIO₁.Implements`) is the audit-once scatter Hoare-triple
combinator: for **every** disjoint flat placement of the four buffers, **every**
program id whose static-mask lanes are in bounds, and **every** launch state
whose windows hold `xs`/`bs`/`ids` at the static-mask lanes, the translated
pointer kernel terminates, the readback leg — guarded by the per-context
`WriteInj` antecedent (no two write-active lanes share a destination, the host
prefix-sum's no-duplicate-destination guarantee) — puts `xs j` at every
write-active lane's scatter cell `ids j − 1`, and every memory cell off the raw
scatter cells is unchanged (the frame holds with or without injectivity).

## Proof architecture

```
masked_select_kernel_correctness               ← TOP SPECIFICATION (maskedSelectIO ⊨ compaction scatter)
  ├─ masked_select_kernel_flattenOk             bridge fragment membership
  ├─ masked_select_kernel_traceSafe             per-execution lane-wise safety walk
  └─ masked_select_kernel_region_run            region-model scatter Hoare triple
       ├─ masked_select_kernel_exec_isSome      termination
       ├─ masked_select_kernel_correct_of_exec  executed-state readback per active lane
       │    └─ masked_select_kernel_correct     algorithm-layer readback per active lane
       └─ masked_select_kernel_frame            scatter-store cell frame
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The destination offset is data-dependent (read from `prefix_sum_ptr`),
so the readback leg of the `⊨` headline is guarded by the struct's per-context
`WriteInj` antecedent — no two **write-active** (in-bounds and selected) lanes
share a destination. This is deliberately weaker than injectivity over *all*
lanes: for a genuine prefix sum, an unselected lane's `prefix_sum − 1` equals
the previous selected lane's destination, so full-lane injectivity is false —
only the write-active restriction captures the host guarantee. The
`.to(tl.int1)` mask cast reduces to identity at the algorithm layer. The frame
and the trace-safety write bound are stated at the *ungated* write-active lanes,
so they hold with or without injectivity.
-/

namespace VeriTile.Bench.TritonBenchG.MaskedSelect

open VeriTile.Triton
open scoped VeriTile.Triton.BoolScatterMasked2DKernelIO₁

/-- Lane `i`'s address in all three read windows of program `s.pid`. Internal
to the readback lemmas (the `⊨` signature declares the same window as
`fun pid₀ _ j => pid₀ * BLOCK_SIZE + j.val`). -/
def maskedSelectOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val

/-- Lane `i`'s state-coupled scatter destination: the masked `prefix_sum` load
minus one (the kernel's `out_offset`). Internal to the readback lemmas; the `⊨`
signature re-expresses it as `ids j − 1` in terms of the pinned index tile. -/
def maskedSelectStoreOffset
    (s : BlockState) (prefix_sum_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  (if maskedSelectOffset s BLOCK_SIZE i < n_elements then
      s.readMemValue .nat prefix_sum_ptr (maskedSelectOffset s BLOCK_SIZE i)
    else
      0) - 1

/-- Lane `i` is **write-active**: in bounds and selected — the state-coupled
form of the signature's `writeMask`. -/
def active
    (s : BlockState) (select_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  maskedSelectOffset s BLOCK_SIZE i < n_elements ∧
    s.readMemValue .bool select_mask_ptr (maskedSelectOffset s BLOCK_SIZE i) = Bool.true

instance activeDecidable
    (s : BlockState) (select_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active s select_mask_ptr n_elements BLOCK_SIZE i) := by
  unfold active
  infer_instance

end VeriTile.Bench.TritonBenchG.MaskedSelect
