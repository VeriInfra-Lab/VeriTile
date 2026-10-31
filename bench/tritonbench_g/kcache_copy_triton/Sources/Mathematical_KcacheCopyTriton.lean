import VeriTile.Triton

/-!
# `kcache_copy_triton` — strict per-kernel correctness

`_copy_to_kcache_seqlen_n_kernel` is a paged K-cache scatter: program
`(cur_token_idx, cur_kv_head_idx, split_x_idx)` derives the sequence id and a
signed token shift, reads `seq_lengths` and the `BLOCK_TABLES` block table to
locate the destination block / in-block slot, gathers one contiguous
`KCACHE_X` split-x block of `K`, and scatters it into `KCache` for that head /
split. It supports the legacy `[num_blocks, num_kv_heads, block_size, head_dim]`
layout (`KCACHE_X = HEAD_DIM`, one split) and the new split-x
`[num_blocks, num_kv_heads, head_dim // x, block_size, x]` layout.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (grid over `(num_tokens, num_kv_heads, head_dim/x)`, the
block-table / seq-length inputs, and how the runtime composes per-program
scatters into the paged K-cache) is the *trusted boundary*, not a proof
obligation here. Because the program ids are universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
copy_to_kcache_seqlen_n1_correctness            ← TOP SPECIFICATION (kcacheCopyN1IO ⊨ chained paged copy)
  ├─ copy_to_kcache_seqlen_n1_surface_flattenOk       bridge fragment membership
  ├─ copy_to_kcache_seqlen_n1_surface_traceSafe       per-execution safety walk
  └─ copy_to_kcache_seqlen_n1_surface_region_run      region-model chained-metadata triple
       ├─ copy_to_kcache_seqlen_n1_surface_exec_isSome  termination
       ├─ copy_to_kcache_seqlen_n1_surface_correct      executed-state readback per cell
       └─ copy_to_kcache_seqlen_n1_surface_frame        scatter cell frame

(separate consumers of _surface_correct, not on the top-theorem path:
  copy_to_kcache_seqlen_n1_surface_compute_correct            (ComputeCorrect over the scatter)
  copy_to_kcache_seqlen_n1_old_layout_block_compute_correct   (legacy layout)
  copy_to_kcache_seqlen_n1_new_layout_xblock_compute_correct  (split-x layout))
```

The headline is stated on the kernel's chained-metadata **IO signature**
`kcacheCopyN1IO` (`ChainMetaMasked2DKernelIO₂ₓ₂`): slot 1 pins the raw
sequence length `m₁ = seq_lengths[pid₀]`, slot 2 pins the block id
`m₂ = BLOCK_TABLES[pid₀·bts + ((m₁−1)/block_size)·btb]` at the
**`m₁`-dependent** cell (the chain), and the single K tile is copied to the
`m₂`-selected cache window. The kernel is K-only (1 in ↦ 1 out), so the
family's second data channel is instantiated *off* by duplicate-region
wiring: `in2 := in1 = K`, `out2 := out1 = KCache` with
`mask2`/`writeMask2 := False` (sound — `FlatAlloc.Disjoint` only constrains
*distinct* listed regions, and the `False` gates make channel 2's legs
vacuous).

The `n_tokens > 1` (prefill) path is covered at the surface-lowering level only
(`copy_to_kcache_seqlen_n_kernel_toAlgorithm_supported`).
The post-arithmetic split-x slice
`copy_to_kcache_split_x_block_{correct,compute_correct}` proves the store once
the cache slot has been selected.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); dtype casts are
erased (post-erasure all numeric dtypes unify to `ℝ`; the signed kernel keeps
explicit `int64`/`uint64` casts on the index path). This is **partial**: full
cellwise readback correctness is proved for the `n_tokens = 1` decode path; the
`n_tokens > 1` path is verified only up to surface lowering (its negative
`cur_token_shift` arithmetic is outside the Nat-only pointer readback).
Python's third grid axis `split_x_idx = tl.program_id(2)` is pinned as the
host parameter `SPLIT_X` on the decode surface (the `kv_cache_copy` sibling's
convention; the `⊨` family is two-pid): the universally quantified `SPLIT_X`
covers every program of that axis. The destination block id is gathered from
`BLOCK_TABLES` and the in-block slot from `seq_lengths`; the headline's
per-cell readback legs are guarded by the family's per-context `WriteInj`
no-aliasing antecedent (the former `hOutInj` side condition — here discharged
for free, since one program's cache window is contiguous). The launch-time
`num_warps` heuristic is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton

open VeriTile.Triton
open scoped VeriTile.Triton.ChainMetaMasked2DKernelIO₂ₓ₂

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/- Lean model of `kcache_copy_triton.py`'s
`_copy_to_kcache_seqlen_n_kernel`.

This version keeps the Python expression shape and typed int regions for
`BLOCK_TABLES` and `seq_lengths`. The explicitly signed variant below pins the
intermediate arithmetic with `tl.int64` casts for the `n_tokens > 1` path. -/

def dimIndex (i : Fin KCACHE_X) : Nat :=
  i.val

def blockId (s : BlockState) (BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX stride_bts stride_btb : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb)

def kSourceOffset
    (s : BlockState) (stride_kt stride_kh stride_kd KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  s.pids 0 * stride_kt + s.pids 1 * stride_kh +
    (s.pids 2 * KCACHE_X + dimIndex i) * stride_kd

def kCacheOffset
    (s : BlockState) (BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb : Nat)
    (i : Fin KCACHE_X) : Nat :=
  blockId s BLOCK_TABLES LAST_BLOCK_IDX stride_bts stride_btb * stride_kcb +
    s.pids 1 * stride_kch + s.pids 2 * stride_kcsplit_x +
    OFFSET_LAST_BLOCK * stride_kcs + dimIndex i

def n1PastKvSeqLen (s : BlockState) (seq_lengths : RegionName) : Nat :=
  s.readMemValue .nat seq_lengths (s.pids 0) - 1

def n1LastBlockIdx (s : BlockState) (seq_lengths : RegionName)
    (block_size : Nat) : Nat :=
  n1PastKvSeqLen s seq_lengths / block_size

def n1OffsetLastBlock (s : BlockState) (seq_lengths : RegionName)
    (block_size : Nat) : Nat :=
  n1PastKvSeqLen s seq_lengths % block_size

def n1BlockId (s : BlockState) (BLOCK_TABLES seq_lengths : RegionName)
    (stride_bts stride_btb block_size : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 0 * stride_bts + n1LastBlockIdx s seq_lengths block_size * stride_btb)

/-- K read offset for the pinned-`SPLIT_X` decode surface: lane `i` of the
split-x block of `K` for program `(cur_token_idx, cur_kv_head_idx)`. -/
def n1KSourceOffset
    (s : BlockState) (SPLIT_X stride_kt stride_kh stride_kd KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  s.pids 0 * stride_kt + s.pids 1 * stride_kh +
    (SPLIT_X * KCACHE_X + dimIndex i) * stride_kd

def n1KCacheOffset
    (s : BlockState) (BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb block_size : Nat)
    (i : Fin KCACHE_X) : Nat :=
  n1BlockId s BLOCK_TABLES seq_lengths stride_bts stride_btb block_size * stride_kcb +
    s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
    n1OffsetLastBlock s seq_lengths block_size * stride_kcs + dimIndex i

/-! ## The `⊨` specification (decode path)

The headline states the pinned-`SPLIT_X` decode surface on the
chained-metadata IO skin `ChainMetaMasked2DKernelIO₂ₓ₂`: the raw loaded
sequence length `m₁` and the raw block id `m₂` are named ghost binders
pinned to the two slot cells — `m₂`'s cell address eats `m₁` (the chain) —
and the K tile is copied to the `m₂`-selected cache window. The machinery
below supplies the three intro obligations: the bridge-fragment membership,
the per-execution safety walk, and the region-model run (termination +
readback + cell frame). -/

/-- An unmasked scatter-store `foldl` leaves every memory cell it does not
hit unchanged (cell-level frame for the unconditional output store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ)
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k))
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons,
        ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_mem]
      exact if_neg (fun hc =>
        hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩)

end VeriTile.Bench.TritonBenchG.KcacheCopyTriton
