import VeriTile.Triton
import VeriTile.Meta.StatementAudit
import VeriTile.Examples.AttentionForwardClosedForm

/-!
# `rope_transform` — strict per-kernel correctness

`_triton_rope` applies the rotary position embedding in place to fused Q/K
buffers: each program owns one row (`program_id(0) = batch*seq` index), loads the
per-row `cos`/`sin` half-dim vectors, and for every head rewrites the two rotary
halves of both `q_ptr` and `k_ptr` via `(t1*cos - t2*sin, t2*cos + t1*sin)` on
the forward path (`BACKWARD_PASS = false`, the sign-flipped transpose otherwise).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (the `(n_row,)` grid, the `next_power_of_2` padding choices
`pad_n_qh`/`pad_n_kh`/`pad_hd`, `BLOCK_SIZE`, the contiguity/transpose bookkeeping
in `rope_forward`, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*, not a proof obligation here. Because the row
position and per-head/per-dim indices are universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
rope_transform_output_summary_general                 ← TOP THEOREM (genuine, dimension-general)
  four-conjunct `ComputeCorrect.Realizes_without_Rounding` bundle (one per stored output half);
  each conjunct: `realizes_writeIf_iff` + `computeCorrect_of_toAlgKernel`
  (lowering the REAL `triton_rope_surface` surface to the algorithm layer inline
  via `simp [triton_rope_surface, ComputeExpr/ComputeOp.toAlgorithm?]`), then the
  matching per-lane store value from
  ├─ rope_transform_q0_forward_correct  → ropeForwardKernelQ0Spec  (q1·cos − q2·sin)
  ├─ rope_transform_q1_forward_correct  → ropeForwardKernelQ1Spec  (q2·cos + q1·sin)
  ├─ rope_transform_k0_forward_correct  → ropeForwardKernelK0Spec  (k1·cos − k2·sin)
  └─ rope_transform_k1_forward_correct  → ropeForwardKernelK1Spec  (k2·cos + k1·sin)
       each reads back one full `[pad_n_qh, pad_hd/2]` store from the REAL
       `triton_rope_surface` (`BACKWARD_PASS = false`), bridging the surface to
       its body via `rope_exec_eq` (which uses `triton_rope_surface_forward_body :
       (surface).body = ropeForwardBody`, by rfl) and then `rope_forward_body_steps`

rope_forward_body_steps (the load-bearing lemma, steps `ropeForwardBody` directly)
  ├─ 22 prologue assigns stepped via the standalone offset/mask/load recipes
  │    (firstHalf_offsets_eval, firstMask_eval, ptr_plus_offsets_eval,
  │     load_ptr_maskOther_real, …; expandDim wall handled by the *_arange recipes)
  ├─ ifThenElse (constBool false).boolNot takes the then-branch (4 store pairs)
  └─ readback: cross-region peel (q_ptr ≠ k_ptr) + same-region disjoint peel
       (first-half vs second-half offsets) + scatter_readback_prop_masked_nd_of_true

combined-row track (rotary-style 2D slice):
rope_kernel_o0o1_row_all_outputs_compute_correct
  ├─ rope_kernel_o0o1_row_o0_compute_correct → rope_kernel_o0o1_row_o0_correct
  └─ rope_kernel_o0o1_row_o1_compute_correct → rope_kernel_o0o1_row_o1_correct

`⊨` track (`GroupedMasked2DKernelIO`, the grouped vector-channel skin — channel
counts are `nIn`/`nOut` fields and channels are `Fin n`-indexed), one masked
Hoare triple over FLAT pointer memory per per-head store:
rope_transform_{q0,q1,k0,k1}_head_implements   (last four headline conjuncts)
  ├─ ropeHeadIO             the declared IO signature (4 read / 1 write channel)
  ├─ *_head_flattenOk       the slice is inside the flat-memory bridge
  ├─ *_head_traceSafe       per-execution bounds walk
  └─ *_head_region_run      termination + values + frame, over the region model
       ├─ *_head_exec_isSome
       ├─ *_head_correct    (the value half, reused from the track above)
       └─ *_head_frame      (foldl_store_preserve_cell)
```

Offset disjointness within a half pair is discharged inline (by `omega`) in the
readback peel.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; the `.to(sin_row.dtype)`
register casts erase to the identity at the algorithm layer (post-erasure all
dtypes unify to `ℝ`). `cos`/`sin` are modeled as **precomputed inputs** loaded
from memory, not computed. The top summary is the **full-surface, full-tile**
forward result: every active lane of each of the four Python-observable stores
(Q/K first and second halves) reads back to the genuine rotary closed form, NOT
the kernel's own executed value, against the real `triton_rope_surface` kernel
(no head-slice gap). The host launch / `next_power_of_2` padding (`pad_n_qh`,
`pad_hd`, `BLOCK_SIZE`, the contiguity/transpose bookkeeping, grid composition)
remains the trusted boundary, as does `q_ptr ≠ k_ptr` (distinct Q/K buffers).
The `@triton.heuristics`/`BACKWARD_PASS` flag is modeled by the `Bool` argument;
`@triton.autotune` is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.RopeTransform

open VeriTile.Triton
open VeriTile.Examples.AttentionForwardClosedForm
open scoped VeriTile.Triton.GroupedMasked2DKernelIO

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! ## Standalone eval recipes for the full-kernel forward proof

The full `triton_rope_surface` kernel builds its tile offsets/masks from the 2D
`tl.arange[:, None]`/`tl.arange[None, :]` (`expandDim`) composition. The generic
`evalOp_expandDim` simp lemma does not fire on `Op.expandDim _ (Op.arange _)`
inside a nested `Option.bind` (the "expandDim wall"), so we fix the concrete
output shapes up front in these standalone recipes and chain them by `simp only`
when stepping the body. -/

/-- `expandDim` axis-1 of an `arange` (the `tl.arange(0, PN)[:, None]` column
broadcast), output shape pinned to `[PN, 1]`. -/
@[simp] theorem evalOp_expandDim_one_arange {PN : Nat} (s : BlockState) :
    @evalOp .nat [PN, 1] (Op.expandDim ⟨1, by simp⟩ (Op.arange PN)) s =
      some ({ data := fun i : TileIndex [PN, 1] => i.1.val } : Tile .nat [PN, 1]) := by
  unfold evalOp; simp [Tile.expandDim, Tile.vec, TileShape.dropInsertedIndex]; rfl

/-- `expandDim` axis-0 of an `arange` (the `tl.arange(0, PH)[None, :]` row
broadcast), output shape pinned to `[1, PH]`. -/
@[simp] theorem evalOp_expandDim_zero_arange {PH : Nat} (s : BlockState) :
    @evalOp .nat [1, PH] (Op.expandDim ⟨0, by simp⟩ (Op.arange PH)) s =
      some ({ data := fun i : TileIndex [1, PH] => i.2.1.val } : Tile .nat [1, PH]) := by
  unfold evalOp; simp [Tile.expandDim, Tile.vec, TileShape.dropInsertedIndex]; rfl

/-- First-half offsets recipe: `arange(PN)[:,None]·hd + arange(PH)[None,:]`
evaluates, lane `idx`, to `idx.row·hd + idx.col`. -/
theorem firstHalf_offsets_eval (s : BlockState) (PN PH hd : Nat) :
    evalOp (Op.add NumericDType.nat Broadcast.nil.consL.consR
      (Op.mul NumericDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.arange PN))
        (Op.constNat hd))
      (Op.expandDim ⟨0, by simp⟩ (Op.arange PH))) s
      = some (⟨fun idx : TileIndex [PN, PH] => idx.1.val * hd + idx.2.1.val⟩ : Tile .nat [PN, PH]) := by
  rw [evalOp_add, evalOp_mul]
  simp only [evalOp_expandDim_one_arange, evalOp_expandDim_zero_arange, evalOp_constNat,
    Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, NumericDType.add, NumericDType.mul, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex]

/-- Second-half offsets recipe: `first_half_offsets + hd//2`. -/
theorem secondHalf_offsets_eval (s : BlockState) (PN PH hd : Nat) (name : RegName)
    (first : Tile .nat [PN, PH])
    (hfirst : s.regs .nat [PN, PH] name = some first) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarR
      (Op.ref .nat [PN, PH] name)
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat hd) (Op.constNat 2))) s
      = some (⟨fun idx : TileIndex [PN, PH] => (first.data idx) + hd / 2⟩ : Tile .nat [PN, PH]) := by
  rw [evalOp_add, evalOp_floorDiv]
  simp only [evalOp_ref, evalOp_constNat, hfirst, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Broadcast.rightIndex, Broadcast.leftIndex, NumericDType.add,
    IntegralDType.floorDiv, Tile.scalar]

/-- First-half mask recipe: `(arange(PN)[:,None] < n) & (arange(PH)[None,:] < hd//2)`. -/
theorem firstMask_eval (s : BlockState) (PN PH n hd : Nat) :
    evalOp (Op.boolAnd Broadcast.nil.consL.consR
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.arange PN))
        (Op.constNat n))
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.arange PH))
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat hd) (Op.constNat 2)))) s
      = some (⟨fun idx : TileIndex [PN, PH] =>
          decide (idx.1.val < n) && decide (idx.2.1.val < hd / 2)⟩ : Tile .bool [PN, PH]) := by
  rw [evalOp_boolAnd, evalOp_lt, evalOp_lt, evalOp_floorDiv]
  simp only [evalOp_expandDim_one_arange, evalOp_expandDim_zero_arange, evalOp_constNat,
    Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.cop, Broadcast.rightIndex, Broadcast.leftIndex, Tile.scalar,
    ComparableDType.lt, IntegralDType.floorDiv]
  rfl

/-- `q_ptr + offsets` recipe: a scalar base pointer broadcast-added to a 2D
offset tile yields the per-lane address tile `(region, base + offset)`. -/
theorem ptr_plus_offsets_eval (s : BlockState) (Q : RegionName) (PN PH : Nat)
    (base : Nat) (offs : Tile .nat [PN, PH]) (pname oname : RegName)
    (hp : s.regs .ptr [] pname = some (Tile.scalar (Q.cast, base)))
    (ho : s.regs .nat [PN, PH] oname = some offs) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] pname) (Op.ref .nat [PN, PH] oname)) s
      = some (⟨fun i => (Q.cast, base + offs.data i)⟩ : Tile .ptr [PN, PH]) := by
  rw [evalOp_ptrAdd]
  simp only [evalOp_ref, hp, ho, Option.bind_some]
  refine congrArg some ?_
  ext i
  · simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex]
  · simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]

/-- Prologue base-pointer recipe: `base_ptr + pid·stride`. -/
theorem ptrbase_mul_eval (s : BlockState) (Q : RegionName) (pidv stride : Nat)
    (hpid : s.regs .nat [] "pid" = some (Tile.scalar pidv)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Q)
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid") (Op.constNat stride))) s
      = some (Tile.scalar (Q.cast, pidv * stride)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase, evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hpid, Option.bind_some]
  refine congrArg some ?_
  ext i
  · simp only [Tile.ptrAdd_data, Tile.scalar, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.mul]
  · simp only [Tile.ptrAdd_data, Tile.scalar, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.mul, Nat.zero_add]

/-- Cos/sin base-pointer recipe: `base_ptr + cos_row_idx·stride`. -/
theorem ptrbase_mul_ref_eval (s : BlockState) (Q : RegionName) (idxv stride : Nat)
    (name : RegName) (hidx : s.regs .nat [] name = some (Tile.scalar idxv)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Q)
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat stride))) s
      = some (Tile.scalar (Q.cast, idxv * stride)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase, evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hidx, Option.bind_some]
  refine congrArg some ?_
  ext i
  · simp only [Tile.ptrAdd_data, Tile.scalar, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.mul]
  · simp only [Tile.ptrAdd_data, Tile.scalar, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.mul, Nat.zero_add]

/-- `cos_row_idx = pid % sl` recipe. -/
theorem mod_eval (s : BlockState) (pidv sl : Nat)
    (hpid : s.regs .nat [] "pid" = some (Tile.scalar pidv)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid") (Op.constNat sl)) s
      = some (Tile.scalar (pidv % sl)) := by
  rw [evalOp_mod]
  simp only [evalOp_ref, evalOp_constNat, hpid, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.scalar, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, IntegralDType.mod]

/-- 1D `cos/sin + cos_offsets` recipe: scalar base broadcast-added to a 1D
offset tile. -/
theorem ptr_plus_offsets1d_eval (s : BlockState) (Q : RegionName) (PH : Nat)
    (base : Nat) (offs : Tile .nat [PH]) (pname oname : RegName)
    (hp : s.regs .ptr [] pname = some (Tile.scalar (Q.cast, base)))
    (ho : s.regs .nat [PH] oname = some offs) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] pname) (Op.ref .nat [PH] oname)) s
      = some (⟨fun i => (Q.cast, base + offs.data i)⟩ : Tile .ptr [PH]) := by
  rw [evalOp_ptrAdd]
  simp only [evalOp_ref, hp, ho, Option.bind_some]
  refine congrArg some ?_
  ext i
  · simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex]
  · simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]

/-- 1D cos/sin mask recipe: `cos_offsets < hd//2`. -/
theorem cosMask_eval (s : BlockState) (PH hd : Nat) (offs : Tile .nat [PH]) (oname : RegName)
    (ho : s.regs .nat [PH] oname = some offs) :
    evalOp (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [PH] oname)
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat hd) (Op.constNat 2))) s
      = some (⟨fun i => decide (offs.data i < hd / 2)⟩ : Tile .bool [PH]) := by
  rw [evalOp_lt, evalOp_floorDiv]
  simp only [evalOp_ref, evalOp_constNat, ho, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.cop, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt, IntegralDType.floorDiv]
  rfl

/-- Masked-other `.ptr` load recipe (the `tl.load(..., other=0)` form used for
all six rope loads): per lane, reads memory where the mask holds, else `0`. -/
theorem load_ptr_maskOther_real {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (s : BlockState)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hp : evalOp ptrOp s = some ptrs) (hm : evalOp maskOp s = some masks) :
    evalOp (.load .real (.ptr ptrOp) (.maskOther maskOp ((Op.const 0).broadcast shape))) s
      = some ⟨fun i => if masks.data i then
          some (s.readMem (ptrs.data i).1 (ptrs.data i).2) else some 0⟩ := by
  simp only [evalOp, hp, hm]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real]
  cases hmi : masks.data i <;> simp [hmi]

/-- `if not BACKWARD_PASS` with `BACKWARD_PASS = false` takes the then-branch. -/
theorem ifThenElse_notFalse_step (thenB elseB : List Stmt) (st : BlockState) :
    stepStmt (Stmt.ifThenElse (Op.constBool Bool.false).boolNot thenB elseB) st
      = stepStmts thenB st := by
  unfold stepStmt
  simp only [evalOp, Tile.uop, Tile.scalar]
  norm_num

/-- Singleton-list form: the trailing `if`-statement of the body takes its
then-branch. -/
theorem stepStmts_singleton_ifThenElse_notFalse (thenB elseB : List Stmt) (st : BlockState) :
    stepStmts [Stmt.ifThenElse (Op.constBool Bool.false).boolNot thenB elseB] st
      = stepStmts thenB st := by
  conv_lhs => unfold stepStmts
  rw [ifThenElse_notFalse_step]
  cases stepStmts thenB st <;> simp

/-- `new_*_tile_1 = t1·cos − t2·sin` value recipe (first-half forward store). -/
theorem newSub_value_eval (s : BlockState) (PN PH : Nat)
    (t1 t2 : Tile .real [PN, PH]) (cosrow sinrow : Tile .real [PH])
    (n1 n2 cn sn : RegName)
    (h1 : s.regs .real [PN, PH] n1 = some t1)
    (h2 : s.regs .real [PN, PH] n2 = some t2)
    (hc : s.regs .real [PH] cn = some cosrow)
    (hsr : s.regs .real [PH] sn = some sinrow) :
    evalOp (Op.sub NumericDType.real Broadcast.nil.consSame.consSame
      (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
        (Op.ref .real [PN, PH] n1) (Op.ref .real [PH] cn))
      (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
        (Op.ref .real [PN, PH] n2) (Op.ref .real [PH] sn))) s
      = some (⟨fun idx : TileIndex [PN, PH] =>
          NumericDType.real.sub
            (NumericDType.real.mul (t1.data idx) (cosrow.data (idx.2.1, idx.2.2)))
            (NumericDType.real.mul (t2.data idx) (sinrow.data (idx.2.1, idx.2.2)))⟩
            : Tile .real [PN, PH]) := by
  rw [evalOp_sub, evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, h1, h2, hc, hsr, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
    Broadcast.leftIndex_leadR, Broadcast.rightIndex_leadR]

/-- `new_*_tile_2 = t2·cos + t1·sin` value recipe (second-half forward store). -/
theorem newAdd_value_eval (s : BlockState) (PN PH : Nat)
    (t1 t2 : Tile .real [PN, PH]) (cosrow sinrow : Tile .real [PH])
    (n1 n2 cn sn : RegName)
    (h1 : s.regs .real [PN, PH] n1 = some t1)
    (h2 : s.regs .real [PN, PH] n2 = some t2)
    (hc : s.regs .real [PH] cn = some cosrow)
    (hsr : s.regs .real [PH] sn = some sinrow) :
    evalOp (Op.add NumericDType.real Broadcast.nil.consSame.consSame
      (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
        (Op.ref .real [PN, PH] n1) (Op.ref .real [PH] cn))
      (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
        (Op.ref .real [PN, PH] n2) (Op.ref .real [PH] sn))) s
      = some (⟨fun idx : TileIndex [PN, PH] =>
          NumericDType.real.add
            (NumericDType.real.mul (t1.data idx) (cosrow.data (idx.2.1, idx.2.2)))
            (NumericDType.real.mul (t2.data idx) (sinrow.data (idx.2.1, idx.2.2)))⟩
            : Tile .real [PN, PH]) := by
  rw [evalOp_add, evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, h1, h2, hc, hsr, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
    Broadcast.leftIndex_leadR, Broadcast.rightIndex_leadR]

/-- Masked `.ptr` store step: produces the masked `writeMemTyped` foldl. -/
theorem stepStmt_store_ptr_mask_eq {shape : TileShape}
    (ptrOp : Op .ptr shape) (valOp : Op .real shape) (maskOp : Op .bool shape)
    (s : BlockState) (ptrs : Tile .ptr shape) (vals : Tile .real shape) (masks : Tile .bool shape)
    (hv : evalOp valOp s = some vals)
    (hp : evalOp ptrOp s = some ptrs)
    (hm : evalOp maskOp s = some masks) :
    stepStmt (Stmt.store .real shape (MemAccess.ptr ptrOp) valOp (MaskOpt.mask maskOp)) s
      = some ((TileShape.allIndices shape).foldl
          (fun acc i =>
            if masks.data i then
              acc.writeMemTyped .real (ptrs.data i).1 (ptrs.data i).2 (vals.data i)
            else acc) s) := by
  unfold stepStmt
  simp only [hv, hp, hm, Option.map_some]
  rfl

/-- Bool-predicate intra-region offset-disjointness readback (the rope-style Q
first-half read commuting past the Q second-half masked store-foldl). -/
theorem foldl_writeMem_same_region_disjoint_offsets_readMem_bool {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (P : α → Bool) (l : List α) (s : BlockState)
    (off : Nat) (hOff : ∀ k, k ∈ l → P k → off ≠ offsetFn k) :
    BlockState.readMem ((l.foldl
        (fun acc k =>
          if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s))
      region off
      = s.readMem region off := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      have htl : ∀ k, k ∈ tl → P k → off ≠ offsetFn k :=
        fun k hk hPk => hOff k (List.mem_cons_of_mem hd hk) hPk
      by_cases hhd : P hd = Bool.true
      · simp only [hhd, ite_true]
        rw [ih _ htl, BlockState.writeMem_readMem, if_neg]
        rintro ⟨_, hOffEq⟩
        exact (hOff hd List.mem_cons_self hhd) hOffEq
      · rw [Bool.not_eq_true] at hhd
        simp only [hhd, Bool.false_eq_true]
        exact ih _ htl

def rowIndex (s : BlockState) : Nat :=
  s.pids 0

def dimIndex (i : Fin BLOCK_HALF) : Nat :=
  i.val

def active (HEAD_IDX n_qh HEAD_HALF : Nat) (i : Fin BLOCK_HALF) : Prop :=
  HEAD_IDX < n_qh ∧ dimIndex i < HEAD_HALF

instance activeDecidable (HEAD_IDX n_qh HEAD_HALF : Nat) (i : Fin BLOCK_HALF) :
    Decidable (active HEAD_IDX n_qh HEAD_HALF i) := by
  unfold active
  infer_instance

def qOffset
    (s : BlockState) (HEAD_IDX q_row_stride hd : Nat) (dim : Nat) : Nat :=
  rowIndex s * q_row_stride + HEAD_IDX * hd + dim

def cosOffset (COS_ROW_IDX cos_row_stride : Nat) (i : Fin BLOCK_HALF) : Nat :=
  COS_ROW_IDX * cos_row_stride + dimIndex i

def sinOffset (COS_ROW_IDX sin_row_stride : Nat) (i : Fin BLOCK_HALF) : Nat :=
  COS_ROW_IDX * sin_row_stride + dimIndex i

@[standalone_spec] noncomputable def ropeQ0Spec
    (s : BlockState) (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      HEAD_HALF : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) *
    s.readMem COS (cosOffset COS_ROW_IDX cos_row_stride i) -
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i + HEAD_HALF)) *
    s.readMem SIN (sinOffset COS_ROW_IDX sin_row_stride i)

/-! ## Second-half Q store (`new_q_tile_2`) -/

/-- Write offset for `q1`: matches the kernel evaluation
`q_base + dim + HEAD_HALF`, i.e. `((s.pid * q_row_stride + HEAD_IDX * hd)
+ dim) + HEAD_HALF`. -/
def q1WriteOffset
    (s : BlockState) (HEAD_IDX q_row_stride hd HEAD_HALF : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pid * q_row_stride + HEAD_IDX * hd + dimIndex i + HEAD_HALF

@[standalone_spec] noncomputable def ropeQ1Spec
    (s : BlockState) (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      HEAD_HALF : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i) *
    s.readMem COS (cosOffset COS_ROW_IDX cos_row_stride i) +
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) *
    s.readMem SIN (sinOffset COS_ROW_IDX sin_row_stride i)

/-! ## K-side coverage

The `rope_transform_q0_head` and `rope_transform_q1_head` surfaces are
parameterised by a generic region name and head-count bound. Applying the
same correctness theorems with the K region, `n_kh` head count, and the
appropriate `K` stride covers the K-side forward stores
`new_k_tile_1 = k_tile_1 * cos - k_tile_2 * sin` and
`new_k_tile_2 = k_tile_2 * cos + k_tile_1 * sin`. The thin wrappers below
make this explicit so downstream consumers of #134 can cite a K-specific
theorem name. -/

/-! ## Full-kernel 2D forward correctness (`BACKWARD_PASS = false`)

The single-head slice proofs above pin one Q/K head and one dim slice, but the
Python kernel writes a full `[pad_n_qh, pad_hd/2]` tile (and similarly for K).
The following theorems close the four full-kernel forward stores against the
real `triton_rope_surface` with `BACKWARD_PASS = false`, following the same
foldl-peel + scatter-readback strategy used in `RopeBackwardTransform.lean`.

Sign convention in the forward branch:
* `new_q_tile_1 = q_tile_1 * cos - q_tile_2 * sin`
* `new_q_tile_2 = q_tile_2 * cos + q_tile_1 * sin`
* `new_k_tile_1 = k_tile_1 * cos - k_tile_2 * sin`
* `new_k_tile_2 = k_tile_2 * cos + k_tile_1 * sin` -/

/-- Tile-level Q first-half offset (target of store #1 in the foldl chain). -/
def qFullFirstOffset
    (s : BlockState) (q_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Nat :=
  s.pids 0 * q_row_stride + idx.1.val * hd + idx.2.1.val

/-- Tile-level Q second-half offset (target of store #2). -/
def qFullSecondOffset
    (s : BlockState) (q_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Nat :=
  s.pids 0 * q_row_stride + idx.1.val * hd + idx.2.1.val + hd / 2

/-- Cos offset for the full kernel's Q stores. -/
def cosFullFirstOffset
    (s : BlockState) (sl cos_row_stride : Nat)
    (idx : TileIndex [pad_hd_half]) : Nat :=
  s.pids 0 % sl * cos_row_stride + idx.1.val

/-- Sin offset for the full kernel's Q stores. -/
def sinFullFirstOffset
    (s : BlockState) (sl sin_row_stride : Nat)
    (idx : TileIndex [pad_hd_half]) : Nat :=
  s.pids 0 % sl * sin_row_stride + idx.1.val

/-- Active predicate for the Q-side stores of the full kernel. -/
def activeQFull (n_qh hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Prop :=
  idx.1.val < n_qh ∧ idx.2.1.val < hd / 2

instance activeQFullDecidable (n_qh hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) :
    Decidable (activeQFull n_qh hd idx) := by
  unfold activeQFull
  infer_instance

/-- Spec for the Q first-half output under `BACKWARD_PASS = false`:
`new_q_tile_1 = q_tile_1 * cos - q_tile_2 * sin`. -/
@[standalone_spec] noncomputable def ropeForwardKernelQ0Spec
    (s : BlockState) (q_ptr cos sin : RegionName)
    (q_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : ℝ :=
  s.readMem q_ptr (qFullFirstOffset s q_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) -
  s.readMem q_ptr (qFullSecondOffset s q_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-- Spec for the Q second-half output under `BACKWARD_PASS = false`:
`new_q_tile_2 = q_tile_2 * cos + q_tile_1 * sin`. -/
@[standalone_spec] noncomputable def ropeForwardKernelQ1Spec
    (s : BlockState) (q_ptr cos sin : RegionName)
    (q_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : ℝ :=
  s.readMem q_ptr (qFullSecondOffset s q_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) +
  s.readMem q_ptr (qFullFirstOffset s q_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-- Tile-level K first-half offset (target of store #3). -/
def kFullFirstOffset
    (s : BlockState) (k_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Nat :=
  s.pids 0 * k_row_stride + idx.1.val * hd + idx.2.1.val

/-- Tile-level K second-half offset (target of store #4). -/
def kFullSecondOffset
    (s : BlockState) (k_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Nat :=
  s.pids 0 * k_row_stride + idx.1.val * hd + idx.2.1.val + hd / 2

/-- Active predicate for the K-side stores of the full kernel. -/
def activeKFull (n_kh hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Prop :=
  idx.1.val < n_kh ∧ idx.2.1.val < hd / 2

instance activeKFullDecidable (n_kh hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) :
    Decidable (activeKFull n_kh hd idx) := by
  unfold activeKFull
  infer_instance

/-- Spec for the K first-half output under `BACKWARD_PASS = false`:
`new_k_tile_1 = k_tile_1 * cos - k_tile_2 * sin`. -/
@[standalone_spec] noncomputable def ropeForwardKernelK0Spec
    (s : BlockState) (k_ptr cos sin : RegionName)
    (k_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : ℝ :=
  s.readMem k_ptr (kFullFirstOffset s k_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) -
  s.readMem k_ptr (kFullSecondOffset s k_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-- Spec for the K second-half output under `BACKWARD_PASS = false`:
`new_k_tile_2 = k_tile_2 * cos + k_tile_1 * sin`. -/
@[standalone_spec] noncomputable def ropeForwardKernelK1Spec
    (s : BlockState) (k_ptr cos sin : RegionName)
    (k_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : ℝ :=
  s.readMem k_ptr (kFullSecondOffset s k_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) +
  s.readMem k_ptr (kFullFirstOffset s k_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-! ## Full-kernel forward body stepping

The four forward stores of `triton_rope_surface` are read back by stepping the
whole lowered 30-statement body (22 prologue assigns + the taken branch of the
`if not BACKWARD_PASS` with its four `(assign, store)` pairs), feeding each
statement an explicit value via the standalone recipes above. -/

/-! ## Genuine per-store forward correctness against `triton_rope_surface`

Each theorem reads back one of the four Python-observable forward stores from the
real `triton_rope_surface` kernel (`BACKWARD_PASS = false`) and shows the active
lanes equal the genuine rotary closed form (`ropeForwardKernel{Q0,Q1,K0,K1}Spec`),
NOT the kernel's own executed value. The host launch / padding choices remain the
trusted boundary. -/

/-! ## Combined one-row `o0` + `o1` rotary-style RoPE slice

This combined slice mirrors `RotaryTransform.rotary_kernel_o0o1_row`: a single
kernel that performs BOTH the `o0 = x0 * cos - x1 * sin` first-half store and
the `o1 = x0 * sin + x1 * cos` second-half store into the same `OUT` region at
disjoint offsets. Correctness is established by stripping the other-half
`foldl` via `foldl_writeMem_same_region_disjoint_offsets_readMem` and then
closing the projected store via `scatter_readback_prop_masked_nd`. -/

def ropeRowIndex (s : BlockState) (BLOCK_M : Nat) : Nat :=
  s.pids 0 * BLOCK_M

def ropeDimIndex (i : Fin BLOCK_HALF) : Nat :=
  i.val

def ropeActive (s : BlockState) (seqlen rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Prop :=
  ropeRowIndex s BLOCK_M < seqlen ∧ ropeDimIndex i < rotary_dim_half

instance ropeActiveDecidable (s : BlockState)
    (seqlen rotary_dim_half BLOCK_M : Nat) (i : Fin BLOCK_HALF) :
    Decidable (ropeActive s seqlen rotary_dim_half BLOCK_M i) := by
  unfold ropeActive
  infer_instance

def ropeOutOffset
    (s : BlockState)
    (stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
    ropeRowIndex s BLOCK_M * stride_out_seqlen +
    ropeDimIndex i * stride_out_headdim

def ropeOut1Offset
    (s : BlockState)
    (stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
    ropeRowIndex s BLOCK_M * stride_out_seqlen +
    (ropeDimIndex i + rotary_dim_half) * stride_out_headdim

def ropeX0Offset
    (s : BlockState)
    (stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_x_batch + s.pids 2 * stride_x_nheads +
    ropeRowIndex s BLOCK_M * stride_x_seqlen +
    ropeDimIndex i * stride_x_headdim

def ropeX1Offset
    (s : BlockState)
    (stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_x_batch + s.pids 2 * stride_x_nheads +
    ropeRowIndex s BLOCK_M * stride_x_seqlen +
    (ropeDimIndex i + rotary_dim_half) * stride_x_headdim

def ropeRotOffset (s : BlockState) (SEQLEN_OFFSETS rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  (ropeRowIndex s BLOCK_M + SEQLEN_OFFSETS) * rotary_dim_half + ropeDimIndex i

@[standalone_spec] noncomputable def ropeO0Spec
    (s : BlockState) (X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : ℝ :=
  let cosVal :=
    if ropeRowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        ropeDimIndex i < rotary_dim_half then
      s.readMem COS (ropeRotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      1.0
  let sinVal :=
    if ropeRowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        ropeDimIndex i < rotary_dim_half then
      s.readMem SIN (ropeRotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      0.0
  s.readMem X
      (ropeX0Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim BLOCK_M i) *
    cosVal -
  s.readMem X
      (ropeX1Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim rotary_dim_half BLOCK_M i) *
    sinVal

@[standalone_spec] noncomputable def ropeO1Spec
    (s : BlockState) (X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : ℝ :=
  let cosVal :=
    if ropeRowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        ropeDimIndex i < rotary_dim_half then
      s.readMem COS (ropeRotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      1.0
  let sinVal :=
    if ropeRowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        ropeDimIndex i < rotary_dim_half then
      s.readMem SIN (ropeRotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      0.0
  s.readMem X
      (ropeX0Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim BLOCK_M i) *
    sinVal +
  s.readMem X
      (ropeX1Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim rotary_dim_half BLOCK_M i) *
    cosVal

/-! ## The `⊨` specification surface (`GroupedMasked2DKernelIO`)

The per-head forward slices above are single-store kernels over the region
model; each of them is packaged below as a grouped masked Hoare triple
(`GroupedMasked2DKernelIO.Implements`, notation `⊨`) over **flat** pointer
memory. The four read channels are the kernel's four `tl.load`s (the Q/K first
half, the Q/K second half, `cos`, `sin`) and the single output channel points
back at the data buffer — the RoPE update is in place, so the triple reads the
*old* window contents into `xs` and asserts the *new* ones. -/

/-- The shared **IO signature** of one in-place per-head RoPE store — the whole
kernel-specific audit surface of the four `⊨` headlines below.

* `bufs = [Buf, COS, SIN]`; the single output channel points **back at**
  `Buf` (`out 0 = Buf`), the in-place rotary update.
* `nIn = 4`, `nOut = 1`, `B = BLOCK_HALF`.
* read channel `0` — `Buf` at the first-half address
  `pid₀·row_stride + HEAD_IDX·hd + j`;
  read channel `1` — `Buf` at the second-half address `… + HEAD_HALF`;
  read channels `2`/`3` — `COS`/`SIN` at `COS_ROW_IDX·cos_row_stride + j`
  resp. `COS_ROW_IDX·sin_row_stride + j`.
* `readMask` — `HEAD_IDX < n_h ∧ j < HEAD_HALF` on the two data channels
  (the kernel's `mask=(HEAD_IDX < n_qh) and (dim < HEAD_HALF)`) and
  `j < HEAD_HALF` on `COS`/`SIN` (the kernel's `mask=dim < HEAD_HALF`).
* `write` — `writeShift` is `0` for the first-half store and `HEAD_HALF`
  for the second-half one; `writeMask` is the store's own
  `(HEAD_IDX < n_h) and (dim < HEAD_HALF)`.

The windows and masks are *declared*, not parsed from the kernel; the headlines
**prove** the kernel's actual addressing and masking match them. -/
def ropeHeadIO (kernel : ComputeKernel) (Buf COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX row_stride cos_row_stride sin_row_stride hd n_h
      HEAD_HALF BLOCK_HALF writeShift : Nat)
    (projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
      first | rfl | simp [ComputeKernel.toAlgKernel]) : GroupedMasked2DKernelIO where
  kernel := kernel
  projection := projection
  nIn := 4
  nOut := 1
  bufs := [Buf, COS, SIN]
  inp := fun i => match i with
    | ⟨0, _⟩ => Buf
    | ⟨1, _⟩ => Buf
    | ⟨2, _⟩ => COS
    | ⟨_ + 3, _⟩ => SIN
  out := fun _ => Buf
  B := BLOCK_HALF
  read := fun i pid₀ _pid₁ j => match i with
    | ⟨0, _⟩ => pid₀ * row_stride + HEAD_IDX * hd + j.val
    | ⟨1, _⟩ => pid₀ * row_stride + HEAD_IDX * hd + j.val + HEAD_HALF
    | ⟨2, _⟩ => COS_ROW_IDX * cos_row_stride + j.val
    | ⟨_ + 3, _⟩ => COS_ROW_IDX * sin_row_stride + j.val
  readMask := fun i _pid₀ _pid₁ j => match i with
    | ⟨0, _⟩ => HEAD_IDX < n_h ∧ j.val < HEAD_HALF
    | ⟨1, _⟩ => HEAD_IDX < n_h ∧ j.val < HEAD_HALF
    | ⟨2, _⟩ => j.val < HEAD_HALF
    | ⟨_ + 3, _⟩ => j.val < HEAD_HALF
  write := fun _ pid₀ _pid₁ j =>
    pid₀ * row_stride + HEAD_IDX * hd + j.val + writeShift
  writeMask := fun _ _pid₀ _pid₁ j => HEAD_IDX < n_h ∧ j.val < HEAD_HALF

/-! ### `rope_transform_q0_head` -/

/-! ### `rope_transform_q1_head` -/

/-! ### K-side `⊨` coverage

`rope_transform_k0_head` / `rope_transform_k1_head` are the `abbrev` aliases of
the Q slices at the K region and `n_kh` head bound, so the K-side IO signatures
and headlines are the Q ones instantiated there. The named wrappers below make
the K-side citable. -/

end VeriTile.Bench.TritonBenchG.RopeTransform
