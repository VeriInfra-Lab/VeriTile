import bench.tritonbench_g.layer_norm_fwd.Proofs.Contracts
import VeriTile.Triton.Semantics.Step

/-! Exact tagged-cell scatter lemmas for the real and boolean outputs. These
lemmas describe the actual store folds; executing the preceding source phases
and identifying their values remains a separate obligation. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Memory
open VeriTile.Triton
noncomputable section

/-- A store with an already encoded tagged value. -/
def writeCell (s : BlockState) (region : RegionName) (offset : Nat) (value : MemCell) : BlockState :=
  { s with mem := fun r o => if r = region ∧ o = offset then value else s.mem r o }

@[simp] theorem writeCell_mem (s : BlockState) (region : RegionName) (offset : Nat)
    (value : MemCell) (r : RegionName) (o : Nat) :
    (writeCell s region offset value).mem r o =
      if r = region ∧ o = offset then value else s.mem r o := rfl

theorem real_store (s : BlockState) (r : RegionName) (o : Nat) (v : ℝ) :
    s.writeMemTyped .real r o (some v) = writeCell s r o (MemCell.real v) := by
  rfl

theorem bool_store (s : BlockState) (r : RegionName) (o : Nat) (v : Bool) :
    s.writeMemTyped .bool r o v = writeCell s r o (MemCell.of .bool v) := by
  rfl

def writeAt (y : RegionName) (start N : Nat) (values : Nat → MemCell)
    (s : BlockState) (o : Nat) : BlockState :=
  if Contract.inRow start N o then writeCell s y o (values (o-start)) else s

def writeSequence (y : RegionName) (start N : Nat) (values : Nat → MemCell)
    (addresses : List Nat) (s : BlockState) : BlockState :=
  addresses.foldl (writeAt y start N values) s

theorem sequence_cell (y : RegionName) (start N : Nat) (values : Nat → MemCell)
    (addresses : List Nat) (s : BlockState) (r : RegionName) (o : Nat) :
    (writeSequence y start N values addresses s).mem r o =
      if r = y ∧ o ∈ addresses ∧ Contract.inRow start N o
      then values (o-start) else s.mem r o := by
  induction addresses generalizing s with
  | nil => simp [writeSequence]
  | cons a rest ih =>
    simp only [writeSequence, List.foldl_cons] at ⊢
    change (writeSequence y start N values rest (writeAt y start N values s a)).mem r o = _
    rw [ih]
    by_cases hr : r = y
    · subst r
      by_cases ho : o = a
      · subst o
        by_cases hn : Contract.inRow start N a <;> by_cases hm : a ∈ rest
        all_goals simp [writeAt, hn, hm]
      · by_cases hm : o ∈ rest <;> by_cases hn : Contract.inRow start N o <;>
          by_cases ha : Contract.inRow start N a
        all_goals simp [writeAt, ho, hm, hn, ha]
    · by_cases ha : Contract.inRow start N a
      all_goals simp [writeAt, hr, ha]

def addresses (start block : Nat) : List Nat :=
  (TileShape.allIndices [block]).map fun j => start + j.1.val

theorem mem_addresses (start block o : Nat) :
    o ∈ addresses start block ↔ Contract.inRow start block o := by
  constructor
  · intro h
    obtain ⟨j, _, hj⟩ := List.mem_map.mp h
    rw [← hj]
    exact Contract.inRow_of_col j.1.isLt
  · intro h
    obtain ⟨col, hc, rfl⟩ := Contract.inRow_iff_col.mp h
    exact List.mem_map.mpr ⟨(⟨col, hc⟩, PUnit.unit), TileShape.mem_allIndices _ _, rfl⟩

def scatter (y : RegionName) (start N block : Nat) (values : Nat → MemCell)
    (s : BlockState) : BlockState := writeSequence y start N values (addresses start block) s

theorem scatter_fold (y : RegionName) (start N block : Nat) (values : Nat → MemCell)
    (s : BlockState) :
    scatter y start N block values s =
      (TileShape.allIndices [block]).foldl
        (fun acc j => if j.1.val < N then writeCell acc y (start+j.1.val) (values j.1.val) else acc) s := by
  simp only [scatter, writeSequence, addresses, List.foldl_map, writeAt]
  congr 1
  funext acc j
  have h : Contract.inRow start N (start+j.1.val) ↔ j.1.val < N := by
    unfold Contract.inRow
    omega
  simp only [h, Nat.add_sub_cancel_left]

theorem scatter_cell (y : RegionName) (start N block : Nat) (hN : N ≤ block)
    (values : Nat → MemCell) (s : BlockState) (r : RegionName) (o : Nat) :
    (scatter y start N block values s).mem r o =
      if r = y ∧ Contract.inRow start N o then values (o-start) else s.mem r o := by
  rw [scatter, sequence_cell]
  simp only [mem_addresses]
  have h : Contract.inRow start N o → Contract.inRow start block o := by
    unfold Contract.inRow
    omega
  by_cases hn : Contract.inRow start N o
  · simp [hn, h hn]
  · simp [hn]

theorem scatter_active (y : RegionName) (start N block : Nat) (hN : N ≤ block)
    (values : Nat → MemCell) (s : BlockState) (col : Nat) (hcol : col < N) :
    (scatter y start N block values s).mem y (start+col) = values col := by
  rw [scatter_cell _ _ _ _ hN, if_pos ⟨rfl, Contract.inRow_of_col hcol⟩]
  simp

theorem scatter_frame (y : RegionName) (start N block : Nat) (hN : N ≤ block)
    (values : Nat → MemCell) (s : BlockState) (r : RegionName) (o : Nat)
    (h : ¬ (r = y ∧ Contract.inRow start N o)) :
    (scatter y start N block values s).mem r o = s.mem r o := by
  rw [scatter_cell _ _ _ _ hN, if_neg h]

theorem sequence_state (y : RegionName) (start N : Nat) (values : Nat → MemCell)
    (addresses : List Nat) (s : BlockState) :
    (writeSequence y start N values addresses s).regs = s.regs ∧
    (writeSequence y start N values addresses s).pids = s.pids ∧
    (writeSequence y start N values addresses s).undef = s.undef ∧
    (writeSequence y start N values addresses s).numPids = s.numPids := by
  induction addresses generalizing s with
  | nil => exact ⟨rfl, rfl, rfl, rfl⟩
  | cons a rest ih =>
    simp only [writeSequence, List.foldl_cons]
    change (writeSequence y start N values rest (writeAt y start N values s a)).regs = _ ∧ _
    by_cases ha : Contract.inRow start N a
    · simpa only [writeAt, ha, if_true, writeCell] using
        ih (writeCell s y a (values (a-start)))
    · simpa only [writeAt, ha, if_false] using ih s

@[simp] theorem scatter_regs (y : RegionName) (start N block : Nat)
    (values : Nat → MemCell) (s : BlockState) :
    (scatter y start N block values s).regs = s.regs :=
  (sequence_state y start N values (addresses start block) s).1

@[simp] theorem scatter_pids (y : RegionName) (start N block : Nat)
    (values : Nat → MemCell) (s : BlockState) :
    (scatter y start N block values s).pids = s.pids :=
  (sequence_state y start N values (addresses start block) s).2.1

/-- Exact source scalar store used for both Mean and Rstd. -/
theorem scalar_store (region : RegionName) (name : RegName) (row : Nat) (value : ℝ)
    (s : BlockState)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hvalue : s.regs .real [] name = some (Tile.scalar (some value))) :
    stepStmt (.store .real [] (.region region (.ref .nat [] "row"))
      (.ref .real [] name) .none) s =
      some (writeCell s region row (MemCell.real value)) := by
  simp [stepStmt, evalOp.eq_def, hrow, hvalue, TileShape.allIndices,
    BlockState.writeMemTyped, BlockState.writeMemAs, MemCell.real, writeCell]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Memory
