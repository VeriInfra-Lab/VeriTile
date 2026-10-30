import bench.tritonbench_g.seeded_dropout.Proofs.Index
import bench.tritonbench_g.seeded_dropout.Proofs.Formula
import VeriTile.Triton.Semantics.State

/-! The mathematical writeback phase uses values from the initial input memory
and the fixed canonical generator. This is a store-phase helper, not a claim
that the complete source has executed. Full tagged cells and all state fields
other than the deliberately changed memory cells are preserved. -/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.Memory
open VeriTile.Triton
noncomputable section

def writeAt (input : Nat → ℝ) (y : RegionName) (N : Nat) (p : ℝ) (seed : Int)
    (s : BlockState) (o : Nat) : BlockState :=
  if o < N then s.writeMem y o (Formula.expected input p seed o) else s

def writeSequence (input : Nat → ℝ) (y : RegionName) (N : Nat) (p : ℝ) (seed : Int)
    (addresses : List Nat) (base : BlockState) : BlockState :=
  addresses.foldl (writeAt input y N p seed) base

theorem writeSequence_cons (input : Nat → ℝ) (y : RegionName) (N : Nat)
    (p : ℝ) (seed : Int) (a : Nat) (rest : List Nat) (base : BlockState) :
    writeSequence input y N p seed (a :: rest) base =
      writeSequence input y N p seed rest (writeAt input y N p seed base a) := rfl

/-- Exact tagged-cell result, including inactive lanes and every other region.
Repeated offsets do not weaken the result: the value at each address is fixed
by the original input snapshot and original seed/p. -/
theorem writeSequence_cell (input : Nat → ℝ) (y : RegionName) (N : Nat)
    (p : ℝ) (seed : Int) (addresses : List Nat) (base : BlockState)
    (r : RegionName) (o : Nat) :
    (writeSequence input y N p seed addresses base).mem r o =
      if r = y ∧ o ∈ addresses ∧ o < N
      then MemCell.real (Formula.expected input p seed o) else base.mem r o := by
  induction addresses generalizing base with
  | nil => simp [writeSequence]
  | cons a rest ih =>
    rw [writeSequence_cons, ih]
    by_cases hr : r = y
    · subst r
      by_cases ho : o = a
      · subst o
        by_cases hn : a < N <;> by_cases hm : a ∈ rest
        all_goals simp [writeAt, BlockState.writeMem_mem, hn, hm]
      · by_cases hm : o ∈ rest <;> by_cases hn : o < N <;> by_cases ha : a < N
        all_goals simp [writeAt, BlockState.writeMem_mem, ho, hm, hn, ha]
    · by_cases ha : a < N
      all_goals simp [writeAt, BlockState.writeMem_mem, hr, ha]

theorem writeSequence_state (input : Nat → ℝ) (y : RegionName) (N : Nat)
    (p : ℝ) (seed : Int) (addresses : List Nat) (base : BlockState) :
    (writeSequence input y N p seed addresses base).regs = base.regs ∧
    (writeSequence input y N p seed addresses base).pids = base.pids ∧
    (writeSequence input y N p seed addresses base).undef = base.undef ∧
    (writeSequence input y N p seed addresses base).numPids = base.numPids := by
  induction addresses generalizing base with
  | nil => exact ⟨rfl, rfl, rfl, rfl⟩
  | cons a rest ih =>
    simp only [writeSequence_cons]
    by_cases ha : a < N
    · simpa only [writeAt, ha, if_true, BlockState.writeMem] using
        ih (base.writeMem y a (Formula.expected input p seed a))
    · simpa only [writeAt, ha, if_false] using ih base

def addresses (pid B : Nat) : List Nat :=
  (TileShape.allIndices [B]).map fun j => Index.offset pid B j.1

theorem mem_addresses (pid B o : Nat) :
    o ∈ addresses pid B ↔ ∃ j : Fin B, Index.offset pid B j = o := by
  constructor
  · intro h
    obtain ⟨j, _, hj⟩ := List.mem_map.mp h
    exact ⟨j.1, hj⟩
  · rintro ⟨j, hj⟩
    exact List.mem_map.mpr ⟨(j, PUnit.unit), TileShape.mem_allIndices _ _, hj⟩

/-- Store phase after arbitrary preceding register assignments. Values always
refer to the original input, even if x and y are the same region. -/
def storeBlock (initial base : BlockState) (x y : RegionName) (N B : Nat)
    (p : ℝ) (seed : Int) : BlockState :=
  writeSequence (initial.readMem x) y N p seed (addresses initial.pid B) base

theorem storeBlock_scatter (initial base : BlockState) (x y : RegionName)
    (N B : Nat) (p : ℝ) (seed : Int) :
    storeBlock initial base x y N B p seed =
      (TileShape.allIndices [B]).foldl (fun acc j =>
        if Index.offset initial.pid B j.1 < N
        then acc.writeMem y (Index.offset initial.pid B j.1)
          (Formula.expected (initial.readMem x) p seed (Index.offset initial.pid B j.1))
        else acc) base := by
  simp only [storeBlock, writeSequence, addresses, List.foldl_map, writeAt]

theorem storeBlock_cell (initial base : BlockState) (x y : RegionName)
    (N B : Nat) (p : ℝ) (seed : Int) (r : RegionName) (o : Nat) :
    (storeBlock initial base x y N B p seed).mem r o =
      if r = y ∧ (∃ j : Fin B, Index.offset initial.pid B j = o) ∧ o < N
      then MemCell.real (Formula.expected (initial.readMem x) p seed o)
      else base.mem r o := by
  exact (writeSequence_cell _ _ _ _ _ _ _ _ _).trans (by simp only [mem_addresses])

theorem storeBlock_active_cell (initial base : BlockState) (x y : RegionName)
    (N B : Nat) (p : ℝ) (seed : Int) (j : Fin B)
    (hj : Index.offset initial.pid B j < N) :
    (storeBlock initial base x y N B p seed).mem y (Index.offset initial.pid B j) =
      MemCell.real (Formula.expected (initial.readMem x) p seed (Index.offset initial.pid B j)) := by
  rw [storeBlock_cell, if_pos ⟨rfl, ⟨j, rfl⟩, hj⟩]

theorem storeBlock_frame (initial base : BlockState) (x y : RegionName)
    (N B : Nat) (p : ℝ) (seed : Int) (r : RegionName) (o : Nat)
    (h : ∀ j : Fin B, Index.offset initial.pid B j < N →
      ¬ (y = r ∧ Index.offset initial.pid B j = o)) :
    (storeBlock initial base x y N B p seed).mem r o = base.mem r o := by
  rw [storeBlock_cell, if_neg]
  rintro ⟨hr, ⟨j, hj⟩, hn⟩
  exact h j (hj ▸ hn) ⟨hr.symm, hj⟩

theorem storeBlock_state (initial base : BlockState) (x y : RegionName)
    (N B : Nat) (p : ℝ) (seed : Int) :
    (storeBlock initial base x y N B p seed).regs = base.regs ∧
    (storeBlock initial base x y N B p seed).pids = base.pids ∧
    (storeBlock initial base x y N B p seed).undef = base.undef ∧
    (storeBlock initial base x y N B p seed).numPids = base.numPids :=
  writeSequence_state _ _ _ _ _ _ _

end
end VeriTile.Bench.TritonBenchG.SeededDropout.Memory
