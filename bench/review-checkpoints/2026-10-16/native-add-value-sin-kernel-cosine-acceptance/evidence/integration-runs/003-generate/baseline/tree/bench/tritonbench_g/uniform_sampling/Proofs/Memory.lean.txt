import bench.tritonbench_g.uniform_sampling.Proofs.Formula
import VeriTile.Triton.Semantics.State

/-! Four-store mathematical writeback and exact tagged-cell frame.
This store-phase helper is independent of a successful source execution.
Connecting it to the complete frontend kernel remains an execution obligation.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.Memory
open VeriTile.Triton
noncomputable section

def writeAt (y : RegionName) (N B : Nat) (seed offset : Int) (from_ upper : ℝ)
    (s : BlockState) (o : Nat) : BlockState :=
  if o < N then s.writeMem y o (Formula.expected B seed offset from_ upper o) else s

def writeSequence (y : RegionName) (N B : Nat) (seed offset : Int) (from_ upper : ℝ)
    (addresses : List Nat) (base : BlockState) : BlockState :=
  addresses.foldl (writeAt y N B seed offset from_ upper) base

theorem writeSequence_cons (y : RegionName) (N B : Nat) (seed offset : Int)
    (from_ upper : ℝ) (a : Nat) (rest : List Nat) (base : BlockState) :
    writeSequence y N B seed offset from_ upper (a :: rest) base =
      writeSequence y N B seed offset from_ upper rest
        (writeAt y N B seed offset from_ upper base a) := rfl

theorem writeSequence_cell (y : RegionName) (N B : Nat) (seed offset : Int)
    (from_ upper : ℝ) (addresses : List Nat) (base : BlockState) (r : RegionName) (o : Nat) :
    (writeSequence y N B seed offset from_ upper addresses base).mem r o =
      if r = y ∧ o ∈ addresses ∧ o < N
      then MemCell.real (Formula.expected B seed offset from_ upper o) else base.mem r o := by
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

theorem writeSequence_state (y : RegionName) (N B : Nat) (seed offset : Int)
    (from_ upper : ℝ) (addresses : List Nat) (base : BlockState) :
    (writeSequence y N B seed offset from_ upper addresses base).regs = base.regs ∧
    (writeSequence y N B seed offset from_ upper addresses base).pids = base.pids ∧
    (writeSequence y N B seed offset from_ upper addresses base).undef = base.undef ∧
    (writeSequence y N B seed offset from_ upper addresses base).numPids = base.numPids := by
  induction addresses generalizing base with
  | nil => exact ⟨rfl, rfl, rfl, rfl⟩
  | cons a rest ih =>
      simp only [writeSequence_cons]
      by_cases ha : a < N
      · simpa only [writeAt, ha, if_true, BlockState.writeMem] using
          ih (base.writeMem y a (Formula.expected B seed offset from_ upper a))
      · simpa only [writeAt, ha, if_false] using ih base

def fieldAddresses (pid B : Nat) (f : Fin 4) : List Nat :=
  (TileShape.allIndices [B]).map fun j => SourceDomain.outputIndex pid B f j.1

def addresses (pid B : Nat) : List Nat :=
  fieldAddresses pid B 0 ++ fieldAddresses pid B 1 ++
    fieldAddresses pid B 2 ++ fieldAddresses pid B 3

theorem mem_fieldAddresses (pid B : Nat) (f : Fin 4) (o : Nat) :
    o ∈ fieldAddresses pid B f ↔ ∃ j : Fin B, SourceDomain.outputIndex pid B f j = o := by
  constructor
  · intro h
    obtain ⟨j, _, hj⟩ := List.mem_map.mp h
    exact ⟨j.1, hj⟩
  · rintro ⟨j, hj⟩
    exact List.mem_map.mpr ⟨(j, PUnit.unit), TileShape.mem_allIndices _ _, hj⟩

theorem mem_addresses (pid B o : Nat) :
    o ∈ addresses pid B ↔ ∃ f : Fin 4, ∃ j : Fin B, SourceDomain.outputIndex pid B f j = o := by
  simp only [addresses, List.mem_append, mem_fieldAddresses]
  constructor
  · intro h
    rcases h with ((h | h) | h) | h
    · exact ⟨0, h⟩
    · exact ⟨1, h⟩
    · exact ⟨2, h⟩
    · exact ⟨3, h⟩
  · rintro ⟨f, h⟩
    fin_cases f <;> tauto

def storeBlock (pid : Nat) (base : BlockState) (y : RegionName) (N B : Nat)
    (seed offset : Int) (from_ upper : ℝ) : BlockState :=
  writeSequence y N B seed offset from_ upper (addresses pid B) base

/-- One complete source-sized output stream, including its independent mask. -/
def storeField (pid : Nat) (base : BlockState) (y : RegionName) (N B : Nat)
    (seed offset : Int) (from_ upper : ℝ) (f : Fin 4) : BlockState :=
  (TileShape.allIndices [B]).foldl (fun s j =>
    let o := SourceDomain.outputIndex pid B f j.1
    if o < N then s.writeMem y o
      (ScalarSpec.value seed offset (SourceDomain.counterIndex pid B j.1.val)
        (Formula.field f.val) from_ upper) else s) base

theorem writeSequence_append (y : RegionName) (N B : Nat) (seed offset : Int)
    (from_ upper : ℝ) (before after : List Nat) (base : BlockState) :
    writeSequence y N B seed offset from_ upper (before ++ after) base =
      writeSequence y N B seed offset from_ upper after
        (writeSequence y N B seed offset from_ upper before base) := by
  exact List.foldl_append

theorem writeSequence_field {B : Nat} (hB : 0 < B) (pid : Nat) (base : BlockState)
    (y : RegionName) (N : Nat) (seed offset : Int) (from_ upper : ℝ) (f : Fin 4) :
    writeSequence y N B seed offset from_ upper (fieldAddresses pid B f) base =
      storeField pid base y N B seed offset from_ upper f := by
  unfold writeSequence fieldAddresses storeField
  rw [List.foldl_map]
  congr 1
  funext s j
  simp only [writeAt, Formula.expected_at hB]

/-- The same mathematical writeback as four consecutive complete tile stores.
No field, inactive-lane mask, or per-field scalar formula is omitted. -/
theorem storeBlock_scatter {B : Nat} (hB : 0 < B) (pid : Nat) (base : BlockState)
    (y : RegionName) (N : Nat) (seed offset : Int) (from_ upper : ℝ) :
    storeBlock pid base y N B seed offset from_ upper =
      storeField pid
        (storeField pid
          (storeField pid
            (storeField pid base y N B seed offset from_ upper 0)
            y N B seed offset from_ upper 1)
          y N B seed offset from_ upper 2)
        y N B seed offset from_ upper 3 := by
  simp only [storeBlock, addresses, writeSequence_append, writeSequence_field hB]

theorem storeBlock_cell (pid : Nat) (base : BlockState) (y : RegionName) (N B : Nat)
    (seed offset : Int) (from_ upper : ℝ) (r : RegionName) (o : Nat) :
    (storeBlock pid base y N B seed offset from_ upper).mem r o =
      if r = y ∧ (∃ f : Fin 4, ∃ j : Fin B, SourceDomain.outputIndex pid B f j = o) ∧ o < N
      then MemCell.real (Formula.expected B seed offset from_ upper o) else base.mem r o := by
  exact (writeSequence_cell _ _ _ _ _ _ _ _ _ _ _).trans (by simp only [mem_addresses])

theorem storeBlock_active_cell (pid : Nat) (base : BlockState) (y : RegionName) (N B : Nat)
    (seed offset : Int) (from_ upper : ℝ) (f : Fin 4) (j : Fin B)
    (hj : SourceDomain.outputIndex pid B f j < N) :
    (storeBlock pid base y N B seed offset from_ upper).mem y (SourceDomain.outputIndex pid B f j) =
      MemCell.real (Formula.expected B seed offset from_ upper (SourceDomain.outputIndex pid B f j)) := by
  rw [storeBlock_cell, if_pos ⟨rfl, ⟨f, j, rfl⟩, hj⟩]

theorem storeBlock_frame (pid : Nat) (base : BlockState) (y : RegionName) (N B : Nat)
    (seed offset : Int) (from_ upper : ℝ) (r : RegionName) (o : Nat)
    (h : ∀ f : Fin 4, ∀ j : Fin B, SourceDomain.outputIndex pid B f j < N →
      ¬ (y = r ∧ SourceDomain.outputIndex pid B f j = o)) :
    (storeBlock pid base y N B seed offset from_ upper).mem r o = base.mem r o := by
  rw [storeBlock_cell, if_neg]
  rintro ⟨hr, ⟨f, j, hj⟩, hn⟩
  exact h f j (hj ▸ hn) ⟨hr.symm, hj⟩

#axiomsClean writeSequence_cell
#axiomsClean writeSequence_state
#axiomsClean mem_addresses
#axiomsClean writeSequence_field
#axiomsClean storeBlock_scatter
#axiomsClean storeBlock_active_cell
#axiomsClean storeBlock_frame

end
end VeriTile.Bench.TritonBenchG.UniformSampling.Memory
