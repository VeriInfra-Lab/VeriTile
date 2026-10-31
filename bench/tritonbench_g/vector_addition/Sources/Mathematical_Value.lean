import bench.tritonbench_g.vector_addition.Sources.Mathematical_Kernels
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Complete original per-program execution in the structural scalar
interpreter. The reference reads input cells before any store and retains
the original mask, including unspecified inactive load values. Arithmetic
is the actual scalar backend addition, without a commutation or Real law. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure Parameters where
  (x y output : RegionName)
  (n block : Nat)

def offset {α : Type} (p : Parameters) (s : State α) (i : TileIndex [p.block]) : Nat :=
  s.pids 0 * p.block + i.1.val

def resultCell {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (mem : RegionName → Nat → Cell α) (o : Nat) : Cell α :=
  .mk .real (M.binary none .real .add ((mem p.x o).read .real) ((mem p.y o).read .real))

def expectedMemory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) : RegionName → Nat → Cell α :=
  (TileShape.allIndices [p.block]).foldl (fun mem i =>
    if offset p s i < p.n then fun r o =>
      if r = p.output ∧ o = offset p s i then resultCell M p s.mem (offset p s i) else mem r o
    else mem) s.mem

def Outside {α : Type} (p : Parameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ i : TileIndex [p.block], offset p s i < p.n → o ≠ offset p s i

theorem masked_scatter_memory {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (region : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write region (off i) (value i)
      else t) s).mem =
      xs.foldl (fun mem i => if active i then fun r o =>
        if r = region ∧ o = off i then value i else mem r o else mem) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    by_cases hi : active i <;> simp only [hi, ↓reduceIte] <;> rfl

theorem expectedMemory_eq_scatter {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    expectedMemory M p s =
      ((TileShape.allIndices [p.block]).foldl (fun t i =>
        if offset p s i < p.n then t.write p.output (offset p s i)
          (resultCell M p s.mem (offset p s i)) else t) s).mem := by
  rw [masked_scatter_memory]
  rfl

theorem offset_injective {α : Type} (p : Parameters) (s : State α) :
    Function.Injective (offset p s) := by
  intro i j h
  exact Prod.ext (Fin.ext (Nat.add_left_cancel h)) (Subsingleton.elim _ _)

theorem expected_readback {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (i : TileIndex [p.block]) (hi : offset p s i < p.n) :
    expectedMemory M p s p.output (offset p s i) = resultCell M p s.mem (offset p s i) := by
  rw [expectedMemory_eq_scatter]
  exact State.masked_scatter_readback p.output (offset p s)
    (fun i => resultCell M p s.mem (offset p s i))
    (fun i => offset p s i < p.n) s i hi (offset_injective p s)

theorem expected_frame {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat) (h : Outside p s r o) :
    expectedMemory M p s r o = s.mem r o := by
  rw [expectedMemory_eq_scatter]
  apply State.masked_scatter_frame
  exact h.imp id (fun h i _ hi => h i hi)

/-- Closed address formula for one whole source program. -/
theorem expected_cell {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (r : RegionName) (o : Nat) :
    expectedMemory M p s r o =
      if r = p.output ∧ s.pids 0 * p.block ≤ o ∧
          o < s.pids 0 * p.block + p.block ∧ o < p.n then
        resultCell M p s.mem o else s.mem r o := by
  by_cases h : r = p.output ∧ s.pids 0 * p.block ≤ o ∧
      o < s.pids 0 * p.block + p.block ∧ o < p.n
  · rw [if_pos h]
    let i : TileIndex [p.block] := (⟨o - s.pids 0 * p.block, by omega⟩, PUnit.unit)
    have ho : offset p s i = o := by simp only [offset, i]; omega
    rw [h.1, ← ho]
    exact expected_readback M p s i (by rw [ho]; exact h.2.2.2)
  · rw [if_neg h]
    apply expected_frame
    by_cases hr : r = p.output
    · right
      intro i hi ho
      apply h
      have bound := i.1.isLt
      unfold offset at hi ho
      exact ⟨hr, by omega, by omega, by omega⟩
    · exact Or.inl hr

#specNonCircular expectedMemory avoiding [FP.Structural.exec, FP.Structural.run]
#axiomsClean masked_scatter_memory
#axiomsClean expectedMemory_eq_scatter
#axiomsClean offset_injective
#axiomsClean expected_readback
#axiomsClean expected_frame
#axiomsClean expected_cell
end
end VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs
