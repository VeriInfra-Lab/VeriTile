import bench.tritonbench_g.matrix_reduction.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.KernelMath
open VeriTile.Triton
noncomputable section
/-- Independent order-theoretic maximum, without any reference to execution
or to the interpreter's reduction operator. -/
def IsMaximum {N : Nat} (xs : Fin N → ℝ) (v : ℝ) : Prop :=
  (∃ j, xs j=v) ∧ ∀ j, xs j≤v

/-- The greatest member of a nonempty finite row, independent of kernel
execution and of any reduction implementation. -/
def maximumValue {N : Nat} (positive : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  (Finset.univ : Finset (Fin N)).sup'
    ⟨⟨0, positive⟩, Finset.mem_univ _⟩ xs

theorem maximumValue_isMaximum {N : Nat} (positive : 0 < N) (xs : Fin N → ℝ) :
    IsMaximum xs (maximumValue positive xs) := by
  constructor
  · obtain ⟨j, _, hj⟩ := Finset.exists_mem_eq_sup'
      (s := (Finset.univ : Finset (Fin N))) ⟨⟨0, positive⟩, Finset.mem_univ _⟩ xs
    exact ⟨j, hj.symm⟩
  · intro j
    exact Finset.le_sup' xs (Finset.mem_univ j)

theorem maximum_eq {N : Nat} (xs : Fin N → ℝ) (a b : ℝ)
    (ha : IsMaximum xs a) (hb : IsMaximum xs b) : a=b := by
  obtain ⟨⟨ia, rfl⟩,ha⟩ := ha
  obtain ⟨⟨ib, rfl⟩,hb⟩ := hb
  exact le_antisymm (hb ia) (ha ib)

theorem reduction_maximum (M N : Nat) (hN : 0<N) (xs : Fin M → Fin N → ℝ) (i : Fin M) :
    IsMaximum (xs i)
      (matrixReduceSpecOf M N (fun k => xs k.1 k.2.1) (i,⟨0,hN⟩,PUnit.unit)) := by
  have heq : matrixReduceSpecOf M N (fun k => xs k.1 k.2.1) (i,⟨0,hN⟩,PUnit.unit)=
      (Finset.univ : Finset (Fin N)).sup' ⟨⟨0,hN⟩,Finset.mem_univ _⟩ (xs i) := by
    simp [matrixReduceSpecOf,Tile.reduceMax,Tile.reduceMaxDrop,TileShape.axisDim,
      TileShape.eraseAxis,TileShape.insertAxisIndex,hN]
    rfl
  rw [heq]
  constructor
  · obtain ⟨j,_,hj⟩ := Finset.exists_mem_eq_sup'
      (s := (Finset.univ : Finset (Fin N))) ⟨⟨0,hN⟩,Finset.mem_univ _⟩ (xs i)
    exact ⟨j,hj.symm⟩
  · intro j
    exact Finset.le_sup' (xs i) (Finset.mem_univ j)


end
end VeriTile.Bench.TritonBenchG.MatrixReduction.KernelMath
