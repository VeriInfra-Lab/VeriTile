import bench.tritonbench_g.matmul_persistent_triton.Proofs.Mathematical_ScheduleInjective
import Mathlib.Data.Fintype.EquivFin

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.Schedule

/-- Bounded grouped coordinates from the original source schedule. -/
def coordinate (rows columns group : Nat) (hg : 0 < group)
    (tile : Fin (rows*columns)) : Fin rows × Fin columns :=
  ⟨⟨row rows columns group tile.val, (grouped_bounds rows columns group tile.val hg tile.isLt).2.1⟩,
   ⟨column rows columns group tile.val, (grouped_bounds rows columns group tile.val hg tile.isLt).2.2⟩⟩

theorem coordinate_injective (rows columns group : Nat) (hg : 0 < group) :
    Function.Injective (coordinate rows columns group hg) := by
  intro a b he
  apply Fin.ext
  exact grouped_injective rows columns group a.val b.val hg a.isLt b.isLt
    (congrArg (fun x => x.1.val) he) (congrArg (fun x => x.2.val) he)

/-- Every valid output tile coordinate occurs in the grouped source schedule,
including zero-size spaces and the shortened last group. -/
theorem coordinate_bijective (rows columns group : Nat) (hg : 0 < group) :
    Function.Bijective (coordinate rows columns group hg) := by
  apply (Fintype.bijective_iff_injective_and_card _).mpr
  exact ⟨coordinate_injective rows columns group hg, by simp⟩

theorem grouped_covered (rows columns group rowIndex columnIndex : Nat)
    (hg : 0 < group) (hr : rowIndex < rows) (hc : columnIndex < columns) :
    ∃ tile, tile < rows*columns ∧ row rows columns group tile = rowIndex ∧
      column rows columns group tile = columnIndex := by
  obtain ⟨tile, ht⟩ := (coordinate_bijective rows columns group hg).2 (⟨rowIndex,hr⟩,⟨columnIndex,hc⟩)
  exact ⟨tile.val,tile.isLt,congrArg (fun x => x.1.val) ht,congrArg (fun x => x.2.val) ht⟩

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.Schedule
