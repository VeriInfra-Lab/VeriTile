import bench.tritonbench_g.index_select_cat.IO
import bench.tritonbench_g.index_select_cat.Proofs.IOCorrect
import bench.tritonbench_g.index_select_cat.Proofs.GatherFormula

namespace VeriTile.Bench.Repairs.IndexSelectCat.Correct
open VeriTile.Triton FP.Structural
noncomputable section

/-- The complete metadata-dependent gather judgment in logical coordinates. -/
theorem logical_gather_iff (rest : Algebra ℝ) (p : Layout) (rowPid colPid : Nat) :
    (logicalGatherIO rest p rowPid colPid).Implements (fun xs i j => xs.payload i j) ↔
      (gatherIO rest p rowPid colPid).Implements
        (fun xs => Fin.cases (fun j => xs (1 : Fin 2) j) (fun i => Fin.elim0 i)) := by
  have eq : (logicalGatherIO rest p rowPid colPid).lift (fun xs i j => xs.payload i j) =
      (fun xs => Fin.cases (fun j => xs (1 : Fin 2) j) (fun i => Fin.elim0 i)) := by
    funext xs o j
    fin_cases o
    cases j with | mk a b => cases b with | mk b c => cases c; rfl
  change (gatherIO rest p rowPid colPid).Implements _ ↔ _
  rw [eq]
  rfl

theorem logical_gather_correct (rest : Algebra ℝ) (p : Layout) (rowPid colPid : Nat)
    (unique : LogicalIO.Unique p rowPid colPid) :
    (logicalGatherIO rest p rowPid colPid).Implements (fun xs i j => xs.payload i j) := by
  apply (logical_gather_iff rest p rowPid colPid).mpr
  simpa only [gathered_eq_copy] using IOCorrect.correct rest p rowPid colPid unique
end
end VeriTile.Bench.Repairs.IndexSelectCat.Correct
