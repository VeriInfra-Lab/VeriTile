import bench.repairs.tritonbench_g.index_select_cat.IO

namespace VeriTile.Bench.Repairs.IndexSelectCat.Correct
open VeriTile.Triton FP.Structural
noncomputable section

/-- The existing dependent output assembly is exactly the selected input
payload at the same logical row/column coordinate. -/
theorem gathered_eq_copy (rest : Algebra ℝ) (p : Layout) (rowPid colPid : Nat) :
    gathered rest p rowPid colPid =
      (fun xs => Fin.cases (fun j => xs (1 : Fin 2) j) (fun o => Fin.elim0 o)) := by
  funext xs port
  have eq : port = (0 : Fin 1) := Fin.eq_zero port
  subst port
  rfl

end
end VeriTile.Bench.Repairs.IndexSelectCat.Correct
