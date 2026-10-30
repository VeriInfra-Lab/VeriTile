import bench.repairs.tritonbench_g.destindex_copy_kv2.IO
import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.PublicIO
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option autoImplicit false

theorem logical_iff (rest : Algebra ℝ) (c : Config) (ps : List Nat) :
    (logicalIO rest c ps).ImplementsOn (LogicalAdmissible c)
      (fun xs row head dim => xs.payload row head dim) ↔
      (copyIO rest c ps).ImplementsOn (Admissible c) (target c) := by
  change (copyIO rest c ps).ImplementsOn
    (fun raw => LogicalAdmissible c (DestindexCopyKv1.CopyIO.inputEquiv ℝ c raw))
    (fun raw => (DestindexCopyKv1.CopyIO.outputEquiv ℝ c).symm (fun row head dim =>
      (DestindexCopyKv1.CopyIO.inputEquiv ℝ c raw).payload row head dim)) ↔ _
  exact Iff.of_eq (congrArg₂ (fun domain formula =>
    (copyIO rest c ps).ImplementsOn domain formula)
      (DestindexCopyKv1.CopyIO.admissible_pullback ℝ c)
      (DestindexCopyKv1.CopyIO.target_transport ℝ c))

theorem logical_correct (rest : Algebra ℝ) (c : Config)
    (width : block c.dims = c.dims) (ps : List Nat) (schedule : ∀ p,p∈ps ↔ p<c.count) :
    (logicalIO rest c ps).ImplementsOn (LogicalAdmissible c)
      (fun xs row head dim => xs.payload row head dim) :=
  (logical_iff rest c ps).mpr (implements rest c width ps schedule)

end
end VeriTile.Bench.Repairs.DestindexCopyKv2.CopyIO
