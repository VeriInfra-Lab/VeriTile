import bench.tritonbench_g.destindex_copy_kv1.Proofs.LogicalReader
import bench.tritonbench_g.destindex_copy.IO
import bench.tritonbench_g.destindex_copy_kv1.Proofs.PublicIO
import bench.tritonbench_g.destindex_copy.Proofs.RawResults
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.DestindexCopy.Correct
open VeriTile.Triton FP.Structural
open DestindexCopyKv1
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The shared finite-channel IO implements the independent row copy for every
complete program order. The nonnegative injective destination map is explicit. -/
specification destindex_copy_real (rest : Algebra ℝ) (c : Config) (ps : List Nat)
    (order : ∀ p,p∈ps ↔ p<c.count) :
    Spec.Real ((logicalCopyIO rest c ps) ⊨{CopyIO.LogicalAdmissible c}
      (fun xs row head dim => xs.payload row head dim)) :=
  CopyIO.logical_correct rest c ps order

#axiomsClean destindex_copy_real
#specNonCircular CopyIO.target avoiding [kernel,Grid.run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
#stmtSurfaceSubset destindex_copy_real ⊆
  [VeriTile.Bench.Repairs.DestindexCopy.Config,
   VeriTile.Bench.Repairs.DestindexCopy.logicalCopyIO,
   VeriTile.Bench.Repairs.DestindexCopyKv1.Config.count,
   VeriTile.Bench.Repairs.DestindexCopyKv1.Config.dims,
   VeriTile.Bench.Repairs.DestindexCopyKv1.Config.heads,
   VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO.LogicalAdmissible,
   VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO.LogicalInputs,
   VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO.LogicalInputs.payload,
   VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO.LogicalOutputs,
   VeriTile.Spec.Real,
   VeriTile.Triton.FP.Structural.Algebra,
   VeriTile.Triton.FP.Structural.LogicalProgramIO.ImplementsOn]

end
end VeriTile.Bench.Repairs.DestindexCopy.Correct
