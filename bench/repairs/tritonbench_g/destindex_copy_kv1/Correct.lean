import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.LogicalReader
import bench.repairs.tritonbench_g.destindex_copy_kv1.IO
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.PublicIO
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.KernelRealResult

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete mathematical row copy on the finite public IO in IO.lean. The
logical destination map is nonnegative and injective; every source access is
proved safe and the exact exterior-cell frame is preserved. -/
specification destindex_copy_kv1_kernel_real (rest : Algebra ℝ) (c : Config) (ps : List Nat)
    (order : ∀ p,p∈ps ↔ p<c.count) :
    Spec.Real ((CopyIO.logicalIO rest c ps) ⊨{CopyIO.LogicalAdmissible c}
      (fun xs row head dim => xs.payload row head dim)) :=
  CopyIO.logical_correct rest c ps order

#specNonCircular CopyIO.target avoiding [kernel,Grid.run,FP.Structural.exec]
#axiomsClean destindex_copy_kv1_kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
#stmtSurfaceSubset destindex_copy_kv1_kernel_real ⊆
  [VeriTile.Bench.Repairs.DestindexCopyKv1.Config,
   VeriTile.Bench.Repairs.DestindexCopyKv1.Config.count,
   VeriTile.Bench.Repairs.DestindexCopyKv1.Config.dims,
   VeriTile.Bench.Repairs.DestindexCopyKv1.Config.heads,
   VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO.LogicalAdmissible,
   VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO.LogicalInputs,
   VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO.LogicalInputs.payload,
   VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO.LogicalOutputs,
   VeriTile.Bench.Repairs.DestindexCopyKv1.CopyIO.logicalIO,
   VeriTile.Spec.Real,
   VeriTile.Triton.FP.Structural.Algebra,
   VeriTile.Triton.FP.Structural.LogicalProgramIO.ImplementsOn]

end
end VeriTile.Bench.Repairs.DestindexCopyKv1.Correct
