import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.Rewrite
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.PublicIO
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.KernelNativeResult
import VeriTile.Meta.Specification

/-! Auxiliary complete native payload-copy observation. The source index-commutation equivalence is in FPEquiv. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv1.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

/-- Complete FP32 payload copy under the same finite IO as the mathematical
entry. Integer metadata selects destinations; every FP32 word is admitted.
The source trace, all row outputs and the full exterior frame are included. -/
theorem destindex_copy_kv1_kernel_fp32 (rest : Algebra Word) (c : Config) (ps : List Nat)
    (order : ∀ p,p∈ps ↔ p<c.count) :
    (CopyIO.copyIO rest c ps).ImplementsOn (CopyIO.Admissible c) (CopyIO.target c) :=
  CopyIO.implements rest c ps order

#stmtSurfaceSubset destindex_copy_kv1_kernel_fp32 ⊆
  [FP.Structural.Algebra,DestindexCopyKv1.Word,Config,Config.count,CopyIO.copyIO,ProgramIO.ImplementsOn,CopyIO.Admissible,CopyIO.target]
#specNonCircular CopyIO.target avoiding [kernel,Grid.run,FP.Structural.exec]

#axiomsClean destindex_copy_kv1_kernel_fp32

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.FPEquiv
