import bench.repairs.tritonbench_g.destindex_copy.IO
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.PublicIO
import bench.repairs.tritonbench_g.destindex_copy.Proofs.RawResults
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.Rewrite
import VeriTile.Meta.Specification

/-! Auxiliary complete native payload-copy observation. The source index-commutation equivalence is in FPEquiv. -/
namespace VeriTile.Bench.Repairs.DestindexCopy.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open DestindexCopyKv1
open scoped VeriTile.Spec
noncomputable section

/-- The same complete finite-channel IO preserves every FP32 payload word,
with actual source access safety, exact typed outputs and the exterior frame. -/
theorem destindex_copy_fp32 (rest : Algebra Word) (c : Config) (ps : List Nat)
    (order : ∀ p,p∈ps ↔ p<c.count) :
    (copyIO rest c ps).ImplementsOn (CopyIO.Admissible c) (CopyIO.target c) :=
  CopyIO.implements rest c ps order

#stmtSurfaceSubset destindex_copy_fp32 ⊆
  [VeriTile.Triton.FP.Structural.Algebra,Word,VeriTile.Bench.Repairs.DestindexCopy.Config,DestindexCopyKv1.Config.count,copyIO,ProgramIO.ImplementsOn,
   CopyIO.Admissible,CopyIO.target]

#specNonCircular CopyIO.target avoiding [kernel,Grid.run,FP.Structural.exec]

#axiomsClean destindex_copy_fp32

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopy.FPEquiv
