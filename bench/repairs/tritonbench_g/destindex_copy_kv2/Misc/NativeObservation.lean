import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.Rewrite
import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.PublicIO
import VeriTile.Meta.Specification

/-! Auxiliary complete native payload-copy observation. The source index-commutation equivalence is in FPEquiv. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv2.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section
set_option autoImplicit false

/-- Complete source observation on every FP32 word. There is no floating
arithmetic or finite-word restriction; metadata remains the logical signed
nonnegative injective destination map from the shared typed IO. -/
def copyIO (rest : Algebra (BitVec 32)) (c : Config) (ps : List Nat) := CopyIO.copyIO rest c ps

theorem destindex_copy_kv2_fp32 (rest : Algebra (BitVec 32)) (c : Config)
    (width : block c.dims = c.dims) (ps : List Nat)
    (schedule : ∀ p, p∈ps ↔ p<c.count) :
    (copyIO rest c ps).ImplementsOn (CopyIO.Admissible c) (CopyIO.target c) :=
  CopyIO.implements rest c width ps schedule

#stmtSurfaceSubset destindex_copy_kv2_fp32 ⊆
  [copyIO, Config, block, DestindexCopyKv1.Config.dims, DestindexCopyKv1.Config.count,
   FP.Structural.Algebra, ProgramIO.ImplementsOn, CopyIO.Admissible, CopyIO.target]

#axiomsClean destindex_copy_kv2_fp32

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv2.FPEquiv
