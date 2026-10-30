import bench.tritonbench_g.destindex_copy_kv1.Proofs.ReaderFP
import bench.tritonbench_g.destindex_copy_kv1.Proofs.Rewrite
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

def original (p : Layout) : Program := ⟨kernel p, []⟩
def indexCommuted (p : Layout) : Program := ⟨indexCommutedKernel p, []⟩
def assumptions : VeriTile.Spec.Assumptions FP.GuardedFragment := []

/-- The existing source-row integer product rewrite preserves successful
execution and every typed memory cell, without any floating numerical law. -/
specification index_commuted_equiv (p : Layout) :
    original p ≡[assumptions] indexCommuted p :=
  ReaderProof.index_commuted_equiv p


#print_fp_assumptions index_commuted_equiv
#axiomsClean index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.FPEquiv
