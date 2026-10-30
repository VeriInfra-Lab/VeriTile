import bench.tritonbench_g.destindex_copy.Proofs.ReaderFP
import bench.tritonbench_g.destindex_copy.IO
import bench.tritonbench_g.destindex_copy_kv1.Proofs.Rewrite
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.DestindexCopy.FPEquiv
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open DestindexCopyKv1
open scoped VeriTile.Spec
noncomputable section

def original (p : Layout) : Program := ⟨kernel p, []⟩
def indexCommuted (p : Layout) : Program := ⟨indexCommutedKernel p, []⟩
def assumptions : VeriTile.Spec.Assumptions FP.GuardedFragment := []

/-- Existing index-commuted source for either repaired plane, with successful
execution and all typed memory preserved. No numerical assumption is used. -/
specification index_commuted_equiv (p : Layout) :
    original p ≡[assumptions] indexCommuted p :=
  ReaderProof.index_commuted_equiv p


#print_fp_assumptions index_commuted_equiv
#axiomsClean index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopy.FPEquiv
