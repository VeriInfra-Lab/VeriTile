import bench.repairs.tritonbench_g.index_select_cat.Kernels
import bench.repairs.tritonbench_g.index_select_cat.Proofs.Rewrite

namespace VeriTile.Bench.Repairs.IndexSelectCat.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (p : Layout) : Program where
  kernel := kernel p
  domain := []
def indexCommuted (p : Layout) : Program where
  kernel := indexCommutedKernel p
  domain := []

specification index_select_index_commuted_equiv (p : Layout) :
    original p ≡[assumptions] indexCommuted p := Rewrite.equivalent p

#print_fp_assumptions index_select_index_commuted_equiv
#axiomsClean index_select_index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IndexSelectCat.FPEquiv
