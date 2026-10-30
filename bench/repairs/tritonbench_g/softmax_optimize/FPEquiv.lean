import bench.repairs.tritonbench_g.softmax_optimize.Kernels
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.KernelRewrite
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Config

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (Y X : RegionName) (M N : Nat) : Program where
  kernel := kernel Y X M N (Config.tile N)
  domain := []
def indexCommuted (Y X : RegionName) (M N : Nat) : Program where
  kernel := indexCommutedKernel Y X M N (Config.tile N)
  domain := []

/-- The complete typed FP32 source is equivalent after commuting the integer
row product in all four loops. No floating transformation law is assumed. -/
specification softmax_source_equiv (Y X : RegionName) (M N : Nat) :
    original Y X M N ≡[assumptions] indexCommuted Y X M N :=
  KernelRewrite.equivalent Y X M N (Config.tile N)

#print_fp_assumptions softmax_source_equiv
#axiomsClean softmax_source_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.FPEquiv
