import bench.repairs.tritonbench_g.index_select_cat.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.Repairs.IndexSelectCat.Rewrite
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (p : Layout) : Program where
  kernel := kernel p
  domain := []
def indexCommuted (p : Layout) : Program where
  kernel := indexCommutedKernel p
  domain := []

theorem equivalent (p : Layout) : original p ≡[assumptions] indexCommuted p := by
  equiv_decompose <;> fp_prove

/-- Exact integer-prefix execution equality composes with the unchanged
remainder, so it also transports successful complete execution. -/
theorem execution_eq {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Layout) (s : FP.Structural.State α) :
    FP.Structural.exec M (kernel p) s = FP.Structural.exec M (indexCommutedKernel p) s := by
  have firstSteps : FP.Structural.run M ((kernel p).surfaceBody.take 3) s =
      FP.Structural.run M ((indexCommutedKernel p).surfaceBody.take 3) s := by
    simp [kernel,indexCommutedKernel,ComputeKernel.surfaceBody,FP.Structural.run,
      FP.Structural.step,FP.Structural.evalExpr,FP.Structural.evalOp_unfold,
      FP.Structural.numeric,ComputeDType.eraseDType,Option.bind]
    congr 1
    funext i
    simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
  change FP.Structural.run M (kernel p).surfaceBody s =
    FP.Structural.run M (indexCommutedKernel p).surfaceBody s
  rw [←List.take_append_drop 3 (kernel p).surfaceBody,
    ←List.take_append_drop 3 (indexCommutedKernel p).surfaceBody]
  rw [FP.Structural.run_append,FP.Structural.run_append,firstSteps]
  rfl

#print_fp_assumptions equivalent
#axiomsClean equivalent
#axiomsClean execution_eq
end VeriTile.Bench.Repairs.IndexSelectCat.Rewrite
