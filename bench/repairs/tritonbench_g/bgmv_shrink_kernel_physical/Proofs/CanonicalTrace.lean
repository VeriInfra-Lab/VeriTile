import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPWorkset

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.CanonicalTrace
open VeriTile.Triton FP.Structural FPValues
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Linearization
noncomputable section
variable {α : Type} [Inhabited α]

/-- The old local transition semantics constructs its finite ordered result. -/
theorem fold_trace (step : Fin 64 → α → α) (order : List (Fin 64)) (value : α) :
    Trace step order value (order.foldl (fun old split => step split old) value) := by
  induction order generalizing value with
  | nil => exact .nil value
  | cons split rest ih => exact .cons split rest value (step split value) _ rfl (ih _)

def run (M : Algebra α) (P : Primitives α) (h : Call) (orders : Orders h) (data : Data α) :
    FPWorkset.Run M P h orders data where
  output := FPValues.target M P h orders data
  cells := by
    intro b n
    unfold events FPValues.target
    by_cases sentinel : data.indices b.val = -1
    · simp only [sentinel,if_true]
      exact .nil _
    · simp only [sentinel,if_false]
      exact fold_trace (FPWorkset.step M P h data b n) (orders.order b n) (FPWorkset.initial h data b n)

#axiomsClean fold_trace
#axiomsClean run
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.CanonicalTrace
