import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Reference
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.TypedReadback

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 40000000
set_option maxRecDepth 64000
set_option linter.unusedSimpArgs false

private theorem fold_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

private theorem fold_pids {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).pids = s.pids := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

/-- Complete direct forward execution at the checked output width and column offset. -/
def forward {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (forwardKernel p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp (config := {maxSteps := 8000000}) [forwardKernel, forwardSource, Original.kernel,
    Original.source, forwardParameters, FP.Structural.exec,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    State.setReg, store, ofFloat, toFloat, Region.cast, ComputeDType.eraseDType,
    numeric, numericLt, natLt, bop, Option.bind, fold_regs, fold_pids]
  try rfl

/-- The complete repaired backward source executes all loads, polynomial
operations, original-zero accumulation and three stores. -/
def backward {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (backwardKernel p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp (config := {maxSteps := 8000000}) [backwardKernel, backwardSource, FP.Structural.exec,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    State.setReg, store, ofFloat, toFloat, Region.cast, ComputeDType.eraseDType,
    numeric, numericLt, natLt, bop, Option.bind, fold_regs, fold_pids]
  try rfl

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Execution
