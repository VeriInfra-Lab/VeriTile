import bench.repairs.tritonbench_g.diag_ssm_triton.Kernels
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourcePair
open VeriTile Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 3000000

inductive Family where
  | forward | backward | complexForward | complexBackward

/-- Proof-only argument grouping; the authoritative sources remain the eight
complete quotations in Kernels.lean. -/
structure Arguments where
  s : RegionName
  x : RegionName
  transition : RegionName
  y : RegionName
  gradS : RegionName
  gradX : RegionName
  gradTransition : RegionName
  gradY : RegionName
  length : Nat
  batch : Nat
  dim : Nat
  block : Nat

def source (family : Family) (candidate : Bool) (p : Arguments) : ComputeKernel :=
  match family, candidate with
  | .forward, Bool.false => Kernels.forwardTyped p.s p.x p.transition p.y p.length p.batch p.dim p.block
  | .forward, Bool.true => Kernels.forwardIndexCommuted p.s p.x p.transition p.y p.length p.batch p.dim p.block
  | .backward, Bool.false => Kernels.backwardTyped p.s p.transition p.y p.gradS p.gradX p.gradTransition p.gradY p.length p.batch p.dim p.block
  | .backward, Bool.true => Kernels.backwardIndexCommuted p.s p.transition p.y p.gradS p.gradX p.gradTransition p.gradY p.length p.batch p.dim p.block
  | .complexForward, Bool.false => Kernels.complexForwardTyped p.s p.x p.y p.transition p.length p.batch p.dim p.block
  | .complexForward, Bool.true => Kernels.complexForwardIndexCommuted p.s p.x p.y p.transition p.length p.batch p.dim p.block
  | .complexBackward, Bool.false => Kernels.complexBackwardTyped p.s p.transition p.y p.gradS p.gradX p.gradTransition p.gradY p.length p.batch p.dim p.block
  | .complexBackward, Bool.true => Kernels.complexBackwardIndexCommuted p.s p.transition p.y p.gradS p.gradX p.gradTransition p.gradY p.length p.batch p.dim p.block

def index (candidate : Bool) (block : Nat) : ComputeStmt :=
  let pid : Op .int [] := .castNatToInt (.programId 0)
  let width : Op .int [] := .constInt (block : Int)
  .assign .int [] "col_idx" (.alg
    (if candidate then .mul .int .nil width pid else .mul .int .nil pid width))

/-- All three original widening assignments, every loop and every store are
retained on both sides. Only the existing integer product is different. -/
theorem decomposition (family : Family) (candidate : Bool) (p : Arguments) :
    (source family candidate p).surfaceBody =
      (source family Bool.false p).surfaceBody.take 3 ++
      index candidate p.block :: (source family Bool.false p).surfaceBody.drop 4 := by
  cases family <;> cases candidate <;> rfl

theorem index_step {α : Type} [Inhabited α] (M : Algebra α) (block : Nat) (s : State α) :
    step M (index Bool.false block) s = step M (index Bool.true block) s := by
  simp [index, step, evalExpr, evalOp_unfold, numeric, FP.Structural.bop, Int.mul_comm]
  congr 1
  funext i
  exact Int.mul_comm _ _

theorem execution {α : Type} [Inhabited α] (M : Algebra α)
    (family : Family) (p : Arguments) (s : State α) :
    exec M (source family Bool.false p) s = exec M (source family Bool.true p) s := by
  change run M (source family Bool.false p).surfaceBody s =
    run M (source family Bool.true p).surfaceBody s
  conv_lhs => rw [decomposition family Bool.false p]
  conv_rhs => rw [decomposition family Bool.true p]
  simp only [run_append, run, index_step]

theorem safety {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (family : Family) (p : Arguments) (s : State α) :
    TraceSafe M bounds (source family Bool.false p) s ↔
      TraceSafe M bounds (source family Bool.true p) s := by
  change Safety.traceSafeList M bounds (source family Bool.false p).surfaceBody s ↔
    Safety.traceSafeList M bounds (source family Bool.true p).surfaceBody s
  conv_lhs => rw [decomposition family Bool.false p]
  conv_rhs => rw [decomposition family Bool.true p]
  simp only [Safety.traceSafeList_append,
    Safety.traceSafeList_cons, index_step]
  simp [index, Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeOp_unfold]

#axiomsClean decomposition
#axiomsClean index_step
#axiomsClean execution
#axiomsClean safety
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourcePair
