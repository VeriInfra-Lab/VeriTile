import VeriTile.Triton.DSL
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit

/-! A diagnostic for the source's conditional `o.to(tl.float32)`.
This isolates the frontend branch-typing question; it is not a substitute
kernel or a claimed backward correctness theorem.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.DTypeProbe
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
set_option maxHeartbeats 1600000
set_option maxRecDepth 8000

def conditionalCast (X : RegionName) (promote : Bool) : ComputeKernel := triton {
  tl.region X = tl.bfloat16
  o = tl.load(X)
  if promote {
    o = o.to(tl.float32)
  }
  y = o
}

def initial {α : Type} [Inhabited α] (value : α) : State α where
  mem := fun _ _ => .mk .bf16 (ofFloat .bf16 value)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => defaultValue dtype

/-- The false branch retains the bf16 register after constexpr continuation
expansion. The pre-fix failure witness is frozen in the review checkpoint. -/
theorem false_branch_runs {α : Type} [Inhabited α]
    (M : Algebra α) (value : α) :
    ∃ t, run (engine M) (conditionalCast "X" Bool.false).surfaceBody (initial value) = some t := by
  simp [conditionalCast, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, initial,
    State.setReg, Cell.read, ofFloat, toFloat, Region.cast, engine,
    Algebra.withDefaultPrecision, resolvePrecision, ComputeDType.eraseDType, Option.bind]

/-- The same typed source succeeds on the branch that executes the cast. -/
theorem true_branch_runs {α : Type} [Inhabited α] (M : Algebra α) (value : α) :
    ∃ t, run (engine M) (conditionalCast "X" Bool.true).surfaceBody (initial value) = some t := by
  simp [conditionalCast, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, initial,
    State.setReg, Cell.read, ofFloat, toFloat, Region.cast, engine,
    Algebra.withDefaultPrecision, resolvePrecision, ComputeDType.eraseDType, Option.bind]

#axiomsClean false_branch_runs
#axiomsClean true_branch_runs
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.DTypeProbe
