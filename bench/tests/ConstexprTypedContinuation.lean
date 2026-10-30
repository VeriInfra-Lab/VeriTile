import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ConstexprTypedContinuation
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def initial {α : Type} [Inhabited α] (value : α) : State α where
  mem := fun _ _ => .mk .bf16 value
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => defaultValue dtype

/-- A caller Bool is substituted as a typed term, not a bare identifier.
The continuation must use each path's actual register carrier. -/
def oneArm (X : RegionName) (promote : Bool) : ComputeKernel := triton {
  tl.region X = tl.bfloat16
  o = tl.load(X)
  if $((promote : Bool)) { o = o.to(tl.float32) }
  y = o
}

def twoArms (X : RegionName) (promote : Bool) : ComputeKernel := triton {
  tl.region X = tl.bfloat16
  o = tl.load(X)
  if $((promote : Bool)) { o = o.to(tl.float32) } else { o = o.to(tl.float16) }
  y = o
}

def signature : CallSignature where
  pointers := [("X",.inputFloat)]
  constexpr := ["PROMOTE"]

def called (X : RegionName) (PROMOTE : Bool) : ComputeKernel :=
  let dtype : FloatDType := .bf16
  with_triton_call signature call_args [] in triton {
    o = tl.load(X)
    if PROMOTE:
        o = o.to(tl.float32)
    y = o
  }

theorem complete_call (X : RegionName) (flag : Bool) :
    called X flag = oneArm X flag := rfl

macro "continuation_simp" : tactic => `(tactic|
  simp [oneArm,twoArms,ComputeKernel.surfaceBody,run,step,
    FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,initial,
    State.setReg,Cell.read,ofFloat,toFloat,Region.cast,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    Option.bind,FP.Structural.numeric,FP.Structural.bop])

theorem false_retains_bf16 {α : Type} [Inhabited α] (M : Algebra α) (value : α) :
    ∃ t,run (engine M) (oneArm "X" Bool.false).surfaceBody (initial value)=some t ∧
      t.regs .bf16 [] "y"=some (fun _ => value) := by continuation_simp

theorem true_casts_fp32 {α : Type} [Inhabited α] (M : Algebra α) (value : α) :
    ∃ t,run (engine M) (oneArm "X" Bool.true).surfaceBody (initial value)=some t ∧
      t.regs .real [] "y"=some (fun _ => M.cast (some .fp32) .bf16 .real value) := by continuation_simp

theorem both_arms_succeed {α : Type} [Inhabited α] (M : Algebra α) (value : α) (flag : Bool) :
    ∃ t,run (engine M) (twoArms "X" flag).surfaceBody (initial value)=some t := by
  cases flag <;> continuation_simp

#axiomsClean complete_call
#axiomsClean false_retains_bf16
#axiomsClean true_casts_fp32
#axiomsClean both_arms_succeed
end VeriTile.Tests.ConstexprTypedContinuation
