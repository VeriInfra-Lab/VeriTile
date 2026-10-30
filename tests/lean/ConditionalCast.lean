import VeriTile.Triton.DSL
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit

/-! A constexpr cast must not change the false path's register carrier or
precision. These checks execute the quoted source under arbitrary algebras. -/
namespace VeriTile.Tests.ConditionalCast
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def initial {α : Type} [Inhabited α] (dtype : FloatDType) (value : α) : State α where
  mem := fun _ _ => .mk dtype.toTileDType (ofFloat dtype value)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun dtype _ _ => defaultValue dtype

def oneArm (X : RegionName) (promote : Bool) : ComputeKernel := triton {
  tl.region X = tl.bfloat16
  o = tl.load(X)
  if promote { o = o.to(tl.float32) }
  y = o
}

def twoArms (X : RegionName) (promote : Bool) : ComputeKernel := triton {
  tl.region X = tl.bfloat16
  o = tl.load(X)
  if promote { o = o.to(tl.float32) } else { o = o.to(tl.float16) }
  y = o
}

def nested (X : RegionName) (outer inner : Bool) : ComputeKernel := triton {
  tl.region X = tl.bfloat16
  o = tl.load(X)
  if outer {
    if inner { o = o.to(tl.float32) }
  }
  y = o
}

def fresh (X : RegionName) (promote : Bool) : ComputeKernel := triton {
  tl.region X = tl.bfloat16
  if promote { o = tl.load(X).to(tl.float32) } else { o = tl.load(X) }
  y = o
}

def wide (X : RegionName) (promote : Bool) : ComputeKernel := triton {
  tl.region X = tl.float64
  o = tl.load(X)
  if promote { o = o.to(tl.float32) }
  y = o + o
}

macro "cast_run_simp" : tactic => `(tactic|
  simp [oneArm, twoArms, nested, fresh, wide, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, initial,
    State.setReg, Cell.read, ofFloat, toFloat, Region.cast, engine,
    Algebra.withDefaultPrecision, resolvePrecision, ComputeDType.eraseDType,
    Option.bind, FP.Structural.numeric, FP.Structural.bop])

theorem false_retains_bf16 {α : Type} [Inhabited α] (M : Algebra α) (value : α) :
    ∃ t, run (engine M) (oneArm "X" Bool.false).surfaceBody
      (initial .bf16 value) = some t ∧
      t.regs .bf16 [] "y" = some (fun _ => value) := by
  cast_run_simp

theorem true_casts_fp32 {α : Type} [Inhabited α] (M : Algebra α) (value : α) :
    ∃ t, run (engine M) (oneArm "X" Bool.true).surfaceBody
      (initial .bf16 value) = some t ∧
      t.regs .real [] "y" = some (fun _ => M.cast (some .fp32) .bf16 .real value) := by
  cast_run_simp

theorem two_arms_succeed {α : Type} [Inhabited α] (M : Algebra α) (value : α)
    (promote : Bool) :
    ∃ t, run (engine M) (twoArms "X" promote).surfaceBody (initial .bf16 value) = some t := by
  cases promote <;> cast_run_simp

theorem nested_succeeds {α : Type} [Inhabited α] (M : Algebra α) (value : α)
    (outer inner : Bool) :
    ∃ t, run (engine M) (nested "X" outer inner).surfaceBody (initial .bf16 value) = some t := by
  cases outer <;> cases inner <;> cast_run_simp

theorem newly_bound_succeeds {α : Type} [Inhabited α] (M : Algebra α) (value : α)
    (promote : Bool) :
    ∃ t, run (engine M) (fresh "X" promote).surfaceBody (initial .bf16 value) = some t := by
  cases promote <;> cast_run_simp

theorem false_keeps_fp64_arithmetic {α : Type} [Inhabited α] (M : Algebra α) (value : α) :
    ∃ t, run (engine M) (wide "X" Bool.false).surfaceBody (initial .real value) = some t ∧
      t.regs .real [] "y" = some (fun _ => M.binary (some .fp64) .real .add value value) := by
  cast_run_simp
  funext i
  rfl

theorem true_uses_fp32_arithmetic {α : Type} [Inhabited α] (M : Algebra α) (value : α) :
    ∃ t, run (engine M) (wide "X" Bool.true).surfaceBody (initial .real value) = some t ∧
      t.regs .real [] "y" = some (fun _ =>
        M.binary (some .fp32) .real .add
          (M.cast (some .fp32) .real .real value)
          (M.cast (some .fp32) .real .real value)) := by
  cast_run_simp
  funext i
  rfl

#axiomsClean false_retains_bf16
#axiomsClean true_casts_fp32
#axiomsClean two_arms_succeed
#axiomsClean nested_succeeds
#axiomsClean newly_bound_succeeds
#axiomsClean false_keeps_fp64_arithmetic
#axiomsClean true_uses_fp32_arithmetic
end VeriTile.Tests.ConditionalCast
