import VeriTile.Triton.DSL
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit

/-! Correlated constexpr branches must use only the reachable dtype/shape;
straight-line temporary type changes do not change the branch exit type.
Execution checks range over arbitrary floating algebras and initial states. -/
namespace VeriTile.Tests.StaticConditions
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

def repeated (N : Region .nat) (S : RegionName) (flag : Bool) : ComputeKernel := triton {
  if flag {
    x = tl.load(N)
    scale = tl.load(S)
    x = x * scale
  }
  if not flag {
    x = tl.load(N)
    scale = tl.load(S)
    x = x * scale
  }
  y = x + x
}

def repeatedLoop (N : Region .nat) (S : RegionName) (flag : Bool) : ComputeKernel := triton {
  if flag {
    x = tl.load(N)
    scale = tl.load(S)
    x = x * scale
  }
  tl.for i in 1 {
    if not flag {
      ptr = N + $(0)
      x = tl.load(ptr)
      scale = tl.load(S)
      x = x * scale
    }
    y = x + x
  }
}

def correlatedShape (flag : Bool) : ComputeKernel := triton {
  if flag { x = tl.zeros([], tl.float32) }
  else { x = tl.zeros([1], tl.float32) }
  if flag { y = x }
  else { y = x[:, None] }
}

-- A local register with the same spelling shadows the external parameter.
def shadowed (flag : Bool) : ComputeKernel := triton {
  if flag {
    flag = $((Bool.false : Bool))
    if flag { x = tl.zeros([1], tl.float32) }
    else { x = tl.full([1], 1.0, dtype=tl.float32) }
  }
}

def stringRepeated (kind : String) : ComputeKernel := triton {
  if kind == "wide" { x = tl.zeros([], tl.float32) }
  else { x = tl.zeros([1], tl.float32) }
  if kind != "wide" { y = x[:, None] }
  else { y = x }
}

-- Address inference must see loads nested beneath a result cast.
def readOffsets (X : RegionName) (I : Region .nat) (offset : Nat)
    (scalar : Bool) : ComputeKernel := triton {
  pid = tl.program_id(0)
  if scalar { offs = pid + $(offset) }
  else { offs = pid + tl.load(I) }
  p = X + offs
  x = tl.load(p).to(tl.float32)
}

macro "static_run_simp" : tactic => `(tactic|
  simp [repeated, repeatedLoop, correlatedShape, shadowed, stringRepeated, readOffsets,
    ComputeKernel.surfaceBody, run, step, FP.Structural.evalExpr, evalComputeOp,
    evalOp_unfold, State.setReg, Cell.read, ofFloat, toFloat, Region.cast,
    engine, Algebra.withDefaultPrecision, resolvePrecision,
    ComputeDType.eraseDType, Option.bind, FP.Structural.numeric,
    FP.Structural.bop, loop])

theorem repeated_exec {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (flag : Bool) :
    ∃ t, run (engine M) (repeated "N" "S" flag).surfaceBody s = some t := by
  cases flag <;> static_run_simp

theorem repeatedLoop_exec {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (flag : Bool) :
    ∃ t, run (engine M) (repeatedLoop "N" "S" flag).surfaceBody s = some t := by
  cases flag <;> static_run_simp

theorem correlatedShape_exec {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (flag : Bool) :
    ∃ t, run (engine M) (correlatedShape flag).surfaceBody s = some t := by
  cases flag <;> static_run_simp

theorem shadowed_false_arm {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) :
    ∃ t, run (engine M) (shadowed Bool.true).surfaceBody s = some t ∧
      t.regs .real [1] "x" = some (fun _ => M.literal (some .fp32) .real 1.0) := by
  static_run_simp

theorem stringRepeated_exec {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (kind : String) :
    ∃ t, run (engine M) (stringRepeated kind).surfaceBody s = some t := by
  by_cases h : kind = "wide" <;> static_run_simp <;> simp_all

theorem readOffsets_exec {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (flag : Bool) (offset : Nat) :
    ∃ t, run (engine M) (readOffsets "X" "I" offset flag).surfaceBody s = some t := by
  cases flag <;> static_run_simp

#axiomsClean readOffsets_exec
#axiomsClean repeated_exec
#axiomsClean repeatedLoop_exec
#axiomsClean correlatedShape_exec
#axiomsClean shadowed_false_arm
#axiomsClean stringRepeated_exec
end VeriTile.Tests.StaticConditions
