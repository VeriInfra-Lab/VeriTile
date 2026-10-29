import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.StatementSourceFunction
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def mixing : SourceFunction := triton_function mix(a, b, LEVELS) constexpr [LEVELS] in triton {
  old = a
  a = b
  b = old + LEVELS
  return a, b, old
}
def swapping : SourceFunction := triton_function swap(a, b) in triton {
  return b, a
}
def signature : CallSignature where
  pointers := [("P",.i64)]
  functions := [mixing,swapping]

def source (P : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    old = tl.load(P + 2)
    __veritile_helper_0_arg_a = tl.load(P + 3)
    a, b, c = mix(tl.load(P), tl.load(P + 1), 7)
    a, b = swap(a, b)
  }

def discarded (P : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    a, _, _ = mix(tl.load(P), tl.load(P + 1), 7)
    observed = _
  }

/-- Locals and generated names cannot clobber the caller. Tuple outputs are
simultaneous even when they overwrite their own argument names. -/
theorem local_scope_and_parallel_results {α : Type} [Inhabited α] (M : Algebra α)
    (P : RegionName) (s : State α) :
    ∃ t,exec M (source P) s=some t ∧
      t.regs .int [] "a"=some (fun _ => Cell.read .int (s.mem P 0)+7) ∧
      t.regs .int [] "b"=some (fun _ => Cell.read .int (s.mem P 1)) ∧
      t.regs .int [] "c"=some (fun _ => Cell.read .int (s.mem P 0)) ∧
      t.regs .int [] "old"=some (fun _ => Cell.read .int (s.mem P 2)) ∧
      t.regs .int [] "__veritile_helper_0_arg_a"=some (fun _ => Cell.read .int (s.mem P 3)) ∧
      t.mem=s.mem := by
  simp [source,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,State.setReg,Option.bind,numeric,bop,Region.cast]
  funext i
  rfl

/-- Python underscore is an ordinary variable; the last tuple assignment is readable. -/
theorem ignored_results_retain_first {α : Type} [Inhabited α] (M : Algebra α)
    (P : RegionName) (s : State α) :
    ∃ t,exec M (discarded P) s=some t ∧
      t.regs .int [] "a"=some (fun _ => Cell.read .int (s.mem P 1)) ∧
      t.regs .int [] "observed"=some (fun _ => Cell.read .int (s.mem P 0)) ∧ t.mem=s.mem := by
  simp [discarded,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,State.setReg,Option.bind,numeric,bop,Region.cast]

/-- error: source helper result count differs from its assignment -/
#guard_msgs in
def wrongResults (P : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    a, b = mix(tl.load(P), tl.load(P + 1), 7)
  }

/-- error: helper constexpr arguments require a known constant -/
#guard_msgs in
def runtimeConstant (P : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    n = tl.load(P)
    a, b, c = mix(tl.load(P), tl.load(P + 1), n)
  }

def captured : SourceFunction := triton_function bad(a) in triton {
  temporary = a + outer
  return temporary, a
}
/-- error: unbound name in source helper: outer -/
#guard_msgs in
def noCapture (P : RegionName) (outer : Int) : ComputeKernel :=
  with_triton_signature ({ signature with functions := [captured] } : CallSignature) in triton {
    x = tl.load(P)
    a, b = bad(x)
  }

#axiomsClean local_scope_and_parallel_results
#axiomsClean ignored_results_retain_first
end VeriTile.Tests.StatementSourceFunction
