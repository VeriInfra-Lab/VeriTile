import VeriTile.Triton.DSL.IntegerJitControl
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.IntegerJitControl
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

def divide : StatementJitFunction := triton_function (x,LEVELS) constexpr_args [LEVELS] {
  y = x / LEVELS
  return y, x, y
}
def signature : CallSignature where
  scalars := [("x",.fp32)]
  constexpr := ["POLICY"]

def selected8 : ComputeKernel :=
  let POLICY : Nat := 8
  with_triton_call signature call_args [] jit_functions [f := divide] in triton {
    if POLICY == 0:
      out = x
    elif POLICY == 8:
      out, _, _ = f(x, 255)
    else:
      out, _, _ = f(x, 15)
  }
/-- A literal helper constexpr remains a weak scalar: the complete execution
uses the FP32 literal callback and introduces no integer-conversion callback.
Ignored tuple destinations retain their original return evaluation order. -/
theorem literal_constexpr_and_ignored_returns {A : Type} [Inhabited A]
    (M : Algebra A) (s : State A) (x : Values A .real [])
    (hx : s.regs .real [] "x"=some x) :
    ∃ t, exec M selected8 s=some t ∧
      t.regs .real [] "out"=some (fun i=>M.binary (some .fp32) .real .div
        (x i) (M.literal (some .fp32) .real 255)) ∧
      t.regs .real [] "_"=some (fun i=>M.binary (some .fp32) .real .div
        (x i) (M.literal (some .fp32) .real 255)) := by
  simp [selected8,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,hx,Option.bind,State.setReg,ofFloat,toFloat,ComputeDType.eraseDType,bop]
  rfl

def selected4 : ComputeKernel :=
  let POLICY : Nat := 4
  with_triton_call signature call_args [] jit_functions [f := divide] in triton {
    if POLICY == 0:
      out = x
    elif POLICY == 8:
      out, _, _ = f(x, 255)
    else:
      out, _, _ = f(x, 15)
  }
theorem fallback_constexpr_branch {A : Type} [Inhabited A]
    (M : Algebra A) (s : State A) (x : Values A .real [])
    (hx : s.regs .real [] "x"=some x) :
    ∃ t, exec M selected4 s=some t ∧
      t.regs .real [] "out"=some (fun i=>M.binary (some .fp32) .real .div
        (x i) (M.literal (some .fp32) .real 15)) := by
  simp [selected4,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,hx,Option.bind,State.setReg,ofFloat,toFloat,ComputeDType.eraseDType,bop]
  rfl

def selected0 : ComputeKernel :=
  let POLICY : Nat := 0
  with_triton_call signature call_args [] jit_functions [f := divide] in triton {
    if POLICY == 0:
      out = x
    elif POLICY == 8:
      out, _, _ = f(x, 255)
    else:
      out, _, _ = f(x, 15)
    if x > 0.0:
      out = out + x
  }
def expected0 : ComputeKernel := triton {
  tl.param x : tl.float32
  out = x
  if x > 0.0 {
    out = out + x
  }
}
/-- The disabled helpers disappear only through actual constexpr selection;
the independent floating runtime comparison remains in the complete program. -/
theorem runtime_branch_retained : selected0=expected0 := rfl

example : True := by
  fail_if_success
    let bad : ComputeKernel :=
      let POLICY : Nat := 8
      with_triton_call signature call_args [] jit_functions [f := divide] in triton {
        POLICY = 0
        out, _, _ = f(x, 255)
      }
  trivial

example : True := by
  fail_if_success
    let bad : ComputeKernel :=
      let POLICY : Nat := 8
      with_triton_call signature call_args [] jit_functions [f := divide] in triton {
        if x > 0.0:
          out, _, _ = f(x, 255)
      }
  trivial

#axiomsClean literal_constexpr_and_ignored_returns
#axiomsClean fallback_constexpr_branch
#axiomsClean runtime_branch_retained
end
end VeriTile.Tests.IntegerJitControl
