import VeriTile.Triton.DSL.IntegralJitFunctions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.IntegralJitFunctions
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 1000000

def duplicate : StatementJitFunction :=
  triton_function (value) constexpr_args [] {
    return value, value
  }

def abi : CallSignature where
  scalars := [("N",.integer)]
  constexpr := ["D"]

def copiedCaller (N : Int) : ComputeKernel :=
  let D : Nat := 64
  with_integral_operands in
  with_triton_call abi call_args [N := N] jit_functions [duplicate := duplicate] in triton {
    N = tl.cast(N, tl.int64)
    value = N // D + (N % D != 0)
    left, right = duplicate(value=value)
  }

/-- A true Boolean contributes integer one before entering a real source
helper; both tuple results remain separate runtime registers. -/
theorem ceil65 {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (copiedCaller 65) s=some t ∧
      t.regs .int [] "left"=some (fun _=>2) ∧
      t.regs .int [] "right"=some (fun _=>2) ∧ t.mem=s.mem := by
  simp [copiedCaller,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,numeric,numericEq,numericLt,FP.Structural.bop,
    integralFloorDiv,integralMod,Option.bind,State.setReg]
  rfl

/-- Runtime signed division truncates toward zero, with signed remainder:
https://triton-lang.org/main/python-api/triton-semantics.html#differences-with-numpy
Thus -1//64 + (-1%64 != 0) is 1 for a runtime Triton tensor. -/
theorem ceil_negative {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (copiedCaller (-1)) s=some t ∧
      t.regs .int [] "left"=some (fun _=>1) ∧ t.mem=s.mem := by
  simp [copiedCaller,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,numeric,numericEq,numericLt,FP.Structural.bop,
    integralFloorDiv,integralMod,Option.bind,State.setReg]
  rfl

theorem reject_non_invocation : True := by
  fail_if_success
    let invalid : Nat := with_integral_operands in (5 : Nat)
  trivial

#axiomsClean ceil65
#axiomsClean ceil_negative
end
end VeriTile.Tests.IntegralJitFunctions
