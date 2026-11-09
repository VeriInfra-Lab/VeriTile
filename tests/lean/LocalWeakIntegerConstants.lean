import VeriTile.Triton.DSL.IntegralJitFunctions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.NormNum
open VeriTile.Triton FP.Structural
namespace VeriTile.Tests.LocalWeakIntegerConstants
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 1000000
def helper : StatementJitFunction :=
  triton_function (x) constexpr_args [] {
    DIM: tl.constexpr = 128
    y = x.to(tl.float32) / DIM
    return y, y
  }
def abi : CallSignature where
  scalars := [("value",.fp32)]
def caller : ComputeKernel :=
  with_integral_operands in
  with_triton_call abi call_args [] jit_functions [helper := helper] in triton {
    left, right = helper(x=value)
  }
theorem local_literal {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (x : α)
    (hx : s.regs .real [] "value"=some (fun _=>x)) :
    ∃ t, exec M caller s=some t ∧ t.regs .real [] "left"=some (fun _=>
      M.binary (some .fp32) .real .div (M.cast (some .fp32) .real .real x)
        (M.literal (some .fp32) .real 128)) := by
  have constant_value : (128.0:ℝ)=128 := by norm_num
  simp [caller,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,ComputeDType.eraseDType,constant_value,hx,numeric,FP.Structural.bop,Option.bind,State.setReg,ofFloat,toFloat]
  rfl

def runtimeHelper : StatementJitFunction :=
  triton_function (x, D) constexpr_args [] {
    y = x.to(tl.float32) / D
    return y, y
  }
def runtimeABI : CallSignature where
  scalars := [("value",.fp32),("D",.integer)]
def runtimeCaller (D : Int) : ComputeKernel :=
  with_integral_operands in
  with_triton_call runtimeABI call_args [D:=D] jit_functions [helper := runtimeHelper] in triton {
    left, right = helper(x=value, D=D)
  }
theorem runtime_conversion {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (x : α) (D : Int) (hx : s.regs .real [] "value"=some (fun _=>x)) :
    ∃ t, exec M (runtimeCaller D) s=some t ∧ t.regs .real [] "left"=some (fun _=>
      M.binary (some .fp32) .real .div (M.cast (some .fp32) .real .real x)
        (M.fromInt (some .fp32) D)) := by
  simp [runtimeCaller,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,ComputeDType.eraseDType,hx,numeric,FP.Structural.bop,Option.bind,
    State.setReg,ofFloat,toFloat]
  rfl

theorem reject_mutable_constexpr : True := by
  fail_if_success
    let invalid : ComputeKernel :=
      let mutableHelper : StatementJitFunction :=
        triton_function (x) constexpr_args [] {
          DIM: tl.constexpr = 128
          DIM = 256
          y = x.to(tl.float32) / DIM
          return y, y
        }
      with_integral_operands in
      with_triton_call abi call_args [] jit_functions [helper := mutableHelper] in triton {
        left, right = helper(x=value)
      }
  trivial

#axiomsClean local_literal
#axiomsClean runtime_conversion
end
end VeriTile.Tests.LocalWeakIntegerConstants
