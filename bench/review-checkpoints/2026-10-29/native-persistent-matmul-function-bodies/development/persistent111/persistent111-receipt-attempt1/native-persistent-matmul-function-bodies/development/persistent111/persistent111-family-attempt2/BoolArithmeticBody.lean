import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
namespace VeriTile.Tests.BoolArithmeticBody
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  scalars := [("n",.i64)]

def source (n : Int) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in triton {
    flag = n != 0
    right = n + flag
    left = flag + n
    delta = n - flag
    product = flag * n
    inline = n + (n != 0)
    tuple = right + left + delta + product + inline
  }

/-- Both operand positions and a computed predicate contribute 0/1. The
negative and zero cases are retained; no floating arithmetic law is used. -/
theorem arithmetic {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (n : Int) :
    (exec M (source n) s).bind (fun t => t.regs .int [] "tuple") =
      some (fun _ => if n = 0 then 0 else 5*n+2) := by
  by_cases h : n = 0 <;>
    simp [exec,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,State.setReg,numeric,numericEq,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,Op.boolToInt,h]
  all_goals first
  | rfl
  | congr 1
    funext i
    simp only [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    omega

#axiomsClean arithmetic
end VeriTile.Tests.BoolArithmeticBody
