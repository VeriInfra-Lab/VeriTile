import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
namespace VeriTile.Tests.BoolInt64Body
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  scalars := [("n",.i64)]

def source (n : Int) : ComputeKernel :=
  with_triton_call signature call_args [n := n] in triton {
    flag = n != 0
    from_name = flag.to(tl.int64)
    from_expression = (n != 0).to(tl.int64)
    from_function = tl.cast(n != 0, tl.int64)
    result = from_name + from_expression + from_function
  }

/-- Boolean widening must produce signed 0/1 for all three source cast forms,
including a named comparison and an inline predicate used in arithmetic. -/
theorem widening {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (n : Int) :
    (exec M (source n) s).bind (fun t => t.regs .int [] "result") =
      some (fun _ => if n = 0 then 0 else 3) := by
  by_cases h : n = 0 <;>
    simp [exec,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,State.setReg,numeric,numericEq,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,Op.boolToInt,h]
  all_goals rfl

#axiomsClean widening
end VeriTile.Tests.BoolInt64Body
