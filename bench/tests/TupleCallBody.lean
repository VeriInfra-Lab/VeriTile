import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
namespace VeriTile.Tests.TupleCallBody
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  scalars := [("a",.i64),("b",.i64),("c",.i64)]

def rotate (a b c : Int) : ComputeKernel :=
  with_triton_call signature call_args [a := a,b := b,c := c] in triton {
    a, b, c = tl.cast(b, tl.int64), tl.cast(c, tl.int64), tl.cast(a, tl.int64)
    value = a - b + c
  }

/-- Every tuple RHS reads the original call, including a later RHS that names
an earlier target. Signed values and the following local reads are preserved. -/
theorem rotate_executes {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (a b c : Int) :
    (exec M (rotate a b c) s).bind (fun t => t.regs .int [] "value") =
      some (fun _ => b - c + a) := by
  simp [exec,rotate,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,State.setReg,numeric,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

/-- error: caller tuple assignment requires distinct targets and matching arity -/
#guard_msgs in
example (a b c : Int) : ComputeKernel :=
  with_triton_call signature call_args [a := a,b := b,c := c] in triton {
    a, a = b, c
  }
#axiomsClean rotate_executes
end VeriTile.Tests.TupleCallBody
