import bench.tritonbench_g.softmax_flaggems.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Int.Init
import Mathlib.Tactic.Common

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Helpers
open VeriTile Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

private def signature : CallSignature where
  scalars := [("a",.i64),("b",.i64)]

/-- Invocation harness for the retained, uncalled source helper. Its body is
the canonical declaration in Kernels; this harness supplies its integer ABI. -/
private def nextInvocation (a b : Int) : ComputeKernel :=
  with_triton_call signature call_args [a := a,b := b]
    jit_functions [next_multiple_of := next_multiple_ofSource] in triton {
      answer = next_multiple_of(a,b)
    }

/-- The complete source helper returns the ceiling multiple, including a=0,
and only updates registers. No floating callback law is required. -/
theorem next_multiple {α : Type} [Inhabited α] (M : Algebra α)
    (a b : Nat) (hb : 0<b) (s : State α) :
    ∃ t, exec M (nextInvocation (a : Int) (b : Int)) s=some t ∧
      t.regs .int [] "answer"=some (fun _ => (((a+b-1)/b*b : Nat) : Int)) ∧ t.mem=s.mem := by
  have hn : ¬(a : Int)+((b : Int)-1)<0 := by omega
  have hp : ((a : Int)+((b : Int)-1))/(b : Int)*(b : Int) = (((a+b-1)/b*b : Nat) : Int) := by
    have he : (a : Int)+((b : Int)-1) = ((a+b-1 : Nat) : Int) := by omega
    rw [he,←Int.natCast_ediv,←Int.natCast_mul]
  simp [nextInvocation,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,numericLt,natLt,integralFloorDiv,State.setReg,
    ComputeDType.eraseDType,hn]
  funext i
  cases i
  simpa [FP.Structural.bop,IntegerExpr.sameShape,Broadcast.leftIndex,Broadcast.rightIndex,hn] using hp

#axiomsClean next_multiple
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Helpers
