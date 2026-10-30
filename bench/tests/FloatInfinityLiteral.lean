import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.FloatInfinityLiteral
open VeriTile.Triton FP.Structural
noncomputable section

def signature : CallSignature where
  pointers := []
  scalars := []
  constexpr := []
def positive : ComputeKernel :=
  with_triton_signature signature in
  triton {
    value = float('inf')
  }
def negativePositive : ComputeKernel :=
  with_triton_signature signature in
  triton {
    value = -float('-inf')
  }
def negative : ComputeKernel :=
  with_triton_signature signature in
  triton {
    value = float('-inf')
  }

/-- Unary minus of negative infinity is positive infinity, not a finite Real. -/
example : negativePositive = positive := rfl
example : negative.surfaceBody = [.assign .real [] "value" (.alg .negInf)] := rfl

/-- The actual infinity word reaches the backend for every interpretation. -/
theorem execution {A : Type} [Inhabited A] (M : Algebra A) (s : State A) :
    (FP.Structural.exec M positive s).map (fun t => t.regs .real [] "value") =
      some (some (fun _ => M.fp32Bits ⟨BitVec.ofNat 32 2139095040⟩)) := by
  simp [positive,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,State.setReg]

/-- error: float(...): only literal inf and -inf are supported -/
#guard_msgs (error, drop info) in
#check (with_triton_signature signature in triton {
  value = float('not_inf')
} : ComputeKernel)
#axiomsClean execution
end
end VeriTile.Tests.FloatInfinityLiteral
