import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PythonIntegerWhere
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  constexpr := ["PICK"]

def selected (PICK : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    value = tl.where(PICK, -3, 7)
  }

def loop (PICK : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    count = tl.where(PICK, 1, 0)
    for i in range(0, count, 1):
        result = count
  }

theorem selected_integer {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (pick : Bool) :
    ∃ t,exec M (selected pick) s=some t ∧
      t.regs .int [] "value"=some (fun _ => if pick then (-3 : Int) else 7) := by
  cases pick <;> simp [selected,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,State.setReg,Option.bind,numeric,bop]

theorem zero_iterations {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t,exec M (loop Bool.false) s=some t ∧
      t.regs .int [] "result"=s.regs .int [] "result" := by
  simp [loop,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,State.setReg,Option.bind,numeric,bop]
  rw [rangeInt]
  simp

theorem one_iteration {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t,exec M (loop Bool.true) s=some t ∧
      t.regs .int [] "result"=some (fun _ => (1 : Int)) := by
  simp [loop,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,State.setReg,Option.bind,numeric,bop]
  rw [rangeInt]
  simp [run,step,evalExpr,evalOp_unfold,State.setReg,Option.bind]
  rw [rangeInt]
  simp

/-- error: tl.where Python integer literal pairs currently require signed32 values -/
#guard_msgs in
def unsignedPair (PICK : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    value = tl.where(PICK, 2147483648, 0)
  }

#axiomsClean selected_integer
#axiomsClean zero_iterations
#axiomsClean one_iteration
end VeriTile.Tests.PythonIntegerWhere
