import VeriTile.Triton.DSL.RecursiveJitFunctions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.RecursiveJitFunctions
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000

def previous : StatementJitFunction :=
  triton_function (a,b) constexpr_args [] {
    return tl.cdiv(a,b)*b-b
  }
def signature : CallSignature where
  scalars := [("N",.i64)]
  constexpr := ["B","FLAG"]

def nested (N : Int) (B : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [N := N]
    jit_functions [previous := previous] in triton {
      if FLAG:
          temporary = tl.full([B], value=0.0, dtype=tl.float32)
          result = previous(N,B)
      else:
          result = previous(N,B)
      saved = result
    }

/-- The helper executes inside either branch. Its signed negative result is
not clipped by natural subtraction; intrinsic keywords do not mutate B. -/
theorem negative_previous {α : Type} [Inhabited α] (M : Algebra α)
    (flag : Bool) (s : State α) :
    ∃ t, exec M (nested 0 8 flag) s=some t ∧
      t.regs .int [] "saved"=some (fun _ => -8) := by
  cases flag <;>
    simp [nested,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,numeric,numericLt,natLt,integralFloorDiv,bop,State.setReg,ComputeDType.eraseDType]
  all_goals funext i; cases i; rfl

def augmented (N : Int) (B : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    offsets = tl.arange(0,1).to(tl.int64)
    for i in range(0,N,1):
        offsets += B
    saved = offsets
  }

/-- A signed tensor plus a constexpr integer remains signed across the loop. -/
theorem promoted_update {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (augmented 1 8 Bool.false) s=some t ∧
      t.regs .int [1] "saved"=some (fun _ => 8) := by
  simp [augmented,exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,State.setReg,
    ComputeDType.eraseDType]
  rw [rangeInt]
  simp [run,step,evalExpr,evalOp_unfold,Option.bind,numeric,numericLt,natLt,State.setReg,bop]
  rw [rangeInt]
  simp [State.setReg]
  funext i
  simp [bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem reject_constexpr_write (N : Int) (B : Nat) (FLAG : Bool) : True := by
  fail_if_success
    let bad : ComputeKernel :=
      with_triton_call signature call_args [N := N]
        jit_functions [previous := previous] in triton {
          if FLAG:
              B = 4
              result = previous(N,B)
        }
  trivial

theorem reject_helper_escape (N : Int) (B : Nat) (FLAG : Bool) : True := by
  fail_if_success
    let bad : ComputeKernel :=
      with_triton_call signature call_args [N := N]
        jit_functions [previous := previous] in triton {
          if FLAG:
              result = previous(N,B) + 1
        }
  trivial

#axiomsClean negative_previous
#axiomsClean promoted_update
end
end VeriTile.Tests.RecursiveJitFunctions
