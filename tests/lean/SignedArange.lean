import VeriTile.Triton.DSL.SignedArange
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Tests.SignedArange
open VeriTile.Triton FP.Structural
noncomputable section

def profile : CallSignature where
  constexpr := ["N"]

def implicitLanes (N : Nat) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    lane = tl.arange(0, N)
    difference = lane[:, None] - lane[None, :]
  }

def explicitLanes (N : Nat) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    lane = tl.arange(0, N).to(tl.int32)
    difference = lane[:, None] - lane[None, :]
  }

/-- Implicit source typing agrees with the explicit source cast and retains
all statements and the signed subtraction operation. -/
theorem implicit_eq_explicit (N : Nat) : implicitLanes N=explicitLanes N := rfl

theorem difference_is_signed {α : Type} [Inhabited α] (M : Algebra α)
    (N : Nat) (bound : N≤2^31) (s : State α) :
    ∃ t, run M (implicitLanes N).surfaceBody s=some t ∧
      t.regs .int [N,N] "difference"=some (fun i => (i.1.val : Int)-(i.2.1.val : Int)) := by
  have lanes : ∀ (i : Fin N), (i.val : Int)%4294967296=(i.val : Int) := by
    intro i
    apply Int.emod_eq_of_lt <;> omega
  have narrow : ∀ (i : Fin N), i.val%4294967296=i.val := by
    intro i
    apply Nat.mod_eq_of_lt
    omega
  refine ⟨(s.setReg "lane" .int [N] (fun i => (i.1.val : Int))).setReg
    "difference" .int [N,N] (fun i => (i.1.val : Int)-(i.2.1.val : Int)),?_,?_⟩
  · simp [implicitLanes,ComputeKernel.surfaceBody,run,step,evalExpr,evalOp_unfold,
      Option.bind,numeric,bop,State.setReg,integralMod,natLt,lanes,narrow,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,IntegerExpr.sameShape]
    have small : ∀ i : Fin N, 2*i.val<4294967296 := by intro i; omega
    simp [small]
    rfl
  · simp [State.setReg]


#axiomsClean implicit_eq_explicit
#axiomsClean difference_is_signed
end
end VeriTile.Tests.SignedArange
