import VeriTile.Triton.DSL.SignedLaneSubtraction
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Tests.SignedLaneSubtraction
open VeriTile.Triton FP.Structural
noncomputable section

def profile : CallSignature where
  constexpr := ["N"]

def copied (N : Nat) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    lane = tl.arange(0, N)
    difference = lane[:, None] - lane[None, :]
  }

/-- Address lanes retain their natural carrier; their difference is signed,
including negative entries, for any numerical backend. -/
theorem difference_is_signed {α : Type} [Inhabited α] (M : Algebra α)
    (N : Nat) (bound : N≤2^31) (s : State α) :
    ∃ t, run M (copied N).surfaceBody s=some t ∧
      t.regs .nat [N] "lane"=some (fun i=>i.1.val) ∧
      t.regs .int [N,N] "difference"=some (fun i=>(i.1.val : Int)-(i.2.1.val : Int)) := by
  have lanes : ∀ (i : Fin N), (i.val : Int)%4294967296=(i.val : Int) := by
    intro i
    apply Int.emod_eq_of_lt <;> omega
  have narrow : ∀ (i : Fin N), i.val%4294967296=i.val := by
    intro i
    apply Nat.mod_eq_of_lt
    omega
  have small : ∀ i : Fin N, 2*i.val<4294967296 := by intro i; omega
  refine ⟨(s.setReg "lane" .nat [N] (fun i=>i.1.val)).setReg
    "difference" .int [N,N] (fun i=>(i.1.val : Int)-(i.2.1.val : Int)),?_,?_,?_⟩
  · simp [copied,ComputeKernel.surfaceBody,run,step,evalExpr,evalOp_unfold,
      Option.bind,numeric,bop,State.setReg,integralMod,natLt,lanes,narrow,small,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,IntegerExpr.sameShape]
    rfl
  · simp [State.setReg]
  · simp [State.setReg]

/-- An explicit floating conversion keeps subtraction on the floating path. -/
def floating (N : Nat) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    lane = tl.arange(0, N).to(tl.float32)
    difference = lane[:, None] - lane[None, :]
  }

theorem floating_source (N : Nat) : floating N=triton {
    lane = tl.arange(0, $(N)).to(tl.float32)
    difference = lane[:, None] - lane[None, :]
  } := rfl

#axiomsClean difference_is_signed
#axiomsClean floating_source
end
end VeriTile.Tests.SignedLaneSubtraction
