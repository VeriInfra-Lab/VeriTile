import VeriTile.Triton.DSL.NonnegativeReadViews
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.NonnegativeReadViews
open VeriTile.Triton FP.Structural
noncomputable section

def profile : CallSignature where
  pointers := [("metadata",.i32)]
  scalars := []
  constexpr := []

def copied (metadata : Region .nat) : ComputeKernel :=
  with_triton_natural_reads [metadata] in
  with_triton_call profile call_args [] in triton {
    count = tl.load(metadata)
    next_count = count + 1
  }

/-- The mathematical metadata interpretation retains the exact input read,
addition and current memory, for all natural values and arithmetic backends.
This theorem does not assert a native signed32 representation for large Nat. -/
theorem reads_current_metadata {α : Type} [Inhabited α] (M : Algebra α)
    (metadata : Region .nat) (s : State α) :
    ∃ t, run M (copied metadata).surfaceBody s=some t ∧ t.mem=s.mem ∧
      t.regs .nat [] "next_count"=some (fun _=>(s.mem metadata.name 0).read .nat+1) := by
  refine ⟨?_,?_,?_,?_⟩
  rotate_left
  · simp [copied,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,Region.cast,State.setReg,numeric,bop]
    rfl
  · rfl
  · simp [State.setReg,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    rfl

/-- Native equality has its actual nonnegative signed32 domain. -/
theorem native_read (value : Int) (lo : 0≤value) (hi : value<2147483648) :
    (BitVec.ofInt 32 value).toInt=value ∧ (BitVec.ofInt 32 value).toNat=value.toNat :=
  DSL.NonnegativeReadViews.signed32_read value lo hi

example : (BitVec.ofInt 32 (-1)).toInt=(-1) ∧
    (BitVec.ofInt 32 (-1)).toNat=4294967295 := by decide

/-- A mathematical input view cannot authorize a store to signed metadata. -/
theorem readonly_rejected (metadata : Region .nat) : True := by
  fail_if_success
    let bad : ComputeKernel :=
      with_triton_natural_reads [metadata] in
      with_triton_call profile call_args [] in triton {
        tl.store(metadata, 0)
      }
  trivial

#axiomsClean reads_current_metadata
#axiomsClean native_read
#axiomsClean readonly_rejected
end
end VeriTile.Tests.NonnegativeReadViews
