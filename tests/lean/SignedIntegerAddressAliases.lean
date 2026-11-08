import tests.lean.SignedIntegerAddressAliasReference
import VeriTile.Triton.DSL.ScalarInt64While
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedIntegerAddressAliases
open VeriTile.Triton FP.Structural
open SignedIntegerAddressAliasReference (signature)
noncomputable section

def scalar (X Y : RegionName) (base B : Nat) : ComputeKernel :=
  with_triton_signature signature pointer_offsets [X := base] in triton {
    offsets = start + stride
    value = tl.load(X + offsets)
    tl.store(Y, value)
  }

def vector (X Y : RegionName) (base B : Nat) : ComputeKernel :=
  with_triton_signature signature pointer_offsets [X := base] in triton {
    columns = tl.arange(0, B).to(tl.int64)
    mask = columns < N
    offsets = start + stride * tl.where(mask, columns, 0)
    values = tl.load(X + offsets, mask=mask, other=0)
    tl.store(Y + columns, values, mask=mask)
  }

theorem scalar_complete : scalar=SignedIntegerAddressAliasReference.scalar := by rfl
theorem vector_complete : vector=SignedIntegerAddressAliasReference.vector := by rfl

/-- A negative signed displacement is added to the caller base before the
complete memory coordinate is projected. Regions may alias arbitrarily. -/
theorem scalar_address {A : Type} [Inhabited A] (M : Algebra A)
    (X Y : RegionName) (base B : Nat) (start stride : Int) (s : State A)
    (hs : s.regs .int [] "start"=some (fun _=>start))
    (ht : s.regs .int [] "stride"=some (fun _=>stride)) :
    ∃ t, exec M (scalar X Y base B) s=some t ∧
      t.mem Y 0=.mk .real (M.fp32Load ((s.mem X ((base : Int)+(start+stride)).toNat).read .real)) := by
  simp [scalar,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,store,Option.bind,numeric,bop,Region.cast,
    ComputeDType.eraseDType,State.setReg,State.write,hs,ht]

/-- error: nat expression identifier `start`: dtype mismatch -/
#guard_msgs (error) in
example (X Y : RegionName) (base B : Nat) : ComputeKernel :=
  with_triton_signature signature pointer_offsets [X := base] in triton {
    offsets = start + tl.load(X)
    value = tl.load(X + offsets)
    tl.store(Y, value)
  }

#axiomsClean scalar_complete
#axiomsClean vector_complete
#axiomsClean scalar_address
end
end VeriTile.Tests.SignedIntegerAddressAliases
