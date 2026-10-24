import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.CallerOutput

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.HostProtocol
open CallerOutput
noncomputable section

inductive Stage where
  | validation | metadata | input | weights | oldOutput | launch | selection
  deriving DecidableEq

inductive Prefix (ε : Type) where
  | failed (stage : Stage) (error : ε)
  | empty
  | prepared

structure Outcome (ε : Type) where
  result : Except ε Unit
  view : View
  memory : Bytes

/-- All allocations and kernel writes before index_copy_ target fresh scratch.
Prefix failure leaves the caller bytes as on entry; a failed final copy has
only the ordinary destination-confinement guarantee. -/
def run {α ε : Type} (h : Call) (v : View) (encode : Bool → α → Fin v.width → BitVec 8)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (prewrite : Prefix ε) (failure : Option (CopyFailure ε h v indices before)) : Outcome ε :=
  match prewrite with
  | .failed _ error => ⟨.error error,v,before⟩
  | .empty => ⟨.ok (),v,before⟩
  | .prepared => match failure with
    | some error => ⟨.error error.error,v,error.after⟩
    | none => ⟨.ok (),v,copy h v encode indices scratch before⟩

theorem frame {α ε : Type} (h : Call) (v : View) (encode : Bool → α → Fin v.width → BitVec 8)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (prewrite : Prefix ε) (failure : Option (CopyFailure ε h v indices before))
    (owner ptr : Nat) (outside : ¬∃ i : Coord v,∃ byte : Fin v.width,
      Active h v indices i ∧ owner=v.original.owner ∧ ptr=address v i byte) :
    (run h v encode indices scratch before prewrite failure).memory owner ptr=before owner ptr := by
  cases prewrite with
  | failed => rfl
  | empty => rfl
  | prepared =>
    cases failure with
    | none => exact CallerOutput.frame h v encode indices scratch before owner ptr outside
    | some failure => exact failure.frame owner ptr outside

theorem identity {α ε : Type} (h : Call) (v : View) (encode : Bool → α → Fin v.width → BitVec 8)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (prewrite : Prefix ε) (failure : Option (CopyFailure ε h v indices before)) :
    (run h v encode indices scratch before prewrite failure).view=v := by
  cases prewrite <;> cases failure <;> rfl

#axiomsClean frame
#axiomsClean identity
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.HostProtocol
