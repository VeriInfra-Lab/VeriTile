import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.ReturnedStorage

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.HostProtocol
open ReturnedStorage
noncomputable section

/-- Source-order phases before the sole caller-writing index_copy_. A failure
in any phase raises its original error. All destinations so far are fresh
scratch; launch confinement is proved by Preparation.caller_frame. -/
inductive Stage where
  | validation | metadata | input | weights | oldOutput | launch | selection
  deriving DecidableEq

inductive Prefix (ε : Type) where
  | failed (stage : Stage) (error : ε)
  | empty
  | prepared

/-- Ordinary local tensor-copy failure contract: even a partially completed
copy touches only its valid destination coordinates. No failure atomicity or
correct numerical result is assumed on this branch. -/
structure CopyFailure (ε : Type) (h : Call) (view : View) (indices : Nat → Int) (before : Bytes) where
  error : ε
  after : Bytes
  frame : ∀ owner address,(¬∃ i : Coord view,∃ byte : Fin 4,
    Active h view indices i ∧ owner=view.owner ∧ address=view.address i byte) →
    after owner address=before owner address

structure Outcome (ε : Type) where
  result : Except ε Unit
  outputObject : Nat
  outputOwner : Nat
  memory : Bytes

/-- Wrapper outcome adapter for ordinary local API outcomes. `scratch` is the
actual successful source execution's physical readback, supplied by the public
FP proof. This definition adds no kernel success or output-value premise. -/
def run {α ε : Type} (encode : Encode α) (h : Call) (view : View)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (prewrite : Prefix ε) (copyFailure : Option (CopyFailure ε h view indices before)) : Outcome ε :=
  match prewrite with
  | .failed _ error => ⟨.error error,view.object,view.owner,before⟩
  | .empty => ⟨.ok (),view.object,view.owner,before⟩
  | .prepared => match copyFailure with
    | some failure => ⟨.error failure.error,view.object,view.owner,failure.after⟩
    | none => ⟨.ok (),view.object,view.owner,copy encode h view indices scratch before⟩

theorem prewrite_failure {α ε : Type} (encode : Encode α) (h : Call) (view : View)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (stage : Stage) (error : ε) (copyFailure : Option (CopyFailure ε h view indices before)) :
    (run encode h view indices scratch before (.failed stage error) copyFailure).result=.error error ∧
    (run encode h view indices scratch before (.failed stage error) copyFailure).memory=before := ⟨rfl,rfl⟩

theorem empty_return {α ε : Type} (encode : Encode α) (h : Call) (view : View)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (copyFailure : Option (CopyFailure ε h view indices before)) :
    (run encode h view indices scratch before .empty copyFailure).result=.ok () ∧
    (run encode h view indices scratch before .empty copyFailure).memory=before := ⟨rfl,rfl⟩

theorem success {α ε : Type} (encode : Encode α) (h : Call) (view : View)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes) :
    (run (ε:=ε) encode h view indices scratch before .prepared none).result=.ok () ∧
    (run (ε:=ε) encode h view indices scratch before .prepared none).memory=
      (returned encode h view indices scratch before).memory := ⟨rfl,rfl⟩

theorem frame {α ε : Type} (encode : Encode α) (h : Call) (view : View)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (prewrite : Prefix ε) (copyFailure : Option (CopyFailure ε h view indices before))
    (owner address : Nat) (outside : ¬∃ i : Coord view,∃ byte : Fin 4,
      Active h view indices i ∧ owner=view.owner ∧ address=view.address i byte) :
    (run encode h view indices scratch before prewrite copyFailure).memory owner address=before owner address := by
  cases prewrite with
  | failed => rfl
  | empty => rfl
  | prepared =>
    cases copyFailure with
    | none => exact copy_frame encode h view indices scratch before owner address outside
    | some failure => exact failure.frame owner address outside

theorem identity {α ε : Type} (encode : Encode α) (h : Call) (view : View)
    (indices : Nat → Int) (scratch : Nat → Nat → α) (before : Bytes)
    (prewrite : Prefix ε) (copyFailure : Option (CopyFailure ε h view indices before)) :
    (run encode h view indices scratch before prewrite copyFailure).outputObject=view.object ∧
    (run encode h view indices scratch before prewrite copyFailure).outputOwner=view.owner := by
  cases prewrite <;> cases copyFailure <;> exact ⟨rfl,rfl⟩

#axiomsClean prewrite_failure
#axiomsClean empty_return
#axiomsClean success
#axiomsClean frame
#axiomsClean identity
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.HostProtocol
