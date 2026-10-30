import VeriTile.Triton.Float.StructuralSafety

/-! Actual-source access safety: masks, branches, discarded reads and unsupported effects. -/

namespace VeriTile.Triton.FP.Structural.Safety.Regression
noncomputable section
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α] (M : Algebra α) (s : State α)

def loadAt (address : Nat) (active : Bool) : ComputeKernel :=
  .mk ["x"] [] [.assign .nat [] "value"
    (.alg (.load .nat (.region "x" (.constNat address))
      (.mask (.constBool active)) .zero))]

def storeAt (address : Nat) (active : Bool) : ComputeKernel :=
  .mk [] ["out"] [.store .nat [] (.region "out" (.constNat address))
    (.alg (.constNat 7)) (.mask (.constBool active))]

def pointerAt (address : Nat) : ComputeKernel :=
  .mk [] [] [.assign .ptr [] "pointer"
    (.alg (.ptrAdd .nil (.ptrBase (⟨"x"⟩ : Region .nat)) (.constNat address)))]

-- Arbitrarily large pointer formation is not a load/store.
example (address : Nat) : TraceSafe M (fun _ => 0) (pointerAt address) s := by
  simp [TraceSafe, pointerAt, ComputeKernel.surfaceBody, traceSafeList_cons,
    stepSafe, safeExpr, safeOp, step, run]

-- A masked-off access needs no backing, including an arbitrarily large offset.
example (address : Nat) : TraceSafe M (fun _ => 0) (loadAt address false) s := by
  simp [TraceSafe, loadAt, ComputeKernel.surfaceBody, traceSafeList_cons,
    stepSafe, safeExpr, safeOp, addressSafe, maskActive, evalOp_unfold, step, run]

example (address : Nat) : TraceSafe M (fun _ => 0) (storeAt address false) s := by
  simp [TraceSafe, storeAt, ComputeKernel.surfaceBody, traceSafeList_cons,
    stepSafe, safeExpr, safeOp, safeAddress, safeMask, addressSafe, maskActive,
    evalOp_unfold, step, run]

-- In-bounds active accesses are accepted for reads and writes.
example (address capacity : Nat) (inside : address < capacity) :
    TraceSafe M (fun _ => capacity) (loadAt address true) s := by
  simp [TraceSafe, loadAt, ComputeKernel.surfaceBody, traceSafeList_cons,
    stepSafe, safeExpr, safeOp, addressSafe, maskActive, evalOp_unfold, step, run, inside]

example (address capacity : Nat) (inside : address < capacity) :
    TraceSafe M (fun _ => capacity) (storeAt address true) s := by
  simp [TraceSafe, storeAt, ComputeKernel.surfaceBody, traceSafeList_cons,
    stepSafe, safeExpr, safeOp, safeAddress, safeMask, addressSafe, maskActive,
    evalOp_unfold, step, run, inside]

-- An active out-of-bounds read is rejected even if its result is discarded.
example : ¬ TraceSafe M (fun _ => 0) (loadAt 0 true) s := by
  simp [TraceSafe, loadAt, ComputeKernel.surfaceBody, traceSafeList_cons,
    stepSafe, safeExpr, safeOp, addressSafe, maskActive, evalOp_unfold]

example : ¬ TraceSafe M (fun _ => 0) (storeAt 0 true) s := by
  simp [TraceSafe, storeAt, ComputeKernel.surfaceBody, traceSafeList_cons,
    stepSafe, safeExpr, safeOp, safeAddress, safeMask, addressSafe, maskActive,
    evalOp_unfold]

-- Conditional control flow follows the actual branch; the skipped body has no access.
example : TraceSafe M (fun _ => 0)
    (.mk ["x"] [] [.ifThen (.constBool false) (loadAt 0 true).surfaceBody]) s := by
  simp [TraceSafe, ComputeKernel.surfaceBody, traceSafeList_cons, stepSafe, safeOp,
    evalOp_unfold, step, run]

example : ¬ TraceSafe M (fun _ => 0)
    (.mk ["x"] [] [.ifThen (.constBool true) (loadAt 0 true).surfaceBody]) s := by
  simp [TraceSafe, loadAt, ComputeKernel.surfaceBody, traceSafeList_cons,
    stepSafe, safeExpr, safeOp, addressSafe, maskActive, evalOp_unfold]

-- Unknown effects fail closed, even without visible memory outputs.
example : ¬ TraceSafe M (fun _ => 0) (.mk [] [] [.effectMarker "unsupported.async"]) s := by
  simp [TraceSafe, ComputeKernel.surfaceBody, traceSafeList_cons, stepSafe]
end
end VeriTile.Triton.FP.Structural.Safety.Regression
