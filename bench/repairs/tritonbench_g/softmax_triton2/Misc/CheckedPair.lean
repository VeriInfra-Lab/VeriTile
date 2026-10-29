import bench.repairs.tritonbench_g.softmax_triton2.Proofs.GridPair
import bench.repairs.tritonbench_g.softmax_triton2.Proofs.PairSafety
import bench.repairs.tritonbench_g.softmax_triton2.Contract

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.CheckedPair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1
open GuardedReciprocal
open scoped VeriTile.Spec
noncomputable section

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Complete original source over the explicit launch order. -/
def original (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (programs : List Nat) : GridProgram where
  kernel := softmax_kernel out inp ib ob ir ic orow oc n B
  programs := programs

/-- The reciprocal candidate uses the same actual operands, source precision,
load/reduction/store order and masks; its runtime fallback retains every input. -/
def reciprocal (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (programs : List Nat) : GridProgram where
  kernel := guarded_reciprocal_kernel out inp ib ob ir ic orow oc n B
  programs := programs

/-- Successful source execution and whole-memory equivalence for every launch,
including every raw FP32 input word. Exact finitef/order-port bindings are part
of the named model view, not caller input guards. DIV-MUL-RCP is the sole local
experimental relation; this does not assert unconditional IEEE bit equality. -/
theorem softmax_reciprocal_equiv (R : Rules) (out inp : RegionName)
    (ib ob ir ic orow oc n B : Nat) (programs : List Nat)
    (hB : 0 < B) (countBound : B < 2^32) :
    original out inp ib ob ir ic orow oc n B programs ≡[R]
      reciprocal out inp ib ob ir ic orow oc n B programs := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  apply grid_of_row R.assumptions
    (⟨softmax_kernel out inp ib ob ir ic orow oc n B⟩ : Program)
    (⟨guarded_reciprocal_kernel out inp ib ob ir ic orow oc n B⟩ : Program) programs
  intro α _ M D model ports s
  let a := originalExecution M out inp ib ob ir ic orow oc n B hB s
  let b := candidateExecution M D ports out inp ib ob ir ic orow oc n B hB s
  exact ⟨a.val,b.val,a.property,b.property,
    row_memory R M D model ports out inp ib ob ir ic orow oc n B hB countBound s⟩

/-- Exactly the existing wrapper's schedule: empty tensors do not launch. -/
def checkedPrograms (x : Contiguous.TensorMetadata) : List Nat :=
  if x.numel = 0 then [] else List.range (Contiguous.rows x)

/-- The count bound follows from the unchanged frontend input contract. Empty
inputs retain the source wrapper's no-launch behavior with no new restriction. -/
theorem checked_reciprocal_equiv (R : Rules) (out inp : RegionName)
    (ib ob : Nat) (x : Contiguous.TensorMetadata) (h : InputContract x) :
    original out inp ib ob (Contiguous.cols x) 1 (Contiguous.cols x) 1
      (Contiguous.cols x) (SoftmaxTriton1.blockSize (Contiguous.cols x)) (checkedPrograms x) ≡[R]
    reciprocal out inp ib ob (Contiguous.cols x) 1 (Contiguous.cols x) 1
      (Contiguous.cols x) (SoftmaxTriton1.blockSize (Contiguous.cols x)) (checkedPrograms x) := by
  by_cases empty : x.numel = 0
  · apply Spec.FloatingPoint.ofNumerical
    · rfl
    · rfl
    intro α _ M D model ports base mem
    exact ⟨mem,by simp [original,checkedPrograms,empty,gridExec],
      by simp [reciprocal,checkedPrograms,empty,gridExec]⟩
  · apply softmax_reciprocal_equiv
    · unfold SoftmaxTriton1.blockSize
      omega
    · have frontend := h.frontend (Nat.pos_of_ne_zero empty)
      unfold SoftmaxTriton1.blockSize
      change max 1 (Contiguous.block (Contiguous.cols x)) < 2^32
      omega

#axiomsClean softmax_reciprocal_equiv
#axiomsClean checked_reciprocal_equiv
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.CheckedPair
