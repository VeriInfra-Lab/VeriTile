import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Pair
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.PairSafety
import bench.repairs.tritonbench_g.softmax_triton3.Contract

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.CheckedPair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
open scoped VeriTile.Spec
noncomputable section

abbrev Rules := FP.ScalarArithmetic.Rules

def original (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool)
    (programs : List Nat) : GridProgram where
  kernel := softmax_kernel out inp mask stride n B hasMask
  programs := programs

def reciprocal (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool)
    (programs : List Nat) : GridProgram where
  kernel := guarded_reciprocal_kernel out inp mask stride n B hasMask
  programs := programs

/-- Two complete actual programs over every requested row and every input word.
The runtime fallback preserves optional masks and nonfinite/extreme-mask cases.
Only the reached safe branch uses the admitted FP32 DIV-MUL-RCP relation. -/
theorem softmax_reciprocal_equiv (R : Rules) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) (programs : List Nat)
    (hB : 0 < B) (countBound : B < 2^32) :
    original out inp mask stride n B hasMask programs ≡[R]
      reciprocal out inp mask stride n B hasMask programs := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · cases hasMask <;> rfl
  apply grid_of_row R.assumptions
    (⟨softmax_kernel out inp mask stride n B hasMask⟩ : Program)
    (⟨guarded_reciprocal_kernel out inp mask stride n B hasMask⟩ : Program) programs
  intro α _ M D model ports s
  let a := Pair.originalExecution M out inp mask stride n B hasMask hB s
  let b := Pair.candidateExecution M D ports out inp mask stride n B hasMask hB s
  exact ⟨a.val,b.val,a.property,b.property,
    Pair.row_memory R M D model ports out inp mask stride n B hasMask hB countBound s⟩

def checkedPrograms (x : Contiguous.TensorMetadata) : List Nat :=
  if x.numel = 0 then [] else List.range (Contiguous.rows x)

/-- Full existing checked input/mask/dimension contract, including empty axes,
all flattened rows and all FP32 words. No finiteness condition is added. -/
theorem checked_reciprocal_equiv (R : Rules) (out inp maskPtr : RegionName)
    (x : Contiguous.TensorMetadata) (mask : Option Contiguous.TensorMetadata) (dim : Int)
    (h : Contiguous.InputContract x mask dim) :
    original out inp maskPtr (Contiguous.cols x) (Contiguous.cols x)
      (blockSize (Contiguous.cols x)) mask.isSome (checkedPrograms x) ≡[R]
    reciprocal out inp maskPtr (Contiguous.cols x) (Contiguous.cols x)
      (blockSize (Contiguous.cols x)) mask.isSome (checkedPrograms x) := by
  by_cases empty : x.numel = 0
  · apply Spec.FloatingPoint.ofNumerical
    · rfl
    · cases mask <;> rfl
    intro α _ M D model ports base mem
    exact ⟨mem,by simp [original,checkedPrograms,empty,gridExec],
      by simp [reciprocal,checkedPrograms,empty,gridExec]⟩
  · apply softmax_reciprocal_equiv
    · unfold blockSize
      omega
    · have frontend := h.input.frontend (Nat.pos_of_ne_zero empty)
      omega

#axiomsClean softmax_reciprocal_equiv
#axiomsClean checked_reciprocal_equiv
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.CheckedPair
