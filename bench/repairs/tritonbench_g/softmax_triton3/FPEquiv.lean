import bench.repairs.tritonbench_g.softmax_triton3.Proofs.SourceIO

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.FPEquiv
open VeriTile.Triton FP.Structural
open Pair
open scoped VeriTile.Spec
noncomputable section

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Complete original source with its original active typed input/output layout. -/
def original (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool) (p : Nat) : IOProgram where
  kernel := softmax_kernel out inp mask stride n B hasMask
  layout := ⟨out,inp,mask,stride,n,B,hasMask,p⟩

/-- Runtime-guarded reciprocal candidate on exactly the same public layout. -/
def reciprocal (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool) (p : Nat) : IOProgram where
  kernel := guarded_reciprocal_kernel out inp mask stride n B hasMask
  layout := ⟨out,inp,mask,stride,n,B,hasMask,p⟩

/-- Under the original IO bounds and logical input binding, both complete source
traces are safe, both succeed, every active typed output cell agrees and both
preserve all exterior cells. Every raw FP32 word is admitted. DIV-MUL-RCP is the
sole numerical relation; this is not unconditional IEEE bit equality. -/
specification softmax_reciprocal_equiv (R : Rules) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) (p : Nat) (hB : 0 < B) (countBound : B < 2^32) :
    original out inp mask stride n B hasMask p ≡[R]
      reciprocal out inp mask stride n B hasMask p := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  exact io_equivalent R out inp mask stride n B hasMask p hB countBound

#axiomsClean softmax_reciprocal_equiv
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.FPEquiv
