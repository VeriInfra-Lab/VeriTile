import bench.tritonbench_g.softmax_triton2.Proofs.SourceIO

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.FPEquiv
open VeriTile.Triton FP.Structural
open GuardedReciprocal
open scoped VeriTile.Spec
noncomputable section

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Complete original source with its original active typed input/output layout. -/
def original (out inp : RegionName) (ib ob ir ic orow oc n B p : Nat) : IOProgram :=
  sourceProgram out inp ib ob ir ic orow oc n B p

/-- Runtime-guarded reciprocal candidate on exactly the same public layout. -/
def reciprocal (out inp : RegionName) (ib ob ir ic orow oc n B p : Nat) : IOProgram :=
  reciprocalProgram out inp ib ob ir ic orow oc n B p

/-- Under the original IO bounds and logical input binding, both complete source
traces are safe, both succeed, every active typed output cell agrees and both
preserve all exterior cells. Every raw FP32 word is admitted. DIV-MUL-RCP is the
sole numerical relation; this is not unconditional IEEE bit equality. -/
specification softmax_reciprocal_equiv (R : Rules) (out inp : RegionName)
    (ib ob ir ic orow oc n B p : Nat) (hB : 0 < B) (countBound : B < 2^32) :
    original out inp ib ob ir ic orow oc n B p ≡[R]
      reciprocal out inp ib ob ir ic orow oc n B p := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  exact io_equivalent R out inp ib ob ir ic orow oc n B p hB countBound

#axiomsClean softmax_reciprocal_equiv
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.FPEquiv
