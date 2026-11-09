import bench.examples.Welford.IO
import bench.examples.Welford.Contract
import bench.examples.Welford.Proofs.FP
import VeriTile.Triton.Float.CountConversionLaws
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

/- Original online versus two-pass Welford, using strictly certified scalar
and bounded count rules together with admitted scalar relaxations.
The count bound is the integer conversion theorem's domain. Empty Welford
rows divide by zero and are outside this FP contract. -/
namespace VeriTile.Bench.Examples.WelfordFPEquiv
open VeriTile Triton FP.Structural FP.CountConversion
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan)
open scoped VeriTile.Spec

def online (x mean variance : RegionName) (N stride : Nat) (empty : ReductionPlan 0) :
    FP.Kernel₁ₓ₂ where
  io := onlineIO x mean variance N stride
  profile := FP.fp32
  domain := requirements x N stride empty

def twopass (x mean variance : RegionName) (N stride : Nat) (empty : ReductionPlan 0) :
    FP.Kernel₁ₓ₂ :=
  { online x mean variance N stride empty with io := twopassIO x mean variance N stride }

specification welford_equiv (N stride : Nat) (hN : 0 < N) (hBound : N ≤ limit)
    (empty : ReductionPlan 0) (R : Rules) :
    online "x" "mean" "variance" N stride empty ≡[R]
      twopass "x" "mean" "variance" N stride empty := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := online "x" "mean" "variance" N stride empty)
    (rhs := twopass "x" "mean" "variance" N stride empty)
    (structural := fun _ _ => False) rfl rfl
  exact WelfordFPContract.equivalent N stride hN hBound empty R

#print_fp_assumptions welford_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.WelfordFPEquiv
