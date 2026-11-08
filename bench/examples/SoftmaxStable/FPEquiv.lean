import bench.examples.SoftmaxStable.IO
import bench.examples.SoftmaxStable.Contract
import bench.examples.SoftmaxStable.Kernels
/- Stable versus naive softmax using the shared strictly certified scalar
theory and admitted numerical relaxations.
Both implementations use libdevice.exp: the measured tl.exp exp-sub relation
failed the configured bias gate (0.1608954387 ULP > 0.05). -/
import bench.examples.SoftmaxStable.Proofs.FP
import VeriTile.Triton.Float.ExponentialLaws
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.SoftmaxStableFPEquiv
open VeriTile.Bench.Examples.SoftmaxStable.Kernels
open VeriTile Triton FP.Structural FP.Exponential
open SoftmaxStableFPContract SoftmaxStableFPExecution
open scoped VeriTile.Spec
open SoftmaxStableFPContract (rowExpressions center rowPlan)
open _root_.VeriTile.Triton.FP.Equational (Schedules)
open FP.GuardExpression

def naive (x y : RegionName) (B : Nat) : FP.Kernel₁ where
  io := naiveIO x y B
  profile := FP.fp32
  domain := requirements x B
  comparisons := []

def stable (x y : RegionName) (B : Nat) : FP.Kernel₁ :=
  { naive x y B with io := stableIO x y B }

/-- Row length is symbolic. The shared contract retains the intermediate
finite/nonzero domain. `R` contains the admitted numerical relaxations;
strictly certified scalar rules belong to the fixed statement theory. -/
specification softmax_stable_equiv (B : Nat) (hB : 0 < B) (R : Rules) :
    stable "x" "y" B ≡[R] naive "x" "y" B := by
  apply Spec.FloatingPoint.ofNumerical (lhs := stable "x" "y" B) (rhs := naive "x" "y" B)
    (structural := fun _ _ => False) rfl rfl
  apply FP.Kernel₁.ofExecution
  exact SoftmaxStableFPContract.equivalent B hB R

#print_fp_assumptions softmax_stable_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.SoftmaxStableFPEquiv
