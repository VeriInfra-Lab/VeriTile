import bench.examples.FusedLayerNorm.IO
import bench.examples.FusedLayerNorm.Contract
import bench.examples.FusedLayerNorm.Proofs.FP
import VeriTile.Triton.Float.CountConversionLaws
import VeriTile.Triton.Float.Kernel
import VeriTile.Meta.StatementAudit

/- Original online-statistics versus two-pass LayerNorm. The shared affine
suffix retains its sqrt, epsilon, gamma/beta and bf16 cast. The proof combines
strictly certified scalar and bounded count rules with admitted arithmetic
relaxations. Empty output rows remain covered because both sources preserve
every memory cell at N = 0. -/

namespace VeriTile.Bench.Examples.FusedLayerNormFPEquiv
open VeriTile Triton FP.Structural FP.CountConversion
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan)
open scoped VeriTile.Spec

def twopass (x g b y : RegionName) (N stride : Nat) (ε : ℝ) (empty : ReductionPlan 0) :
    FP.Kernel₃ where
  io := twopassIO x g b y N stride ε
  profile := FP.fp32
  domain := requirements x N stride empty

def fused (x g b y : RegionName) (N stride : Nat) (ε : ℝ) (empty : ReductionPlan 0) :
    FP.Kernel₃ :=
  { twopass x g b y N stride ε empty with io := fusedIO x g b y N stride ε }

specification layernorm_equiv (N stride : Nat) (hBound : N ≤ limit) (ε : ℝ)
    (empty : ReductionPlan 0) (R : Rules) :
    fused "x" "gamma" "beta" "y" N stride ε empty ≡[R]
      twopass "x" "gamma" "beta" "y" N stride ε empty := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := fused "x" "gamma" "beta" "y" N stride ε empty)
    (rhs := twopass "x" "gamma" "beta" "y" N stride ε empty)
    (structural := fun _ _ => False) rfl rfl
  exact LayerNormFPContract.equivalent N stride hBound ε empty R

#print_fp_assumptions layernorm_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.FusedLayerNormFPEquiv
