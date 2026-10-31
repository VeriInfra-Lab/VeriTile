import bench.tritonbench_g.pow_scalar_tensor.Kernels
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Mathematical_WholeHost

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole
open VeriTile.Triton FP.Structural WholeSource
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open HostPlan SignedView SignedGrid
noncomputable section

/-- Ordinary real power domain. Zero to zero follows the conventional value 1.
Negative bases require an integer exponent; zero to a negative exponent and
negative bases with a fractional exponent have no finite real result here.
NaN/infinity are FP payloads, not elements of this mathematical domain. -/
def powerDefined (base exponent : ℝ) : Prop :=
  0 < base ∨ (base = 0 ∧ 0 ≤ exponent) ∨
    (base < 0 ∧ ∃ z : Int, exponent = (z : ℝ))

/-- Independent scalar-base tensor power; no execution or output is consulted. -/
def realResult (base : ℝ) (xs : Nat → ℝ) (i : Nat) : ℝ := Real.rpow base (xs i)

def realBackend (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with literal := fun _ _ x => x
              cast := fun _ _ _ x => x
              binary := fun p d op => if p = some .fp32 ∧ d = .real ∧ op = .pow then
                Real.rpow else rest.binary p d op }

theorem real_scalar (rest : Algebra ℝ) (base x : ℝ) :
    WholeTile.scalarResult (realBackend rest) base x = Real.rpow base x := by
  simp [WholeTile.scalarResult, WholeTile.scalarBase, realBackend]

theorem positive_domain {base : ℝ} (positive : 0 < base) (exponent : ℝ) :
    powerDefined base exponent := Or.inl positive

theorem negative_integer_domain {base : ℝ} (negative : base < 0) (z : Int) :
    powerDefined base (z : ℝ) := Or.inr (Or.inr ⟨negative, z, rfl⟩)

theorem zero_negative_excluded {exponent : ℝ} (negative : exponent < 0) :
    ¬ powerDefined 0 exponent := by
  simp [powerDefined, not_le.mpr negative]

#guard_msgs (drop info) in
#specNonCircular realResult avoiding [VeriTile.Bench.Repairs.PowScalarTensor.Checked.kernel]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole
