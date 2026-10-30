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

theorem real_complete (rest : Algebra ℝ) (base : ℝ) (candidate : Bool) (input output : Buffer)
    {n inputCapacity outputCapacity : Nat} (positive : 0 < n) (sx sy : Int)
    (hshapeIn : input.shape = [n]) (hshapeOut : output.shape = [n])
    (hstrideIn : input.strides = [sx]) (hstrideOut : output.strides = [sy])
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity) (hstride : sy ≠ 0)
    (hsep : FootprintsDisjoint (input.view sx) (output.view sy) n)
    (s : State ℝ) (xs : Nat → ℝ)
    (mathematical : ∀ i, i < n → powerDefined base (xs i))
    (values : ∀ i, i < n → s.mem input.region ((input.view sx).address i) = .mk .fp32 (xs i)) :
    ∃ final, runHost (realBackend rest) base candidate input output Bool.true s = .returned output final ∧
      (∀ i, i < n → final.mem output.region ((output.view sy).address i) =
        Cell.mk .fp32 (realResult base xs i)) ∧
      (∀ i, i < n → final.mem input.region ((input.view sx).address i) =
        s.mem input.region ((input.view sx).address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → (input.view sx).address i < inputCapacity ∧
        (output.view sy).address i < outputCapacity) := by
  obtain ⟨final, run, outputValues, inputValues, frame, bounds⟩ :=
    WholeHost.rank_one_success (realBackend rest) base candidate input output positive sx sy hshapeIn hshapeOut hstrideIn hstrideOut
      hin hout hstride hsep s
  refine ⟨final, run, ?_, inputValues, frame, bounds⟩
  intro i hi
  rw [outputValues i hi, values i hi]
  simp [Cell.read, real_scalar, realResult]

#guard_msgs (drop info) in
#specNonCircular realResult avoiding [WholeSource.kernel, WholeSource.runHost, WholeSource.executeArguments]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole
