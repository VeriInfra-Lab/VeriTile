import bench.tritonbench_g.relu_strided_buffer.Kernels
import bench.tritonbench_g.relu_strided_buffer.Proofs.StructuralHost

/-! Independent Real specification of the shared typed source. Precision tags
remain source metadata; their mathematical interpretation here is exact Real.
This is not an assertion about NaN, signed-zero words or native rounding. -/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
open VeriTile.Triton FP.Structural HostPlan SignedView SignedGrid
noncomputable section

/-- Mathematical tensor ReLU, independently of the executable kernel. -/
def realResult (xs : Nat → ℝ) (i : Nat) : ℝ := max 0 (xs i)

/-- Exact mathematical interpretation of the source's scalar primitives. Only
literal, identity cast and strict comparison slots are used by this source. -/
def realBackend (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with literal := fun _ _ x => x
              cast := fun _ _ _ x => x
              compareLt := fun p d => if p = none ∧ d = .fp32 then
                some (fun x y => decide (x < y)) else rest.compareLt p d }

theorem real_compare (rest : Algebra ℝ) :
    (realBackend rest).compareLt none .fp32 = some (fun x y => decide (x < y)) := by
  simp [realBackend]

theorem real_scalar (rest : Algebra ℝ) (x : ℝ) :
    StructuralTile.scalarResult (realBackend rest) (fun x y => decide (x < y)) x = max 0 x := by
  simp only [StructuralTile.scalarResult, StructuralTile.scalarZero, realBackend]
  by_cases positive : 0 < x
  · simp [positive, max_eq_right (le_of_lt positive)]
  · simp [positive, max_eq_left (le_of_not_gt positive)]

theorem real_complete (rest : Algebra ℝ) (input output : Buffer)
    {n inputCapacity outputCapacity : Nat} (positive : 0 < n) (sx sy : Int)
    (hshapeIn : input.shape = [n]) (hshapeOut : output.shape = [n])
    (hstrideIn : input.strides = [sx]) (hstrideOut : output.strides = [sy])
    (hin : (input.view sx).Valid n inputCapacity)
    (hout : (output.view sy).Valid n outputCapacity) (hstride : sy ≠ 0)
    (hsep : FootprintsDisjoint (input.view sx) (output.view sy) n)
    (s : State ℝ) (xs : Nat → ℝ)
    (values : ∀ i, i < n → s.mem input.region ((input.view sx).address i) = .mk .fp32 (xs i)) :
    ∃ final, runHost (realBackend rest) input output Bool.true s = .returned output final ∧
      (∀ i, i < n → final.mem output.region ((output.view sy).address i) =
        Cell.mk .fp32 (realResult xs i)) ∧
      (∀ i, i < n → final.mem input.region ((input.view sx).address i) =
        s.mem input.region ((input.view sx).address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ (output.view sy).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → (input.view sx).address i < inputCapacity ∧
        (output.view sy).address i < outputCapacity) := by
  obtain ⟨final, run, outputValues, inputValues, frame, bounds⟩ :=
    StructuralHost.rank_one_success (realBackend rest) (fun x y => decide (x < y))
      (real_compare rest) input output positive sx sy hshapeIn hshapeOut hstrideIn hstrideOut
      hin hout hstride hsep s
  refine ⟨final, run, ?_, inputValues, frame, bounds⟩
  intro i hi
  rw [outputValues i hi, values i hi]
  simp [Cell.read, real_scalar, realResult]

#guard_msgs (drop info) in
#specNonCircular realResult avoiding [kernel, runHost, StructuralHost.execute]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
