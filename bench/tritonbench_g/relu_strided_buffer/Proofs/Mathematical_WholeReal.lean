import bench.tritonbench_g.relu_strided_buffer.Kernels
import VeriTile.Triton
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_WholeSource
import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_StructuralHost

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

#guard_msgs (drop info) in
#specNonCircular realResult avoiding [VeriTile.Bench.Repairs.ReluStridedBuffer.kernel]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.Whole
