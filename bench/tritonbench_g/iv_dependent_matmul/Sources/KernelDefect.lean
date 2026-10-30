import bench.tritonbench_g.iv_dependent_matmul.Sources.Contract

/-! The selected wrapper admits a live tile rejected by the pinned NVIDIA dot
frontend. The compiler-side lower bound is independently pinned and exercised
by check_block_defect.py; this module proves the exact checked-domain gap. -/
namespace VeriTile.Bench.Repairs.IvDependentMatmul.KernelDefect
open TritonBenchG.IvDependentMatmul Whole Kernels

def request (schedule : Schedule) : Request :=
  ⟨"A", "B", "C", 1, 1, 1, 1, 1, 1, by decide, by decide, by decide,
    schedule, "cuda"⟩

/-- Shape lower bounds returned for FP32 by the pinned NVIDIA frontend.
This records a necessary frontend condition, not a sufficient launch contract. -/
def FP32DotShape (r : Request) : Prop :=
  1 ≤ r.blockRows ∧ 1 ≤ r.blockColumns ∧ 16 ≤ r.blockInner

theorem all_schedules_admitted_unsupported (schedule : Schedule) :
    Contract.check (request schedule) Bool.true Bool.true = .ok () ∧
    programs (request schedule) = 1 ∧
    0 < (request schedule).inner ∧
    ¬ FP32DotShape (request schedule) := by
  cases schedule <;> simp only [FP32DotShape] <;> decide

theorem all_schedules_preconditions (schedule : Schedule) :
    Contract.Preconditions (request schedule) Bool.true Bool.true := by
  cases schedule <;> refine ⟨rfl, rfl, ?_⟩ <;> constructor <;> decide

theorem contract_does_not_imply_dot_shape :
    ¬ (∀ r : Request, Contract.Preconditions r Bool.true Bool.true →
      FP32DotShape r) := by
  intro implication
  exact (all_schedules_admitted_unsupported .preLoad).2.2.2
    (implication (request .preLoad) (all_schedules_preconditions .preLoad))

#axiomsClean all_schedules_admitted_unsupported
#axiomsClean all_schedules_preconditions
#axiomsClean contract_does_not_imply_dot_shape
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.IvDependentMatmul.KernelDefect
