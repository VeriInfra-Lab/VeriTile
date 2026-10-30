import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Real
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.TensorBinding

/-! Complete mathematical source projection on the checked contiguous FP32
CUDA domain. Arithmetic in this entry is Real; it makes no FP-to-Real claim. -/
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.HistoricalCorrect
open VeriTile VeriTile.Triton VeriTile.Torch
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeHost
noncomputable section

/-- Independent intended formula, with all columns and no implementation tile. -/
def matrixVector (n m : Nat) (matrix : Nat → Nat → ℝ)
    (vector : Nat → ℝ) (i : Fin n) : ℝ := ∑ j : Fin m, matrix i.val j.val * vector j.val

theorem python_checks_iff_preconditions (x y : ArgumentMetadata) :
    checkInputs x y = .ok () ↔ AdmittedArguments x y := checkInputs_iff x y

/-- All loop iterations, row/column tails, empty extents, complete program
orders and autotune trials. Ordinary successful setup supplies legal launch
program ids; allocator bounds supply the fresh output span. -/
specification mv_real_outcomes
    (x y : Torch.FP32.Tensor) (lx : Torch.FP32.TensorLaws x) (ly : Torch.FP32.TensorLaws y)
    (h : InputContract x.metadata y.metadata)
    (read : Nat → Torch.FP32.Word) (realValue : Torch.FP32.Word → ℝ) (old : Nat → Prop)
    (rt : Runtime ℝ (TensorBinding.descriptor x y lx ly h)
      (TensorBinding.memory realValue read) old)
    (outputSpan : ∀ a, rt.allocate = some a →
      4*a.base + 4*rows x.metadata ≤ 2^64)
    (launchable : ∀ a, rt.allocate = some a → rt.launchReady a = Bool.true →
      ∀ t, t ∈ (rt.plan a).trials ∨ t = (rt.plan a).selected →
        WholeGrid.gridSize ((TensorBinding.descriptor x y lx ly h).parameters t.config) < 2^31) :
    Spec.Real (
      executeChecked (WholeHost.execute WholeReal.family) rt =
        .ok (WholeHost.execute WholeReal.family rt) ∧
      WholePublic.Outcomes (fun _ i => matrixVector (rows x.metadata) (columns x.metadata)
        (TensorBinding.matrix x lx realValue read) (TensorBinding.vector y ly realValue read) i)
        rt (WholeHost.execute WholeReal.family rt) ∧
      ∀ a, rt.allocate = some a → rt.launchReady a = Bool.true →
        Index.Correspondence a ∧
        Index.PlanCorrespondence (TensorBinding.descriptor x y lx ly h) (rt.plan a)) := by
  refine ⟨checked_run h _ rt, ?_, ?_⟩
  · exact Real.outcomes (TensorBinding.direct x y lx ly h realValue read) h.dimensions rt
  · intro a allocated ready
    have bounds := TensorBinding.bounds x y lx ly h a (outputSpan a allocated)
    exact ⟨Index.correspondence a bounds,
      Index.plan_correspondence ⟨bounds, launchable a allocated ready⟩⟩

#axiomsClean mv_real_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.HistoricalCorrect
