import bench.tritonbench_g.matmul_persistent_triton.Proofs.ZeroK
import bench.tritonbench_g.matmul_persistent_triton.Proofs.NumericModel
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealMath

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.ZeroK
open VeriTile.Triton FP.Structural FP.ProfiledRewrite NumericExecution
noncomputable section
open scoped Classical

def finiteInitial (storage : TypedSource.Storage) : State (WithBot ℝ) :=
  { initial storage with mem := fun r o =>
      if r = "C" then .mk (channel storage).toTileDType (ofFloat (channel storage) (some 37))
      else (initial storage).mem r o }

def emptyInputs : RealMath.Inputs where
  a := fun row _ => row + 1
  b := fun k column => k + column + 1

/-- Positive M,N and an actual launched PID do not make K=0 a GEMM: the
source leaves a correctly typed nonzero output sentinel intact, whereas the
independent empty matrix-product sum is zero. Both original/candidate and
both wrapper storage profiles exhibit this source defect. -/
theorem zero_K_counterexample (storage : TypedSource.Storage) (v : Bool) :
    ∃ t, run (engine model)
      (({sample with K := 0} : Parameters).source storage v).surfaceBody
        (finiteInitial storage) = some t ∧
      t.mem "C" 14 = .mk (channel storage).toTileDType (ofFloat (channel storage) (some 37)) ∧
      RealMath.formula 0 emptyInputs 2 0 = 0 ∧
      t.mem "C" 14 ≠ .mk (channel storage).toTileDType
        (ofFloat (channel storage) (some (RealMath.formula 0 emptyInputs 2 0))) := by
  obtain ⟨t, ht, hm⟩ := run_unchanged sample storage v (by decide) model (finiteInitial storage)
  refine ⟨t, ht, ?_, RealMath.zero_K _ _ _, ?_⟩
  · rw [hm]
    simp [finiteInitial]
  · rw [hm, RealMath.zero_K]
    cases storage <;> norm_num [finiteInitial, channel, ofFloat, FloatDType.toTileDType]
    all_goals change (some (37 : ℝ) : WithBot ℝ) ≠ some 0
    all_goals norm_num

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.ZeroK
