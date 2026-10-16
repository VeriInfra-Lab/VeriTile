import bench.tritonbench_g.matmul_persistent_triton.Proofs.NumericValues
import bench.tritonbench_g.matmul_persistent_triton.Proofs.NumericVariants
import bench.tritonbench_g.matmul_persistent_triton.Proofs.NumericFrame
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealMath

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical

def inputs : RealMath.Inputs where
  a := fun row _ => row + 1
  b := fun k column => k + column + 1

/-- The independent arrays' GEMM formula, without any execution-derived input. -/
theorem input_formula :
    RealMath.formula 3 inputs 2 0 = 18 ∧
    RealMath.formula 3 inputs 2 1 = 27 ∧
    RealMath.formula 3 inputs 2 2 = 36 := by
  norm_num [RealMath.formula, inputs, Finset.sum_range_succ]

/-- Both complete source variants execute the four persistent iterations in
both storage channels, produce the independent matrix-product values, and
preserve all raw memory outside their three active output stores. -/
theorem nonempty_typed_execution (storage : TypedSource.Storage) (v : Bool) :
    ∃ t, run (engine model) (sample.source storage v).surfaceBody (initial storage) = some t ∧
      t.mem "C" 14 = .mk (channel storage).toTileDType
        (ofFloat (channel storage) (some (RealMath.formula 3 inputs 2 0))) ∧
      t.mem "C" 16 = .mk (channel storage).toTileDType
        (ofFloat (channel storage) (some (RealMath.formula 3 inputs 2 1))) ∧
      t.mem "C" 18 = .mk (channel storage).toTileDType
        (ofFloat (channel storage) (some (RealMath.formula 3 inputs 2 2))) ∧
      (∀ (r : RegionName) o, r ≠ "C" ∨ (o ≠ 14 ∧ o ≠ 16 ∧ o ≠ 18) → t.mem r o = (initial storage).mem r o) := by
  have hc := numeric_cells storage
  rw [input_formula.1, input_formula.2.1, input_formula.2.2]
  exact ⟨(numeric_run storage).val, both_execution storage v,
    hc.1, hc.2.1, hc.2.2.1, numeric_frame storage⟩

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
