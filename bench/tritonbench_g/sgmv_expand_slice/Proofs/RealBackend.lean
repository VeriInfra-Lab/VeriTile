import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceBody
import bench.tritonbench_g.sgmv_expand_slice.Proofs.Mathematics

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section

/-- Exact interpretation of every floating scalar operation used by the
source. The independent mathematical target does not mention this evaluator. -/
def algebra (rest : Algebra ℝ) : Algebra ℝ :=
  {rest with
    literal := fun _ _ value => value
    cast := fun _ _ _ value => value
    binary := fun precision dtype operation a b => match operation with
      | .add => a+b
      | other => rest.binary precision dtype other a b
    fp32Load := id}

def dot : SourceBody.DotImplementation ℝ where
  apply := by
    intro precision left right output options batch m k n a b acc
    by_cases hb : batch = []
    · subst batch
      exact fun i => acc i + ∑ j : Fin k, a (i.1,j,PUnit.unit)*b (j,i.2.1,PUnit.unit)
    · exact fun _ => 0

theorem selected_dot (precision : Option ComputeDType) (left right output : FloatDType)
    (options : DotOptions) (a : TileIndex [32,16] → ℝ) (b : TileIndex [16,32] → ℝ)
    (acc : TileIndex [32,32] → ℝ) (i : TileIndex [32,32]) :
    dot.apply precision left right output options
      (batch := []) (m := 32) (k := 16) (n := 32) a b acc i =
      acc i + ∑ j : Fin 16, a (i.1,j,PUnit.unit)*b (j,i.2.1,PUnit.unit) := by
  simp [dot]

#axiomsClean selected_dot
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.RealBackend
