import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Host
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeRealPublic

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Real
open VeriTile.Triton VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeHost
noncomputable section

/-- Independent mathematical specification: the finite dot product of each
logical matrix row with the original logical vector. -/
def matrixVector (rows columns : Nat) (matrix : Nat → Nat → ℝ)
    (vector : Nat → ℝ) (i : Fin rows) : ℝ :=
  ∑ j : Fin columns, matrix i.val j.val * vector j.val

theorem bound_formula {x : Inputs} {entry : Nat → ℝ}
    {matrix : Nat → Nat → ℝ} {vector : Nat → ℝ}
    (b : DirectBinding ℝ x entry matrix vector)
    (compatible : x.columns = x.vectorLength) (i : Fin x.rows) :
    WholeReal.mathematical x entry i = matrixVector x.rows x.columns matrix vector i := by
  unfold WholeReal.mathematical WholeMath.matrixVector matrixVector
  apply Finset.sum_congr rfl
  intro j _
  dsimp only
  rw [b.matrixCoordinate _ _ i.isLt j.isLt,
    b.vectorCoordinate _ (by simpa only [←compatible] using j.isLt)]

theorem outcomes {x : Inputs} {entry : Nat → ℝ} {old : Nat → Prop}
    {matrix : Nat → Nat → ℝ} {vector : Nat → ℝ}
    (b : DirectBinding ℝ x entry matrix vector)
    (compatible : x.columns = x.vectorLength)
    (rt : WholeHost.Runtime ℝ x entry old) :
    WholePublic.Outcomes (fun _ i => matrixVector x.rows x.columns matrix vector i)
      rt (WholeHost.execute WholeReal.family rt) := by
  have formula : (fun (_ : Allocated ℝ x entry old) i => WholeReal.mathematical x entry i) =
      (fun _ i => matrixVector x.rows x.columns matrix vector i) := by
    funext a i
    exact bound_formula b compatible i
  rw [←formula]
  exact WholePublic.real_outcomes rt

#axiomsClean outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Real
