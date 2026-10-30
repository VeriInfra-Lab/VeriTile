import bench.tritonbench_g.token_softmax_llama.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Correct
open VeriTile.Triton FP.Structural Strided
open TokenSoftmaxBloom.SoftmaxIO (realDomain target realRow length)
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Full logical segment softmax, with the selected head/segment as the first
output index. The bounds test only totalizes this function outside realDomain;
its public Cover condition makes every logical denominator term active. -/
def segmentSoftmax (p : Parameters) (work : List Grid.PID)
    (xs : PublicIO.LogicalInputs p work) (r : Fin work.length) (c : Fin p.width) : ℝ :=
  Real.exp (xs.values r c) /
    ∑ j : Fin (xs.lengths r), Real.exp (xs.segment p work r j)

/-- Complete selected segment/head schedule and independent finite softmax.
The IO domain states full-segment coverage and output event uniqueness;
repeated identical launches remain admitted. -/
specification token_softmax_grid_real (p : Parameters) (positive : 0<p.width)
    (separate : Grid.Separate p) (work : List Grid.PID) :
    Spec.Real (PublicIO.logicalRealIO p work ⊨{PublicIO.logicalDomain p work} segmentSoftmax p work) := by
  exact (PublicIO.logical_iff p work).2 (PublicIO.real p positive separate work)

#specNonCircular realRow avoiding [source,Grid.run,outputValues,FP.Structural.exec]
#axiomsClean token_softmax_grid_real
#stmtSurfaceSubset token_softmax_grid_real ⊆
  [Spec.Real,LogicalProgramIO.ImplementsOn,PublicIO.logicalRealIO,PublicIO.logicalDomain,segmentSoftmax,
   Real.Scalar,Parameters,Parameters.width,Grid.Separate,Grid.PID,PublicIO.LogicalInputs]
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Correct
