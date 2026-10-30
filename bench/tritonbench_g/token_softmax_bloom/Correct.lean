import bench.tritonbench_g.token_softmax_bloom.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Each selected segment/head implements softmax over exactly its runtime
logical length. The original full finite tensors, coverage, output-event
uniqueness, repeated identical launches, masks and frame are retained. -/
specification token_softmax_grid_real (p : Parameters) (positive : 0 < p.width)
    (separate : Grid.Separate p) (work : List Grid.PID) :
    Spec.Real (Logical.io p work ⊨{Logical.domain p work} fun xs job i =>
      Real.exp (xs.values job i) / ∑ j : Fin (xs.length job), Real.exp (xs.at job j.val)) :=
  (Logical.judgment_iff p work).2 (IOCorrect.real p positive separate work)

#specNonCircular Logical.domain avoiding [source,Grid.run,outputValues,FP.Structural.exec]
#axiomsClean token_softmax_grid_real
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.Correct
