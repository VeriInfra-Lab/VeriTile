import bench.repairs.tritonbench_g.matmul_kernel.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.MatmulKernel
open VeriTile.Triton FP.Structural
open scoped BigOperators LogicalProgramIO
noncomputable section

/-- Independent matrix mathematics on logical finite input/output coordinates.
The IO retains the actual complete source grid and every original condition. -/
specification correct (p : Config) (hc : p.Valid) :
    VeriTile.Spec.Real (ReaderIO.view p ⊨ fun xs r c => ∑ t : Fin p.K, xs.A r t * xs.B t c) :=
  ReaderProof.correct p hc

#axiomsClean correct
end
end VeriTile.Bench.Repairs.MatmulKernel
