import bench.tritonbench_g.log_softmax.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.LogSoftmax.Correct
open VeriTile.Triton FP.Structural
open scoped BigOperators LogicalProgramIO
noncomputable section

/-- Full logical N-column forward formula at every M×N×K coordinate. The
complete original grid and the reduction plan's coverage/permutation remain. -/
specification forward_grid_correctness (M N K BM : Nat)
    (retained : Grid.RetainedBlock BM) (plan : Plan N (Grid.blockColumns N)) :
    Spec.Real (ReaderIO.forward M N K BM plan ⊨ fun xs m n k => xs m n k - Real.log (∑ j : Fin N, Real.exp (xs m j k))) :=
  ReaderProof.forward_correct M N K BM retained plan

/-- Complete backward VJP from the supplied saved tensor and upstream tensor. -/
specification backward_grid_correctness (M N K BM : Nat)
    (retained : Grid.RetainedBlock BM) (plan : Plan N (Grid.blockColumns N)) :
    Spec.Real (ReaderIO.backward M N K BM plan ⊨ fun xs m n k => xs.upstream m n k - Real.exp (xs.saved m n k) * ∑ j : Fin N, xs.upstream m j k) :=
  ReaderProof.backward_correct M N K BM retained plan

#axiomsClean forward_grid_correctness
#axiomsClean backward_grid_correctness
end
end VeriTile.Bench.Repairs.LogSoftmax.Correct
