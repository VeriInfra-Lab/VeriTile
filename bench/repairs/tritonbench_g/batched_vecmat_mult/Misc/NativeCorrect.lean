import bench.repairs.tritonbench_g.batched_vecmat_mult.IO
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.IOCorrect
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.AdmissionCore

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.FPEquivalence
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open VecmatIO
noncomputable section

/-- Complete source-to-native observation under the retained FP32 reduction
and contraction model. Arbitrary input expressions include address-tagged
words and all prior typed-read defaults. The K loop retains floor division;
no K-divisibility or new floating transformation law is required. -/
theorem batched_vecmat_grid_native (cfg : FP.Config)
    (rest : Algebra SourceGraph.Expr) (profile : KernelProfile.Selector) (p : Parameters)
    (_index : Whole.IntegerAgreement p) (separate : SourceGraph.InputsSeparate p) (hBM : 0<p.BM) (hBN : 0<p.BN)
    (hM : p.BM∣p.M) (hN : p.BN∣p.N) :
    (graphIO rest profile p).Observes
      (nativeObservation cfg p) (nativeGraphTarget cfg profile p) :=
  ProgramIO.Observes.of_implements (IOCorrect.graph rest profile p separate hBM hBN hM hN) _

/- This is a primitive-model observation theorem. Its fixed graph contraction
is part of the native interpretation, not an assumed FP algebraic rewrite. -/
#specNonCircular graphTarget avoiding [source,FP.Structural.exec,SourceGraph.runGrid]
#axiomsClean batched_vecmat_grid_native
#stmtSurfaceSubset batched_vecmat_grid_native ⊆
  [ProgramIO.Observes, graphIO, nativeObservation, nativeGraphTarget,
   SourceGraph.instInhabitedExpr, SourceGraph.Expr, SourceGraph.InputsSeparate, ValueProofs.Word,
   FP.Config, FP.Structural.Algebra, Whole.IntegerAgreement, KernelProfile.Selector, Parameters, ValueProofs.Parameters,
   ValueProofs.Parameters.BM, ValueProofs.Parameters.BN, ValueProofs.Parameters.M, ValueProofs.Parameters.N]
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.FPEquivalence
