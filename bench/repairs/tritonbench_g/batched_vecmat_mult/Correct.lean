import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.LogicalReader
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.AdmissionCore

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.Correctness
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open VecmatIO
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

/-- Every output of the complete selected grid is the full-K dot product.
Named inputs retain arbitrary source graph operands and their mathematical
word decoding, together with all source windows, safety and exterior frame. -/
specification batched_vecmat_grid_real (decode : ValueProofs.Word → ℝ)
    (rest : Algebra SourceGraph.Expr) (profile : KernelProfile.Selector) (p : Parameters)
    (_index : Whole.IntegerAgreement p) (separate : SourceGraph.InputsSeparate p)
    (hBM : 0 < p.BM) (hBN : 0 < p.BN)
    (hM : p.BM ∣ p.M) (hN : p.BN ∣ p.N) (hK : p.BK ∣ p.K) :
    Spec.Real (logicalIO decode rest profile p ⊨ fun xs m n => ∑ k : Fin p.K,
      SourceGraph.realValue decode (xs.a m k) * SourceGraph.realValue decode (xs.b m n k)) :=
  logical_implements decode rest profile p separate hBM hBN hM hN hK

#axiomsClean batched_vecmat_grid_real
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.Correctness
