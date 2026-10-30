import bench.repairs.tritonbench_g.batched_vecmat_mult.IO
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.VecmatIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open SourceGraph (Expr)
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

theorem logical_iff (decode : ValueProofs.Word → ℝ) (rest : Algebra Expr)
    (profile : KernelProfile.Selector) (p : Parameters) :
    (logicalIO decode rest profile p ⊨ fun xs m n => ∑ k : Fin p.K,
      SourceGraph.realValue decode (xs.a m k) * SourceGraph.realValue decode (xs.b m n k)) ↔
    (graphIO rest profile p).Observes (realObservation decode p) (realTarget decode p) := by
  change (graphIO rest profile p).Observes (realObservation decode p)
    (fun vals => (logicalOutputsEquiv p).symm (fun m n => ∑ k : Fin p.K,
      SourceGraph.realValue decode ((logicalInputsEquiv p vals).a m k) *
      SourceGraph.realValue decode ((logicalInputsEquiv p vals).b m n k))) ↔ _
  have eq : (fun vals => (logicalOutputsEquiv p).symm (fun m n => ∑ k : Fin p.K,
      SourceGraph.realValue decode ((logicalInputsEquiv p vals).a m k) *
      SourceGraph.realValue decode ((logicalInputsEquiv p vals).b m n k))) = realTarget decode p := by
    funext vals o
    fin_cases o
    rfl
  exact congrArg (fun f => (graphIO rest profile p).Observes (realObservation decode p) f) eq |>.to_iff

theorem logical_implements (decode : ValueProofs.Word → ℝ) (rest : Algebra Expr)
    (profile : KernelProfile.Selector) (p : Parameters)
    (separate : SourceGraph.InputsSeparate p) (hBM : 0<p.BM) (hBN : 0<p.BN)
    (hM : p.BM∣p.M) (hN : p.BN∣p.N) (hK : p.BK∣p.K) :
    logicalIO decode rest profile p ⊨ fun xs m n => ∑ k : Fin p.K,
      SourceGraph.realValue decode (xs.a m k) * SourceGraph.realValue decode (xs.b m n k) :=
  (logical_iff decode rest profile p).mpr
    (IOCorrect.real decode rest profile p separate hBM hBN hM hN hK)

#axiomsClean logical_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.VecmatIO
