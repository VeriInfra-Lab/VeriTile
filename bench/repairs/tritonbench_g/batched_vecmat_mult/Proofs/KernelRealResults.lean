import bench.repairs.tritonbench_g.batched_vecmat_mult.Kernels
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelCore
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelProfile
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.Real

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelRealResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩
local instance : Inhabited SourceGraph.Expr := ⟨SourceGraph.Expr.zero⟩

/-- Full-K independent mathematical dot products from raw source-graph memory,
complete actual source/finite-index execution, and the full tagged frame. -/
theorem batched_vecmat_grid_real (decode : ValueProofs.Word → ℝ)
    (rest : Algebra SourceGraph.Expr) (profile : KernelProfile.Selector)
    (p : Parameters) (initial : State SourceGraph.Expr) (index : Whole.IntegerAgreement p)
    (separate : SourceGraph.InputsSeparate p) (hBM : 0 < p.BM) (hBN : 0 < p.BN)
    (hM : p.BM ∣ p.M) (hN : p.BN ∣ p.N) (hK : p.BK ∣ p.K) :
    Spec.Real (∃ final, SourceGraph.runGrid rest (KernelProfile.toPlans profile) p initial=some final ∧
      FiniteExecution.grid (KernelProfile.toPlans profile) p initial=some final.mem ∧
      (∀ m : Fin p.M, ∀ n : Fin p.N,
        SourceGraph.realValue decode (SourceGraph.contract
          ((final.mem p.output (m.val*p.N+n.val)).read .real))=
          RealProofs.tensorExpected decode p initial.mem m.val n.val) ∧
      (∀ r i, SourceGraph.GridOutside p r i → final.mem r i=initial.mem r i)) := by
  obtain ⟨final,he,hv,hf⟩ := RealProofs.grid_real_correct decode rest
    (KernelProfile.toPlans profile) p initial separate hBM hBN hM hN hK
  refine ⟨final,he,?_,hv,hf⟩
  rw [FiniteExecution.grid_source_execution rest (KernelProfile.toPlans profile) p initial index,he]
  rfl

#specNonCircular RealProofs.tensorExpected avoiding [source,FP.Structural.exec,SourceGraph.runGrid]
#axiomsClean batched_vecmat_grid_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelRealResults
