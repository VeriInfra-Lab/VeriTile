import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.FiniteExecution

/-! Complete raw-state source and finite-index execution, independent of any
Tensor object or wrapper. The operand graph reads only the initial memory. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelCore
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
noncomputable section
local instance : Inhabited SourceGraph.Expr := ⟨SourceGraph.Expr.zero⟩

theorem graph_complete (rest : Algebra SourceGraph.Expr) (plans : SourceGraph.PlanSelector)
    (p : Parameters) (initial : State SourceGraph.Expr) (index : Whole.IntegerAgreement p)
    (separate : SourceGraph.InputsSeparate p) (hBM : 0 < p.BM) (hBN : 0 < p.BN)
    (hM : p.BM ∣ p.M) (hN : p.BN ∣ p.N) :
    ∃ final, SourceGraph.runGrid rest plans p initial=some final ∧
      FiniteExecution.grid plans p initial=some final.mem ∧
      (∀ m : Fin p.M, ∀ n : Fin p.N, final.mem p.output (m.val*p.N+n.val)=
        .mk .real (SourceGraph.tensorValue plans p initial.mem m.val n.val (p.K/p.BK))) ∧
      (∀ r i, SourceGraph.GridOutside p r i → final.mem r i=initial.mem r i) := by
  obtain ⟨final,he,hm,hf⟩ := SourceGraph.grid_correct rest plans p initial
  refine ⟨final,he,?_,?_,hf⟩
  · rw [FiniteExecution.grid_source_execution rest plans p initial index,he]
    rfl
  · intro m n
    rw [hm,←SourceGraph.complete_initial_memory_eq_grid plans p initial separate]
    exact SourceGraph.complete_tensor_readback plans p initial hBM hBN hM hN m n

#axiomsClean graph_complete
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelCore
