import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.FiniteIndex
import bench.tritonbench_g.batched_vecmat_mult.Proofs.Real
import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphReal

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.RealProofs
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
noncomputable section
local instance : Inhabited SourceGraph.Expr := ⟨SourceGraph.Expr.zero⟩

/-- Independent full logical K sum, with no tile or program identifier. -/
def tensorExpected (decode : ValueProofs.Word → ℝ) (p : Parameters)
    (mem : SourceGraph.Memory) (m n : Nat) : ℝ :=
  ∑ k : Fin p.K,
    SourceGraph.realValue decode ((mem p.A (m*p.K+k.val)).read .real) *
    SourceGraph.realValue decode ((mem p.B (m*p.N*p.K+n*p.K+k.val)).read .real)

theorem blocked_sum_eq_sum (K BK : Nat) (hK : BK ∣ K) (f : Nat → ℝ) :
    (∑ c : Fin (K/BK), ∑ e : Fin BK, f (c.val*BK+e.val)) = ∑ k : Fin K, f k.val := by
  let equiv : Fin (K/BK) × Fin BK ≃ Fin K :=
    finProdFinEquiv.trans (finCongr (Nat.div_mul_cancel hK))
  have h := equiv.sum_comp (fun k : Fin K => f k.val)
  change (∑ q : Fin (K/BK) × Fin BK, f (q.2.val+BK*q.1.val)) = _ at h
  simpa [Fintype.sum_prod_type, Nat.mul_comm, Nat.add_comm] using h

theorem blocked_expected_eq_full (decode : ValueProofs.Word → ℝ) (p : Parameters)
    (mem : SourceGraph.Memory) (m n : Nat) (hK : p.BK ∣ p.K) :
    SourceGraph.realTensorExpected decode p mem m n (p.K/p.BK) = tensorExpected decode p mem m n := by
  exact blocked_sum_eq_sum p.K p.BK hK (fun k =>
    SourceGraph.realValue decode ((mem p.A (m*p.K+k)).read .real) *
    SourceGraph.realValue decode ((mem p.B (m*p.N*p.K+n*p.K+k)).read .real))

/-- Full source grid, actual stored output, independent mathematical block sum,
and complete memory frame. Whole-block K covers exactly 0..K-1 by the reused
`complete_inner_coverage`; no final-output equation is a premise. -/
theorem grid_real_correct (decode : ValueProofs.Word → ℝ)
    (rest : Algebra SourceGraph.Expr) (plans : SourceGraph.PlanSelector)
    (p : Parameters) (initial : State SourceGraph.Expr)
    (separate : SourceGraph.InputsSeparate p)
    (hBM : 0 < p.BM) (hBN : 0 < p.BN) (hM : p.BM ∣ p.M) (hN : p.BN ∣ p.N)
    (hK : p.BK ∣ p.K) :
    ∃ final, SourceGraph.runGrid rest plans p initial = some final ∧
      (∀ m : Fin p.M, ∀ n : Fin p.N,
        SourceGraph.realValue decode (SourceGraph.contract
          ((final.mem p.output (m.val*p.N+n.val)).read .real)) =
          tensorExpected decode p initial.mem m.val n.val) ∧
      (∀ r o, SourceGraph.GridOutside p r o → final.mem r o = initial.mem r o) := by
  obtain ⟨final,he,hm,hf⟩ := SourceGraph.grid_correct rest plans p initial
  refine ⟨final,he,?_,hf⟩
  intro m n
  rw [hm, ← SourceGraph.complete_initial_memory_eq_grid plans p initial separate,
    SourceGraph.complete_tensor_readback plans p initial hBM hBN hM hN m n]
  exact (SourceGraph.selected_graph_real_correct decode plans p initial.mem m.val n.val
    (p.K/p.BK)).trans (blocked_expected_eq_full decode p initial.mem m.val n.val hK)

#specNonCircular tensorExpected avoiding [kernel, FP.Structural.exec]
#axiomsClean blocked_sum_eq_sum
#axiomsClean grid_real_correct
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.RealProofs
