import bench.tritonbench_g.batched_vecmat_mult.Proofs.SourceGraphTensor
import Mathlib.Algebra.BigOperators.Fin

/-! Independent mathematical meaning of every selected graph. Tree validity
supplies each K lane exactly once; contractions preserve Real multiplication
and addition, while no FP-word equality is inferred from this projection. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
open VeriTile.Triton FP.Structural FP.Equational
noncomputable section

private theorem tree_real_leaves (decode : ValueProofs.Word → ℝ) (xs : Fin n → Expr)
    (tree : ReductionTree n) :
    realValue decode (treeValue xs tree) =
      (tree.leaves.map (fun e => e.elim 0 (fun i => realValue decode (xs i)))).sum := by
  induction tree with
  | input i => simp [treeValue, ReductionTree.leaves]
  | zero => simp [treeValue, ReductionTree.leaves, realValue]
  | add a b ha hb => simp [treeValue, ReductionTree.leaves, realValue, ha, hb]

theorem reduction_real_sum (decode : ValueProofs.Word → ℝ) (xs : Fin n → Expr)
    (plan : ReductionPlan n) :
    realValue decode (treeValue xs plan.tree) = ∑ i : Fin n, realValue decode (xs i) := by
  rw [tree_real_leaves]
  have h := (plan.valid.map (fun e => e.elim 0 (fun i => realValue decode (xs i)))).sum_eq
  simpa using h

/-- Independent batched vector-matrix formula on actual source input cells.
Whole K blocks are the original wrapper guard; the sum contains every covered
K element and no execution, output-memory or assumed accumulator equation. -/
def realTensorExpected (decode : ValueProofs.Word → ℝ) (p : Parameters)
    (mem : Memory) (m n count : Nat) : ℝ :=
  ∑ c : Fin count, ∑ e : Fin p.BK,
    realValue decode ((mem p.A (m*p.K+(c.val*p.BK+e.val))).read .real) *
    realValue decode ((mem p.B (m*p.N*p.K+n*p.K+(c.val*p.BK+e.val))).read .real)

theorem selected_graph_real_correct (decode : ValueProofs.Word → ℝ) (plans : PlanSelector)
    (p : Parameters) (mem : Memory) (m n count : Nat) :
    realValue decode (contract (tensorValue plans p mem m n count)) =
      realTensorExpected decode p mem m n count := by
  rw [contraction_preserves_real]
  induction count with
  | zero => simp [tensorValue, realValue, realTensorExpected]
  | succ count ih =>
    rw [tensorValue, realValue, ih]
    unfold tensorBlock
    rw [reduction_real_sum]
    simp only [realValue]
    simp [realTensorExpected, Fin.sum_univ_castSucc]

#specNonCircular realTensorExpected avoiding [tensorValue, expectedMemory, source, FP.Structural.exec]
#axiomsClean reduction_real_sum
#axiomsClean selected_graph_real_correct
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
