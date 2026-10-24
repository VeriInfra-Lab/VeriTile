import bench.repairs.tritonbench_g.layer_norm_fwd.Kernels
import bench.tritonbench_g.layer_norm_fwd.Proofs.Formula
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole
noncomputable section
set_option maxHeartbeats 2000000

theorem sumTree_real (tree : SumTree block) (x : Fin block → ℝ) :
    tree.eval (·+·) x = (tree.leaves.map x).sum := by
  induction tree with
  | leaf j => simp [SumTree.eval,SumTree.leaves]
  | add a b ha hb => simp [SumTree.eval,SumTree.leaves,ha,hb,List.sum_append]

theorem squareTree_real (tree : SquareTree block) (x : Fin block → ℝ) :
    tree.eval (·*·) (·+·) (fun a b c => a*b+c) x =
      (tree.leaves.map (fun j => (x j)^2)).sum := by
  induction tree with
  | square j => simp [SquareTree.eval,SquareTree.leaves,pow_two]
  | add a b ha hb => simp [SquareTree.eval,SquareTree.leaves,ha,hb,List.sum_append]
  | fmaSquare j b hb => simp [SquareTree.eval,SquareTree.leaves,hb,pow_two]

theorem mean_reduction_real (red : Reductions block) (x : Fin block → ℝ) :
    red.mean.eval (·+·) x = ∑ j : Fin block, x j := by
  rw [sumTree_real,(red.meanComplete.map x).sum_eq]
  rw [← List.ofFn_eq_map,List.sum_ofFn]

theorem variance_reduction_real (red : Reductions block) (x : Fin block → ℝ) :
    red.variance.eval (·*·) (·+·) (fun a b c => a*b+c) x =
      ∑ j : Fin block, (x j)^2 := by
  rw [squareTree_real,(red.varianceComplete.map (fun j => (x j)^2)).sum_eq]
  rw [← List.ofFn_eq_map,List.sum_ofFn]

/-- The mathematical bridge uses only the proved leaf permutation. It never
requires FP addition associativity or a false mul/add-to-FMA identity. -/
theorem mean_padded (red : Reductions block) (x : Nat → ℝ) (hN : N ≤ block) :
    red.mean.eval (·+·) (fun j => if j.val < N then x j.val else 0) =
      ∑ j ∈ Finset.range N, x j := by
  rw [mean_reduction_real]
  rw [Fin.sum_univ_eq_sum_range (fun j => if j < N then x j else 0)]
  exact Formula.sum_padded x hN

theorem variance_padded (red : Reductions block) (x : Nat → ℝ) (hN : N ≤ block) :
    red.variance.eval (·*·) (·+·) (fun a b c => a*b+c)
      (fun j => if j.val < N then x j.val else 0) =
      ∑ j ∈ Finset.range N, (x j)^2 := by
  rw [variance_reduction_real]
  rw [Fin.sum_univ_eq_sum_range (fun j => (if j < N then x j else 0)^2)]
  have pointwise : (fun j => (if j < N then x j else (0 : ℝ))^2) =
      (fun j => if j < N then (x j)^2 else 0) := by
    funext j
    split_ifs <;> simp
  rw [pointwise]
  exact Formula.sum_padded (fun j => (x j)^2) hN

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole
