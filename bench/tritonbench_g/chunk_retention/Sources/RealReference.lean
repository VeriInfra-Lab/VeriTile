import bench.tritonbench_g.chunk_retention.Sources.RealBody
import bench.tritonbench_g.chunk_retention.Sources.Whole

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
open scoped BigOperators
namespace RealScalar

theorem row_split (row : Nat) : row/64*64+row%64=row := by
  simpa [Nat.mul_comm] using Nat.div_add_mod row 64

theorem reference_output (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (h row j : Nat)
    (hr : row<c.T) (hj : j<c.V) :
    Reference.output realOps c xs scale h row j=MathTarget.output c xs scale Bool.true h row j := by
  have hc := c.chunk_of_row row hr
  have hi := row_split row
  have hl : row%64<64 := Nat.mod_lt _ (by decide)
  unfold Reference.output
  simp only [real_cast]
  rw [output_real c xs scale h (row/64) (row%64) j _ (by omega) hl hj]
  calc
    _ = ScalarSource.outputTile c xs scale h (row/64) (row%64) j
        (ScalarSource.forwardCarry c xs Bool.true h (row/64)) := by
      unfold ScalarSource.outputTile
      congr 2
      apply Finset.sum_congr rfl
      intro e he
      dsimp only
      rw [history_scalar c xs h (row/64) e j (Nat.le_of_lt hc) (Finset.mem_range.mp he) hj]
    _ = _ := by rw [ScalarSource.output_correct c xs scale Bool.true h (row/64) (row%64) j hc,hi]

theorem reference_query (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (h row e : Nat)
    (hr : row<c.T) (he : e<c.K) :
    Reference.dq realOps c xs scale h row e=MathTarget.queryGradient c xs scale Bool.true h row e := by
  have hc := c.chunk_of_row row hr
  have hi := row_split row
  have hl : row%64<64 := Nat.mod_lt _ (by decide)
  unfold Reference.dq
  simp only [real_cast]
  rw [query_real c xs scale h (row/64) (row%64) e _ (by omega) hl he]
  calc
    _ = ScalarSource.queryTile c xs scale h (row/64) (row%64) e
        (ScalarSource.forwardCarry c xs Bool.true h (row/64)) := by
      unfold ScalarSource.queryTile
      congr 2
      apply Finset.sum_congr rfl
      intro j hj
      dsimp only
      rw [history_scalar c xs h (row/64) e j (Nat.le_of_lt hc) he (Finset.mem_range.mp hj)]
    _ = _ := by rw [ScalarSource.query_correct c xs scale Bool.true h (row/64) (row%64) e hc,hi]

theorem reference_key (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (h row e : Nat)
    (hr : row<c.T) (he : e<c.K) :
    Reference.dk realOps c xs scale h row e=MathTarget.keyGradient c xs scale h row e := by
  have hc := c.chunk_of_row row hr
  have hi := row_split row
  have hl := (time_lane c (row/64) (row%64) hc (Nat.mod_lt _ (by decide))).mp (by omega)
  unfold Reference.dk
  simp only [real_cast]
  rw [key_real c xs scale h (row/64) (row%64) e _ hc hl he]
  calc
    _ = ScalarSource.keyTile c xs scale h (row/64) (row%64) e
        (ScalarSource.reverseCarry c xs scale h (c.chunks-1-row/64)) := by
      unfold ScalarSource.keyTile
      congr 2
      apply Finset.sum_congr rfl
      intro j hj
      exact congrArg (fun x=>xs.v h (row/64*64+row%64) j*x)
        (reverse_scalar c xs scale h (c.chunks-1-row/64) e j (by omega) he (Finset.mem_range.mp hj))
    _ = _ := by rw [ScalarSource.key_correct c xs scale h (row/64) (row%64) e hc hl,hi]

theorem reference_partial (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (block h row j : Nat)
    (hr : row<c.T) (hj : j<c.V) :
    Reference.dvPart realOps c xs scale block h row j=
      ScalarSource.valuePartial c xs scale block h (row/64) (row%64) j
        (ScalarSource.reverseCarry c xs scale h (c.chunks-1-row/64)) := by
  have hc := c.chunk_of_row row hr
  have hi := row_split row
  have hl := (time_lane c (row/64) (row%64) hc (Nat.mod_lt _ (by decide))).mp (by omega)
  unfold Reference.dvPart
  simp only [real_cast]
  rw [value_real c xs scale block h (row/64) (row%64) j _ hc hl hj]
  unfold ScalarSource.valuePartial ScalarSource.localCoefficient
  apply Finset.sum_congr rfl
  intro e he
  by_cases hk : block*c.BK+e<c.K
  · simp only [if_pos hk,Reference.dhistory]
    rw [reverse_scalar c xs scale h (c.chunks-1-row/64) (block*c.BK+e) j (by omega) hk hj]
  · simp only [if_neg hk]

theorem tree_leaves (tree : SumTree) (values : Nat→ℝ) :
    tree.eval realOps values=(tree.leaves.map values).sum := by
  induction tree with
  | zero => rfl
  | leaf i => simp [SumTree.eval,SumTree.leaves]
  | add a b ia ib => simp [SumTree.eval,SumTree.leaves,real_add,ia,ib]

theorem tree_complete (tree : SumTree) (n : Nat) (values : Nat→ℝ) (complete : tree.Complete n) :
    tree.eval realOps values=∑ i∈Finset.range n,values i := by
  rw [tree_leaves,(complete.map values).sum_eq]
  clear complete
  induction n with
  | zero => simp
  | succ n ih => simp [List.range_succ,Finset.sum_range_succ,ih]

theorem reference_value (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (tree : SumTree)
    (complete : tree.Complete c.keyBlocks) (h row j : Nat) (hr : row<c.T) (hj : j<c.V) :
    Reference.dv realOps c xs scale tree h row j=MathTarget.valueGradient c xs scale h row j := by
  rw [Reference.dv,tree_complete tree c.keyBlocks _ complete]
  simp_rw [reference_partial c xs scale _ h row j hr hj]
  have hc := c.chunk_of_row row hr
  have hi := row_split row
  have hl := (time_lane c (row/64) (row%64) hc (Nat.mod_lt _ (by decide))).mp (by omega)
  exact (ScalarSource.value_correct c xs scale h (row/64) (row%64) j hc hl).trans
    (congrArg (fun i=>MathTarget.valueGradient c xs scale h i j) hi)

theorem reference_final (c : Shape) (xs : Inputs ℝ) (h e j : Nat)
    (he : e<c.K) (hj : j<c.V) :
    Reference.finalState realOps c xs h e j=MathTarget.finalState c xs Bool.true h e j := by
  rw [Reference.finalState,real_cast,history_scalar c xs h c.chunks e j le_rfl he hj,
    ScalarSource.final_state_correct]

#axiomsClean reference_output
#axiomsClean reference_query
#axiomsClean reference_key
#axiomsClean reference_value
#axiomsClean reference_final
end RealScalar
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
