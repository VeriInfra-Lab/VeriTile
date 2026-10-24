import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathBackwardLoop
namespace VeriTile.Bench.Repairs.LayerNormOps.MathReturn
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 1000000

 theorem update_y_read (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (row : Nat) (a : Backward.Acc ℝ p) (h : LoopInvariant rest p s a)
    (hy : p.recompute=Bool.true) (hx : p.y≠p.dx) (hd : p.y≠p.dresIn)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.update (Logical.algebra rest) p row a).state.mem p.y (row*p.yStride+i.1.val) =
      .mk .fp32 (rowY p s row i.1.val) := by
  unfold Backward.update
  rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hx)]
  have hrec : (Backward.recomputed (Logical.algebra rest) p a row).mem p.y (row*p.yStride+i.1.val) =
      .mk .fp32 (rowY p s row i.1.val) := by
    simp only [Backward.recomputed,hy,if_true]
    rw [Backward.scatter_read _ _ _ _ _ _ i hi,y_value rest p s a h row i hi]
    rfl
  cases hr : p.storeDres
  · simpa only [hr,Bool.false_eq_true,if_false] using hrec
  · simp only [if_true]
    rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hd)]
    exact hrec

 theorem update_y_other (M : Algebra α) [Inhabited α] (p : BackwardParameters)
    (row other : Nat) (a : Backward.Acc α p) (hs : p.columns≤p.yStride)
    (hx : p.y≠p.dx) (hd : p.y≠p.dresIn) (hne : other≠row)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.update M p other a).state.mem p.y (row*p.yStride+i.1.val) =
      a.state.mem p.y (row*p.yStride+i.1.val) := by
  unfold Backward.update
  rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hx)]
  have hrec : (Backward.recomputed M p a other).mem p.y (row*p.yStride+i.1.val) =
      a.state.mem p.y (row*p.yStride+i.1.val) := by
    cases hy : p.recompute
    · simp [Backward.recomputed,hy]
    · simp only [Backward.recomputed,hy,if_true]
      apply Backward.scatter_frame
      right
      intro j hj
      exact Grid.row_offset_ne _ _ _ (Nat.lt_of_lt_of_le hi hs) (Nat.lt_of_lt_of_le hj hs) row other hne.symm
  cases hr : p.storeDres
  · simpa only [hr,Bool.false_eq_true,if_false] using hrec
  · simp only [if_true]
    rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hd)]
    exact hrec

 theorem loop_y_projection (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (fresh : ∀ r ∈ backwardReads p, Backward.RegionOutside p r)
    (hy : p.recompute=Bool.true) (hs : p.columns≤p.yStride)
    (hx : p.y≠p.dx) (hd : p.y≠p.dresIn)
    (row : Nat) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.accumulated (Logical.algebra rest) p s).state.mem p.y (row*p.yStride+i.1.val) =
      foldRange (fun n value => if n=row then Cell.mk .fp32 (rowY p s row i.1.val) else value)
        (Backward.start p s) (Backward.stop p s) 1 (s.mem p.y (row*p.yStride+i.1.val)) := by
  apply Folds.project (Backward.update (Logical.algebra rest) p) _ (LoopInvariant rest p s)
    (fun a => a.state.mem p.y (row*p.yStride+i.1.val))
    (invariant_update rest p s fresh) ?_ _ _ _ (invariant_initial rest p s)
  intro n a ha
  dsimp only
  by_cases hn : n=row
  · subst n
    rw [if_pos rfl,update_y_read rest p s row a ha hy hx hd i hi]
  · rw [if_neg hn,update_y_other _ _ _ _ _ hs hx hd hn i hi]

 theorem loop_y_inside (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (fresh : ∀ r ∈ backwardReads p, Backward.RegionOutside p r)
    (hy : p.recompute=Bool.true) (hs : p.columns≤p.yStride)
    (hx : p.y≠p.dx) (hd : p.y≠p.dresIn)
    (row : Nat) (lo : Backward.start p s≤row) (hirow : row<Backward.stop p s)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.accumulated (Logical.algebra rest) p s).state.mem p.y (row*p.yStride+i.1.val) =
      .mk .fp32 (rowY p s row i.1.val) := by
  rw [loop_y_projection rest p s fresh hy hs hx hd row i hi]
  exact Folds.set_inside _ _ _ _ _ lo hirow

 theorem loop_y_outside (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (fresh : ∀ r ∈ backwardReads p, Backward.RegionOutside p r)
    (hy : p.recompute=Bool.true) (hs : p.columns≤p.yStride)
    (hx : p.y≠p.dx) (hd : p.y≠p.dresIn)
    (row : Nat) (outside : row<Backward.start p s ∨ Backward.stop p s≤row)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.accumulated (Logical.algebra rest) p s).state.mem p.y (row*p.yStride+i.1.val) =
      s.mem p.y (row*p.yStride+i.1.val) := by
  rw [loop_y_projection rest p s fresh hy hs hx hd row i hi]
  exact Folds.set_outside _ _ _ _ _ outside

 theorem finish_region_frame (M : Algebra α) (p : BackwardParameters) (pid : Nat)
    (a : Backward.Acc α p) (r : RegionName) (o : Nat) (hw : r≠p.dw) (hb : r≠p.db) :
    (Backward.finish M p pid a).mem r o=a.state.mem r o := by
  unfold Backward.finish
  cases h : p.hasBias
  · simpa only [h,Bool.false_eq_true,if_false] using Backward.scatter_frame p pid p.dw p.columns _ a.state r o (Or.inl hw)
  · simp only [if_true]
    rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hb),
      Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hw)]

#axiomsClean loop_y_inside
#axiomsClean loop_y_outside
end
end VeriTile.Bench.Repairs.LayerNormOps.MathReturn
