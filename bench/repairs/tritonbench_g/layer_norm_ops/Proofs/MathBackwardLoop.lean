import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathForward
namespace VeriTile.Bench.Repairs.LayerNormOps.MathReturn
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.LayerNormLiger
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 1000000

def backwardReads (p : BackwardParameters) : List RegionName := [p.x,p.dy,p.mean,p.rstd,p.dres]
def readRow (s : State ℝ) (r : RegionName) (stride row col : Nat) : ℝ := (s.mem r (row*stride+col)).read .fp32

def rowDX (p : BackwardParameters) (s : State ℝ) (row col : Nat) : ℝ :=
  Logical.dx p.columns p.rms p.hasDres (readRow s p.x p.xStride row)
    (fun j => (s.mem p.weight j).read .fp32) (readRow s p.dy p.dyStride row)
    (readRow s p.dres p.dresStride row) ((s.mem p.mean (row*p.meanStride)).read .fp32)
    ((s.mem p.rstd (row*p.rstdStride)).read .fp32) col

def rowY (p : BackwardParameters) (s : State ℝ) (row col : Nat) : ℝ :=
  (readRow s p.x p.xStride row col - if p.rms then 0 else (s.mem p.mean (row*p.meanStride)).read .fp32) *
    (s.mem p.rstd (row*p.rstdStride)).read .fp32 * (s.mem p.weight col).read .fp32 +
    if p.hasBias then (s.mem p.bias col).read .fp32 else 0

structure LoopInvariant (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ) (a : Backward.Acc ℝ p) : Prop where
  reads : ∀ r ∈ backwardReads p, a.state.mem r = s.mem r
  weight : a.weight = (Backward.initial (Logical.algebra rest) p s).weight
  bias : a.bias = (Backward.initial (Logical.algebra rest) p s).bias

 theorem invariant_initial (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ) :
    LoopInvariant rest p s (Backward.initial (Logical.algebra rest) p s) :=
  ⟨fun _ _ => rfl,rfl,rfl⟩

 theorem invariant_update (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (fresh : ∀ r ∈ backwardReads p, Backward.RegionOutside p r)
    (row : Nat) (a : Backward.Acc ℝ p) (h : LoopInvariant rest p s a) :
    LoopInvariant rest p s (Backward.update (Logical.algebra rest) p row a) := by
  refine ⟨?_,h.weight,h.bias⟩
  intro r hr
  funext offset
  rw [Backward.update_frame _ _ _ _ _ _ (fresh r hr)]
  exact congrFun (h.reads r hr) offset

 theorem dx_value_active (rest : Algebra ℝ) (p : BackwardParameters) (a : Backward.Acc ℝ p)
    (row : Nat) (w : Nat → ℝ) (hw : ∀ i : TileIndex [p.block], i.1.val<p.columns → a.weight i=w i.1.val)
    (hb : p.columns≤p.block) (i : TileIndex [p.block]) (hi : i.1.val<p.columns)
    (fresh : p.recompute=Bool.true → p.dres≠p.y) :
    Backward.dxValue (Logical.algebra rest) p a row i =
      Logical.dx p.columns p.rms p.hasDres (Logical.bx p a row) w (Logical.bdy p a row)
        (fun j => (a.state.mem p.dres (row*p.dresStride+j)).read .fp32)
        (Logical.bmu p a row) (Logical.binv p a row) i.1.val := by
  have hf : (Backward.recomputed (Logical.algebra rest) p a row).mem p.dres (row*p.dresStride+i.1.val) =
      a.state.mem p.dres (row*p.dresStride+i.1.val) := by
    cases hy : p.recompute
    · simp [Backward.recomputed,hy]
    · simp only [Backward.recomputed,hy,if_true]
      exact Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl (fresh hy))
  unfold Backward.dxValue
  rw [Logical.backward_dxBase rest p a row w hw hb i hi]
  cases hd : p.hasDres <;>
    simp [hd,Backward.loaded,hi,hf,Forward.cast,Forward.binary,
      Logical.algebra,LayerNormLiger.Logical.algebra,Logical.dx]
  all_goals exact congrArg (Cell.read .fp32) hf

 theorem dx_value (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (a : Backward.Acc ℝ p) (h : LoopInvariant rest p s a) (row : Nat)
    (hb : p.columns≤p.block) (i : TileIndex [p.block]) (hi : i.1.val<p.columns)
    (hdy : p.recompute=Bool.true → p.dres≠p.y) :
    Backward.dxValue (Logical.algebra rest) p a row i = rowDX p s row i.1.val := by
  have hw : ∀ j : TileIndex [p.block], j.1.val<p.columns → a.weight j=(s.mem p.weight j.1.val).read .fp32 := by
    intro j hj
    rw [h.weight]
    simp [Backward.initial,Backward.loaded,Forward.cast,Logical.algebra,LayerNormLiger.Logical.algebra,hj]
  rw [dx_value_active rest p a row (fun j => (s.mem p.weight j).read .fp32) hw hb i hi hdy]
  simp only [rowDX,Logical.dx,Logical.bx,Logical.bdy,Logical.bmu,Logical.binv,readRow,
    h.reads p.x (by simp [backwardReads]),h.reads p.dy (by simp [backwardReads]),
    h.reads p.mean (by simp [backwardReads]),h.reads p.rstd (by simp [backwardReads]),
    h.reads p.dres (by simp [backwardReads])]

 theorem y_value (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (a : Backward.Acc ℝ p) (h : LoopInvariant rest p s a) (row : Nat)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    Backward.yValue (Logical.algebra rest) p a row i = rowY p s row i.1.val := by
  unfold Backward.yValue
  rw [Logical.backward_xhat,h.weight,h.bias]
  cases hb : p.hasBias <;>
    simp [rowY,Logical.bhat,Logical.bx,Logical.bmu,Logical.binv,readRow,
      Backward.initial,Backward.loaded,Forward.cast,Forward.unhintedBinary,Backward.binary,Forward.binary,
      Logical.algebra,LayerNormLiger.Logical.algebra,hi,hb,
      h.reads p.x (by simp [backwardReads]),h.reads p.mean (by simp [backwardReads]),
      h.reads p.rstd (by simp [backwardReads])]

 theorem update_region_frame (M : Algebra α) [Inhabited α] (p : BackwardParameters)
    (row : Nat) (a : Backward.Acc α p) (r : RegionName) (o : Nat)
    (hx : r≠p.dx) (hy : r≠p.y) (hr : r≠p.dresIn) :
    (Backward.update M p row a).state.mem r o=a.state.mem r o := by
  unfold Backward.update
  rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hx)]
  have hrec : (Backward.recomputed M p a row).mem r o=a.state.mem r o := by
    cases h : p.recompute
    · simp [Backward.recomputed,h]
    · simpa only [Backward.recomputed,h,if_true] using
        Backward.scatter_frame p row p.y p.yStride _ a.state r o (Or.inl hy)
  cases h : p.storeDres
  · simpa only [h,if_false,Bool.false_eq_true] using hrec
  · simp only [if_true]
    rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hr)]
    exact hrec

 theorem update_dx_other (M : Algebra α) [Inhabited α] (p : BackwardParameters)
    (row other : Nat) (a : Backward.Acc α p) (hs : p.columns≤p.dxStride)
    (hy : p.dx≠p.y) (hr : p.dx≠p.dresIn) (hne : other≠row)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.update M p other a).state.mem p.dx (row*p.dxStride+i.1.val)=
      a.state.mem p.dx (row*p.dxStride+i.1.val) := by
  unfold Backward.update
  rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inr ?_)]
  · have hrec : (Backward.recomputed M p a other).mem p.dx (row*p.dxStride+i.1.val)=
        a.state.mem p.dx (row*p.dxStride+i.1.val) := by
      cases h : p.recompute
      · simp [Backward.recomputed,h]
      · simpa only [Backward.recomputed,h,if_true] using
          Backward.scatter_frame p other p.y p.yStride _ a.state _ _ (Or.inl hy)
    cases h : p.storeDres
    · simpa only [h,if_false,Bool.false_eq_true] using hrec
    · simp only [if_true]
      rw [Backward.scatter_frame _ _ _ _ _ _ _ _ (Or.inl hr)]
      exact hrec
  · intro j hj
    exact Grid.row_offset_ne _ _ _ (Nat.lt_of_lt_of_le hi hs) (Nat.lt_of_lt_of_le hj hs) row other hne.symm

 theorem loop_dx_projection (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (fresh : ∀ r ∈ backwardReads p, Backward.RegionOutside p r)
    (hb : p.columns≤p.block) (hs : p.columns≤p.dxStride)
    (hy : p.dx≠p.y) (hr : p.dx≠p.dresIn) (hdy : p.recompute=Bool.true → p.dres≠p.y)
    (row : Nat) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.accumulated (Logical.algebra rest) p s).state.mem p.dx (row*p.dxStride+i.1.val) =
      foldRange (fun n value => if n=row then Cell.mk .fp32 (rowDX p s row i.1.val) else value)
        (Backward.start p s) (Backward.stop p s) 1 (s.mem p.dx (row*p.dxStride+i.1.val)) := by
  apply Folds.project (Backward.update (Logical.algebra rest) p) _ (LoopInvariant rest p s)
    (fun a => a.state.mem p.dx (row*p.dxStride+i.1.val))
    (invariant_update rest p s fresh) ?_ _ _ _ (invariant_initial rest p s)
  intro n a ha
  dsimp only
  by_cases hn : n=row
  · subst n
    rw [if_pos rfl,Backward.update_dx _ _ _ _ i hi,dx_value rest p s a ha row hb i hi hdy]
    rfl
  · rw [if_neg hn,update_dx_other _ _ _ _ _ hs hy hr hn i hi]

 theorem loop_dx_inside (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (fresh : ∀ r ∈ backwardReads p, Backward.RegionOutside p r)
    (hb : p.columns≤p.block) (hs : p.columns≤p.dxStride)
    (hy : p.dx≠p.y) (hr : p.dx≠p.dresIn) (hdy : p.recompute=Bool.true → p.dres≠p.y)
    (row : Nat) (lo : Backward.start p s≤row) (hirow : row<Backward.stop p s)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.accumulated (Logical.algebra rest) p s).state.mem p.dx (row*p.dxStride+i.1.val) =
      .mk .fp32 (rowDX p s row i.1.val) := by
  rw [loop_dx_projection rest p s fresh hb hs hy hr hdy row i hi]
  exact Folds.set_inside _ _ _ _ _ lo hirow

 theorem loop_dx_outside (rest : Algebra ℝ) (p : BackwardParameters) (s : State ℝ)
    (fresh : ∀ r ∈ backwardReads p, Backward.RegionOutside p r)
    (hb : p.columns≤p.block) (hs : p.columns≤p.dxStride)
    (hy : p.dx≠p.y) (hr : p.dx≠p.dresIn) (hdy : p.recompute=Bool.true → p.dres≠p.y)
    (row : Nat) (outside : row<Backward.start p s ∨ Backward.stop p s≤row)
    (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    (Backward.accumulated (Logical.algebra rest) p s).state.mem p.dx (row*p.dxStride+i.1.val) =
      s.mem p.dx (row*p.dxStride+i.1.val) := by
  rw [loop_dx_projection rest p s fresh hb hs hy hr hdy row i hi]
  exact Folds.set_outside _ _ _ _ _ outside

#axiomsClean dx_value
#axiomsClean y_value
#axiomsClean loop_dx_inside
#axiomsClean loop_dx_outside
end
end VeriTile.Bench.Repairs.LayerNormOps.MathReturn
